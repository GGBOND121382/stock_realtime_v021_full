#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live inference for AS1455 Chapter 17 checkpoint ensemble.

This reproduces the scaler/model semantics of scripts/run_ashare_ch17_nn_reproduce.py:
for each selected fold, fit StandardScaler on that fold's X_train from model_data,
transform the live feature matrix, load the selected ckpt_{fold}_{epoch}.weights.h5,
and predict.  For live dates that are outside historical CV windows, the default is
mean_all_folds.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ['dollar_vol', 'dollar_vol_rank', 'rsi', 'bb_high', 'bb_low', 'NATR', 'ATR', 'PPO', 'MACD', 'sector', 'r01', 'r05', 'r10', 'r21', 'r42', 'r63', 'r01dec', 'r05dec', 'r10dec', 'r21dec', 'r42dec', 'r63dec', 'r01q_sector', 'r05q_sector', 'r10q_sector', 'r21q_sector', 'r42q_sector', 'r63q_sector', 'year', 'month', 'weekday']
EXPECTED_OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]
# The training script's HDF column order is:
#   28 feature columns, 3 forward-label columns, then year/month/weekday.
# After pop/drop of the forward labels, X_cv is exactly FEATURE_COLUMNS.
# Do not require a full-column order match here; select by column name to remain
# compatible with both training HDF order and live feature order.
REQUIRED_MODEL_DATA_COLUMNS = FEATURE_COLUMNS + EXPECTED_OUTCOMES
N_SPLITS = 7
TRAIN_PERIOD_LENGTH = 21 * 12 * 4
TEST_PERIOD_LENGTH = 21 * 3
LOOKAHEAD = 1


class MultipleTimeSeriesCV:
    def __init__(self, n_splits: int = 3, train_period_length: int = 126, test_period_length: int = 21, lookahead: int | None = None, date_idx: str = "date", shuffle: bool = False) -> None:
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X: pd.DataFrame, y: pd.Series | None = None, groups: Any = None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        required_days = self.train_length + self.lookahead + self.n_splits * self.test_length
        if len(days) < required_days:
            raise RuntimeError(f"not enough dates for CV folds: need {required_days}, got {len(days)}")
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])
        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > days[train_start]) & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start]) & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()


def normalize_symbol(value: object) -> str:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")
    s = s.replace("sz", "").replace("sh", "") if re.fullmatch(r"(?i)(sz|sh)\d{6}", s) else s
    m = re.search(r"(\d{6})", s)
    if m:
        code = m.group(1)
    elif re.fullmatch(r"\d{1,6}", s):
        code = s.zfill(6)
    else:
        return s.upper()
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def parse_as_date_series(values: pd.Series) -> pd.Series:
    """Parse live date values robustly.

    Handles both ISO dates (2026-06-26) and compact A-share dates
    (20260626).  Plain integers must not be interpreted as unix/ns stamps.
    """
    raw = values.astype(str).str.strip()
    compact = raw.str.replace(r"[^0-9]", "", regex=True)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    mask8 = compact.str.fullmatch(r"\d{8}", na=False)
    if mask8.any():
        out.loc[mask8] = pd.to_datetime(compact.loc[mask8], format="%Y%m%d", errors="coerce")
    rest = ~mask8
    if rest.any():
        out.loc[rest] = pd.to_datetime(raw.loc[rest], errors="coerce")
    return out.dt.strftime("%Y-%m-%d")


def expected_cv_report(X: pd.DataFrame) -> pd.DataFrame:
    cv = MultipleTimeSeriesCV(N_SPLITS, TRAIN_PERIOD_LENGTH, TEST_PERIOD_LENGTH, LOOKAHEAD)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        train_index = X.iloc[train_idx].index
        test_index = X.iloc[test_idx].index
        rows.append({
            "fold": int(fold),
            "train_start": pd.Timestamp(train_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
            "train_end": pd.Timestamp(train_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
            "test_start": pd.Timestamp(test_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
            "test_end": pd.Timestamp(test_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
            "n_train_symbols": int(train_index.get_level_values("symbol").nunique()),
            "n_test_symbols": int(test_index.get_level_values("symbol").nunique()),
        })
    return pd.DataFrame(rows)


def validate_cv_report(bundle_dir: Path, X: pd.DataFrame) -> dict[str, Any]:
    path = bundle_dir / "cv_split_report.csv"
    if not path.exists():
        return {"checked": False, "reason": f"missing {path}"}
    expected = expected_cv_report(X)
    actual = pd.read_csv(path)
    # Tolerate BOM/str/int dtype differences but not value differences.
    cols = ["fold", "train_start", "train_end", "test_start", "test_end", "n_train_rows", "n_test_rows", "n_train_symbols", "n_test_symbols"]
    missing = [c for c in cols if c not in actual.columns]
    if missing:
        raise RuntimeError(f"cv_split_report missing columns: {missing}")
    a = actual[cols].copy()
    e = expected[cols].copy()
    for c in cols:
        a[c] = a[c].astype(str)
        e[c] = e[c].astype(str)
    if len(a) != len(e) or not a.reset_index(drop=True).equals(e.reset_index(drop=True)):
        diff = []
        for i in range(min(len(a), len(e))):
            if not a.iloc[i].equals(e.iloc[i]):
                diff.append({"row": i, "actual": a.iloc[i].to_dict(), "expected": e.iloc[i].to_dict()})
                if len(diff) >= 3:
                    break
        raise RuntimeError(f"cv_split_report does not match reconstructed CV splits; first_diffs={diff}")
    return {"checked": True, "path": str(path), "rows": int(len(actual))}


def make_model(input_dim: int, dense_layers: Iterable[int], activation: str, dropout: float):
    from tensorflow.keras.layers import Activation, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    for i, layer_size in enumerate(dense_layers, 1):
        if i == 1:
            model.add(Dense(int(layer_size), input_dim=input_dim))
            model.add(Activation(activation))
        else:
            model.add(Dense(int(layer_size)))
            model.add(Activation(activation))
    model.add(Dropout(float(dropout)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Adam")
    return model


def clear_model_session() -> None:
    from tensorflow.keras.backend import clear_session
    clear_session()
    gc.collect()


def load_training_X(model_data_path: Path, train_end: str | None) -> pd.DataFrame:
    if not model_data_path.exists():
        raise FileNotFoundError(model_data_path)
    data = pd.read_hdf(model_data_path, "model_data")
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected model_data index names: {data.index.names}")

    # Compatibility rule: the real training HDF stores forward-label columns before
    # year/month/weekday, while live feature files store only FEATURE_COLUMNS.
    # The original training script ultimately feeds the model with FEATURE_COLUMNS
    # after dropping r01_fwd/r05_fwd/r21_fwd.  Therefore we validate by column name
    # and then explicitly reorder X by FEATURE_COLUMNS.
    missing = [c for c in REQUIRED_MODEL_DATA_COLUMNS if c not in data.columns]
    if missing:
        raise RuntimeError(
            "model_data is missing required columns: "
            f"{missing}; got={list(data.columns)}"
        )

    if train_end:
        dates = data.index.get_level_values("date")
        data = data.loc[dates <= pd.Timestamp(train_end)]

    # Keep the same dropna semantics as the training script: rows with NA in any
    # model feature or forward label are excluded before scaler fitting.
    data = data[REQUIRED_MODEL_DATA_COLUMNS].dropna().sort_index()
    X = data[FEATURE_COLUMNS].copy()
    if list(X.columns) != FEATURE_COLUMNS:
        raise RuntimeError(f"unexpected feature columns after selection: {list(X.columns)}")
    return X


def load_live_features(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise RuntimeError(f"live features must contain symbol column: {path}")
    if "date" not in df.columns:
        raise RuntimeError(f"live features must contain date column: {path}")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"live features missing columns: {missing}")
    out = df.copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["date"] = parse_as_date_series(out["date"])
    if out["date"].isna().any():
        bad = df.loc[out["date"].isna(), "date"].head(5).tolist()
        raise RuntimeError(f"live features contain unparseable date values: {bad}")
    X = out[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_cols = [c for c in FEATURE_COLUMNS if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
    if bad_cols:
        raise RuntimeError(f"live features contain NaN/inf in columns: {bad_cols}")
    return out, X.astype(float)


def resolve_path(path_text: str | None, bundle_dir: Path, repo_root: Path) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    if p.is_absolute():
        return p
    # Prefer path relative to current repo root; fall back to bundle dir.
    p1 = repo_root / p
    if p1.exists():
        return p1
    p2 = bundle_dir / p
    if p2.exists():
        return p2
    return p1


def select_checkpoint(cp_info: dict[str, Any], bundle_dir: Path, repo_root: Path) -> Path:
    bundle_path = cp_info.get("bundle_path")
    if bundle_path:
        p = bundle_dir / str(bundle_path)
        if p.exists():
            return p
    src = cp_info.get("source_path")
    p = resolve_path(str(src), bundle_dir, repo_root)
    if p is None:
        raise FileNotFoundError(src)
    return p


def fit_fold_scalers(X_train_all: pd.DataFrame, active_folds: list[int]) -> dict[int, StandardScaler]:
    cv = MultipleTimeSeriesCV(N_SPLITS, TRAIN_PERIOD_LENGTH, TEST_PERIOD_LENGTH, LOOKAHEAD)
    scalers: dict[int, StandardScaler] = {}
    for fold, (train_idx, _test_idx) in enumerate(cv.split(X_train_all)):
        if fold not in active_folds:
            continue
        scaler = StandardScaler()
        scaler.fit(X_train_all.iloc[train_idx])
        scalers[fold] = scaler
    missing = sorted(set(active_folds).difference(scalers))
    if missing:
        raise RuntimeError(f"failed to fit scalers for folds: {missing}")
    return scalers


def predict_one_checkpoint(X_live: pd.DataFrame, scaler: StandardScaler, checkpoint_path: Path, dense_layers: list[int], activation: str, dropout: float) -> np.ndarray:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    X_scaled = scaler.transform(X_live)
    model = make_model(X_live.shape[1], dense_layers, activation, dropout)
    status = model.load_weights(checkpoint_path.as_posix())
    if hasattr(status, "expect_partial"):
        status.expect_partial()
    pred = model.predict(X_scaled, verbose=0).squeeze()
    pred = np.asarray(pred, dtype=float).reshape(-1)
    del model
    clear_model_session()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AS1455 live checkpoint ensemble inference")
    ap.add_argument("--live-dir", required=True)
    ap.add_argument("--deploy-dir", required=True)
    ap.add_argument("--model-data", default=None, help="override manifest model_data_path")
    ap.add_argument("--feature-file", default=None, help="default: LIVE_DIR/11_live_model_features_for_prediction.csv")
    ap.add_argument("--out-predictions", default=None, help="default: LIVE_DIR/14_live_predictions.csv")
    ap.add_argument("--out-report", default=None, help="default: LIVE_DIR/14_live_predictions_report.json")
    ap.add_argument("--fold-mode", default=None, choices=["mean_all_folds", "single_fold"], help="override manifest fold_mode")
    ap.add_argument("--single-fold", type=int, default=None)
    ap.add_argument("--train-end", default=None, help="override training cutoff for scaler reconstruction")
    ap.add_argument("--dry-run", action="store_true", help="validate inputs/checkpoints and fit scalers; do not load tensorflow/checkpoints")
    args = ap.parse_args()

    start = time.time()
    repo_root = Path.cwd()
    live_dir = Path(args.live_dir)
    deploy_dir = Path(args.deploy_dir)
    manifest_path = deploy_dir / "manifest.json"
    out_pred = Path(args.out_predictions) if args.out_predictions else live_dir / "14_live_predictions.csv"
    out_report = Path(args.out_report) if args.out_report else live_dir / "14_live_predictions_report.json"
    feature_file = Path(args.feature_file) if args.feature_file else live_dir / "11_live_model_features_for_prediction.csv"

    report: dict[str, Any] = {
        "passed": False,
        "live_dir": str(live_dir),
        "deploy_dir": str(deploy_dir),
        "feature_file": str(feature_file),
    }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_features = manifest.get("feature_columns")
        if manifest_features is not None and list(manifest_features) != FEATURE_COLUMNS:
            raise RuntimeError(f"manifest feature_columns mismatch. expected={FEATURE_COLUMNS}; got={manifest_features}")
        model_data_text = args.model_data or manifest.get("model_data_path")
        model_data_path = resolve_path(model_data_text, deploy_dir, repo_root)
        if model_data_path is None:
            raise RuntimeError("model_data_path is missing; pass --model-data or set it in manifest")
        report["model_data_path"] = str(model_data_path)

        live_df, X_live = load_live_features(feature_file)
        report["n_live_rows"] = int(len(live_df))
        report["n_live_symbols"] = int(live_df["symbol"].nunique())

        fold_mode = args.fold_mode or manifest.get("ensemble", {}).get("fold_mode", "mean_all_folds")
        manifest_folds = [int(x) for x in manifest.get("ensemble", {}).get("folds", list(range(7)))]
        if fold_mode == "single_fold":
            sf = args.single_fold
            if sf is None:
                sf = int(manifest_folds[0])
            active_folds = [int(sf)]
        else:
            active_folds = manifest_folds
        report["fold_mode"] = fold_mode
        report["active_folds"] = active_folds

        train_end = args.train_end
        if train_end is None:
            # Prefer train_data_summary train_end if copied into manifest indirectly not always present.
            train_end = manifest.get("train_end")
        X_train_all = load_training_X(model_data_path, train_end)
        report["training_X_shape"] = [int(X_train_all.shape[0]), int(X_train_all.shape[1])]
        report["training_date_min"] = str(X_train_all.index.get_level_values("date").min().date())
        report["training_date_max"] = str(X_train_all.index.get_level_values("date").max().date())
        report["cv_split_validation"] = validate_cv_report(deploy_dir, X_train_all)

        scalers = fit_fold_scalers(X_train_all, active_folds)
        report["scalers_fit"] = sorted(scalers)

        models = manifest.get("models", [])
        if not models:
            raise RuntimeError("manifest contains no models")
        checkpoint_status = []
        model_preds: dict[str, np.ndarray] = {}
        if args.dry_run:
            for m in models:
                for fold in active_folds:
                    cp_info = m["checkpoints"].get(str(fold))
                    if not cp_info:
                        raise FileNotFoundError(f"missing checkpoint entry for {m.get('model_name')} fold {fold}")
                    cp = select_checkpoint(cp_info, deploy_dir, repo_root)
                    checkpoint_status.append({"model": m.get("model_name"), "fold": fold, "path": str(cp), "exists": cp.exists()})
                    if not cp.exists():
                        raise FileNotFoundError(cp)
        else:
            for m in models:
                mname = str(m["model_name"])
                fold_preds = []
                for fold in active_folds:
                    cp_info = m["checkpoints"].get(str(fold))
                    if not cp_info:
                        raise FileNotFoundError(f"missing checkpoint entry for {mname} fold {fold}")
                    cp = select_checkpoint(cp_info, deploy_dir, repo_root)
                    checkpoint_status.append({"model": mname, "fold": fold, "path": str(cp), "exists": cp.exists()})
                    pred = predict_one_checkpoint(
                        X_live,
                        scalers[int(fold)],
                        cp,
                        [int(x) for x in m["dense_layers"]],
                        str(m["activation"]),
                        float(m["dropout"]),
                    )
                    if len(pred) != len(X_live) or not np.isfinite(pred).all():
                        raise RuntimeError(f"bad prediction for {mname} fold {fold}: shape={pred.shape}, finite={np.isfinite(pred).all()}")
                    fold_preds.append(pred)
                model_preds[mname] = np.mean(np.vstack(fold_preds), axis=0)

        report["checkpoint_status"] = checkpoint_status
        report["n_checkpoints_used"] = len(checkpoint_status)
        expected_checkpoint_count = len(models) * len(active_folds)
        if len(checkpoint_status) != expected_checkpoint_count:
            raise RuntimeError(f"checkpoint count mismatch: got={len(checkpoint_status)}, expected={expected_checkpoint_count}")

        if not args.dry_run:
            out = pd.DataFrame({
                "date": live_df["date"],
                "symbol": live_df["symbol"],
            })
            model_names = []
            for m in models:
                mname = str(m["model_name"])
                model_names.append(mname)
                out[f"pred_{mname}"] = model_preds[mname]
            pred_cols = [f"pred_{m}" for m in model_names]
            out["pred_score"] = out[pred_cols].mean(axis=1)
            out["signal_name"] = manifest.get("ensemble", {}).get("signal_name", "ensemble_all5_mean")
            out["bundle_id"] = manifest.get("bundle_id", deploy_dir.name)
            out["fold_mode"] = fold_mode
            if not np.isfinite(out["pred_score"].to_numpy(dtype=float)).all():
                raise RuntimeError("pred_score contains NaN/inf")
            if out["pred_score"].nunique(dropna=True) <= 1:
                raise RuntimeError("pred_score has <=1 unique values; refusing to generate rank")
            out_pred.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_pred, index=False, encoding="utf-8-sig")
            report["prediction_file"] = str(out_pred)
            report["prediction_rows"] = int(len(out))
            report["prediction_columns"] = pred_cols
            report["pred_score_min"] = float(out["pred_score"].min())
            report["pred_score_max"] = float(out["pred_score"].max())

        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        raise
    finally:
        report["elapsed_seconds"] = round(time.time() - start, 3)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    print(json.dumps({
        "passed": report["passed"],
        "prediction_file": report.get("prediction_file"),
        "prediction_rows": report.get("prediction_rows"),
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
