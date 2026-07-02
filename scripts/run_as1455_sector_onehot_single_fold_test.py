#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 single-fold sector one-hot ablation.

Reuses existing model_data_as1455.h5 and compares:
1) original single numeric sector feature; and
2) sector one-hot features with only non-one-hot columns standardized.

The candidate NN structures are unchanged. Fold 0 is the newest fold; fold 6
is the oldest fold because the ML4T splitter iterates from newest to oldest.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from ast import literal_eval
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "model_data_as1455.h5"
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_sector_onehot_fold_test"
LOOKAHEAD = 1
N_SPLITS = 7
TRAIN_PERIOD_LENGTH = 21 * 12 * 4
TEST_PERIOD_LENGTH = 21 * 3
EXPECTED_OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]
EXPECTED_MODEL_COLUMNS = [
    "dollar_vol", "dollar_vol_rank", "rsi", "bb_high", "bb_low", "NATR", "ATR", "PPO", "MACD", "sector",
    "r01", "r05", "r10", "r21", "r42", "r63",
    "r01dec", "r05dec", "r10dec", "r21dec", "r42dec", "r63dec",
    "r01q_sector", "r05q_sector", "r10q_sector", "r21q_sector", "r42q_sector", "r63q_sector",
    "r01_fwd", "r05_fwd", "r21_fwd", "year", "month", "weekday",
]
FORBIDDEN_MODEL_COLUMNS = {
    "open", "high", "low", "close", "volume", "board", "industry", "is_mainboard", "tradestatus", "isST",
    "raw_open_as1455", "raw_high_as1455", "raw_low_as1455", "raw_close_as1455", "raw_volume_as1455",
    "raw_amount_as1455", "last_bar_time", "open_limit_up", "open_limit_down",
}
DEFAULT_PARAMS = [
    {"dense_layers": "(32, 16)", "activation": "tanh", "dropout": 0.2, "batch_size": 64, "epoch": 2},
    {"dense_layers": "(64, 32)", "activation": "tanh", "dropout": 0.1, "batch_size": 64, "epoch": 13},
    {"dense_layers": "(16, 8)", "activation": "tanh", "dropout": 0.2, "batch_size": 256, "epoch": 6},
]


class MultipleTimeSeriesCV:
    def __init__(self, n_splits: int, train_period_length: int, test_period_length: int, lookahead: int, date_idx: str = "date"):
        self.n_splits = n_splits
        self.train_length = train_period_length
        self.test_length = test_period_length
        self.lookahead = lookahead
        self.date_idx = date_idx

    def split(self, X: pd.DataFrame):
        days = sorted(X.index.get_level_values(self.date_idx).unique(), reverse=True)
        required = self.train_length + self.lookahead + self.n_splits * self.test_length
        if len(days) < required:
            raise RuntimeError(f"not enough dates for {self.n_splits} folds: need {required}, got {len(days)}")
        dates = X.reset_index()[[self.date_idx]]
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            train_idx = dates[(dates[self.date_idx] > days[train_start_idx]) & (dates[self.date_idx] <= days[train_end_idx])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start_idx]) & (dates[self.date_idx] <= days[test_end_idx])].index
            yield train_idx.to_numpy(), test_idx.to_numpy()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def require_deps() -> None:
    missing = []
    try:
        import tables  # noqa: F401
    except Exception:
        missing.append("tables/PyTables")
    try:
        import tensorflow  # noqa: F401
    except Exception:
        missing.append("tensorflow")
    if missing:
        raise SystemExit("Missing required dependency: " + ", ".join(missing))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def clear_session() -> None:
    from tensorflow.keras.backend import clear_session as keras_clear_session
    keras_clear_session()
    gc.collect()


def make_model(input_dim: int, dense_layers: Iterable[int], activation: str, dropout: float):
    from tensorflow.keras.layers import Activation, Dense, Dropout
    from tensorflow.keras.models import Sequential
    model = Sequential()
    for i, n_units in enumerate(dense_layers, 1):
        model.add(Dense(n_units, input_dim=input_dim) if i == 1 else Dense(n_units))
        model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Adam")
    return model


def parse_train_end(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(f"{value}-12-31" if len(value) == 4 and value.isdigit() else value)


def load_xy(path: Path, train_end: str | None, dropna_mode: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    data = pd.read_hdf(path, "model_data")
    n_before = int(len(data))
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    if list(data.columns) != EXPECTED_MODEL_COLUMNS:
        raise RuntimeError(f"unexpected model_data columns: {list(data.columns)}")
    forbidden = sorted(FORBIDDEN_MODEL_COLUMNS.intersection(data.columns))
    if forbidden:
        raise RuntimeError(f"forbidden columns in model_data: {forbidden}")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")

    dates = data.index.get_level_values("date")
    end_ts = parse_train_end(train_end)
    effective_end = pd.Timestamp(dates.max()) if end_ts is None else end_ts
    if end_ts is not None:
        data = data.loc[dates <= end_ts]

    if dropna_mode == "strict_original":
        data = data.dropna()
    elif dropna_mode == "r01_only":
        data = data.dropna(subset=[c for c in data.columns if c not in ["r05_fwd", "r21_fwd"]])
    else:
        raise RuntimeError(f"bad dropna_mode: {dropna_mode}")
    data = data.sort_index()
    if data.empty:
        raise RuntimeError("empty data after filtering/dropna")

    y = data["r01_fwd"].copy()
    X = data.drop(EXPECTED_OUTCOMES, axis=1)
    if X.shape[1] != 31 or any("fwd" in c for c in X.columns):
        raise RuntimeError(f"bad X columns: shape={X.shape}, columns={list(X.columns)}")
    meta = {"rows_before_dropna": n_before, "rows_after_dropna": int(len(data)), "train_end_effective": effective_end.strftime("%Y-%m-%d")}
    return X, y, meta


def make_onehot(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    sectors = sorted(int(v) for v in X["sector"].astype(int).unique())
    width = max(3, len(str(max(sectors))))
    cat = pd.Categorical(X["sector"].astype(int), categories=sectors, ordered=False)
    dummies = pd.get_dummies(cat, prefix="sector").astype("uint8")
    dummies.index = X.index
    dummies = dummies.rename(columns={c: f"sector_{int(str(c).split('_')[-1]):0{width}d}" for c in dummies.columns})
    sector_cols = list(dummies.columns)
    before = list(X.columns[: list(X.columns).index("sector")])
    after = list(X.columns[list(X.columns).index("sector") + 1 :])
    X_oh = pd.concat([X[before], dummies, X[after]], axis=1)
    return X_oh, sector_cols, before, after


def get_fold(X: pd.DataFrame, fold_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cv = MultipleTimeSeriesCV(N_SPLITS, TRAIN_PERIOD_LENGTH, TEST_PERIOD_LENGTH, LOOKAHEAD)
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        if i == fold_index:
            train_index = X.iloc[train_idx].index
            test_index = X.iloc[test_idx].index
            report = {
                "fold_index": i,
                "train_start": pd.Timestamp(train_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
                "train_end": pd.Timestamp(train_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
                "test_start": pd.Timestamp(test_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(test_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_train_symbols": int(train_index.get_level_values("symbol").nunique()),
                "n_test_symbols": int(test_index.get_level_values("symbol").nunique()),
            }
            return train_idx, test_idx, report
    raise RuntimeError(f"fold_index must be between 0 and {N_SPLITS - 1}, got {fold_index}")


def load_params(params_csv: str | None, top_n: int, smoke: bool) -> list[dict[str, Any]]:
    rows = pd.read_csv(params_csv).head(top_n).to_dict("records") if params_csv else DEFAULT_PARAMS[:top_n]
    if smoke:
        rows = rows[:1]
    out = []
    for row in rows:
        out.append({
            "dense_layers": str(row["dense_layers"]),
            "activation": str(row.get("activation", "tanh")),
            "dropout": float(row.get("dropout", 0.0)),
            "batch_size": int(row.get("batch_size", 64)),
            "epoch": min(int(row.get("epoch", 19)), 1) if smoke else int(row.get("epoch", 19)),
        })
    if not out:
        raise RuntimeError("empty parameter list")
    return out


def transform_numeric(X: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X.iloc[train_idx]), scaler.transform(X.iloc[test_idx])


def transform_onehot(X_oh: pd.DataFrame, sector_cols: list[str], before: list[str], after: list[str], train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler_before = StandardScaler()
    scaler_after = StandardScaler()
    train_before = scaler_before.fit_transform(X_oh.iloc[train_idx][before]) if before else np.empty((len(train_idx), 0))
    test_before = scaler_before.transform(X_oh.iloc[test_idx][before]) if before else np.empty((len(test_idx), 0))
    train_after = scaler_after.fit_transform(X_oh.iloc[train_idx][after]) if after else np.empty((len(train_idx), 0))
    test_after = scaler_after.transform(X_oh.iloc[test_idx][after]) if after else np.empty((len(test_idx), 0))
    train_sector = X_oh.iloc[train_idx][sector_cols].to_numpy(dtype=np.float32)
    test_sector = X_oh.iloc[test_idx][sector_cols].to_numpy(dtype=np.float32)
    return np.concatenate([train_before, train_sector, train_after], axis=1), np.concatenate([test_before, test_sector, test_after], axis=1)


def train_predict(x_train: np.ndarray, y_train: pd.Series, x_test: np.ndarray, input_dim: int, params: dict[str, Any], seed: int) -> np.ndarray:
    set_seed(seed)
    layers = literal_eval(params["dense_layers"])
    if not isinstance(layers, tuple):
        raise RuntimeError(f"dense_layers must parse to tuple: {params['dense_layers']}")
    model = make_model(input_dim, layers, params["activation"], params["dropout"])
    pred = None
    for _ in range(params["epoch"] + 1):
        model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=1, verbose=0, shuffle=True)
        pred = model.predict(x_test, verbose=0).squeeze()
    del model
    clear_session()
    return np.asarray(pred, dtype=float)


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    val = spearmanr(x, y)[0]
    return float(val) if val is not None and not np.isnan(val) else float("nan")


def write_metrics(pred: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    actual = y.rename("actual")
    daily_rows, summary_rows, top_rows = [], [], []
    for col in pred.columns:
        both = pd.concat([actual, pred[col].rename("score")], axis=1).dropna()
        by_date = both.groupby(level="date")
        daily = by_date.apply(lambda g: safe_spearman(g["actual"], g["score"]))
        for dt, ic in daily.items():
            daily_rows.append({"score_col": col, "date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "spearman_ic": ic})
        summary_rows.append({
            "score_col": col,
            "n_rows": int(len(both)),
            "n_dates": int(both.index.get_level_values("date").nunique()),
            "pooled_spearman": safe_spearman(both["actual"], both["score"]),
            "daily_ic_mean": float(daily.mean()),
            "daily_ic_median": float(daily.median()),
            "daily_ic_positive_rate": float((daily > 0).mean()),
        })
        for n in [25, 100, 300]:
            tops, bottoms = [], []
            for _, g in by_date:
                if len(g) < n:
                    continue
                g = g.sort_values("score", ascending=False)
                tops.append(float(g.head(n)["actual"].mean()))
                bottoms.append(float(g.tail(n)["actual"].mean()))
            if tops:
                top_rows.append({
                    "score_col": col,
                    "n": n,
                    "n_dates": len(tops),
                    "top_mean_r01_fwd": float(np.mean(tops)),
                    "bottom_mean_r01_fwd": float(np.mean(bottoms)),
                    "top_minus_bottom": float(np.mean(np.asarray(tops) - np.asarray(bottoms))),
                })
    pd.DataFrame(daily_rows).to_csv(out_dir / "daily_ic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(out_dir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top_rows).to_csv(out_dir / "top_bottom_returns.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AS1455 single-fold numeric-sector vs one-hot-sector NN test")
    p.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--train-end", default=None)
    p.add_argument("--fold-index", type=int, default=0, help="0=newest fold, 6=oldest fold")
    p.add_argument("--params-csv", default=None)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--variant", choices=["numeric", "onehot", "both"], default="both")
    p.add_argument("--dropna-mode", choices=["strict_original", "r01_only"], default="strict_original")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input-check-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, data_meta = load_xy(Path(args.model_data), args.train_end, args.dropna_mode)
    X_oh, sector_cols, before_cols, after_cols = make_onehot(X)
    params = load_params(args.params_csv, args.top_n, args.smoke)
    train_idx, test_idx, fold_report = get_fold(X, args.fold_index)

    write_json(out_dir / "run_summary.json", {
        **data_meta,
        "model_data": str(Path(args.model_data).resolve()),
        "out_dir": str(out_dir.resolve()),
        "variant": args.variant,
        "dropna_mode": args.dropna_mode,
        "numeric_feature_count": int(X.shape[1]),
        "onehot_feature_count": int(X_oh.shape[1]),
        "sector_onehot_count": len(sector_cols),
        "params": params,
    })
    write_json(out_dir / "fold_report.json", fold_report)
    pd.DataFrame([fold_report]).to_csv(out_dir / "fold_report.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "feature_cols_numeric.json", list(X.columns))
    write_json(out_dir / "feature_cols_onehot.json", list(X_oh.columns))
    write_json(out_dir / "sector_onehot_cols.json", sector_cols)
    write_json(out_dir / "params_used.json", params)

    print(f"[DATA] rows={data_meta['rows_after_dropna']} numeric_features={X.shape[1]} onehot_features={X_oh.shape[1]}")
    print(f"[FOLD] {fold_report}")
    if args.input_check_only:
        print(f"[OK] input reports written to {out_dir}")
        return

    require_deps()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    frames = []
    if args.variant in {"numeric", "both"}:
        x_train, x_test = transform_numeric(X, train_idx, test_idx)
        preds = {}
        for i, prm in enumerate(params):
            print(f"[TRAIN] numeric_p{i}: {prm}", flush=True)
            preds[f"numeric_p{i}"] = train_predict(x_train, y_train, x_test, X.shape[1], prm, args.seed + i)
        df = pd.DataFrame(preds, index=y_test.index)
        if len(df.columns) > 1:
            df["numeric_ensemble"] = df.mean(axis=1)
        frames.append(df)
    if args.variant in {"onehot", "both"}:
        x_train, x_test = transform_onehot(X_oh, sector_cols, before_cols, after_cols, train_idx, test_idx)
        preds = {}
        for i, prm in enumerate(params):
            print(f"[TRAIN] onehot_p{i}: {prm}", flush=True)
            preds[f"onehot_p{i}"] = train_predict(x_train, y_train, x_test, X_oh.shape[1], prm, args.seed + i)
        df = pd.DataFrame(preds, index=y_test.index)
        if len(df.columns) > 1:
            df["onehot_ensemble"] = df.mean(axis=1)
        frames.append(df)

    pred = pd.concat(frames, axis=1)
    pred.to_hdf(out_dir / "single_fold_predictions.h5", "predictions")
    pred.to_csv(out_dir / "single_fold_predictions.csv", encoding="utf-8-sig")
    y_test.to_frame("r01_fwd").to_csv(out_dir / "single_fold_actual.csv", encoding="utf-8-sig")
    write_metrics(pred, y_test, out_dir)
    summary = pd.read_csv(out_dir / "ic_summary.csv")
    print(summary.to_string(index=False))
    print(f"[OK] written to {out_dir}")


if __name__ == "__main__":
    main()
