#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fold-0 NN parameter search with AS1455 sector-rotation features.

Adds same-date sector-level rotation features to the existing AS1455 model_data.
Default mode keeps the original numeric sector feature and searches the unchanged
Chapter-17 NN candidate grid on fold 0, the newest ML4T time-series split.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from ast import literal_eval
from itertools import product
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "model_data_as1455.h5"
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_sector_rotation_fold0_search"
LOOKAHEAD = 1
N_SPLITS = 7
TRAIN_PERIOD_LENGTH = 21 * 12 * 4
TEST_PERIOD_LENGTH = 21 * 3
RETURN_COLS = ["r01", "r05", "r10", "r21", "r42", "r63"]
EXPECTED_OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]
EXPECTED_MODEL_COLUMNS = [
    "dollar_vol", "dollar_vol_rank", "rsi", "bb_high", "bb_low", "NATR", "ATR", "PPO", "MACD", "sector",
    "r01", "r05", "r10", "r21", "r42", "r63",
    "r01dec", "r05dec", "r10dec", "r21dec", "r42dec", "r63dec",
    "r01q_sector", "r05q_sector", "r10q_sector", "r21q_sector", "r42q_sector", "r63q_sector",
    "r01_fwd", "r05_fwd", "r21_fwd", "year", "month", "weekday",
]
DENSE_LAYER_OPTS = [(16, 8), (32, 16), (32, 32), (64, 32)]
ACTIVATION_OPTS = ["tanh"]
DROPOUT_OPTS = [0, 0.1, 0.2]
BATCH_SIZE_OPTS = [64, 256]
PARAM_COLS = ["dense_layers", "activation", "dropout", "batch_size", "epoch"]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_train_end(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(f"{value}-12-31" if len(value) == 4 and value.isdigit() else value)


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


def clear_model_session() -> None:
    from tensorflow.keras.backend import clear_session
    clear_session()
    gc.collect()


def fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"


class MultipleTimeSeriesCV:
    def __init__(self, n_splits: int, train_length: int, test_length: int, lookahead: int, date_idx: str = "date"):
        self.n_splits = n_splits
        self.train_length = train_length
        self.test_length = test_length
        self.lookahead = lookahead
        self.date_idx = date_idx

    def split(self, X: pd.DataFrame):
        days = sorted(X.index.get_level_values(self.date_idx).unique(), reverse=True)
        required = self.train_length + self.lookahead + self.n_splits * self.test_length
        if len(days) < required:
            raise RuntimeError(f"not enough dates: need {required}, got {len(days)}")
        dates = X.reset_index()[[self.date_idx]]
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            train_idx = dates[(dates[self.date_idx] > days[train_start_idx]) & (dates[self.date_idx] <= days[train_end_idx])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start_idx]) & (dates[self.date_idx] <= days[test_end_idx])].index
            yield train_idx.to_numpy(), test_idx.to_numpy()


def load_xy(path: Path, train_end: str | None, dropna_mode: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    data = pd.read_hdf(path, "model_data")
    n_before = int(len(data))
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    if list(data.columns) != EXPECTED_MODEL_COLUMNS:
        raise RuntimeError(f"unexpected model_data columns: {list(data.columns)}")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")
    end_ts = parse_train_end(train_end)
    dates = data.index.get_level_values("date")
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
    y = data["r01_fwd"].copy()
    X = data.drop(EXPECTED_OUTCOMES, axis=1)
    if X.shape[1] != 31 or any("fwd" in c for c in X.columns):
        raise RuntimeError(f"bad X shape/columns: {X.shape}")
    meta = {"rows_before_dropna": n_before, "rows_after_dropna": int(len(data)), "train_end_effective": effective_end.strftime("%Y-%m-%d")}
    return X, y, meta


def add_sector_rotation_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Use same-date sector means/ranks; no forward-return columns are used."""
    out = X.copy()
    dates = out.index.get_level_values("date")
    sectors = out["sector"].astype(int)
    key = pd.MultiIndex.from_arrays([dates, sectors], names=["date", "sector"])
    base = out.assign(__date=dates, __sector=sectors)
    sector_ret_mean = base.groupby(["__date", "__sector"], sort=False)[RETURN_COLS].mean()
    market_ret_mean = base.groupby("__date", sort=False)[RETURN_COLS].mean()
    sector_ret_rank = sector_ret_mean.groupby(level=0).rank(pct=True, ascending=True)
    new_cols: list[str] = []
    for c in RETURN_COLS:
        sector_values = sector_ret_mean[c].reindex(key).to_numpy()
        market_values = market_ret_mean[c].reindex(dates).to_numpy()
        out[f"sector_{c}_mean"] = sector_values
        out[f"sector_{c}_rel_mkt"] = sector_values - market_values
        out[f"sector_{c}_rank_pct"] = sector_ret_rank[c].reindex(key).to_numpy()
        new_cols += [f"sector_{c}_mean", f"sector_{c}_rel_mkt", f"sector_{c}_rank_pct"]
    sector_dv_sum = base.groupby(["__date", "__sector"], sort=False)["dollar_vol"].sum()
    market_dv_sum = base.groupby("__date", sort=False)["dollar_vol"].sum()
    sector_dv_rank = sector_dv_sum.groupby(level=0).rank(pct=True, ascending=True)
    dv_values = sector_dv_sum.reindex(key).to_numpy()
    market_dv_values = market_dv_sum.reindex(dates).to_numpy()
    out["sector_dollar_vol_sum"] = dv_values
    out["sector_dollar_vol_share"] = np.divide(dv_values, market_dv_values, out=np.zeros_like(dv_values, dtype=float), where=market_dv_values != 0)
    out["sector_dollar_vol_rank_pct"] = sector_dv_rank.reindex(key).to_numpy()
    new_cols += ["sector_dollar_vol_sum", "sector_dollar_vol_share", "sector_dollar_vol_rank_pct"]
    if out[new_cols].isna().any().any():
        bad = out[new_cols].isna().sum()
        raise RuntimeError(f"NA in rotation features: {bad[bad > 0].to_dict()}")
    return out, new_cols


def apply_sector_encoding(X: pd.DataFrame, encoding: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    if encoding == "numeric":
        return X.copy(), [], []
    if encoding != "onehot":
        raise RuntimeError(f"bad sector encoding: {encoding}")
    sectors = sorted(int(v) for v in X["sector"].astype(int).unique())
    width = max(3, len(str(max(sectors))))
    cat = pd.Categorical(X["sector"].astype(int), categories=sectors, ordered=False)
    dummies = pd.get_dummies(cat, prefix="sector").astype("uint8")
    dummies.index = X.index
    dummies = dummies.rename(columns={c: f"sector_{int(str(c).split('_')[-1]):0{width}d}" for c in dummies.columns})
    pos = list(X.columns).index("sector")
    final = pd.concat([X.iloc[:, :pos], dummies, X.iloc[:, pos + 1 :]], axis=1)
    return final, list(dummies.columns), list(dummies.columns)


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
            }
            return train_idx, test_idx, report
    raise RuntimeError(f"fold_index must be 0..{N_SPLITS - 1}, got {fold_index}")


def make_model(input_dim: int, dense_layers: tuple[int, ...], activation: str, dropout: float):
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


def param_grid(smoke: bool) -> list[dict[str, Any]]:
    rows = []
    for dense_layers, activation, dropout, batch_size in product(DENSE_LAYER_OPTS, ACTIVATION_OPTS, DROPOUT_OPTS, BATCH_SIZE_OPTS):
        rows.append({"dense_layers": str(dense_layers), "activation": activation, "dropout": float(dropout), "batch_size": int(batch_size)})
    return rows[:1] if smoke else rows


def transform_X(X: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, no_scale_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    scale_cols = [c for c in X.columns if c not in no_scale_cols]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X.iloc[train_idx][scale_cols]).astype(np.float32)
    x_test = scaler.transform(X.iloc[test_idx][scale_cols]).astype(np.float32)
    if not no_scale_cols:
        return x_train, x_test
    raw_train = X.iloc[train_idx][no_scale_cols].to_numpy(dtype=np.float32)
    raw_test = X.iloc[test_idx][no_scale_cols].to_numpy(dtype=np.float32)
    return np.concatenate([x_train, raw_train], axis=1), np.concatenate([x_test, raw_test], axis=1)


def safe_spearman(actual: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> float:
    val = spearmanr(actual, score)[0]
    return float(val) if val is not None and not np.isnan(val) else float("nan")


def daily_ic(actual: pd.Series, pred: pd.Series) -> pd.Series:
    both = pd.concat([actual.rename("actual"), pred.rename("score")], axis=1).dropna()
    return both.groupby(level="date").apply(lambda g: safe_spearman(g["actual"], g["score"]))


def score_top_bottom(pred: pd.Series, actual: pd.Series) -> pd.DataFrame:
    both = pd.concat([actual.rename("actual"), pred.rename("score")], axis=1).dropna()
    rows = []
    for n in [25, 100, 300]:
        tops, bottoms = [], []
        for _, g in both.groupby(level="date"):
            if len(g) < n:
                continue
            g = g.sort_values("score", ascending=False)
            tops.append(float(g.head(n)["actual"].mean()))
            bottoms.append(float(g.tail(n)["actual"].mean()))
        if tops:
            rows.append({"n": n, "n_dates": len(tops), "top_mean_r01_fwd": float(np.mean(tops)), "bottom_mean_r01_fwd": float(np.mean(bottoms)), "top_minus_bottom": float(np.mean(np.asarray(tops) - np.asarray(bottoms)))})
    return pd.DataFrame(rows)


def train_search(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray, no_scale_cols: list[str], grid: list[dict[str, Any]], epochs: int, seed: int, out_dir: Path) -> pd.DataFrame:
    x_train, x_test = transform_X(X, train_idx, test_idx, no_scale_cols)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    rows, daily_rows = [], []
    start = time()
    for pid, params in enumerate(grid):
        set_seed(seed + pid)
        model = make_model(x_train.shape[1], literal_eval(params["dense_layers"]), params["activation"], params["dropout"])
        print(f"[TRAIN] param_id={pid} {params}", flush=True)
        for epoch in range(epochs):
            model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=1, verbose=0, shuffle=True)
            pred = pd.Series(model.predict(x_test, verbose=0).squeeze(), index=y_test.index)
            ic = daily_ic(y_test, pred)
            base = {"param_id": pid, **params, "epoch": epoch}
            rows.append({**base, "pooled_spearman": safe_spearman(y_test, pred), "daily_ic_mean": float(ic.mean()), "daily_ic_median": float(ic.median()), "daily_ic_positive_rate": float((ic > 0).mean()), "n_dates": int(len(ic)), "n_rows": int(len(pred)), "elapsed": fmt(time() - start)})
            daily_rows.extend([{**base, "date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "spearman_ic": float(v)} for dt, v in ic.items()])
            print(f"{fmt(time() - start)} | param {pid:02d} | epoch {epoch + 1:02d} | mean {ic.mean(): .5f} | median {ic.median(): .5f}", flush=True)
        del model
        clear_model_session()
        pd.DataFrame(rows).to_csv(out_dir / "scores_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(daily_rows).to_csv(out_dir / "scores_by_day.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows).sort_values("daily_ic_median", ascending=False)
    summary.to_csv(out_dir / "scores_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_rows).to_hdf(out_dir / "scores.h5", "ic_by_day", mode="w")
    return summary


def retrain_best(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray, no_scale_cols: list[str], best: pd.DataFrame, seed: int, out_dir: Path) -> None:
    x_train, x_test = transform_X(X, train_idx, test_idx, no_scale_cols)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pred_cols = []
    for i, row in best.reset_index(drop=True).iterrows():
        params = row.to_dict()
        set_seed(seed + 1000 + i)
        model = make_model(x_train.shape[1], literal_eval(str(params["dense_layers"])), str(params["activation"]), float(params["dropout"]))
        for _ in range(int(params["epoch"]) + 1):
            model.fit(x_train, y_train, batch_size=int(params["batch_size"]), epochs=1, verbose=0, shuffle=True)
        pred_cols.append(pd.Series(model.predict(x_test, verbose=0).squeeze(), index=y_test.index, name=f"best_{i}"))
        del model
        clear_model_session()
    preds = pd.concat(pred_cols, axis=1)
    if preds.shape[1] > 1:
        preds["ensemble"] = preds.mean(axis=1)
    preds.to_hdf(out_dir / "best_predictions.h5", "predictions", mode="w")
    preds.to_csv(out_dir / "best_predictions.csv", encoding="utf-8-sig")
    y_test.to_frame("r01_fwd").to_csv(out_dir / "best_actual.csv", encoding="utf-8-sig")
    ic_rows, tb_rows = [], []
    for col in preds.columns:
        ic = daily_ic(y_test, preds[col])
        ic_rows.append({"score_col": col, "pooled_spearman": safe_spearman(y_test, preds[col]), "daily_ic_mean": float(ic.mean()), "daily_ic_median": float(ic.median()), "daily_ic_positive_rate": float((ic > 0).mean())})
        tb = score_top_bottom(preds[col], y_test)
        tb.insert(0, "score_col", col)
        tb_rows.append(tb)
    pd.DataFrame(ic_rows).to_csv(out_dir / "best_ic_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(tb_rows, ignore_index=True).to_csv(out_dir / "best_top_bottom_returns.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AS1455 sector-rotation fold-0 NN parameter search")
    p.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--train-end", default=None)
    p.add_argument("--fold-index", type=int, default=0, help="0=newest fold, 6=oldest fold")
    p.add_argument("--sector-encoding", choices=["numeric", "onehot"], default="numeric")
    p.add_argument("--dropna-mode", choices=["strict_original", "r01_only"], default="r01_only")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--best-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input-check-only", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir already has files; pass --force or choose another --out-dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    X_base, y, meta = load_xy(Path(args.model_data), args.train_end, args.dropna_mode)
    X_rot, rotation_cols = add_sector_rotation_features(X_base)
    X_final, no_scale_cols, sector_onehot_cols = apply_sector_encoding(X_rot, args.sector_encoding)
    grid = param_grid(args.smoke)
    train_idx, test_idx, fold = get_fold(X_final, args.fold_index)
    write_json(out_dir / "run_summary.json", {**meta, "model_data": str(Path(args.model_data).resolve()), "out_dir": str(out_dir.resolve()), "base_feature_count": int(X_base.shape[1]), "rotation_feature_count": len(rotation_cols), "final_feature_count": int(X_final.shape[1]), "sector_encoding": args.sector_encoding, "dropna_mode": args.dropna_mode, "fold_index": args.fold_index, "epochs": args.epochs, "param_grid_size": len(grid)})
    write_json(out_dir / "fold_report.json", fold)
    pd.DataFrame([fold]).to_csv(out_dir / "fold_report.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "feature_cols_base.json", list(X_base.columns))
    write_json(out_dir / "rotation_feature_cols.json", rotation_cols)
    write_json(out_dir / "feature_cols_final.json", list(X_final.columns))
    write_json(out_dir / "sector_onehot_cols.json", sector_onehot_cols)
    pd.DataFrame(grid).to_csv(out_dir / "param_grid.csv", index=False, encoding="utf-8-sig")
    print(f"[DATA] base={X_base.shape[1]} rotation={len(rotation_cols)} final={X_final.shape[1]}")
    print(f"[FOLD] {fold}")
    if args.input_check_only:
        print(f"[OK] input reports written to {out_dir}")
        return
    require_deps()
    summary = train_search(X_final, y, train_idx, test_idx, no_scale_cols, grid, args.epochs, args.seed, out_dir)
    best = summary.head(args.best_n).copy()
    best.to_csv(out_dir / "best_params.csv", index=False, encoding="utf-8-sig")
    print("[BEST]")
    print(best[PARAM_COLS + ["pooled_spearman", "daily_ic_mean", "daily_ic_median", "daily_ic_positive_rate"]].to_string(index=False))
    retrain_best(X_final, y, train_idx, test_idx, no_scale_cols, best[PARAM_COLS], args.seed, out_dir)
    print(f"[OK] written to {out_dir}")


if __name__ == "__main__":
    main()
