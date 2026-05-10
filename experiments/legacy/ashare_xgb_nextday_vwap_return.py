#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Next-day VWAP return XGBoost regression.

This script reuses the local artifacts produced by the dual-opportunity
pipeline.  It builds a daily target from intraday 5-minute bars:

    next_day_vwap_ret_close = next_trading_day_daily_vwap / current_close - 1

The daily VWAP is aggregated from signal_samples.csv using adjusted
5-minute VWAP and volume/PV fields, so the target is on the same adjusted
price scale as daily_features.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception as e:  # pragma: no cover
    raise SystemExit(f"未安装 xgboost，请先 pip install -U xgboost。原始错误: {e}")

RANDOM_STATE = 42
EPS = 1e-12


@dataclass
class XGBRegModelConfig:
    train_ratio: float = 0.60
    valid_ratio: float = 0.20
    test_ratio: float = 0.20
    n_splits: int = 5
    xgb_n_jobs: int = 4
    max_cv_param_combos: int = 24
    min_cv_train_samples: int = 180
    min_cv_val_samples: int = 50

    max_depth_grid: Tuple[int, ...] = (2, 3, 4)
    learning_rate_grid: Tuple[float, ...] = (0.03, 0.05)
    n_estimators_grid: Tuple[int, ...] = (200, 500)
    min_child_weight_grid: Tuple[int, ...] = (5, 20)
    subsample_grid: Tuple[float, ...] = (0.8,)
    colsample_bytree_grid: Tuple[float, ...] = (0.8,)
    reg_lambda_grid: Tuple[float, ...] = (1.0, 5.0, 10.0)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(np.asarray(y_true, dtype=float))
    s2 = pd.Series(np.asarray(y_pred, dtype=float))
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return float(s1.corr(s2, method="spearman"))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(np.asarray(y_true, dtype=float))
    s2 = pd.Series(np.asarray(y_pred, dtype=float))
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return float(s1.corr(s2, method="pearson"))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err * err))) if len(err) else np.nan


def quantile_spread(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.2) -> Dict[str, float]:
    df = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 20:
        return {"top_mean": np.nan, "bottom_mean": np.nan, "spread": np.nan, "q": q}
    n = max(1, int(np.floor(len(df) * q)))
    top = df.nlargest(n, "pred")
    bot = df.nsmallest(n, "pred")
    return {
        "top_mean": float(top["y"].mean()),
        "bottom_mean": float(bot["y"].mean()),
        "spread": float(top["y"].mean() - bot["y"].mean()),
        "q": q,
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {}
    err = yp - yt
    return {
        "rows": int(len(yt)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse(yt, yp),
        "median_abs_error": float(np.median(np.abs(err))),
        "p75_abs_error": float(np.quantile(np.abs(err), 0.75)),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
        "max_abs_error": float(np.max(np.abs(err))),
        "rank_ic": safe_spearman(yt, yp),
        "pearson_ic": safe_pearson(yt, yp),
        "direction_accuracy": float(np.mean(np.sign(yt) == np.sign(yp))),
        "target_mean": float(np.mean(yt)),
        "pred_mean": float(np.mean(yp)),
        "target_std": float(np.std(yt)),
        "pred_std": float(np.std(yp)),
        "pct_abs_error_le_0.005": float(np.mean(np.abs(err) <= 0.005)),
        "pct_abs_error_le_0.010": float(np.mean(np.abs(err) <= 0.010)),
        "top_bottom_spread_q20": quantile_spread(yt, yp, q=0.2),
    }


def build_daily_vwap(signal_samples_path: Path) -> pd.DataFrame:
    usecols = ["trade_date", "bar_bar_vwap", "bar_bar_volume", "bar_bar_pv"]
    bars = pd.read_csv(signal_samples_path, usecols=usecols, parse_dates=["trade_date"])
    bars = bars.replace([np.inf, -np.inf], np.nan).dropna(subset=["trade_date", "bar_bar_vwap", "bar_bar_volume"])
    bars = bars[bars["bar_bar_volume"] > 0].copy()
    if "bar_bar_pv" in bars.columns and bars["bar_bar_pv"].notna().any():
        daily = bars.groupby("trade_date").agg(
            daily_vwap_pv=("bar_bar_pv", "sum"),
            daily_vwap_volume=("bar_bar_volume", "sum"),
            n_intraday_bars=("bar_bar_vwap", "size"),
        )
        daily["daily_vwap"] = daily["daily_vwap_pv"] / daily["daily_vwap_volume"]
    else:
        bars["pv"] = bars["bar_bar_vwap"] * bars["bar_bar_volume"]
        daily = bars.groupby("trade_date").agg(
            daily_vwap_pv=("pv", "sum"),
            daily_vwap_volume=("bar_bar_volume", "sum"),
            n_intraday_bars=("bar_bar_vwap", "size"),
        )
        daily["daily_vwap"] = daily["daily_vwap_pv"] / daily["daily_vwap_volume"]
    return daily.reset_index().rename(columns={"trade_date": "date"})


def feature_columns(df: pd.DataFrame) -> List[str]:
    leak = {
        "date",
        "daily_vwap",
        "daily_vwap_pv",
        "daily_vwap_volume",
        "n_intraday_bars",
        "next_date",
        "next_day_vwap",
        "next_day_close",
        "next_day_vwap_ret_close",
        "next_day_vwap_ret_vwap",
        "next_day_close_ret_close",
        "label_rev",
    }
    cols = [c for c in df.columns if c not in leak and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def build_dataset(daily_features_path: Path, signal_samples_path: Path, min_bars: int) -> pd.DataFrame:
    daily_feat = pd.read_csv(daily_features_path, parse_dates=["date"])
    daily_vwap = build_daily_vwap(signal_samples_path)
    df = daily_feat.merge(daily_vwap, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df = df[df["n_intraday_bars"] >= min_bars].copy()
    df["next_date"] = df["date"].shift(-1)
    df["next_day_vwap"] = df["daily_vwap"].shift(-1)
    df["next_day_close"] = df["close"].shift(-1)
    df["next_day_vwap_ret_close"] = df["next_day_vwap"] / df["close"] - 1.0
    df["next_day_vwap_ret_vwap"] = df["next_day_vwap"] / df["daily_vwap"] - 1.0
    df["next_day_close_ret_close"] = df["next_day_close"] / df["close"] - 1.0
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["next_day_vwap_ret_close"]).reset_index(drop=True)
    return df


def split_chrono(df: pd.DataFrame, cfg: XGBRegModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)
    train = df.iloc[:n_train].copy()
    valid = df.iloc[n_train:n_train + n_valid].copy()
    test = df.iloc[n_train + n_valid:].copy()
    return train, valid, test


def make_expanding_splits(n_samples: int, n_splits: int, min_train: int, min_val: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < min_train + min_val:
        return []
    val_size = max(min_val, (n_samples - min_train) // n_splits)
    splits = []
    for i in range(n_splits):
        val_end = n_samples - (n_splits - 1 - i) * val_size
        val_start = val_end - val_size
        train_end = val_start
        if train_end >= min_train and val_end <= n_samples and val_start >= 0:
            splits.append((np.arange(train_end), np.arange(val_start, val_end)))
    return splits


def fit_one_xgb(X: pd.DataFrame, y: np.ndarray, params: Dict[str, object]) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=int(params.get("n_jobs", 4)),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
    )
    model.fit(X, y)
    return model


def cv_search_params(X: pd.DataFrame, y: np.ndarray, cfg: XGBRegModelConfig) -> Tuple[Dict[str, object], pd.DataFrame]:
    combos = []
    for md in cfg.max_depth_grid:
        for lr in cfg.learning_rate_grid:
            for ne in cfg.n_estimators_grid:
                for mcw in cfg.min_child_weight_grid:
                    for ss in cfg.subsample_grid:
                        for cs in cfg.colsample_bytree_grid:
                            for rl in cfg.reg_lambda_grid:
                                combos.append({
                                    "max_depth": md,
                                    "learning_rate": lr,
                                    "n_estimators": ne,
                                    "min_child_weight": mcw,
                                    "subsample": ss,
                                    "colsample_bytree": cs,
                                    "reg_lambda": rl,
                                    "n_jobs": cfg.xgb_n_jobs,
                                })
    if cfg.max_cv_param_combos > 0 and len(combos) > cfg.max_cv_param_combos:
        keep_idx = np.unique(np.linspace(0, len(combos) - 1, cfg.max_cv_param_combos).round().astype(int))
        combos = [combos[int(i)] for i in keep_idx]

    splits = make_expanding_splits(len(X), cfg.n_splits, cfg.min_cv_train_samples, cfg.min_cv_val_samples)
    rows = []
    for params in combos:
        rmses, maes, rics = [], [], []
        for tr_idx, va_idx in splits:
            model = fit_one_xgb(X.iloc[tr_idx], y[tr_idx], params)
            pred = model.predict(X.iloc[va_idx])
            metrics = evaluate_regression(y[va_idx], pred)
            rmses.append(metrics["rmse"])
            maes.append(metrics["mae"])
            rics.append(metrics["rank_ic"])
        if rmses:
            rows.append({
                **params,
                "cv_rmse": float(np.mean(rmses)),
                "cv_mae": float(np.mean(maes)),
                "cv_rank_ic": float(np.nanmean(rics)),
                "folds_used": len(rmses),
            })
    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        fallback = {
            "max_depth": 2,
            "learning_rate": 0.03,
            "n_estimators": 200,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": cfg.xgb_n_jobs,
        }
        return fallback, pd.DataFrame([{**fallback, "note": "no_valid_cv_split"}])
    cv_df = cv_df.sort_values(["cv_rmse", "cv_rank_ic"], ascending=[True, False]).reset_index(drop=True)
    best = cv_df.iloc[0][[
        "max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "n_jobs"
    ]].to_dict()
    best["max_depth"] = int(best["max_depth"])
    best["n_estimators"] = int(best["n_estimators"])
    best["min_child_weight"] = int(best["min_child_weight"])
    best["n_jobs"] = int(best["n_jobs"])
    return best, cv_df


def add_predictions(df: pd.DataFrame, pred: np.ndarray, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["target"] = out[target_col]
    out["pred"] = pred
    out["error"] = out["pred"] - out["target"]
    out["abs_error"] = out["error"].abs()
    out["pred_direction"] = np.sign(out["pred"])
    out["target_direction"] = np.sign(out["target"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="XGB next-day daily VWAP return regression")
    p.add_argument("--daily-features", default="dual_opp_out_002714_v12/daily_features.csv")
    p.add_argument("--signal-samples", default="dual_opp_out_002714_v12/signal_samples.csv")
    p.add_argument("--out-dir", default="nextday_vwap_return_out")
    p.add_argument("--target", default="next_day_vwap_ret_close", choices=["next_day_vwap_ret_close", "next_day_vwap_ret_vwap", "next_day_close_ret_close"])
    p.add_argument("--min-bars", type=int, default=40)
    p.add_argument("--xgb-n-jobs", type=int, default=4)
    p.add_argument("--max-cv-param-combos", type=int, default=24)
    args = p.parse_args()

    start = time.time()
    out_dir = ensure_dir(args.out_dir)
    cfg = XGBRegModelConfig(xgb_n_jobs=args.xgb_n_jobs, max_cv_param_combos=args.max_cv_param_combos)

    data = build_dataset(Path(args.daily_features), Path(args.signal_samples), args.min_bars)
    cols = feature_columns(data)
    train, valid, test = split_chrono(data, cfg)

    med = train[cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    X_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    X_valid = valid[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    X_test = test[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    y_train = train[args.target].to_numpy(dtype=float)
    y_valid = valid[args.target].to_numpy(dtype=float)
    y_test = test[args.target].to_numpy(dtype=float)

    best_params, cv_df = cv_search_params(X_train, y_train, cfg)
    model = fit_one_xgb(X_train, y_train, best_params)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    baseline_mean = float(np.mean(y_train))
    baseline_zero = 0.0
    metrics = {
        "train": evaluate_regression(y_train, pred_train),
        "valid": evaluate_regression(y_valid, pred_valid),
        "test": evaluate_regression(y_test, pred_test),
        "valid_baseline_train_mean": evaluate_regression(y_valid, np.full_like(y_valid, baseline_mean)),
        "test_baseline_train_mean": evaluate_regression(y_test, np.full_like(y_test, baseline_mean)),
        "valid_baseline_zero": evaluate_regression(y_valid, np.full_like(y_valid, baseline_zero)),
        "test_baseline_zero": evaluate_regression(y_test, np.full_like(y_test, baseline_zero)),
    }

    pred_train_df = add_predictions(train, pred_train, args.target)
    pred_valid_df = add_predictions(valid, pred_valid, args.target)
    pred_test_df = add_predictions(test, pred_test, args.target)
    data.to_csv(out_dir / "training_samples.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(out_dir / "cv_results.csv", index=False, encoding="utf-8-sig")
    pred_train_df.to_csv(out_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")
    pred_valid_df.to_csv(out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    pred_test_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame({
        "feature": cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    model.save_model(str(out_dir / "xgb_nextday_vwap_return_model.json"))

    summary = {
        "target": args.target,
        "elapsed_seconds": round(time.time() - start, 3),
        "rows": int(len(data)),
        "features": int(len(cols)),
        "date_min": str(data["date"].min().date()),
        "date_max": str(data["date"].max().date()),
        "split": {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
            "train_date_min": str(train["date"].min().date()),
            "train_date_max": str(train["date"].max().date()),
            "valid_date_min": str(valid["date"].min().date()),
            "valid_date_max": str(valid["date"].max().date()),
            "test_date_min": str(test["date"].min().date()),
            "test_date_max": str(test["date"].max().date()),
        },
        "best_params": best_params,
        "metrics": metrics,
        "outputs": {
            "training_samples": str(out_dir / "training_samples.csv"),
            "cv_results": str(out_dir / "cv_results.csv"),
            "valid_predictions": str(out_dir / "valid_predictions.csv"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "feature_importance": str(out_dir / "feature_importance.csv"),
            "model": str(out_dir / "xgb_nextday_vwap_return_model.json"),
        },
    }
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
