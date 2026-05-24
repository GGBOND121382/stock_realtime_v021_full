#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search model complexity for the next-day hit strategy.

This keeps the evaluation protocol fixed:
  - walk-forward train/valid/test windows
  - train only on entry rows
  - validation chooses score threshold
  - test is scored by selected trades with target-or-close exits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from model_training.optimize_nextday_vwap_model import (
    add_market_state_features,
    add_reversal_features,
    add_trade_returns,
    choose_valid_threshold,
    ensure_dir,
    feature_groups,
    iter_walk_forward_windows,
    prepare_x_by_median,
    regime_mask,
    trade_metrics,
)


RANDOM_STATE = 42


def model_configs() -> List[Tuple[str, object]]:
    configs: List[Tuple[str, object]] = [
        (
            "xgb_d2_200_lr003_mcw5",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=2,
                learning_rate=0.03,
                n_estimators=200,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
            ),
        ),
        (
            "xgb_d3_400_lr003_mcw3",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=3,
                learning_rate=0.03,
                n_estimators=400,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
            ),
        ),
        (
            "xgb_d3_300_lr004_mcw3",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=3,
                learning_rate=0.04,
                n_estimators=300,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
            ),
        ),
        (
            "xgb_d3_600_lr002_mcw3",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=3,
                learning_rate=0.02,
                n_estimators=600,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
            ),
        ),
        (
            "xgb_d3_800_lr0015_mcw5",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=3,
                learning_rate=0.015,
                n_estimators=800,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
            ),
        ),
        (
            "xgb_d4_700_lr002_mcw2",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=4,
                learning_rate=0.02,
                n_estimators=700,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
            ),
        ),
        (
            "xgb_d4_500_lr002_mcw5",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=4,
                learning_rate=0.02,
                n_estimators=500,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=8.0,
            ),
        ),
        (
            "xgb_d5_900_lr002_mcw1",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=4,
                max_depth=5,
                learning_rate=0.02,
                n_estimators=900,
                min_child_weight=1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=8.0,
            ),
        ),
        (
            "extra_trees_600_d3",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=3,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "random_forest_600_d4",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=4,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ]
    if LGBMClassifier is not None:
        configs.extend([
            (
                "lgbm_leaves7_400",
                LGBMClassifier(
                    objective="binary",
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=20,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    verbose=-1,
                ),
            ),
            (
                "lgbm_leaves15_700",
                LGBMClassifier(
                    objective="binary",
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                    n_estimators=700,
                    learning_rate=0.02,
                    num_leaves=15,
                    max_depth=4,
                    min_child_samples=12,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=5.0,
                    verbose=-1,
                ),
            ),
        ])
    return configs


def clone_model(model):
    import copy

    return copy.deepcopy(model)


def predict_positive(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def make_dataset(args) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = pd.read_csv(args.samples, parse_dates=["date"])
    feature_time_mode = getattr(args, "feature_time_mode", "eod")
    if str(feature_time_mode).lower() not in {"asof", "asof1455"}:
        df = add_reversal_features(df, args.intraday_bars)
    df = add_market_state_features(df)
    entry_col = "close_asof1455" if str(feature_time_mode).lower() in {"asof", "asof1455"} and "close_asof1455" in df.columns else "close"
    vwap_col = "vwap_asof1455" if str(feature_time_mode).lower() in {"asof", "asof1455"} and "vwap_asof1455" in df.columns else "daily_vwap"
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["next_day_close", entry_col, vwap_col]
    ).reset_index(drop=True)
    df = add_trade_returns(
        df,
        args.round_trip_cost_bps,
        args.target_hit_bps,
        getattr(args, "entry_policy", "vwap_low"),
        getattr(args, "entry_vwap_premium_bps", 50.0),
        feature_time_mode,
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["trade_net_close_return", "trade_net_high_return", "trade_target_or_close_return"]
    ).reset_index(drop=True)
    return df, feature_groups(df, args.max_missing, feature_time_mode)


def run_one(
    df: pd.DataFrame,
    cols: List[str],
    group_name: str,
    model_name: str,
    model_template,
    args,
) -> Tuple[List[Dict], List[pd.DataFrame], List[pd.DataFrame]]:
    rows = []
    pred_parts = []
    importance_parts = []
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    if args.label_mode == "hit":
        label_col = "trade_hit_label"
        return_col = "trade_target_or_close_return"
    elif args.label_mode == "close_profit":
        label_col = "trade_close_profit_label"
        return_col = "trade_net_close_return"
        if label_col not in df.columns:
            df[label_col] = (df["trade_net_close_return"] > 0).astype(int)
    else:
        raise ValueError(f"unknown label_mode={args.label_mode}")
    windows = iter_walk_forward_windows(df, args.train_rows, args.valid_rows, args.test_rows)
    for window_id, (start, train_end, valid_end, test_end) in enumerate(windows, start=1):
        train = df.iloc[start:train_end].copy()
        valid = df.iloc[train_end:valid_end].copy()
        test = df.iloc[valid_end:test_end].copy()
        entry_train = train["entry_signal"].to_numpy(bool)
        fit_train = train.loc[entry_train].copy()
        if len(fit_train) < args.min_train_entries or fit_train[label_col].nunique() < 2:
            continue
        apply = pd.concat([valid, test], ignore_index=False)
        x_train, x_apply = prepare_x_by_median(fit_train, apply, cols)
        y = fit_train[label_col].to_numpy(int)
        model = clone_model(model_template)
        params = model.get_params() if hasattr(model, "get_params") else {}
        if "scale_pos_weight" in params:
            pos = max(float(np.sum(y == 1)), 1.0)
            neg = max(float(np.sum(y == 0)), 1.0)
            model.set_params(scale_pos_weight=neg / pos)
        model.fit(x_train, y)
        score = predict_positive(model, x_apply)
        score_cols = list(dict.fromkeys([
            "date",
            "entry_signal",
            "trade_target_or_close_return",
            "trade_net_close_return",
            "trade_hit_label",
            label_col,
        ]))
        scored = apply[score_cols].copy()
        scored["selected_eval_return"] = scored[return_col]
        scored["feature_group"] = group_name
        scored["model_name"] = model_name
        scored["window_id"] = window_id
        scored["split"] = np.where(scored.index < valid_end, "valid", "test")
        scored["score"] = score
        scored["target_hit_bps"] = args.target_hit_bps
        scored["entry_policy"] = getattr(args, "entry_policy", "vwap_low")
        valid_scored = scored[scored["split"] == "valid"].copy()
        test_scored = scored[scored["split"] == "test"].copy()
        threshold = choose_valid_threshold(
            valid_scored.rename(columns={"score": "hit_score"}),
            "hit_score",
            quantiles,
            args.min_valid_trades,
            "selected_eval_return",
            "none",
        )
        if threshold is None:
            continue
        thr = threshold["threshold"]
        for split, part in [("valid", valid_scored), ("test", test_scored)]:
            chosen = part[part["entry_signal"] & (part["score"] >= thr)].copy()
            row = {
                "feature_group": group_name,
                "model_name": model_name,
                "window_id": window_id,
                "split": split,
                "target_hit_bps": args.target_hit_bps,
                "entry_policy": getattr(args, "entry_policy", "vwap_low"),
                "quantile": threshold["quantile"],
                "threshold": thr,
                "auc": float(roc_auc_score(part[label_col], part["score"]))
                if part[label_col].nunique() > 1 else np.nan,
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "valid_start": valid["date"].min(),
                "valid_end": valid["date"].max(),
                "test_start": test["date"].min(),
                "test_end": test["date"].max(),
            }
            row.update(trade_metrics(chosen["selected_eval_return"]))
            rows.append(row)
        selected = scored["entry_signal"] & (scored["score"] >= thr)
        scored["selected"] = selected.astype(int)
        scored["selected_return"] = np.where(selected, scored["selected_eval_return"], np.nan)
        scored["chosen_threshold"] = thr
        scored["chosen_quantile"] = threshold["quantile"]
        pred_parts.append(scored)
        if hasattr(model, "feature_importances_"):
            importance_parts.append(pd.DataFrame({
                "feature_group": group_name,
                "model_name": model_name,
                "window_id": window_id,
                "feature": cols,
                "importance": model.feature_importances_,
            }))
    return rows, pred_parts, importance_parts


def summarize(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = pred_df[pred_df["split"] == "test"].copy()
    group_cols = ["feature_group", "model_name", "target_hit_bps"]
    if "entry_policy" in test.columns:
        group_cols.insert(0, "entry_policy")
    for key_vals, part in test.groupby(group_cols, dropna=False):
        ret = part.loc[part["selected"] == 1, "selected_return"].dropna()
        row = dict(zip(group_cols, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        row.update(trade_metrics(ret))
        row["windows"] = int(part["window_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Search walk-forward model complexity")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "fundamental_features_out" / "training_samples_with_fundamentals.csv"))
    p.add_argument("--intraday-bars", default=str(SAVED_DATA_DIR / "dual_opp_out_002714_v12" / "raw_cache" / "002714_5m_raw.csv"))
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "walk_forward_model_complexity_out"))
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--target-hit-bps", type=float, default=50.0)
    p.add_argument("--label-mode", choices=["hit", "close_profit"], default="hit")
    p.add_argument("--entry-policy", choices=["vwap_low", "all_days"], default="vwap_low",
                   help="Entry universe: vwap_low=close<=daily_vwap*(1+premium), all_days=all valid labeled days")
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0,
                   help="VWAP premium threshold for entry-policy=vwap_low; 50 means close <= VWAP*1.005")
    p.add_argument("--feature-time-mode", choices=["eod", "asof", "asof1455"], default="eod")
    p.add_argument("--feature-cutoff-time", default="")
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--groups", default="reversal_fundamental_regime,all_no_ak")
    p.add_argument("--models", default="xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d4_700_lr002_mcw2,xgb_d5_900_lr002_mcw1,lgbm_leaves7_400,lgbm_leaves15_700")
    p.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    p.add_argument("--train-rows", type=int, default=756)
    p.add_argument("--valid-rows", type=int, default=126)
    p.add_argument("--test-rows", type=int, default=63)
    p.add_argument("--min-valid-trades", type=int, default=8)
    p.add_argument("--min-train-entries", type=int, default=80)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df, groups = make_dataset(args)
    selected_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    wanted_models = {m.strip() for m in args.models.split(",") if m.strip()}
    configs = [(name, model) for name, model in model_configs() if name in wanted_models]
    metric_rows = []
    pred_parts = []
    importance_parts = []
    for group_name in selected_groups:
        cols = groups.get(group_name, [])
        if not cols:
            continue
        for model_name, model in configs:
            rows, preds, imps = run_one(df, cols, group_name, model_name, model, args)
            metric_rows.extend(rows)
            pred_parts.extend(preds)
            importance_parts.extend(imps)
            pd.DataFrame(metric_rows).to_csv(out_dir / f"metrics_{int(args.target_hit_bps)}bps.partial.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / f"metrics_{int(args.target_hit_bps)}bps.csv", index=False, encoding="utf-8-sig")
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    if not pred_df.empty:
        pred_df.to_csv(out_dir / f"predictions_{int(args.target_hit_bps)}bps.csv", index=False, encoding="utf-8-sig")
    if importance_parts:
        pd.concat(importance_parts, ignore_index=True).to_csv(
            out_dir / f"feature_importance_{int(args.target_hit_bps)}bps.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = summarize(pred_df) if not pred_df.empty else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(["compound_return", "avg_return"], ascending=False)
        summary.to_csv(out_dir / f"summary_{int(args.target_hit_bps)}bps.csv", index=False, encoding="utf-8-sig")
    result = {
        "rows": int(len(df)),
        "target_hit_bps": args.target_hit_bps,
        "label_mode": args.label_mode,
        "entry_policy": args.entry_policy,
        "entry_vwap_premium_bps": args.entry_vwap_premium_bps,
        "feature_time_mode": args.feature_time_mode,
        "feature_cutoff_time": args.feature_cutoff_time,
        "groups": {k: len(groups.get(k, [])) for k in selected_groups},
        "models": [name for name, _ in configs],
        "top": summary.head(20).to_dict(orient="records") if not summary.empty else [],
    }
    with open(out_dir / f"summary_{int(args.target_hit_bps)}bps.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
