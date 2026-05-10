#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and save a stock-specific next-day model artifact."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"

from model_training.optimize_nextday_vwap_model import (
    choose_valid_threshold,
    ensure_dir,
    prepare_x_by_median,
    trade_metrics,
)
from model_training.search_walk_forward_model_complexity import (
    clone_model,
    make_dataset,
    model_configs,
    predict_positive,
)


DEFAULT_STOCK_CODE = "002714.SZ"
DEFAULT_FEATURE_GROUP = "all_no_ak"
DEFAULT_MODEL_NAME = "xgb_d4_500_lr002_mcw5"


def get_model_template(model_name: str):
    models = dict(model_configs())
    if model_name not in models:
        raise KeyError(f"unknown model_name={model_name}; available={sorted(models)}")
    return models[model_name]


def train_saved_model(args: argparse.Namespace) -> Dict:
    out_root = ensure_dir(args.out_dir)
    artifact_dir = ensure_dir(out_root / args.stock_code / args.artifact_name)

    df, groups = make_dataset(args)
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
    cols = groups.get(args.feature_group, [])
    if not cols:
        raise ValueError(f"empty feature group: {args.feature_group}")

    n = len(df)
    valid_rows = min(args.valid_rows, max(1, n // 5))
    train = df.iloc[:-valid_rows].copy()
    valid = df.iloc[-valid_rows:].copy()
    fit_train = train.loc[train["entry_signal"].to_numpy(bool)].copy()
    if len(fit_train) < args.min_train_entries:
        raise ValueError(f"not enough train entry rows: {len(fit_train)}")
    if fit_train[label_col].nunique() < 2:
        raise ValueError("training target has one class only")

    template = get_model_template(args.model_name)
    x_train, x_valid = prepare_x_by_median(fit_train, valid, cols)
    median = fit_train[cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    y = fit_train[label_col].to_numpy(int)
    model = clone_model(template)
    params = model.get_params() if hasattr(model, "get_params") else {}
    if "scale_pos_weight" in params:
        pos = max(float(np.sum(y == 1)), 1.0)
        neg = max(float(np.sum(y == 0)), 1.0)
        model.set_params(scale_pos_weight=neg / pos)
    model.fit(x_train, y)

    score = predict_positive(model, x_valid)
    valid_scored = valid[["date", "entry_signal", return_col, label_col]].copy()
    valid_scored = valid_scored.rename(columns={return_col: "selected_eval_return", label_col: "eval_label"})
    valid_scored["hit_score"] = score
    threshold_info = choose_valid_threshold(
        valid_scored,
        "hit_score",
        [float(x) for x in args.quantiles.split(",") if x.strip()],
        args.min_valid_trades,
        "selected_eval_return",
        "none",
    )
    if threshold_info is None:
        raise ValueError("could not choose validation threshold")
    threshold = float(threshold_info["threshold"])
    quantile = float(threshold_info["quantile"])
    valid_scored["selected"] = (valid_scored["entry_signal"].to_numpy(bool)) & (valid_scored["hit_score"] >= threshold)
    valid_scored["selected_return"] = np.where(
        valid_scored["selected"],
        valid_scored["selected_eval_return"],
        np.nan,
    )

    final_train = df.loc[df["entry_signal"].to_numpy(bool)].copy()
    x_final, _ = prepare_x_by_median(final_train, final_train, cols)
    final_median = final_train[cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    y_final = final_train[label_col].to_numpy(int)
    final_model = clone_model(template)
    params = final_model.get_params() if hasattr(final_model, "get_params") else {}
    if "scale_pos_weight" in params:
        pos = max(float(np.sum(y_final == 1)), 1.0)
        neg = max(float(np.sum(y_final == 0)), 1.0)
        final_model.set_params(scale_pos_weight=neg / pos)
    final_model.fit(x_final, y_final)

    model_path = artifact_dir / "model.joblib"
    med_path = artifact_dir / "feature_median.csv"
    cols_path = artifact_dir / "feature_columns.txt"
    valid_path = artifact_dir / "validation_tail_predictions.csv"
    meta_path = artifact_dir / "metadata.json"

    joblib.dump(final_model, model_path)
    final_median.rename("median").to_csv(med_path, encoding="utf-8-sig")
    cols_path.write_text("\n".join(cols) + "\n", encoding="utf-8")
    valid_scored.to_csv(valid_path, index=False, encoding="utf-8-sig")

    selected_ret = valid_scored.loc[valid_scored["selected"], "selected_return"].dropna()
    metadata = {
        "artifact_created_at": datetime.now().isoformat(timespec="seconds"),
        "stock_code": args.stock_code,
        "artifact_name": args.artifact_name,
        "samples": str(Path(args.samples).resolve()),
        "intraday_bars": str(Path(args.intraday_bars).resolve()),
        "feature_group": args.feature_group,
        "model_name": args.model_name,
        "label_mode": args.label_mode,
        "entry_policy": getattr(args, "entry_policy", "vwap_low"),
        "entry_vwap_premium_bps": getattr(args, "entry_vwap_premium_bps", 50.0),
        "label_col": label_col,
        "return_col": return_col,
        "target_hit_bps": args.target_hit_bps,
        "round_trip_cost_bps": args.round_trip_cost_bps,
        "threshold": float(threshold),
        "threshold_quantile": float(quantile),
        "feature_count": len(cols),
        "rows": int(len(df)),
        "train_rows_for_threshold": int(len(train)),
        "valid_rows_for_threshold": int(len(valid)),
        "final_train_entry_rows": int(len(final_train)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "validation_tail_auc": float(roc_auc_score(valid_scored["eval_label"], valid_scored["hit_score"]))
        if valid_scored["eval_label"].nunique() > 1 else None,
        "validation_tail_trade_metrics": trade_metrics(selected_ret),
        "files": {
            "model": str(model_path),
            "feature_median": str(med_path),
            "feature_columns": str(cols_path),
            "validation_tail_predictions": str(valid_path),
            "metadata": str(meta_path),
        },
        "usage_note": f"This artifact is stock-specific. Use only for {args.stock_code} with the same feature-building pipeline.",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save a stock-specific next-day model artifact")
    p.add_argument("--stock-code", default=DEFAULT_STOCK_CODE)
    p.add_argument("--artifact-name", default="nextday_hit_50bps_xgb_d4_hog_v1")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "hog_industry_features_out" / "training_samples_with_hog_industry.csv"))
    p.add_argument("--intraday-bars", default=str(SAVED_DATA_DIR / "dual_opp_out_002714_v12" / "raw_cache" / "002714_5m_raw.csv"))
    p.add_argument("--out-dir", default=str(SAVED_MODELS_DIR))
    p.add_argument("--feature-group", default=DEFAULT_FEATURE_GROUP)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--label-mode", choices=["hit", "close_profit"], default="hit")
    p.add_argument("--entry-policy", choices=["vwap_low", "all_days"], default="vwap_low",
                   help="Must match the entry policy used when selecting this configuration")
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--target-hit-bps", type=float, default=50.0)
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--valid-rows", type=int, default=252)
    p.add_argument("--min-train-entries", type=int, default=80)
    p.add_argument("--min-valid-trades", type=int, default=8)
    p.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    return p.parse_args()


if __name__ == "__main__":
    result = train_saved_model(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
