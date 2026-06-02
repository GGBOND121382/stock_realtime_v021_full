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
from model_training.missing_feature_logging import (
    feature_missing_report,
    log_and_write_feature_missing_report,
    matrix_missing_stats,
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
    missing_report = feature_missing_report(
        df,
        {args.feature_group: cols},
        max_missing=args.max_missing,
        sample_path=args.samples,
    )
    log_and_write_feature_missing_report(
        missing_report,
        artifact_dir,
        filename="feature_missing_report.csv",
        context=f"stock={args.stock_code} artifact={args.artifact_name}",
    )

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
    train_missing = matrix_missing_stats(fit_train, cols, "train")
    valid_missing = matrix_missing_stats(valid, cols, "valid")
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

    model_path = artifact_dir / "model.joblib"
    med_path = artifact_dir / "feature_median.csv"
    cols_path = artifact_dir / "feature_columns.txt"
    valid_path = artifact_dir / "validation_tail_predictions.csv"
    meta_path = artifact_dir / "metadata.json"

    # Save the exact model and feature medians used to produce
    # validation_tail_predictions.csv and choose the threshold.  Re-fitting a
    # second "final" model on all rows changes score calibration, so the saved
    # threshold/validation metrics no longer describe model.joblib.
    joblib.dump(model, model_path)
    median.rename("median").to_csv(med_path, encoding="utf-8-sig")
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
        "feature_time_mode": getattr(args, "feature_time_mode", "eod"),
        "feature_cutoff_time": getattr(args, "feature_cutoff_time", ""),
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
        "train_entry_rows_for_saved_model": int(len(fit_train)),
        "feature_missing_stats": {
            **train_missing,
            **valid_missing,
        },
        "saved_model_scope": "validation_train",
        "saved_model_matches_validation_tail_predictions": True,
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
        "usage_note": (
            f"This artifact is stock-specific. Use only for {args.stock_code} "
            "with the same feature-building pipeline. model.joblib is the "
            "validation-train model whose scores produced "
            "validation_tail_predictions.csv and threshold."
        ),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save a stock-specific next-day model artifact")
    p.add_argument("--stock-code", default=DEFAULT_STOCK_CODE)
    p.add_argument("--artifact-name", default="nextday_hit_50bps_xgb_d4_hog_v1")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "002714_pipeline_out" / "03_sector" / "training_samples_with_sector.csv"))
    p.add_argument("--intraday-bars", default=str(SAVED_DATA_DIR / "002714_pipeline_out" / "00_base" / "002714_5m.csv"))
    p.add_argument("--out-dir", default=str(SAVED_MODELS_DIR))
    p.add_argument("--feature-group", default=DEFAULT_FEATURE_GROUP)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--label-mode", choices=["hit", "close_profit"], default="hit")
    p.add_argument("--entry-policy", choices=["vwap_low", "all_days"], default="vwap_low",
                   help="Must match the entry policy used when selecting this configuration")
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)
    p.add_argument("--feature-time-mode", choices=["eod", "asof", "asof1455"], default="eod")
    p.add_argument("--feature-cutoff-time", default="")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--target-hit-bps", type=float, default=50.0)
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--valid-rows", type=int, default=126)
    p.add_argument("--min-train-entries", type=int, default=80)
    p.add_argument("--min-valid-trades", type=int, default=8)
    p.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    return p.parse_args()


if __name__ == "__main__":
    result = train_saved_model(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
