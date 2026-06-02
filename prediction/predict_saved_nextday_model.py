#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load a stock-specific saved next-day model artifact and score prepared rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"

from model_training.optimize_nextday_vwap_model import add_market_state_features, add_reversal_features, compute_entry_signal
from model_training.search_walk_forward_model_complexity import make_dataset
from model_training.search_walk_forward_model_complexity import validate_saved_data_pipeline_input


def load_artifact(artifact_dir: str | Path, stock_code: str) -> tuple[object, list[str], pd.Series, dict]:
    artifact = Path(artifact_dir)
    meta = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("stock_code") != stock_code:
        raise ValueError(f"stock_code mismatch: artifact={meta.get('stock_code')} requested={stock_code}")
    model = joblib.load(artifact / "model.joblib")
    cols = [line.strip() for line in (artifact / "feature_columns.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    med = pd.read_csv(artifact / "feature_median.csv", index_col=0)["median"]
    return model, cols, med, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Score rows with a saved stock-specific next-day model")
    p.add_argument("--artifact-dir", default=str(SAVED_MODELS_DIR / "002714.SZ" / "nextday_hit_50bps_xgb_d4_hog_v1"))
    p.add_argument("--stock-code", default="002714.SZ")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "002714_pipeline_out" / "03_sector" / "training_samples_with_sector.csv"))
    p.add_argument("--intraday-bars", default=str(SAVED_DATA_DIR / "002714_pipeline_out" / "00_base" / "002714_5m.csv"))
    p.add_argument("--entry-policy", choices=["vwap_low", "all_days"], default="vwap_low",
                   help="Used only when artifact metadata has no entry_policy")
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--target-hit-bps", type=float, default=50.0)
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--date", help="Optional YYYY-MM-DD date to score")
    p.add_argument("--allow-unlabeled", action="store_true", help="Score rows without next-day labels; useful for latest close prediction")
    p.add_argument("--out", default=str(SAVED_DATA_DIR / "predictions" / "002714.SZ_nextday_hit_50bps_xgb_d4_hog_v1_latest_scores.csv"))
    args = p.parse_args()

    model, cols, med, meta = load_artifact(args.artifact_dir, args.stock_code)
    if args.allow_unlabeled:
        validate_saved_data_pipeline_input(args.samples, "samples")
        validate_saved_data_pipeline_input(args.intraday_bars, "intraday-bars")
        df = pd.read_csv(args.samples, parse_dates=["date"])
        df = add_reversal_features(df, args.intraday_bars)
        df = add_market_state_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    else:
        df, _groups = make_dataset(args)
    if args.date:
        target_date = pd.to_datetime(args.date)
        df = df[df["date"] == target_date].copy()
    if df.empty:
        raise ValueError("no rows to score")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"samples missing required features: {missing[:20]} ... total={len(missing)}")
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    score = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
    entry_policy = str(meta.get("entry_policy") or getattr(args, "entry_policy", "vwap_low"))
    entry_vwap_premium_bps = float(meta.get("entry_vwap_premium_bps", getattr(args, "entry_vwap_premium_bps", 50.0)))
    # Always recompute entry_signal from artifact metadata instead of trusting a
    # stale column in the sample file. This matters when scoring all_days models
    # from samples created by the older vwap_low pipeline.
    entry_signal = compute_entry_signal(df, entry_policy, entry_vwap_premium_bps).astype(bool)

    out = df[["date"]].copy()
    out["stock_code"] = args.stock_code
    out["artifact_name"] = meta["artifact_name"]
    out["feature_status"] = "offline_complete_or_sample_based"
    out["entry_policy"] = entry_policy
    out["entry_vwap_premium_bps"] = entry_vwap_premium_bps
    out["entry_signal"] = entry_signal.to_numpy(bool)
    if "close" in df.columns:
        out["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "daily_vwap" in df.columns:
        out["daily_vwap"] = pd.to_numeric(df["daily_vwap"], errors="coerce")
    out["hit_score"] = score
    out["threshold"] = float(meta["threshold"])
    out["signal_raw_score_pass"] = out["hit_score"] >= out["threshold"]
    out["signal"] = out["entry_signal"] & out["signal_raw_score_pass"]
    out["missing_feature_count"] = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.tail(10).to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
