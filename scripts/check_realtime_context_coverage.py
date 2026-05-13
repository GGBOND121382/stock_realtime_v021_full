#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether saved models have realtime context coverage.

This is a thin wrapper around data_collection/collect_realtime_context.py plan.
It writes/reads realtime_context_plan.csv and fails if any model has missing
context config features.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check realtime context coverage for saved models")
    p.add_argument("--models-dir", default="saved_models")
    p.add_argument("--watchlist", default="selected_watchlist.txt")
    p.add_argument("--config", default="configs/realtime_context_sources.toml")
    p.add_argument("--out-dir", default="saved_data/realtime_context")
    p.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    p.add_argument("--model-policy", choices=["all", "preferred"], default="all")
    p.add_argument("--allow-missing", action="store_true", help="Do not exit nonzero when missing config features exist")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "data_collection" / "collect_realtime_context.py"),
        "plan",
        "--models-dir", args.models_dir,
        "--watchlist", args.watchlist,
        "--config", args.config,
        "--out-dir", args.out_dir,
        "--date", args.date,
        "--model-policy", args.model_policy,
        "--refresh-plan",
    ]
    print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(PROJECT_DIR))
    if rc != 0:
        return rc

    plan_path = PROJECT_DIR / args.out_dir / args.date / "realtime_context_plan.csv"
    if not plan_path.exists():
        print(f"ERROR: plan not found: {plan_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(plan_path)
    if df.empty:
        print("No saved models in watchlist; realtime context plan is empty.")
        return 0

    cols = [
        "stock_code", "artifact_name", "requires_sector_context", "sector_symbols",
        "context_groups", "required_context_features", "missing_context_config_features",
    ]
    print("\n[REALTIME CONTEXT COVERAGE]")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    missing = df[df.get("missing_context_config_features", "").fillna("").astype(str).str.strip() != ""]
    if not missing.empty:
        print("\n[MISSING CONTEXT CONFIG FEATURES]")
        print(missing[["stock_code", "artifact_name", "missing_context_config_features"]].to_string(index=False))
        return 0 if args.allow_missing else 1

    print("\nOK: no missing realtime context config features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
