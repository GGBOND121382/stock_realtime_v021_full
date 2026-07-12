#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for r1 rotation+one-hot one-fold-lag backtests.

The implementation lives in ``run_as1455_target_one_lag_backtest.py``.  This
file only supplies the historical r1/A defaults so existing commands keep
working without maintaining a second prediction pipeline.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_MODEL_DATA = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch12_as1455"
    / "model_data_as1455.h5"
)
DEFAULT_FOLD_DIR_TEMPLATE = str(
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_sector_rotation_onehot_fold{fold}_search"
)
DEFAULT_RAW_DAILY_CACHE_DIR = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch12_as1455"
    / "baostock_raw_daily_cache"
)
DEFAULT_GRID_SCRIPT = (
    PROJECT_DIR
    / "code"
    / "backtest"
    / "run_as1455_close_auction_grid_inprocess.py"
)
DEFAULT_OUT_ROOT = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / f"ch17_as1455_rotation_one_lag_daily_backtest_{datetime.now():%Y%m%d}"
)


def main() -> None:
    from scripts.run_as1455_target_one_lag_backtest import main as target_main

    defaults = [
        "--feature-preset",
        "rotation_onehot",
        "--target-col",
        "r01_fwd",
        "--rebalance-every",
        "1",
        "--offset-mode",
        "zero",
        "--fold-dir-template",
        DEFAULT_FOLD_DIR_TEMPLATE,
        "--out-root",
        str(DEFAULT_OUT_ROOT),
        "--model-family",
        "AS1455 rotation one-lag NN",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    target_main()


if __name__ == "__main__":
    main()
