#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for r1 rotation+one-hot one-fold-lag backtests.

The implementation lives in ``run_as1455_target_one_lag_backtest.py``. This
file only supplies historical r1/A defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_paths  # noqa: E402

DEFAULT_MODEL_DATA = as1455_paths.DEFAULT_MODEL_DATA
DEFAULT_FOLD_DIR_TEMPLATE = common.default_fold_dir_template(
    "rotation_onehot", "r01_fwd"
)
DEFAULT_RAW_DAILY_CACHE_DIR = as1455_paths.DEFAULT_RAW_DAILY_CACHE_DIR
DEFAULT_GRID_SCRIPT = as1455_paths.DEFAULT_GRID_SCRIPT
DEFAULT_OUT_ROOT = common.default_one_lag_out_root(
    "rotation_onehot", "r01_fwd", 1
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
