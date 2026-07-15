#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for r1 add-on one-fold-lag backtests.

The implementation lives in ``run_as1455_target_one_lag_backtest.py``. This
file only supplies historical r1/B defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402

DEFAULT_ADDON_FOLD_DIR_TEMPLATE = common.default_fold_dir_template(
    "rotation_addon_onehot", "r01_fwd"
)
DEFAULT_ADDON_OUT_ROOT = common.default_one_lag_out_root(
    "rotation_addon_onehot", "r01_fwd", 1
)


def main() -> None:
    from scripts.run_as1455_target_one_lag_backtest import main as target_main

    defaults = [
        "--feature-preset",
        "rotation_addon_onehot",
        "--target-col",
        "r01_fwd",
        "--rebalance-every",
        "1",
        "--offset-mode",
        "zero",
        "--fold-dir-template",
        DEFAULT_ADDON_FOLD_DIR_TEMPLATE,
        "--out-root",
        str(DEFAULT_ADDON_OUT_ROOT),
        "--model-family",
        "AS1455 full-rotation compact-add-on one-lag NN",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    target_main()


if __name__ == "__main__":
    main()
