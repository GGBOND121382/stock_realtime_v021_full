#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract-safe entrypoint for the r21 fixed-signal nested matrix.

The implementation reuses helpers from ``run_as1455_nested_fold_protocol``.
Those helpers expect a generic argparse namespace containing ``target_col``.
The r21-specific parser intentionally does not expose a user-selectable target,
so this entrypoint injects the fixed value and validates the shared contract
before any expensive prediction or grid work begins.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_r21_fixed_signal_nested_matrix as implementation  # noqa: E402

ORIGINAL_PARSE_ARGS = implementation.parse_args
TARGET_COL = "r21_fwd"
REQUIRED_SHARED_FIELDS = (
    "target_col",
    "feature_preset",
    "python_bin",
    "raw_daily_cache_dir",
    "profile",
    "capacity_mode",
    "rebalance_every",
    "max_positions_list",
    "sell_rank_list",
    "initial_cash",
    "skip_parity_check",
    "raw_5m_cache_dir",
    "last5_panel",
    "universe",
    "st_symbols",
    "st_status",
    "corporate_actions",
    "target_output_mode",
)


def parse_args_with_shared_contract() -> Namespace:
    args = ORIGINAL_PARSE_ARGS()
    args.target_col = TARGET_COL
    missing = [name for name in REQUIRED_SHARED_FIELDS if not hasattr(args, name)]
    if missing:
        raise RuntimeError(
            "r21 nested namespace is incompatible with shared nested helpers; "
            f"missing fields: {missing}"
        )
    if args.target_col != TARGET_COL:
        raise RuntimeError(
            f"r21 nested target drift: expected={TARGET_COL} actual={args.target_col}"
        )
    return args


def main() -> None:
    implementation.parse_args = parse_args_with_shared_contract
    implementation.main()


if __name__ == "__main__":
    main()
