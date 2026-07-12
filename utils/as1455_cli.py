#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CLI helpers for AS1455 prediction-to-grid entry points."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import as1455_ch17_common as common
from utils import as1455_paths
from utils.as1455_signal_specs import append_signal_specs


def add_prediction_grid_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_mode: str,
) -> None:
    """Add the common prediction/grid arguments used by AS1455 protocols."""
    parser.add_argument("--model-data", default=str(as1455_paths.DEFAULT_MODEL_DATA))
    parser.add_argument("--train-end", default=None)
    parser.add_argument(
        "--dropna-mode",
        choices=["target_only", "strict_original", "r01_only"],
        default="target_only",
    )
    parser.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--grid-script", default=str(as1455_paths.DEFAULT_GRID_SCRIPT))
    parser.add_argument("--grid-out-root", default=None)
    parser.add_argument(
        "--raw-daily-cache-dir",
        default=str(as1455_paths.DEFAULT_RAW_DAILY_CACHE_DIR),
    )
    parser.add_argument("--profile", default="close_auction_skip_limit")
    parser.add_argument(
        "--capacity-mode",
        default="none",
        choices=["none", "last5_amount", "last5_volume", "last5_both"],
    )
    parser.add_argument(
        "--output-mode",
        default=default_output_mode,
        choices=["summary", "compact", "full"],
        help="File-retention level only; it does not change trading logic.",
    )
    parser.add_argument("--max-positions-list", default="5,10,15,20,25")
    parser.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    parser.add_argument("--python-bin", default=sys.executable or "python3")
    parser.add_argument("--force-grid", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--parity-check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def normalize_common_prediction_args(args: argparse.Namespace) -> None:
    if args.dropna_mode == "r01_only":
        if args.target_col != "r01_fwd":
            raise SystemExit("r01_only is valid only with --target-col r01_fwd")
        args.dropna_mode = "target_only"
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")


def run_prediction_grid(
    *,
    args: argparse.Namespace,
    prediction_file: Path,
    model_family: str,
    model_run: str,
) -> None:
    grid_out = (
        Path(args.grid_out_root)
        if args.grid_out_root
        else Path(args.out_root) / "01_close_auction_grid"
    )
    command = common.build_grid_command(
        python_bin=args.python_bin,
        grid_script=Path(args.grid_script),
        grid_out=grid_out,
        prediction_file=prediction_file,
        raw_daily_cache_dir=Path(args.raw_daily_cache_dir),
        profile=args.profile,
        capacity_mode=args.capacity_mode,
        output_mode=args.output_mode,
        offset_mode=args.offset_mode,
        rebalance_every=args.rebalance_every,
        max_positions_list=args.max_positions_list,
        sell_rank_list=args.sell_rank_list,
        model_family=model_family,
        model_run=model_run,
        force_grid=args.force_grid,
        smoke=args.smoke,
        parity_check_only=args.parity_check_only,
    )
    command = append_signal_specs(command, args.top_n)
    common.run_command(command, dry_run=args.dry_run)


def resolve_existing_prediction(args: argparse.Namespace) -> Path:
    if not args.prediction_file:
        raise SystemExit("--skip-predictions requires --prediction-file")
    prediction_file = Path(args.prediction_file)
    if not prediction_file.exists():
        raise FileNotFoundError(prediction_file)
    return prediction_file
