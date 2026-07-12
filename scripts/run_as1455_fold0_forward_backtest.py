#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use fold0 search-time checkpoints after the fold0 test window.

This entry point owns only the fold0-forward protocol:

- source model artifacts: fold0 search-time checkpoints/scaler/manifest;
- prediction dates: strictly later than fold0 ``test_end``;
- portfolio state: initial cash and no positions.

Feature construction, checkpoint inference, artifact writing, and grid command
construction are shared in ``utils.as1455_ch17_common``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import as1455_ch17_common as common

PROJECT_DIR = common.PROJECT_DIR
DEFAULT_MODEL_DATA = common.base.DEFAULT_MODEL_DATA
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


def build_forward_predictions(args: argparse.Namespace) -> Path:
    fold0_dir = Path(args.fold0_dir).expanduser().resolve()
    if not fold0_dir.exists():
        raise FileNotFoundError(fold0_dir)

    features = common.build_target_features(
        Path(args.model_data),
        args.train_end,
        args.dropna_mode,
        args.target_col,
        args.feature_preset,
        args.sector_encoding,
    )
    fold0_test_end = common.resolve_fold_test_end(fold0_dir)

    dates = pd.DatetimeIndex(features.X.index.get_level_values("date"))
    mask = dates > fold0_test_end
    if args.start_date:
        mask &= dates >= pd.Timestamp(args.start_date)
    if args.end_date:
        mask &= dates <= pd.Timestamp(args.end_date)
    forward_idx = np.flatnonzero(mask)
    if len(forward_idx) == 0:
        raise RuntimeError(
            "no forward rows after fold0 test_end; "
            f"fold0_test_end={fold0_test_end:%Y-%m-%d} "
            f"available_max={pd.Timestamp(dates.max()):%Y-%m-%d} "
            f"start_date={args.start_date} end_date={args.end_date}"
        )

    predictions, checkpoint_rows, _source_manifest = (
        common.predict_checkpoint_set(
            features.X,
            forward_idx,
            fold0_dir,
            args.top_n,
            metadata={"source_fold": 0},
        )
    )
    selected_dates = pd.DatetimeIndex(
        predictions.index.get_level_values("date")
    )
    print(
        f"[PRED] fold0 test_end={fold0_test_end:%Y-%m-%d} "
        f"forward={selected_dates.min():%Y-%m-%d}.."
        f"{selected_dates.max():%Y-%m-%d} "
        f"dates={selected_dates.nunique()} rows={len(predictions)}"
    )

    return common.write_prediction_artifacts(
        out_root=Path(args.out_root),
        predictions=predictions,
        y=features.y,
        target_col=args.target_col,
        prediction_filename="fold0_forward_preds.h5",
        manifest_filename="fold0_forward_prediction_manifest.json",
        checkpoint_filename="selected_fold0_checkpoints.csv",
        manifest={
            "protocol": "fold0_search_checkpoint_forward_test",
            "feature_preset": args.feature_preset,
            "fold0_dir": str(fold0_dir),
            "fold0_test_end": fold0_test_end.strftime("%Y-%m-%d"),
            "requested_start_date": args.start_date,
            "requested_end_date": args.end_date,
            "top_n": int(args.top_n),
            "portfolio_initial_state": "empty_positions_and_initial_cash",
            "feature_meta": features.report,
        },
        checkpoint_rows=checkpoint_rows,
        prediction_file=(
            Path(args.prediction_file) if args.prediction_file else None
        ),
    )


def run_grid(args: argparse.Namespace, prediction_file: Path) -> None:
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
        model_family=(
            f"AS1455 fold0 forward {args.feature_preset} {args.target_col}"
        ),
        model_run=(
            "fold0 search-time checkpoints; dates after fold0 test_end; "
            f"rebalance_every={args.rebalance_every}; empty start"
        ),
        force_grid=args.force_grid,
        smoke=args.smoke,
        parity_check_only=args.parity_check_only,
    )
    common.run_command(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest fold0 top checkpoints after fold0 test_end"
    )
    parser.add_argument(
        "--feature-preset",
        choices=list(common.FEATURE_PRESETS),
        required=True,
    )
    parser.add_argument(
        "--target-col",
        choices=list(common.TARGET_SPECS),
        required=True,
    )
    parser.add_argument("--rebalance-every", type=int, default=None)
    parser.add_argument("--offset-mode", choices=["zero", "full"], default=None)
    parser.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    parser.add_argument("--fold0-dir", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--dropna-mode",
        choices=["target_only", "strict_original"],
        default="target_only",
    )
    parser.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--grid-script", default=str(DEFAULT_GRID_SCRIPT))
    parser.add_argument("--grid-out-root", default=None)
    parser.add_argument(
        "--raw-daily-cache-dir",
        default=str(DEFAULT_RAW_DAILY_CACHE_DIR),
    )
    parser.add_argument("--profile", default="close_auction_skip_limit")
    parser.add_argument(
        "--capacity-mode",
        default="none",
        choices=["none", "last5_amount", "last5_volume", "last5_both"],
    )
    parser.add_argument(
        "--output-mode",
        default="full",
        choices=["summary", "compact", "full"],
    )
    parser.add_argument("--max-positions-list", default="5,10,15,20,25")
    parser.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    parser.add_argument("--python-bin", default=sys.executable or "python3")
    parser.add_argument("--force-grid", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--parity-check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = common.target_spec(args.target_col)
    if args.rebalance_every is None:
        args.rebalance_every = spec.rebalance_every
    if args.offset_mode is None:
        args.offset_mode = spec.offset_mode
    if args.fold0_dir is None:
        args.fold0_dir = str(
            common.default_fold0_dir(args.feature_preset, args.target_col)
        )
    if args.out_root is None:
        args.out_root = str(
            common.default_fold0_forward_out_root(
                args.feature_preset,
                args.target_col,
                args.rebalance_every,
            )
        )
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.skip_predictions:
        if not args.prediction_file:
            raise SystemExit("--skip-predictions requires --prediction-file")
        prediction_file = Path(args.prediction_file)
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)
    else:
        prediction_file = build_forward_predictions(args)

    if not args.skip_grid:
        run_grid(args, prediction_file)
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
