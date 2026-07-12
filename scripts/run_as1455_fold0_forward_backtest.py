#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use fold0 search-time checkpoints after the fold0 test window.

This entry point owns only the fold0-forward date selection. Shared feature,
checkpoint, artifact, CLI, signal-spec, and grid logic live in ``utils``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_cli  # noqa: E402


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
    parser.add_argument("--fold0-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    as1455_cli.add_prediction_grid_arguments(
        parser, default_output_mode="full"
    )
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
    as1455_cli.normalize_common_prediction_args(args)
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prediction_file = (
        as1455_cli.resolve_existing_prediction(args)
        if args.skip_predictions
        else build_forward_predictions(args)
    )
    if not args.skip_grid:
        as1455_cli.run_prediction_grid(
            args=args,
            prediction_file=prediction_file,
            model_family=(
                f"AS1455 fold0 forward {args.feature_preset} {args.target_col}"
            ),
            model_run=(
                "fold0 search-time checkpoints; dates after fold0 test_end; "
                f"rebalance_every={args.rebalance_every}; empty start"
            ),
        )
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
