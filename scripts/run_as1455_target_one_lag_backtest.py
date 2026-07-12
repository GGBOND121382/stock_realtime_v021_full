#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-aware one-fold-lag AS1455 prediction and close-auction backtest.

This entry point owns only the one-fold-lag protocol:

    source fold(target+1) -> target fold

Feature construction, checkpoint loading, model inference, artifact writing, and
grid command construction are shared in ``utils.as1455_ch17_common``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def make_one_lag_predictions(args: argparse.Namespace) -> Path:
    features = common.build_target_features(
        Path(args.model_data),
        args.train_end,
        args.dropna_mode,
        args.target_col,
        args.feature_preset,
        args.sector_encoding,
    )

    all_predictions: list[pd.DataFrame] = []
    checkpoint_rows: list[dict[str, Any]] = []
    fold_mapping: list[dict[str, Any]] = []

    for target_fold in args.target_folds_list:
        source_fold = target_fold + 1
        source_dir = common.fold_dir_from_template(
            args.fold_dir_template, source_fold
        )
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)

        _train_idx, test_idx, target_report = common.get_fold_target(
            features.X, target_fold, args.target_col
        )
        predictions, selected_rows, source_manifest = (
            common.predict_checkpoint_set(
                features.X,
                test_idx,
                source_dir,
                args.top_n,
                metadata={
                    "source_fold": int(source_fold),
                    "target_fold": int(target_fold),
                },
            )
        )
        all_predictions.append(predictions)
        checkpoint_rows.extend(selected_rows)
        fold_mapping.append(
            {
                "source_fold": int(source_fold),
                "target_fold": int(target_fold),
                "source_dir": str(source_dir),
                "target_test_start": target_report["test_start"],
                "target_test_end": target_report["test_end"],
                "n_target_rows": int(len(test_idx)),
                "n_models": int(predictions.shape[1]),
                "source_train_start": source_manifest.get("train_start"),
                "source_train_end": source_manifest.get("train_end"),
                "source_test_start": source_manifest.get("test_start"),
                "source_test_end": source_manifest.get("test_end"),
                "n_model_input_features": int(
                    len(source_manifest.get("model_input_cols", []))
                ),
            }
        )
        print(
            f"[PRED] source fold{source_fold} -> target fold{target_fold}: "
            f"rows={len(predictions)} models={predictions.shape[1]}"
        )

    if not all_predictions:
        raise RuntimeError("no fold predictions generated; check --target-folds")

    predictions = pd.concat(all_predictions, axis=0).sort_index()
    return common.write_prediction_artifacts(
        out_root=Path(args.out_root),
        predictions=predictions,
        y=features.y,
        target_col=args.target_col,
        prediction_filename="test_preds.h5",
        manifest_filename="one_lag_prediction_manifest.json",
        checkpoint_filename="selected_checkpoints.csv",
        manifest={
            "protocol": "one_fold_lag_search_checkpoint_transfer",
            "feature_preset": args.feature_preset,
            "fold_mapping": fold_mapping,
            "feature_meta": features.report,
            "top_n": int(args.top_n),
            "portfolio_initial_state": "empty_positions_and_initial_cash",
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
            args.model_family
            or f"AS1455 {args.feature_preset} {args.target_col}"
        ),
        model_run=(
            f"{args.feature_preset} {args.target_col} one-fold-lag "
            f"search checkpoints; rebalance_every={args.rebalance_every}"
        ),
        force_grid=args.force_grid,
        smoke=args.smoke,
        parity_check_only=args.parity_check_only,
    )
    common.run_command(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-aware one-fold-lag AS1455 backtest"
    )
    parser.add_argument(
        "--feature-preset",
        choices=list(common.FEATURE_PRESETS),
        required=True,
    )
    parser.add_argument(
        "--target-col",
        choices=list(common.TARGET_SPECS),
        default="r05_fwd",
    )
    parser.add_argument("--rebalance-every", type=int, default=None)
    parser.add_argument("--offset-mode", choices=["zero", "full"], default=None)
    parser.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    parser.add_argument("--train-end", default=None)
    parser.add_argument(
        "--dropna-mode",
        choices=["target_only", "strict_original", "r01_only"],
        default="target_only",
    )
    parser.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    parser.add_argument("--fold-dir-template", default=None)
    parser.add_argument("--target-folds", default="0,1,2,3,4,5")
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
        default="compact",
        choices=["summary", "compact", "full"],
        help="File-retention level only; it does not change trading logic.",
    )
    parser.add_argument("--max-positions-list", default="5,10,15,20,25")
    parser.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    parser.add_argument("--python-bin", default=sys.executable or "python3")
    parser.add_argument("--model-family", default=None)
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
    if args.fold_dir_template is None:
        args.fold_dir_template = common.default_fold_dir_template(
            args.feature_preset, args.target_col
        )
    if args.out_root is None:
        args.out_root = str(
            common.default_one_lag_out_root(
                args.feature_preset,
                args.target_col,
                args.rebalance_every,
            )
        )
    if args.dropna_mode == "r01_only":
        if args.target_col != "r01_fwd":
            raise SystemExit("r01_only is valid only with --target-col r01_fwd")
        args.dropna_mode = "target_only"
    args.target_folds_list = common.parse_int_list(args.target_folds)
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    for fold in args.target_folds_list:
        if fold < 0 or fold > 5:
            raise SystemExit(
                "target folds must be in 0..5 for one-fold-lag mapping"
            )
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
        prediction_file = make_one_lag_predictions(args)

    if not args.skip_grid:
        run_grid(args, prediction_file)
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
