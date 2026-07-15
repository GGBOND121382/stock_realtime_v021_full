#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-aware one-fold-lag AS1455 prediction and backtest.

This entry point owns only the protocol plan:

    source fold(target+1) -> target fold

Feature construction, checkpoint inference, artifact writing, common CLI
arguments, signal specs, and grid execution live in ``utils``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_cli  # noqa: E402


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
    parser.add_argument("--fold-dir-template", default=None)
    parser.add_argument("--target-folds", default="0,1,2,3,4,5")
    parser.add_argument("--model-family", default=None)
    as1455_cli.add_prediction_grid_arguments(
        parser, default_output_mode="compact"
    )
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

    as1455_cli.normalize_common_prediction_args(args)
    args.target_folds_list = common.parse_int_list(args.target_folds)
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

    prediction_file = (
        as1455_cli.resolve_existing_prediction(args)
        if args.skip_predictions
        else make_one_lag_predictions(args)
    )
    if not args.skip_grid:
        as1455_cli.run_prediction_grid(
            args=args,
            prediction_file=prediction_file,
            model_family=(
                args.model_family
                or f"AS1455 {args.feature_preset} {args.target_col}"
            ),
            model_run=(
                f"{args.feature_preset} {args.target_col} one-fold-lag "
                f"search checkpoints; rebalance_every={args.rebalance_every}"
            ),
        )
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
