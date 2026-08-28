#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R21 fixed-signal nested validation matrix with shared predictions.

For every available source fold k:

1. infer the source fold's own held-out validation window with its locally ranked
   top-five checkpoints;
2. independently grid-search trading parameters for each fixed signal
   (top-five mean, top-three mean, and best single checkpoint);
3. freeze each signal's complete winning configuration;
4. apply it once to target fold k-1, or to the latest strict-forward window when
   k == 0.

The three signal variants share validation/target prediction HDF files. Target
fold and forward outcomes never participate in selection. Existing model
checkpoints are reused; this script does not train models or refresh market data.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_nested_fold_protocol as nested  # noqa: E402
from scripts.resolve_as1455_fixed_signal_matrix_folds import resolve_target  # noqa: E402
from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_forward_features import build_inference_features  # noqa: E402

SIGNALS: dict[str, dict[str, str]] = {
    "all5": {
        "spec": "ensemble_all5_mean:0,1,2,3,4:mean",
        "grid_script": "scripts/run_as1455_close_auction_grid_fixed_all5_ensemble.py",
    },
    "first3": {
        "spec": "ensemble_first3_mean:0,1,2:mean",
        "grid_script": "scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py",
    },
    "best": {
        "spec": "model_0:0:single",
        "grid_script": "scripts/run_as1455_close_auction_grid_fixed_best_model.py",
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=nested.bt.json_default),
        encoding="utf-8",
    )


def parse_source_folds(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid --source-folds={text!r}")
    if any(value < 0 or value > 6 for value in values):
        raise ValueError(f"source folds must lie in 0..6: {values}")
    if 0 not in values:
        raise ValueError("source_fold0 is required for strict-forward evaluation")
    return sorted(values, reverse=True)


def resolve_source_folds(args: argparse.Namespace) -> tuple[list[int], dict[str, Any]]:
    if args.source_folds != "auto":
        folds = parse_source_folds(args.source_folds)
        return folds, {
            "mode": "explicit",
            "source_folds": folds,
            "target_folds": [fold - 1 for fold in folds if fold > 0],
        }

    plan = resolve_target("r21_fwd", args.top_n)
    folds = sorted({0, *[int(value) for value in plan["source_folds"]]}, reverse=True)
    return folds, {
        "mode": "auto_checkpoint_availability",
        **plan,
        "nested_source_folds": folds,
    }


def signal_args(args: argparse.Namespace, signal_kind: str, *, force: bool) -> argparse.Namespace:
    value = copy.copy(args)
    value.grid_script = str((PROJECT_DIR / SIGNALS[signal_kind]["grid_script"]).resolve())
    value.force = bool(force)
    return value


def result_row(record: dict[str, Any], signal_kind: str) -> dict[str, Any]:
    summary = record["summary"]
    selection = record["selection"]
    return {
        "signal_kind": signal_kind,
        "signal_spec": SIGNALS[signal_kind]["spec"],
        "segment": record["segment"],
        "source_fold": record["source_fold"],
        "target_fold": record["target_fold"],
        "selection_signal": selection["signal_spec"],
        "selection_max_positions": selection["historical_max_positions"],
        "selection_sell_rank": selection["historical_sell_rank"],
        "selection_offset": selection["historical_rebalance_offset"],
        "validation_start": selection["historical_date_min"],
        "validation_end": selection["historical_date_max"],
        "validation_n_days": selection["historical_n_days"],
        "target_start": record["target_start"],
        "target_end": record["target_end"],
        "target_n_days": record["target_n_days"],
        "total_return": summary.get("total_return"),
        "annual_return": summary.get("annual_return"),
        "sharpe": summary.get("sharpe"),
        "max_drawdown": summary.get("max_drawdown"),
        "run_dir": record["run_dir"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R21 per-source-fold validation grids for fixed all5/first3/best "
            "signals followed by frozen next-fold and strict-forward backtests"
        )
    )
    parser.add_argument(
        "--historical-model-data",
        default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5",
    )
    parser.add_argument(
        "--forward-model-data",
        default=(
            "saved_data/ashare_ml4t/ch12_as1455_forward_latest/"
            "model_data_as1455.h5"
        ),
    )
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument("--fold-dir-template", default=None)
    parser.add_argument("--source-folds", default="auto")
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--raw-daily-cache-dir",
        default="saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache",
    )
    parser.add_argument("--raw-5m-cache-dir", default=None)
    parser.add_argument("--last5-panel", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--st-symbols", default=None)
    parser.add_argument("--st-status", default=None)
    parser.add_argument("--corporate-actions", default=None)
    parser.add_argument("--python-bin", default=sys.executable or "python3")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--selection-rank-metric", default="sharpe")
    parser.add_argument("--max-positions-list", default="5,10,15,20,25")
    parser.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--profile", default="close_auction_skip_limit")
    parser.add_argument("--capacity-mode", default="none")
    parser.add_argument("--initial-cash", type=float, default=200000.0)
    parser.add_argument(
        "--validation-output-mode",
        choices=["summary", "compact", "full"],
        default="summary",
    )
    parser.add_argument(
        "--target-output-mode", choices=["compact", "full"], default="compact"
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-forward-predictions",
        action="store_true",
        help="Do not rebuild the shared fold0-forward prediction HDF.",
    )
    parser.add_argument(
        "--reuse-forward-results",
        action="store_true",
        help="Do not force replacement of the three strict-forward results.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-parity-check", action="store_true")
    parser.add_argument("--skip-continuous", action="store_true")
    args = parser.parse_args()
    if args.top_n < 5:
        raise SystemExit("--top-n must be at least 5 for the all5 signal")
    if args.rebalance_every != 21:
        raise SystemExit("this experiment requires --rebalance-every=21")
    if args.capacity_mode != "none":
        raise SystemExit("this controlled experiment currently requires capacity_mode=none")
    if args.initial_cash <= 0:
        raise SystemExit("--initial-cash must be positive")
    if args.fold_dir_template is None:
        args.fold_dir_template = common.default_fold_dir_template(
            args.feature_preset, "r21_fwd"
        )
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    shared_root = out_root / "shared_predictions"

    historical_model_data = Path(args.historical_model_data).expanduser().resolve()
    forward_model_data = Path(args.forward_model_data).expanduser().resolve()
    raw_daily_cache = Path(args.raw_daily_cache_dir).expanduser().resolve()
    for path in (historical_model_data, forward_model_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not raw_daily_cache.is_dir():
        raise FileNotFoundError(raw_daily_cache)

    source_folds, fold_plan = resolve_source_folds(args)
    target_folds = [fold - 1 for fold in source_folds if fold > 0]
    write_json(out_root / "fold_availability_plan.json", fold_plan)

    historical_features = common.build_target_features(
        historical_model_data,
        None,
        "target_only",
        "r21_fwd",
        args.feature_preset,
        "onehot",
    )
    forward_features = build_inference_features(
        forward_model_data,
        None,
        "r21_fwd",
        args.feature_preset,
        "onehot",
    )

    records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SIGNALS}
    signal_roots = {kind: out_root / kind for kind in SIGNALS}

    print(
        "[MODE] r21 fixed-signal nested matrix: "
        f"source_folds={source_folds} target_folds={target_folds}"
    )
    print(
        "[MODE] per-source validation grid per signal="
        f"5x6x21={5 * 6 * 21}; signals={list(SIGNALS)}"
    )
    print("[MODE] shared predictions across all three fixed signals")
    print("[MODE] training=false data_refresh=false")

    for source_fold in source_folds:
        target_fold = source_fold - 1 if source_fold > 0 else None
        source_dir = common.fold_dir_from_template(args.fold_dir_template, source_fold)
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)

        shared_source_root = shared_root / f"source_fold{source_fold}"
        validation_idx, source_report, validation_dates = nested.validation_window(
            historical_features, source_fold, "r21_fwd"
        )
        validation_pred = nested.prediction_artifact(
            out_root=shared_source_root / "validation_selection",
            features=historical_features,
            row_idx=validation_idx,
            source_dir=source_dir,
            source_fold=source_fold,
            target_col="r21_fwd",
            top_n=args.top_n,
            protocol="source_fold_validation_grid_shared_fixed_signals",
            target_fold=source_fold,
            filename="validation_preds.h5",
            force=args.force,
        )

        if target_fold is not None:
            target_idx, target_report = nested.target_prediction_indices(
                historical_features, target_fold, "r21_fwd"
            )
            target_pred = nested.prediction_artifact(
                out_root=shared_source_root / f"target_fold{target_fold}",
                features=historical_features,
                row_idx=target_idx,
                source_dir=source_dir,
                source_fold=source_fold,
                target_col="r21_fwd",
                top_n=args.top_n,
                protocol="frozen_source_fold_to_next_target_fold_shared_fixed_signals",
                target_fold=target_fold,
                filename="target_preds.h5",
                force=args.force,
            )
            segment = f"target_fold{target_fold}"
        else:
            fold0_end = common.resolve_fold_test_end(source_dir)
            target_idx = nested.forward_prediction_indices(
                forward_features,
                fold0_end,
                args.start_date,
                args.end_date,
            )
            target_dates_from_features = nested.normalized_dates(
                forward_features.X.iloc[target_idx].index.get_level_values("date")
            )
            target_report = {
                "target_fold": None,
                "test_start": target_dates_from_features[0],
                "test_end": target_dates_from_features[-1],
            }
            target_pred = nested.prediction_artifact(
                out_root=shared_source_root / "forward",
                features=forward_features,
                row_idx=target_idx,
                source_dir=source_dir,
                source_fold=source_fold,
                target_col="r21_fwd",
                top_n=args.top_n,
                protocol="frozen_fold0_to_latest_strict_forward_shared_fixed_signals",
                target_fold=None,
                filename="forward_preds.h5",
                force=args.force or not args.reuse_forward_predictions,
            )
            segment = "strict_oos_forward"

        target_dates = nested.normalized_dates(
            pd.read_hdf(target_pred, "predictions").index.get_level_values("date")
        )

        for signal_kind in SIGNALS:
            source_root = signal_roots[signal_kind] / f"source_fold{source_fold}"
            validation_root = source_root / "validation_selection"
            validation_args = signal_args(args, signal_kind, force=args.force)
            selection = nested.run_validation_grid(
                validation_args,
                root=validation_root,
                prediction_file=validation_pred,
            )
            selection = nested.with_validation_phase(
                selection, source_report, validation_dates
            )
            if selection.signal_spec != SIGNALS[signal_kind]["spec"]:
                raise RuntimeError(
                    f"fixed signal drift for {signal_kind}: "
                    f"expected={SIGNALS[signal_kind]['spec']} "
                    f"actual={selection.signal_spec}"
                )
            selection_payload = selection.to_dict()
            write_json(
                source_root / "selected_for_next_window.json",
                {
                    "protocol": "fixed_signal_source_validation_select_then_freeze",
                    "signal_kind": signal_kind,
                    "signal_spec": SIGNALS[signal_kind]["spec"],
                    "source_fold": source_fold,
                    "target_fold": target_fold,
                    "validation_fold_report": source_report,
                    "validation_dates": {
                        "start": validation_dates[0],
                        "end": validation_dates[-1],
                        "n_days": len(validation_dates),
                    },
                    "selection": selection_payload,
                    "target_results_used_for_selection": False,
                },
            )

            target_root = (
                source_root / f"target_fold{target_fold}"
                if target_fold is not None
                else source_root / "forward"
            )
            target_args = signal_args(
                args,
                signal_kind,
                force=(
                    args.force
                    or (target_fold is None and not args.reuse_forward_results)
                ),
            )
            strict_manifest = nested.run_frozen_target(
                target_args,
                root=target_root,
                prediction_file=target_pred,
                selection=selection,
            )
            run_dir, config_file = nested.retained_run(target_root, strict_manifest)
            summary = nested.read_json(run_dir / "summary.json")
            record = {
                "segment": segment,
                "signal_kind": signal_kind,
                "source_fold": source_fold,
                "target_fold": target_fold,
                "source_dir": str(source_dir),
                "shared_validation_prediction_file": str(validation_pred),
                "validation_root": str(validation_root),
                "selection": selection_payload,
                "prediction_file": str(target_pred),
                "target_root": str(target_root),
                "run_dir": str(run_dir),
                "config_file": str(config_file),
                "summary_file": str(run_dir / "summary.json"),
                "strict_manifest": str(
                    target_root / "01_close_auction_grid" / "strict_oos_manifest.json"
                ),
                "target_start": target_dates[0],
                "target_end": target_dates[-1],
                "target_n_days": len(target_dates),
                "target_fold_report": target_report,
                "summary": summary,
            }
            records[signal_kind].append(record)
            print(
                f"[OK] signal={signal_kind} source_fold{source_fold} "
                f"validation={validation_dates[0]:%Y-%m-%d}.."
                f"{validation_dates[-1]:%Y-%m-%d} -> {segment}"
            )

    matrix_rows: list[dict[str, Any]] = []
    for signal_kind, signal_records in records.items():
        signal_root = signal_roots[signal_kind]
        signal_root.mkdir(parents=True, exist_ok=True)
        rows = [result_row(record, signal_kind) for record in signal_records]
        table = pd.DataFrame(rows)
        table.to_csv(
            signal_root / "nested_fold_target_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        matrix_rows.extend(rows)

        continuous = None
        if not args.skip_continuous:
            continuous_args = signal_args(args, signal_kind, force=False)
            continuous = nested.run_continuous_account(
                continuous_args, signal_records, signal_root
            )

        signal_manifest = {
            "protocol": "r21_fixed_signal_nested_per_source_fold_validation_grid",
            "signal_kind": signal_kind,
            "fixed_signal_spec": SIGNALS[signal_kind]["spec"],
            "feature_preset": args.feature_preset,
            "target_col": "r21_fwd",
            "source_folds": source_folds,
            "target_folds": target_folds,
            "model_training": False,
            "data_refresh": False,
            "historical_model_data": str(historical_model_data),
            "forward_model_data": str(forward_model_data),
            "shared_prediction_root": str(shared_root),
            "validation_grid_count": len(source_folds),
            "validation_configurations_per_fold": 5 * 6 * 21,
            "target_grid_count": 0,
            "target_fixed_backtest_count": len(source_folds),
            "global_concatenated_target_grid": False,
            "target_results_used_for_selection": False,
            "forward_results_used_for_selection": False,
            "records": signal_records,
            "continuous": continuous,
        }
        write_json(signal_root / "nested_fold_protocol_manifest.json", signal_manifest)

    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(
        out_root / "r21_fixed_signal_nested_matrix_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "status": "ok",
        "protocol": "r21_three_fixed_signals_nested_validation_then_next_window",
        "signals": {kind: value["spec"] for kind, value in SIGNALS.items()},
        "source_folds": source_folds,
        "target_folds": target_folds,
        "fold_availability_plan": fold_plan,
        "shared_predictions": True,
        "validation_grid_runs": len(source_folds) * len(SIGNALS),
        "validation_configurations_per_run": 5 * 6 * 21,
        "total_validation_configurations": len(source_folds) * len(SIGNALS) * 5 * 6 * 21,
        "frozen_evaluation_count": len(source_folds) * len(SIGNALS),
        "signal_manifests": {
            kind: str(signal_roots[kind] / "nested_fold_protocol_manifest.json")
            for kind in SIGNALS
        },
        "duration_seconds": int(round(time.time() - started)),
    }
    write_json(out_root / "r21_fixed_signal_nested_matrix_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
