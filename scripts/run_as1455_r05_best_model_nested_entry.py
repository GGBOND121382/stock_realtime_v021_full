#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the r05 nested protocol with each source fold's best checkpoint fixed.

The historical validation and target-fold windows are built from the original
historical model-data file. Only the fold0 strict-forward feature matrix is
built from the separately refreshed forward model-data file.

This entry also adds low-memory resume guards for 8 GB servers:

* completed 150-row validation grids are selected directly without reloading
  their execution panels;
* completed frozen target runs are reused from their strict-OOS manifests;
* the historical feature matrix is released after the fold0 validation
  prediction is materialized;
* the latest forward feature matrix is released after its prediction HDF is
  materialized, before the frozen forward backtest subprocess starts.
"""
from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_nested_fold_protocol as base  # noqa: E402

TARGET_COL = "r05_fwd"
FIXED_SIGNAL_SPEC = "model_0:0:single"
EXPECTED_VALIDATION_GRID_ROWS = 150


def pop_option(name: str) -> str:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[index + 1]
            del args[index : index + 2]
            sys.argv[1:] = args
            return value
        prefix = name + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            del args[index]
            sys.argv[1:] = args
            return value
    raise SystemExit(f"{name} is required")


def option_value(name: str) -> str | None:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def trim_process_memory(label: str) -> None:
    """Collect Python objects and ask glibc to return free arenas to the OS."""
    collected = gc.collect()
    malloc_trimmed = False
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trimmed = bool(libc.malloc_trim(0))
    except (AttributeError, OSError):
        pass
    print(
        f"[MEMORY] {label}: gc_collected={collected} "
        f"malloc_trimmed={malloc_trimmed}",
        flush=True,
    )


def release_feature_matrix(features: Any, label: str) -> None:
    """Drop the large DataFrames while retaining lightweight feature metadata."""
    rows = int(len(features.X))
    columns = int(features.X.shape[1])
    features.X = pd.DataFrame()
    features.y = pd.Series(dtype="float64")
    trim_process_memory(f"released {label} rows={rows} columns={columns}")


def completed_validation_grid(root: Path) -> tuple[bool, int, Path | None]:
    candidates = [
        root / "01_close_auction_grid" / "grid_summary.csv",
        root / "01_close_auction_grid" / "02_summary" / "grid_summary.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        if "status" in table.columns:
            table = table.loc[
                table["status"].astype(str).str.lower().eq("ok")
            ].copy()
        if "run_name" in table.columns:
            count = int(table["run_name"].astype(str).nunique())
        elif "run_dir" in table.columns:
            count = int(
                table["run_dir"]
                .astype(str)
                .map(lambda value: Path(value).name)
                .nunique()
            )
        else:
            count = int(len(table))
        return count == EXPECTED_VALIDATION_GRID_ROWS, count, path
    return False, 0, None


def reusable_strict_manifest(root: Path) -> dict[str, Any] | None:
    manifest_path = (
        root / "01_close_auction_grid" / "strict_oos_manifest.json"
    )
    if not manifest_path.is_file():
        return None
    payload = base.read_json(manifest_path)
    run_name = payload.get("retained_run_name")
    if not run_name:
        return None
    run_dir = root / "01_close_auction_grid" / "01_runs" / str(run_name)
    if not (run_dir / "summary.json").is_file():
        return None
    if not (run_dir / "config.json").is_file():
        return None
    return payload


def install_low_memory_resume_guards() -> None:
    original_prediction_artifact = base.prediction_artifact
    original_run_validation_grid = base.run_validation_grid
    original_run_frozen_target = base.run_frozen_target

    def guarded_prediction_artifact(**kwargs: Any) -> Path:
        path = original_prediction_artifact(**kwargs)
        source_fold = int(kwargs["source_fold"])
        protocol = str(kwargs["protocol"])
        if source_fold == 0 and protocol == "source_fold_validation_grid":
            release_feature_matrix(
                kwargs["features"], "historical feature matrix before fold0 grid"
            )
        elif source_fold == 0 and protocol == "frozen_fold0_to_strict_forward":
            release_feature_matrix(
                kwargs["features"], "forward feature matrix before forward backtest"
            )
        return path

    def resumed_validation_grid(
        args: Any,
        *,
        root: Path,
        prediction_file: Path,
    ) -> Any:
        if not args.force:
            complete, count, summary_path = completed_validation_grid(root)
            if complete:
                print(
                    "[RESUME] complete validation grid: "
                    f"rows={count} summary={summary_path}",
                    flush=True,
                )
                return base.select_historical_signal(
                    backtest_root=root,
                    rank_metric=args.selection_rank_metric,
                )
        return original_run_validation_grid(
            args, root=root, prediction_file=prediction_file
        )

    def resumed_frozen_target(
        args: Any,
        *,
        root: Path,
        prediction_file: Path,
        selection: Any,
    ) -> dict[str, Any]:
        if not args.force:
            payload = reusable_strict_manifest(root)
            if payload is not None:
                print(
                    "[RESUME] frozen target result: "
                    f"root={root} retained_run={payload['retained_run_name']}",
                    flush=True,
                )
                return payload
        return original_run_frozen_target(
            args,
            root=root,
            prediction_file=prediction_file,
            selection=selection,
        )

    base.prediction_artifact = guarded_prediction_artifact
    base.run_validation_grid = resumed_validation_grid
    base.run_frozen_target = resumed_frozen_target


def main() -> None:
    forward_model_data = Path(pop_option("--forward-model-data")).expanduser().resolve()
    if not forward_model_data.is_file():
        raise FileNotFoundError(forward_model_data)

    target_col = option_value("--target-col")
    if target_col != TARGET_COL:
        raise SystemExit(
            f"this entry requires --target-col {TARGET_COL}; got {target_col!r}"
        )

    original_builder = base.build_inference_features

    def build_latest_forward_features(
        _historical_model_data: Path,
        train_end: str | None,
        target_col_arg: str,
        feature_preset: str,
        sector_encoding: str,
    ) -> Any:
        if target_col_arg != TARGET_COL:
            raise RuntimeError(
                f"unexpected forward target: {target_col_arg!r}; expected {TARGET_COL!r}"
            )
        print(f"[FORWARD MODEL DATA] {forward_model_data}")
        return original_builder(
            forward_model_data,
            train_end,
            target_col_arg,
            feature_preset,
            sector_encoding,
        )

    # The generic nested runner uses build_inference_features only for the
    # source_fold0 -> strict-forward branch. Historical validation/target
    # features continue to use --model-data through common.build_target_features.
    base.build_inference_features = build_latest_forward_features
    install_low_memory_resume_guards()
    base.main()


if __name__ == "__main__":
    main()
