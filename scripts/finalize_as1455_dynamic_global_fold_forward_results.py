#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize a fixed-signal global-fold experiment with a dynamic fold set.

The shared finalizer was originally written for exactly target_fold5..target_fold0.
This adapter keeps its calculation and plotting logic, but validates the fold set
from ``prediction_segments.csv`` instead of requiring six segments.  This is
needed for r21_fwd, where sample length supports target_fold0..target_fold4 only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import finalize_as1455_global_fold_forward_results as shared  # noqa: E402
from utils.as1455_model_selection import read_csv_auto  # noqa: E402

SIGNALS = {
    "all5": "ensemble_all5_mean:0,1,2,3,4:mean",
    "first3": "ensemble_first3_mean:0,1,2:mean",
    "best": "model_0:0:single",
}


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


def argument_value(name: str) -> str | None:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def load_segments_dynamic(path: Path) -> pd.DataFrame:
    segments = read_csv_auto(path)
    required = {"source_fold", "target_fold", "start", "end", "n_days"}
    missing = required - set(segments.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    segments = segments.copy()
    segments["start"] = pd.to_datetime(
        segments["start"], errors="coerce"
    ).dt.normalize()
    segments["end"] = pd.to_datetime(
        segments["end"], errors="coerce"
    ).dt.normalize()
    for column in ("source_fold", "target_fold", "n_days"):
        segments[column] = pd.to_numeric(
            segments[column], errors="raise"
        ).astype(int)
    segments = (
        segments.dropna(subset=["start", "end"])
        .sort_values("start")
        .reset_index(drop=True)
    )
    if segments.empty:
        raise RuntimeError(f"no historical target-fold segments in {path}")
    if segments["target_fold"].duplicated().any():
        raise RuntimeError(f"duplicate target folds in {path}")
    if segments["source_fold"].duplicated().any():
        raise RuntimeError(f"duplicate source folds in {path}")
    bad_mapping = segments.loc[
        segments["source_fold"] != segments["target_fold"] + 1
    ]
    if not bad_mapping.empty:
        raise RuntimeError(
            "one-fold-lag mapping failed: "
            + bad_mapping[["source_fold", "target_fold"]].to_dict(
                orient="records"
            ).__repr__()
        )
    if (segments["start"] > segments["end"]).any():
        raise RuntimeError(f"segment start is after end in {path}")
    for previous, current in zip(
        segments.itertuples(index=False),
        segments.iloc[1:].itertuples(index=False),
    ):
        if current.start <= previous.end:
            raise RuntimeError(
                "historical target-fold date ranges overlap: "
                f"target_fold{previous.target_fold} ends {previous.end:%Y-%m-%d}; "
                f"target_fold{current.target_fold} starts {current.start:%Y-%m-%d}"
            )
    return segments


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    signal_kind = pop_option("--signal-kind")
    try:
        signal_spec = SIGNALS[signal_kind]
    except KeyError as exc:
        raise SystemExit(
            f"unsupported --signal-kind={signal_kind!r}; expected {sorted(SIGNALS)}"
        ) from exc

    out_root_value = argument_value("--out-root")
    if not out_root_value:
        raise SystemExit("--out-root is required")

    shared.FIXED_SIGNAL_SPEC = signal_spec
    shared.load_segments = load_segments_dynamic
    shared.main()

    out_root = Path(out_root_value).expanduser().resolve()
    manifest_file = out_root / "global_fold0_to_fold5_forward_manifest.json"
    returns_file = out_root / "historical_fold_segment_returns.csv"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    segment_returns = read_csv_auto(returns_file).copy()
    segment_returns["start"] = pd.to_datetime(
        segment_returns["start"], errors="raise"
    )
    ordered = segment_returns.sort_values("start").reset_index(drop=True)
    target_folds = ordered["target_fold"].astype(int).tolist()
    source_folds = ordered["source_fold"].astype(int).tolist()

    manifest.update(
        {
            "protocol": (
                "global_available_target_folds_fixed_"
                f"{signal_kind}_then_strict_forward"
            ),
            "fixed_signal_kind": signal_kind,
            "fixed_signal_spec": signal_spec,
            "historical_target_folds": target_folds,
            "historical_source_folds": source_folds,
            "historical_segment_count": len(target_folds),
            "historical_fold_range": (
                f"target_fold{max(target_folds)}..target_fold{min(target_folds)}"
            ),
            "target_fold5_skipped_for_sample_shortage": (
                5 not in target_folds
            ),
        }
    )
    write_json(manifest_file, manifest)
    print(
        "[OK] dynamic fold finalization: "
        f"signal={signal_kind} target_folds={target_folds} "
        f"source_folds={source_folds}"
    )


if __name__ == "__main__":
    main()
