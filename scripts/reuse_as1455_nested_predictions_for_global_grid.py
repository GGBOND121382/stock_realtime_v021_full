#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concatenate nested target-fold prediction artifacts for one global grid.

The input mapping is source_fold6->target_fold5 through
source_fold1->target_fold0. Prediction columns keep their rank-slot semantics,
so columns 0,1,2 are each source fold's top-three checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MAPPING = tuple((source, source - 1) for source in range(6, 0, -1))


def prediction_path(root: Path, source_fold: int, target_fold: int) -> Path:
    return (
        root
        / f"source_fold{source_fold}"
        / f"target_fold{target_fold}"
        / "00_predictions"
        / "target_preds.h5"
    )


def normalized_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.MultiIndex):
        raise RuntimeError("prediction frame must use a MultiIndex")
    if "date" not in frame.index.names or "symbol" not in frame.index.names:
        raise RuntimeError(
            f"prediction index must contain date/symbol, got {frame.index.names}"
        )
    return (
        pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("date")))
        .normalize()
        .unique()
        .sort_values()
    )


def normalize_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    mapping: dict[Any, int] = {}
    for column in frame.columns:
        try:
            mapping[column] = int(column)
        except (TypeError, ValueError):
            continue
    out = frame.rename(columns=mapping)
    if out.columns.duplicated().any():
        raise RuntimeError(f"prediction columns collide after normalization: {path}")
    missing = sorted({0, 1, 2} - set(out.columns))
    if missing:
        raise RuntimeError(f"{path} lacks first-three rank slots: {missing}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse nested fold predictions as one fold0..5 grid input"
    )
    parser.add_argument("--nested-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nested_root = Path(args.nested_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not nested_root.is_dir():
        raise FileNotFoundError(nested_root)

    prediction_dir = out_root / "00_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    out_file = prediction_dir / "test_preds.h5"
    manifest_file = prediction_dir / "one_lag_prediction_manifest.json"
    segment_file = prediction_dir / "prediction_segments.csv"
    if out_file.exists() and manifest_file.exists() and not args.force:
        print(f"[RESUME] combined predictions={out_file}")
        return

    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    seen_dates = pd.DatetimeIndex([])
    for source_fold, target_fold in MAPPING:
        path = prediction_path(nested_root, source_fold, target_fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = normalize_columns(pd.read_hdf(path, "predictions"), path)
        if frame.index.duplicated().any():
            raise RuntimeError(f"duplicate symbol/date rows: {path}")
        dates = normalized_dates(frame)
        overlap = seen_dates.intersection(dates)
        if len(overlap):
            raise RuntimeError(
                f"fold date overlap at source_fold{source_fold}: "
                f"{list(map(str, overlap[:10]))}"
            )
        seen_dates = seen_dates.union(dates)
        frames.append(frame)
        rows.append(
            {
                "source_fold": source_fold,
                "target_fold": target_fold,
                "prediction_file": str(path),
                "start": dates[0].strftime("%Y-%m-%d"),
                "end": dates[-1].strftime("%Y-%m-%d"),
                "n_days": len(dates),
                "n_rows": len(frame),
                "prediction_columns": ",".join(map(str, frame.columns)),
            }
        )

    combined = pd.concat(frames, axis=0).sort_index()
    if combined.index.duplicated().any():
        raise RuntimeError("combined predictions contain duplicate symbol/date rows")
    combined_dates = normalized_dates(combined)
    combined.to_hdf(out_file, key="predictions", mode="w")
    combined.to_csv(out_file.with_suffix(".csv"), encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(segment_file, index=False, encoding="utf-8-sig")
    manifest = {
        "protocol": "reused_nested_one_fold_lag_predictions_for_global_fold0_to_fold5_grid",
        "nested_root": str(nested_root),
        "prediction_file": str(out_file),
        "prediction_csv": str(out_file.with_suffix(".csv")),
        "segment_file": str(segment_file),
        "source_folds": [row[0] for row in MAPPING],
        "target_folds": [row[1] for row in MAPPING],
        "rank_slot_semantics": {
            "0": "each source fold's highest-ranked saved checkpoint",
            "1": "each source fold's second-ranked saved checkpoint",
            "2": "each source fold's third-ranked saved checkpoint",
        },
        "fixed_experiment_signal": "ensemble_first3_mean:0,1,2:mean",
        "n_rows": len(combined),
        "n_dates": len(combined_dates),
        "prediction_start": combined_dates[0].strftime("%Y-%m-%d"),
        "prediction_end": combined_dates[-1].strftime("%Y-%m-%d"),
        "segments": rows,
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] combined predictions={out_file}")
    print(f"[OK] dates={manifest['prediction_start']}..{manifest['prediction_end']} n={manifest['n_dates']}")


if __name__ == "__main__":
    main()
