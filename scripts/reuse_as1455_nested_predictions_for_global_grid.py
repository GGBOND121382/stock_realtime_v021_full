#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concatenate nested target-fold prediction artifacts for one global grid.

The requested mapping is always one-fold-lag:

    source_fold(target+1) -> target_fold

By default target_fold0..5 are used.  A shorter subset such as target_fold0..4
is accepted when source_fold6 is unavailable.  Prediction columns preserve
rank-slot semantics across folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def normalize_columns(frame: pd.DataFrame, path: Path, top_n: int) -> pd.DataFrame:
    mapping: dict[Any, int] = {}
    for column in frame.columns:
        try:
            mapping[column] = int(column)
        except (TypeError, ValueError):
            continue
    out = frame.rename(columns=mapping)
    if out.columns.duplicated().any():
        raise RuntimeError(f"prediction columns collide after normalization: {path}")
    missing = sorted(set(range(top_n)) - set(out.columns))
    if missing:
        raise RuntimeError(
            f"{path} lacks required rank slots for top_n={top_n}: {missing}"
        )
    return out.loc[:, list(range(top_n))]


def parse_target_folds(value: str) -> list[int]:
    folds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not folds:
        raise argparse.ArgumentTypeError("target fold list is empty")
    if len(folds) != len(set(folds)):
        raise argparse.ArgumentTypeError(f"duplicate target folds: {folds}")
    invalid = [fold for fold in folds if fold < 0 or fold > 5]
    if invalid:
        raise argparse.ArgumentTypeError(f"target folds must be in 0..5: {invalid}")
    return sorted(folds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse nested fold predictions as one dynamic global-grid input"
    )
    parser.add_argument("--nested-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--target-folds", default="0,1,2,3,4,5")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.target_folds_list = parse_target_folds(args.target_folds)
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    return args


def main() -> None:
    args = parse_args()
    nested_root = Path(args.nested_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not nested_root.is_dir():
        raise FileNotFoundError(nested_root)

    mapping = tuple(
        (target_fold + 1, target_fold)
        for target_fold in sorted(args.target_folds_list, reverse=True)
    )

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
    for source_fold, target_fold in mapping:
        path = prediction_path(nested_root, source_fold, target_fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = normalize_columns(
            pd.read_hdf(path, "predictions"), path, args.top_n
        )
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
    pd.DataFrame(rows).sort_values("start").to_csv(
        segment_file, index=False, encoding="utf-8-sig"
    )

    # Keep both the legacy segment representation and the canonical fold_mapping
    # representation used by target-aware cache validators.  Both are generated
    # from the same rows so metadata cannot drift from the HDF selection.
    fold_mapping = [
        {
            "source_fold": int(row["source_fold"]),
            "target_fold": int(row["target_fold"]),
            "source_dir": str(
                nested_root / f"source_fold{int(row['source_fold'])}"
            ),
            "target_test_start": str(row["start"]),
            "target_test_end": str(row["end"]),
            "n_target_rows": int(row["n_rows"]),
            "n_target_dates": int(row["n_days"]),
            "n_models": args.top_n,
        }
        for row in rows
    ]

    manifest = {
        "protocol": "reused_nested_one_fold_lag_predictions_for_dynamic_global_grid",
        "nested_root": str(nested_root),
        "prediction_file": str(out_file),
        "prediction_csv": str(out_file.with_suffix(".csv")),
        "segment_file": str(segment_file),
        "source_folds": [row[0] for row in mapping],
        "target_folds": [row[1] for row in mapping],
        "target_folds_requested": args.target_folds_list,
        "top_n": args.top_n,
        "fold_mapping": fold_mapping,
        "rank_slot_semantics": {
            str(index): f"each source fold's rank-{index + 1} saved checkpoint"
            for index in range(args.top_n)
        },
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
    print(
        f"[OK] dates={manifest['prediction_start']}.."
        f"{manifest['prediction_end']} n={manifest['n_dates']}"
    )
    print(
        "[OK] fold_mapping="
        + ",".join(
            f"source{item['source_fold']}->target{item['target_fold']}"
            for item in fold_mapping
        )
    )


if __name__ == "__main__":
    main()
