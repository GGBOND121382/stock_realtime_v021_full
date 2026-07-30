#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find an existing historical grid that exactly matches a fixed-signal experiment.

The unified refresh workflow writes new forward results under a stable matrix root,
but historical grids may already exist in older ``requested_v1`` or
``global_fold_selection`` directories.  This resolver searches those trees and
reuses a grid only after validating the target-fold segments, fixed signal,
rebalance period, successful grid coverage, and retained best-run artifacts.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_model_selection import (  # noqa: E402
    find_summary_file,
    read_csv_auto,
    select_historical_signal,
    signal_spec_from_row,
    successful_rows,
)

SIGNALS = {
    "all5": "ensemble_all5_mean:0,1,2,3,4:mean",
    "first3": "ensemble_first3_mean:0,1,2:mean",
    "best": "model_0:0:single",
}
DEFAULT_SEARCH_ROOTS = (
    Path("saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix"),
    Path("saved_data/ashare_ml4t/ch17_as1455_global_fold_selection"),
)


def parse_folds(text: str) -> list[int]:
    folds = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not folds or len(folds) != len(set(folds)):
        raise ValueError(f"invalid target folds: {text!r}")
    return folds


def expected_grid_count(rebalance_every: int) -> int:
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    return 5 * 6 * rebalance_every


def segment_folds(root: Path) -> tuple[list[int], list[int]]:
    path = root / "00_predictions" / "prediction_segments.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = read_csv_auto(path)
    required = {"source_fold", "target_fold", "start", "end"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["source_fold"] = pd.to_numeric(frame["source_fold"], errors="raise").astype(int)
    frame["target_fold"] = pd.to_numeric(frame["target_fold"], errors="raise").astype(int)
    frame["start"] = pd.to_datetime(frame["start"], errors="raise").dt.normalize()
    frame["end"] = pd.to_datetime(frame["end"], errors="raise").dt.normalize()
    frame = frame.sort_values("start").reset_index(drop=True)
    if frame.empty or frame["target_fold"].duplicated().any():
        raise RuntimeError(f"invalid prediction segments: {path}")
    if not (frame["source_fold"] == frame["target_fold"] + 1).all():
        raise RuntimeError(f"bad one-fold-lag mapping: {path}")
    return (
        sorted(frame["target_fold"].tolist()),
        sorted(frame["source_fold"].tolist()),
    )


def fixed_signal_rows(summary: pd.DataFrame, signal_spec: str) -> pd.DataFrame:
    frame = successful_rows(summary).copy()
    keep: list[bool] = []
    for _, row in frame.iterrows():
        try:
            spec, _required = signal_spec_from_row(row)
        except Exception:
            keep.append(False)
            continue
        keep.append(spec == signal_spec)
    return frame.loc[keep].copy()


def distinct_grid_count(frame: pd.DataFrame) -> int:
    keys = [
        column
        for column in (
            "signal_name",
            "signal_cols",
            "signal_mode",
            "max_positions",
            "sell_rank",
            "rebalance_every",
            "rebalance_offset",
        )
        if column in frame.columns
    ]
    if not keys:
        return int(len(frame))
    return int(frame.drop_duplicates(keys).shape[0])


def best_run_dir(root: Path, run_name: str) -> Path:
    _summary, grid_dir = find_summary_file(root)
    direct = grid_dir / "01_runs" / run_name
    if direct.is_dir():
        return direct
    matches = sorted(root.glob(f"**/01_runs/{run_name}"))
    if not matches:
        raise FileNotFoundError(direct)
    return matches[0]


def candidate_manifest(root: Path) -> dict[str, Any]:
    for path in (
        root.parent / "global_fold0_to_fold5_forward_manifest.json",
        root.parent / "global_available_target_folds_forward_manifest.json",
    ):
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    return {}


def validate_candidate(
    root: Path,
    *,
    target_col: str,
    signal_spec: str,
    rebalance_every: int,
    expected_folds: list[int],
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    actual_targets, actual_sources = segment_folds(root)
    if actual_targets != sorted(expected_folds):
        raise RuntimeError(
            f"target-fold mismatch expected={sorted(expected_folds)} actual={actual_targets}"
        )

    summary_file, _grid_dir = find_summary_file(root)
    summary = read_csv_auto(summary_file)
    matching = fixed_signal_rows(summary, signal_spec)
    if matching.empty:
        raise RuntimeError(f"no successful rows for signal={signal_spec}")
    if "rebalance_every" in matching.columns:
        values = pd.to_numeric(matching["rebalance_every"], errors="coerce")
        matching = matching.loc[values.eq(rebalance_every)].copy()
    if matching.empty:
        raise RuntimeError(f"no successful rows for rebalance_every={rebalance_every}")

    expected_count = expected_grid_count(rebalance_every)
    actual_count = distinct_grid_count(matching)
    if actual_count < expected_count:
        raise RuntimeError(
            f"incomplete grid: expected at least {expected_count}, got {actual_count}"
        )

    selection = select_historical_signal(backtest_root=root, rank_metric="sharpe")
    if selection.signal_spec != signal_spec:
        raise RuntimeError(
            f"selected signal mismatch expected={signal_spec} actual={selection.signal_spec}"
        )
    if selection.historical_rebalance_every not in (None, rebalance_every):
        raise RuntimeError(
            "selected rebalance mismatch: "
            f"expected={rebalance_every} actual={selection.historical_rebalance_every}"
        )
    run_dir = best_run_dir(root, selection.run_name)
    for filename in ("config.json", "summary.json", "close_auction_nav.csv"):
        path = run_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest = candidate_manifest(root)
    manifest_target = manifest.get("target_col")
    if manifest_target not in (None, "", target_col):
        raise RuntimeError(
            f"manifest target mismatch expected={target_col} actual={manifest_target}"
        )

    return {
        "historical_root": str(root),
        "summary_file": str(summary_file),
        "target_col": target_col,
        "signal_spec": signal_spec,
        "rebalance_every": rebalance_every,
        "target_folds": actual_targets,
        "source_folds": actual_sources,
        "distinct_successful_grid_rows": actual_count,
        "expected_grid_rows": expected_count,
        "selected_run_name": selection.run_name,
        "selected_sharpe": selection.rank_metric_value,
        "selected_run_dir": str(run_dir),
        "mtime": max(summary_file.stat().st_mtime, run_dir.stat().st_mtime),
    }


def candidate_roots(search_roots: list[Path], preferred: Path | None) -> list[Path]:
    roots: list[Path] = []
    if preferred is not None:
        roots.append(preferred)
    for base in search_roots:
        base = base.expanduser()
        if not base.is_dir():
            continue
        for filename in ("grid_summary_compact.csv", "grid_summary.csv"):
            for path in base.rglob(filename):
                if "strict_oos_forward" in path.parts:
                    continue
                try:
                    root = path.parents[2]
                except IndexError:
                    continue
                roots.append(root)
    dedup: dict[str, Path] = {}
    for root in roots:
        try:
            key = str(root.expanduser().resolve())
        except OSError:
            key = str(root)
        dedup[key] = root
    return list(dedup.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find an exact compatible historical AS1455 grid result"
    )
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--signal-kind", choices=sorted(SIGNALS), required=True)
    parser.add_argument("--rebalance-every", type=int, required=True)
    parser.add_argument("--target-folds", required=True)
    parser.add_argument("--preferred-root", default=None)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--format", choices=["path", "shell", "json"], default="path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_folds = parse_folds(args.target_folds)
    search_roots = [Path(value) for value in args.search_root] or list(DEFAULT_SEARCH_ROOTS)
    preferred = Path(args.preferred_root) if args.preferred_root else None
    signal_spec = SIGNALS[args.signal_kind]

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for root in candidate_roots(search_roots, preferred):
        try:
            item = validate_candidate(
                root,
                target_col=args.target_col,
                signal_spec=signal_spec,
                rebalance_every=args.rebalance_every,
                expected_folds=expected_folds,
            )
        except Exception as exc:
            rejected.append({"root": str(root), "reason": str(exc)})
            continue
        item["preferred"] = bool(
            preferred is not None
            and Path(item["historical_root"]).resolve() == preferred.expanduser().resolve()
        )
        valid.append(item)

    valid.sort(
        key=lambda item: (
            1 if item.get("preferred") else 0,
            float(item.get("mtime", 0.0)),
        ),
        reverse=True,
    )
    selected = valid[0] if valid else None
    payload = {
        "status": "found" if selected else "missing",
        "request": {
            "target_col": args.target_col,
            "signal_kind": args.signal_kind,
            "signal_spec": signal_spec,
            "rebalance_every": args.rebalance_every,
            "target_folds": expected_folds,
        },
        "selected": selected,
        "valid_candidate_count": len(valid),
        "valid_candidates": valid,
        "rejected_candidate_count": len(rejected),
    }
    if args.output_json:
        path = Path(args.output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[HISTORY RESOLUTION] report={path}", file=sys.stderr)

    value = selected["historical_root"] if selected else ""
    if selected:
        print(
            "[HISTORY REUSE] "
            f"target={args.target_col} signal={args.signal_kind} "
            f"folds={expected_folds} root={value}",
            file=sys.stderr,
        )
    else:
        print(
            "[HISTORY MISS] no exact compatible historical grid; "
            "the experiment will build one in the new output root",
            file=sys.stderr,
        )

    if args.format == "path":
        print(value)
    elif args.format == "shell":
        print(f"REUSE_HISTORICAL_ROOT={shlex.quote(value)}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
