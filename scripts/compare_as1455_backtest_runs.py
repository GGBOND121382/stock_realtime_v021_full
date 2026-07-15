#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two AS1455 single-run output directories.

The comparator is intended for refactors: output paths and engine metadata may
change, but trades, fees, holdings, NAV, and performance metrics must not.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_FILES = [
    "close_auction_nav.csv",
    "close_auction_orders.csv",
    "close_auction_rejections.csv",
    "round_trips.csv",
]
JSON_FILE = "summary.json"
IGNORE_JSON_KEYS = {
    "created_at",
    "created_at_utc",
    "grid_engine",
    "model_family",
    "model_run",
}
SORT_KEYS = {
    "close_auction_nav.csv": ["date"],
    "close_auction_orders.csv": ["date", "symbol", "side"],
    "close_auction_rejections.csv": ["date", "symbol", "side", "reason"],
    "round_trips.csv": ["round_trip_id", "symbol", "entry_date", "exit_date"],
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def normalize_frame(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    out = frame.copy()
    keys = [key for key in SORT_KEYS.get(filename, []) if key in out.columns]
    if keys:
        out = out.sort_values(keys, kind="mergesort")
    return out.reset_index(drop=True)


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    filename: str,
    rtol: float,
    atol: float,
) -> list[str]:
    errors: list[str] = []
    left = normalize_frame(left, filename)
    right = normalize_frame(right, filename)

    if list(left.columns) != list(right.columns):
        errors.append(
            f"{filename}: columns differ: left={list(left.columns)} "
            f"right={list(right.columns)}"
        )
        return errors
    if len(left) != len(right):
        errors.append(
            f"{filename}: row count differs: left={len(left)} right={len(right)}"
        )
        return errors

    for column in left.columns:
        left_col = left[column]
        right_col = right[column]
        left_num = pd.to_numeric(left_col, errors="coerce")
        right_num = pd.to_numeric(right_col, errors="coerce")
        numeric_mask = left_num.notna() | right_num.notna()
        numeric_coverage = float(numeric_mask.mean()) if len(left) else 1.0

        if numeric_coverage >= 0.95:
            left_values = left_num.to_numpy(dtype=float)
            right_values = right_num.to_numpy(dtype=float)
            equal = np.isclose(
                left_values,
                right_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        else:
            left_values = left_col.fillna("<NA>").astype(str).to_numpy()
            right_values = right_col.fillna("<NA>").astype(str).to_numpy()
            equal = left_values == right_values

        if not bool(np.all(equal)):
            indices = np.flatnonzero(~equal)[:10]
            samples = [
                {
                    "row": int(index),
                    "left": left.iloc[int(index)][column],
                    "right": right.iloc[int(index)][column],
                }
                for index in indices
            ]
            errors.append(
                f"{filename}: column {column!r} differs at "
                f"{int((~equal).sum())} rows; samples={samples}"
            )
    return errors


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_json(item)
            for key, item in value.items()
            if key not in IGNORE_JSON_KEYS
        }
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return value


def compare_json(
    left: Any,
    right: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> list[str]:
    errors: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            errors.append(
                f"{path}: keys differ: only_left={sorted(left_keys-right_keys)} "
                f"only_right={sorted(right_keys-left_keys)}"
            )
        for key in sorted(left_keys & right_keys):
            errors.extend(
                compare_json(
                    left[key],
                    right[key],
                    path=f"{path}.{key}",
                    rtol=rtol,
                    atol=atol,
                )
            )
        return errors
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: list length differs: {len(left)} != {len(right)}"]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            errors.extend(
                compare_json(
                    left_item,
                    right_item,
                    path=f"{path}[{index}]",
                    rtol=rtol,
                    atol=atol,
                )
            )
        return errors

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
            errors.append(f"{path}: {left!r} != {right!r}")
    elif left != right:
        errors.append(f"{path}: {left!r} != {right!r}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two AS1455 single-run result directories"
    )
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_root = Path(args.left_run)
    right_root = Path(args.right_run)
    if not left_root.is_dir():
        raise FileNotFoundError(left_root)
    if not right_root.is_dir():
        raise FileNotFoundError(right_root)

    errors: list[str] = []
    for filename in CSV_FILES:
        left_path = left_root / filename
        right_path = right_root / filename
        if not left_path.exists() or not right_path.exists():
            errors.append(
                f"{filename}: missing left={left_path.exists()} "
                f"right={right_path.exists()}"
            )
            continue
        errors.extend(
            compare_frames(
                read_csv(left_path),
                read_csv(right_path),
                filename=filename,
                rtol=args.rtol,
                atol=args.atol,
            )
        )

    left_summary = left_root / JSON_FILE
    right_summary = right_root / JSON_FILE
    if not left_summary.exists() or not right_summary.exists():
        errors.append(
            f"{JSON_FILE}: missing left={left_summary.exists()} "
            f"right={right_summary.exists()}"
        )
    else:
        left_payload = normalize_json(
            json.loads(left_summary.read_text(encoding="utf-8"))
        )
        right_payload = normalize_json(
            json.loads(right_summary.read_text(encoding="utf-8"))
        )
        errors.extend(
            compare_json(
                left_payload,
                right_payload,
                path=JSON_FILE,
                rtol=args.rtol,
                atol=args.atol,
            )
        )

    report = {
        "left_run": str(left_root),
        "right_run": str(right_root),
        "rtol": args.rtol,
        "atol": args.atol,
        "passed": not errors,
        "errors": errors,
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    if errors:
        print("[FAIL] AS1455 run comparison")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("[PASS] AS1455 run outputs are equivalent")


if __name__ == "__main__":
    main()
