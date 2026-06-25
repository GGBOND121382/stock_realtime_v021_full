#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair AS1455 live collection quality status and resume feature build.

This script is intended for the case where 06/07/08 live quote files already
exist, but 08_collection_report.json failed because empty missing_core_fields
were interpreted as non-empty strings such as "nan".

It does two mechanical things:
  1. Recompute quality_status in 08_live_raw_row_as1455.csv and rewrite
     08_live_collection_report.csv / 08_collection_report.json.
  2. Optionally invoke features/build_as1455_live_features.py to produce
     09_live_qfq_row_as1455.csv, 10_live_feature_panel_tail, 11_live_model_features.csv,
     and 12_feature_build_report.json, without re-collecting quotes.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

EMPTY_STRINGS = {"", "nan", "none", "null", "na", "n/a", "[]", "{}"}
CORE_RAW_COLS = [
    "raw_open_as1455",
    "raw_high_as1455",
    "raw_low_as1455",
    "raw_close_as1455",
    "raw_volume_as1455",
    "raw_amount_as1455",
    "live_preclose",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_trade_date(value: str) -> str:
    s = str(value).strip()
    if s.lower() == "today":
        return datetime.now().strftime("%Y%m%d")
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    return s


def normalize_missing(value: Any) -> str:
    """Return empty string for truly empty missing-core field markers."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in EMPTY_STRINGS:
        return ""
    return text


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def finite_positive(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x) and x > 0


def recompute_quality(row: pd.Series) -> tuple[str, str]:
    # Start from missing_core_fields, but make NaN/'nan' truly empty.
    explicit_missing = normalize_missing(row.get("missing_core_fields", ""))
    missing = []
    if explicit_missing:
        missing.extend([x for x in explicit_missing.split(",") if x.strip()])

    # Re-check the actual raw fields. This makes the repair independent of the
    # old, possibly corrupted core_complete value.
    required = {
        "open": row.get("raw_open_as1455"),
        "high": row.get("raw_high_as1455"),
        "low": row.get("raw_low_as1455"),
        "last_price": row.get("raw_close_as1455"),
        "volume": row.get("raw_volume_as1455"),
        "amount": row.get("raw_amount_as1455"),
        "prev_close": row.get("live_preclose"),
    }
    for name, value in required.items():
        if not finite_positive(value):
            missing.append(name)

    missing = sorted(set(x.strip() for x in missing if x and x.strip()))
    if missing:
        return "missing_core_fields", ",".join(missing)

    try:
        lo = float(row["raw_low_as1455"])
        hi = float(row["raw_high_as1455"])
        op = float(row["raw_open_as1455"])
        cl = float(row["raw_close_as1455"])
        if not (lo <= op <= hi and lo <= cl <= hi):
            return "price_order_invalid", ""
    except Exception:
        return "invalid_numeric", ""

    source_status = str(row.get("source_status", "ok")).strip().lower()
    if source_status not in {"", "ok", "nan", "none"}:
        return f"source_status_{source_status}", ""

    return "ok", ""


def repair_collection(live_dir: Path, trade_date: str, cutoff_time: str, min_valid_rate: float, expected_symbols: int | None = None) -> dict[str, Any]:
    raw_path = live_dir / "08_live_raw_row_as1455.csv"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing live raw panel: {raw_path}")
    raw = pd.read_csv(raw_path, dtype={"symbol": str}, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError(f"empty live raw panel: {raw_path}")

    # Normalize symbol and dates only lightly; do not modify price/volume values.
    raw["missing_core_fields"] = raw.get("missing_core_fields", "").map(normalize_missing)
    repaired = raw.apply(recompute_quality, axis=1, result_type="expand")
    raw["quality_status"] = repaired[0]
    raw["missing_core_fields"] = repaired[1]
    raw["core_complete"] = raw["quality_status"].eq("ok")

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    raw.to_csv(live_dir / "08_live_collection_report.csv", index=False, encoding="utf-8-sig")

    snapshots_total = None
    latest_rows = None
    source_status_counts = {}
    snap_path = live_dir / "06_live_snapshots_raw.csv"
    if snap_path.exists() and snap_path.stat().st_size > 0:
        try:
            # Read only the source_status column if possible to keep this light.
            snaps = pd.read_csv(snap_path, usecols=lambda c: c in {"symbol", "source_status"}, dtype={"symbol": str}, encoding="utf-8-sig")
            snapshots_total = int(len(snaps))
        except Exception:
            snapshots_total = None
    latest_path = live_dir / "07_live_snapshot_asof1455.csv"
    if latest_path.exists() and latest_path.stat().st_size > 0:
        try:
            latest = pd.read_csv(latest_path, dtype={"symbol": str}, encoding="utf-8-sig")
            latest_rows = int(len(latest))
            if "source_status" in latest.columns:
                source_status_counts = latest["source_status"].value_counts(dropna=False).to_dict()
        except Exception:
            latest_rows = None

    if expected_symbols is None or expected_symbols <= 0:
        universe_path = live_dir / "01_universe.csv"
        if universe_path.exists() and universe_path.stat().st_size > 0:
            try:
                expected_symbols = int(pd.read_csv(universe_path, usecols=lambda c: c == "symbol", dtype={"symbol": str}, encoding="utf-8-sig")["symbol"].nunique())
            except Exception:
                expected_symbols = int(raw["symbol"].nunique()) if "symbol" in raw.columns else len(raw)
        else:
            expected_symbols = int(raw["symbol"].nunique()) if "symbol" in raw.columns else len(raw)

    valid_rows = int(raw["quality_status"].eq("ok").sum())
    valid_rate = valid_rows / expected_symbols if expected_symbols else 0.0
    summary = {
        "trade_date": trade_date,
        "cutoff_time": cutoff_time,
        "expected_symbols": int(expected_symbols or 0),
        "snapshot_rows_total": int(snapshots_total if snapshots_total is not None else -1),
        "symbols_with_valid_asof": int(latest_rows if latest_rows is not None else raw["symbol"].nunique() if "symbol" in raw.columns else len(raw)),
        "valid_panel_rows": valid_rows,
        "valid_panel_rate": valid_rate,
        "min_valid_rate": float(min_valid_rate),
        "collection_passed": bool(valid_rate >= min_valid_rate),
        "quality_status_counts": raw["quality_status"].value_counts(dropna=False).to_dict(),
        "source_status_counts": source_status_counts,
        "repair_applied": True,
        "repair_reason": "normalize empty/NaN missing_core_fields before quality_status evaluation",
        "repair_script": "tools/repair_as1455_live_collect_and_features_v1.py",
    }
    write_json(live_dir / "08_collection_report.json", summary)
    return summary


def run_features(repo_root: Path, trade_date: str, live_dir: Path, out_root: Path, min_feature_rows: int, feature_columns: str | None) -> int:
    script = repo_root / "features" / "build_as1455_live_features.py"
    if not script.exists():
        raise FileNotFoundError(f"missing feature script: {script}")
    args = [
        sys.executable,
        str(script),
        "--trade-date",
        trade_date,
        "--live-dir",
        str(live_dir),
        "--out-root",
        str(out_root),
        "--min-feature-rows",
        str(min_feature_rows),
    ]
    if feature_columns:
        args += ["--training-feature-columns", feature_columns]
    print("[INFO] running feature builder:", " ".join(args), flush=True)
    return subprocess.call(args, cwd=str(repo_root))


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair AS1455 live collection status and optionally resume feature build")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--out-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--cutoff-time", default="14:55:00")
    ap.add_argument("--min-valid-rate", type=float, default=0.98)
    ap.add_argument("--expected-symbols", type=int, default=None)
    ap.add_argument("--min-feature-rows", type=int, default=980)
    ap.add_argument("--feature-columns", default=None)
    ap.add_argument("--no-features", action="store_true", help="only repair 08 collection outputs; do not run feature builder")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    trade_date = parse_trade_date(args.trade_date)
    out_root = (repo_root / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve()
    live_dir = Path(args.live_dir).resolve() if args.live_dir else out_root / trade_date

    print(f"[INFO] repo_root={repo_root}", flush=True)
    print(f"[INFO] live_dir={live_dir}", flush=True)

    summary = repair_collection(live_dir, trade_date, args.cutoff_time, args.min_valid_rate, args.expected_symbols)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary.get("collection_passed"):
        raise SystemExit("collection still failed after repair; see 08_collection_report.json")

    if args.no_features:
        return
    rc = run_features(repo_root, trade_date, live_dir, out_root, args.min_feature_rows, args.feature_columns)
    if rc != 0:
        raise SystemExit(rc)
    report_path = live_dir / "12_feature_build_report.json"
    if report_path.exists():
        print("[INFO] feature report:", report_path, flush=True)
        print(report_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
