#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delete rebuildable AS1455 live history-tail copies after fast state succeeds.

The shared caches under ``ch12_as1455`` remain untouched.  Only large per-day
copies used to construct ``06_live_feature_state_fast.npz`` are removed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_trade_date(value: str) -> str:
    if value.lower() == "today":
        return datetime.now().strftime("%Y%m%d")
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"bad trade date: {value!r}")
    return digits


def read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete rebuildable per-day AS1455 history tails after the compact fast state is valid"
    )
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--out-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--keep-history-tail", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    live_dir = (
        Path(args.live_dir)
        if args.live_dir
        else Path(args.out_root) / parse_trade_date(args.trade_date)
    )
    live_dir.mkdir(parents=True, exist_ok=True)
    report_path = live_dir / "07_intermediate_cleanup_report.json"

    if args.keep_history_tail:
        summary = {
            "passed": True,
            "cleanup_skipped": True,
            "reason": "keep_history_tail",
            "live_dir": str(live_dir),
        }
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    state_path = live_dir / "06_live_feature_state_fast.npz"
    state_report_path = live_dir / "06_live_feature_state_fast_report.json"
    prepare_report_path = live_dir / "05_prepare_report.json"
    state_report = read_json(state_report_path)
    prepare_report = read_json(prepare_report_path)
    missing = [
        str(path)
        for path in (state_path, state_report_path, prepare_report_path)
        if not path.exists() or path.stat().st_size == 0
    ]
    prerequisites_ok = (
        not missing
        and bool(state_report.get("prefast_passed"))
        and bool(prepare_report.get("prepare_passed"))
    )
    if not prerequisites_ok:
        summary = {
            "passed": False,
            "cleanup_skipped": True,
            "reason": "invalid_or_missing_prerequisites",
            "missing": missing,
            "prefast_passed": state_report.get("prefast_passed"),
            "prepare_passed": prepare_report.get("prepare_passed"),
            "live_dir": str(live_dir),
        }
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    names = [
        "04_history_tail_raw.csv",
        "04_history_tail_raw.parquet",
        "05_history_tail_qfq_livebase.csv",
        "05_history_tail_qfq_livebase.parquet",
    ]
    removed: list[dict[str, object]] = []
    absent: list[str] = []
    errors: list[str] = []
    for name in names:
        path = live_dir / name
        if not path.exists():
            absent.append(str(path))
            continue
        size = int(path.stat().st_size)
        if not args.dry_run:
            try:
                path.unlink()
            except Exception as exc:  # pragma: no cover - filesystem-specific
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
        removed.append({"path": str(path), "bytes": size})

    summary = {
        "passed": not errors,
        "cleanup_skipped": False,
        "dry_run": bool(args.dry_run),
        "live_dir": str(live_dir),
        "scope": "per_day_rebuildable_history_tail_copies_only",
        "shared_caches_modified": False,
        "fast_state_retained": str(state_path),
        "removed_count": len(removed),
        "removed_bytes": int(sum(int(row["bytes"]) for row in removed)),
        "removed": removed,
        "missing_or_absent": absent,
        "errors": errors,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
