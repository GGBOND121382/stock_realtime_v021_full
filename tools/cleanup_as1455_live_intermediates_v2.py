#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean AS1455 live intermediate history-tail files after fast state succeeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime


def parse_trade_date(value: str) -> str:
    if value.lower() == "today":
        return datetime.now().strftime("%Y%m%d")
    return value.replace("-", "")[:8]


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete rebuildable AS1455 live intermediate files when final artifacts exist")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--out-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--keep-history-tail", action="store_true")
    ap.add_argument("--require-fast-state", action="store_true", default=True)
    ap.add_argument("--require-prediction-features", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / parse_trade_date(args.trade_date)
    report_path = live_dir / "07_intermediate_cleanup_report.json"
    deleted: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    if args.keep_history_tail:
        summary = {"passed": True, "cleanup_skipped": True, "reason": "keep_history_tail", "live_dir": str(live_dir)}
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    prerequisites = []
    if args.require_fast_state:
        prerequisites.append(live_dir / "06_live_feature_state_fast.npz")
    if args.require_prediction_features:
        prerequisites.append(live_dir / "11_live_model_features_for_prediction.csv")
    missing = [str(p) for p in prerequisites if not p.exists()]
    if missing:
        summary = {"passed": False, "cleanup_skipped": True, "reason": "missing_prerequisites", "missing": missing, "live_dir": str(live_dir)}
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    patterns = [
        "04_history_tail_raw.csv", "04_history_tail_raw.parquet",
        "05_history_tail_qfq_livebase.csv", "05_history_tail_qfq_livebase.parquet",
        "10_live_feature_panel_tail.csv", "10_live_feature_panel_tail.parquet",
    ]
    for name in patterns:
        p = live_dir / name
        if not p.exists():
            skipped.append(str(p))
            continue
        if args.dry_run:
            deleted.append(str(p))
            continue
        try:
            p.unlink()
            deleted.append(str(p))
        except Exception as exc:
            errors.append(f"{p}: {type(exc).__name__}: {exc}")

    summary = {
        "passed": len(errors) == 0,
        "cleanup_skipped": False,
        "live_dir": str(live_dir),
        "deleted_count": len(deleted),
        "deleted": deleted,
        "missing_or_absent": skipped,
        "errors": errors,
        "dry_run": bool(args.dry_run),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
