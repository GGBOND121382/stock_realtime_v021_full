#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect AS1455 live raw row from realtime cumulative quotes.

This is the formal AS1455 collector, separated from the old intraday watcher.
It produces the T-day raw AS1455 row from the latest valid quote at or before
cutoff_time.  It does not build features and does not run models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    collect_sina_quotes,
    ensure_dir,
    load_universe,
    parse_clock,
    parse_trade_date,
    select_latest_asof_snapshot,
    snapshots_to_raw_panel,
    write_csv,
    write_json,
)

DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"


def append_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        df.to_csv(path, index=False, mode="a", header=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def run_one_round(universe: pd.DataFrame, args, round_id: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    started = time.time()
    snapshots, errors = collect_sina_quotes(
        universe,
        batch_size=args.batch_size,
        timeout=args.timeout_seconds,
        batch_sleep=args.batch_sleep_seconds,
    )
    snapshots.insert(0, "round", int(round_id))
    snapshots.insert(0, "run_id", run_id)
    if not errors.empty:
        errors.insert(0, "round", int(round_id))
        errors.insert(0, "run_id", run_id)
    summary = {
        "run_id": run_id,
        "round": int(round_id),
        "snapshot_rows": int(len(snapshots)),
        "payload_rows": int(snapshots["source_status"].eq("ok").sum()) if "source_status" in snapshots else 0,
        "core_complete_rows": int(snapshots["core_complete"].sum()) if "core_complete" in snapshots else 0,
        "core_complete_rate": float(snapshots["core_complete"].mean()) if "core_complete" in snapshots and len(snapshots) else 0.0,
        "error_batches": int(len(errors)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return snapshots, errors, summary


def finalize_day(live_dir: Path, trade_date: str, cutoff_time: str, min_valid_rate: float, expected_symbols: int) -> dict:
    raw_path = live_dir / "06_live_snapshots_raw.csv"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise FileNotFoundError(f"no snapshots file found: {raw_path}")
    snapshots = pd.read_csv(raw_path, dtype={"symbol": str}, encoding="utf-8-sig")
    latest = select_latest_asof_snapshot(snapshots, cutoff_time=cutoff_time)
    raw_panel = snapshots_to_raw_panel(latest, trade_date=trade_date)
    write_csv(live_dir / "07_live_snapshot_asof1455.csv", latest)
    write_csv(live_dir / "08_live_raw_row_as1455.csv", raw_panel)
    report = raw_panel.copy()
    write_csv(live_dir / "08_live_collection_report.csv", report)
    valid_rows = int(raw_panel["quality_status"].eq("ok").sum()) if not raw_panel.empty else 0
    valid_rate = valid_rows / expected_symbols if expected_symbols else 0.0
    summary = {
        "trade_date": trade_date,
        "cutoff_time": cutoff_time,
        "expected_symbols": int(expected_symbols),
        "snapshot_rows_total": int(len(snapshots)),
        "symbols_with_valid_asof": int(raw_panel["symbol"].nunique()) if not raw_panel.empty else 0,
        "valid_panel_rows": valid_rows,
        "valid_panel_rate": valid_rate,
        "min_valid_rate": float(min_valid_rate),
        "collection_passed": bool(valid_rate >= min_valid_rate),
        "quality_status_counts": raw_panel["quality_status"].value_counts(dropna=False).to_dict() if not raw_panel.empty else {},
        "source_status_counts": latest["source_status"].value_counts(dropna=False).to_dict() if not latest.empty and "source_status" in latest else {},
    }
    write_json(live_dir / "08_collection_report.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect AS1455 14:55 live raw panel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["collect-once", "collect-loop", "finalize"]:
        sp = sub.add_parser(name)
        sp.add_argument("--trade-date", default="today")
        sp.add_argument("--universe", default=None)
        sp.add_argument("--max-symbols", type=int, default=None)
        sp.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
        sp.add_argument("--batch-size", type=int, default=250)
        sp.add_argument("--timeout-seconds", type=float, default=8.0)
        sp.add_argument("--batch-sleep-seconds", type=float, default=0.2)
        sp.add_argument("--cutoff-time", default="14:55:00")
        sp.add_argument("--min-valid-rate", type=float, default=0.98)
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--finalize", action="store_true")
        if name == "collect-loop":
            sp.add_argument("--interval-seconds", type=float, default=30.0)
            sp.add_argument("--until", default="14:55:05")
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    universe = load_universe(args.universe, args.max_symbols)
    write_csv(live_dir / "01_universe.csv", universe)
    round_summaries = []

    if args.cmd in {"collect-once", "collect-loop"}:
        round_id = 1
        until_clock = parse_clock(args.until) if args.cmd == "collect-loop" else None
        while True:
            snapshots, errors, summary = run_one_round(universe, args, round_id, run_id)
            append_csv(live_dir / "06_live_snapshots_raw.csv", snapshots)
            if not errors.empty:
                append_csv(live_dir / "06_live_collection_errors.csv", errors)
            round_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if args.cmd == "collect-once":
                break
            now_clock = datetime.now().time()
            if until_clock and now_clock >= until_clock:
                break
            time.sleep(max(0.0, float(args.interval_seconds)))
            round_id += 1
        if round_summaries:
            rs = pd.DataFrame(round_summaries)
            write_csv(live_dir / "06_live_round_summary.csv", rs)
    if args.cmd == "finalize" or getattr(args, "finalize", False) or args.cmd == "collect-loop":
        summary = finalize_day(live_dir, trade_date, args.cutoff_time, args.min_valid_rate, len(universe))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
