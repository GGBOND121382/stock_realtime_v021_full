#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_ths_sector_hedged_cache.py

测试 THS sector summary 的低并发错峰请求 + 一次抓取多板块过滤。

验证目标：
  1. 每轮只抓一次 THS summary 表，而不是每个 sector 抓一次；
  2. 最多 2 个错峰 subprocess 请求，谁先成功用谁；
  3. 能从同一张 summary 表中过滤出多个板块；
  4. 即使失败，也在 timeout 内返回，不拖死流程。

运行：
  python3 tests/test_ths_sector_hedged_cache.py
  python3 tests/test_ths_sector_hedged_cache.py --rounds 3 --timeout 5 --hedge-workers 2 --hedge-delay 1.5 --allow-fail

输出：
  debug_ths_sector_hedged/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_collection.collect_realtime_context import (  # noqa: E402
    fetch_ths_sector_summary_hedged,
    sector_snapshot_from_summary,
)


OUT = Path("debug_ths_sector_hedged")


DEFAULT_SECTORS = [
    "电网设备",
    "贵金属",
    "工业金属",
    "养殖业",
    "农产品加工",
    "建筑材料",
    "化学制品",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--hedge-workers", type=int, default=2)
    ap.add_argument("--hedge-delay", type=float, default=1.5)
    ap.add_argument("--sectors", default=",".join(DEFAULT_SECTORS))
    ap.add_argument("--allow-fail", action="store_true", help="Do not return non-zero if all rounds fail.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
    results = []

    for i in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        df, meta = fetch_ths_sector_summary_hedged(
            timeout_seconds=args.timeout,
            hedge_workers=args.hedge_workers,
            hedge_delay_seconds=args.hedge_delay,
        )
        elapsed = time.perf_counter() - t0

        sector_rows = []
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(OUT / f"round_{i}_sector_summary.csv", index=False, encoding="utf-8-sig")
            for sec in sectors:
                sector_rows.append(sector_snapshot_from_summary(df, sec, meta))
        else:
            for sec in sectors:
                sector_rows.append(sector_snapshot_from_summary(None, sec, meta))

        ok_count = sum(1 for r in sector_rows if r.get("status") == "ok")
        result = {
            "round": i,
            "elapsed_seconds": elapsed,
            "meta": meta,
            "summary_shape": list(df.shape) if isinstance(df, pd.DataFrame) else None,
            "requested_sectors": sectors,
            "ok_count": ok_count,
            "sectors": sector_rows,
        }
        results.append(result)

        print(f"ROUND {i}: elapsed={elapsed:.3f}s, launched={meta.get('launched_workers')}, winner={meta.get('winner')}, summary_shape={result['summary_shape']}, ok_sectors={ok_count}/{len(sectors)}, meta={meta}")

    successful_rounds = sum(1 for r in results if r["ok_count"] > 0)
    summary = {
        "rounds": args.rounds,
        "timeout_seconds": args.timeout,
        "hedge_workers": args.hedge_workers,
        "hedge_delay_seconds": args.hedge_delay,
        "successful_rounds": successful_rounds,
        "results": results,
        "decision_hint": {
            "success": "collect-loop can use hedged sector summary and cache snapshots.",
            "failure": "build-features should remain cache-only; missing sector should reject only sector models.",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {OUT / 'summary.json'}")
    if successful_rounds == 0 and not args.allow_fail:
        print("[FAIL] all hedged THS sector summary rounds failed.")
        return 2
    print("[OK] at least one hedged THS sector summary round succeeded." if successful_rounds else "[WARN] no successful round, but --allow-fail set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
