#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_full_table_sources_timeout.py

目的：
  逐个测试当前工程里可能涉及“全量拉表”的 AKShare 数据源是否会超时。
  不做模型、不写业务数据，只测接口耗时、行数、字段。

重点：
  1. A 股/ETF 全量实时表：可能被 collect_akshare_l1_cache.py 使用；
  2. 板块全行业表：规模小，但仍测试；
  3. A+H 全表：规模中等，牧原 H 股本体曾用；
  4. 港股全量表：默认不测；加 --include-hk-full 才测，因为 14:55 关键路径应禁止使用。

运行：
  python3 tests/test_full_table_sources_timeout.py --timeout 15 --rounds 2

如果脚本放在项目根目录：
  python3 test_full_table_sources_timeout.py --timeout 15 --rounds 2

输出：
  debug_full_table_sources/summary.json
  debug_full_table_sources/*.csv
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd


OUT = Path("debug_full_table_sources")


def _df_brief(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "head": df.head(3).astype(str).to_dict("records"),
        "tail": df.tail(3).astype(str).to_dict("records"),
    }


def run_with_timeout(name: str, fn: Callable[[], pd.DataFrame], timeout: int) -> dict[str, Any]:
    def worker(q):
        t0 = time.perf_counter()
        try:
            df = fn()
            elapsed = time.perf_counter() - t0
            if not isinstance(df, pd.DataFrame):
                q.put({
                    "name": name,
                    "ok": False,
                    "elapsed_seconds": elapsed,
                    "error": f"not dataframe: {type(df)}",
                })
                return
            q.put({
                "name": name,
                "ok": True,
                "elapsed_seconds": elapsed,
                **_df_brief(df),
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            q.put({
                "name": name,
                "ok": False,
                "elapsed_seconds": elapsed,
                "error": repr(e),
                "traceback": traceback.format_exc(limit=8),
            })

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "name": name,
            "ok": False,
            "elapsed_seconds": timeout,
            "error": f"timeout>{timeout}s",
        }

    if q.empty():
        return {
            "name": name,
            "ok": False,
            "elapsed_seconds": None,
            "error": "no child result",
        }

    return q.get()


def save_result_csv(name: str, result: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = result.get("head", []) + result.get("tail", [])
    if not rows:
        return
    safe = name.replace("/", "_").replace(" ", "_").replace(":", "_")
    pd.DataFrame(rows).to_csv(OUT / f"{safe}_sample.csv", index=False, encoding="utf-8-sig")


def get_func(ak, names: list[str]):
    for n in names:
        if hasattr(ak, n):
            return n, getattr(ak, n)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=15, help="单个接口超时秒数。")
    ap.add_argument("--rounds", type=int, default=1, help="每个接口重复测试轮数。")
    ap.add_argument(
        "--include-hk-full",
        action="store_true",
        help="包含 stock_hk_spot / stock_hk_spot_em 全港股接口测试。生产 14:55 路径不应使用它们。",
    )
    ap.add_argument(
        "--sectors",
        default="贵金属,工业金属,小金属,养殖业,农产品加工,电网设备,建筑材料,化学制药,化学制品",
        help="测试指定板块接口用的板块名，逗号分隔。",
    )
    args = ap.parse_args()

    import akshare as ak

    OUT.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, Callable[[], pd.DataFrame], str]] = []

    # A股实时全量源：collect_akshare_l1_cache.py 可能使用
    for label, func_names in [
        ("A股全量_Sina_stock_zh_a_spot", ["stock_zh_a_spot", "stock_zh_a_spot_sina"]),
        ("A股全量_EM_stock_zh_a_spot_em", ["stock_zh_a_spot_em"]),
        ("A股全量_THS_stock_zh_a_spot_ths", ["stock_zh_a_spot_ths"]),
    ]:
        fname, fn = get_func(ak, func_names)
        if fn is not None:
            tasks.append((label, fn, "full_a_stock"))
        else:
            tasks.append((label, lambda: pd.DataFrame(), f"missing_func:{func_names}"))

    # ETF全量源：如果你的 watchlist 有 ETF，会用到
    for label, func_names in [
        ("ETF全量_EM_fund_etf_spot_em", ["fund_etf_spot_em"]),
        ("ETF全量_THS_fund_etf_spot_ths", ["fund_etf_spot_ths"]),
    ]:
        fname, fn = get_func(ak, func_names)
        if fn is not None:
            tasks.append((label, fn, "full_etf"))
        else:
            tasks.append((label, lambda: pd.DataFrame(), f"missing_func:{func_names}"))

    # 行业板块全表：规模小
    for label, func_names in [
        ("行业全表_THS_stock_board_industry_summary_ths", ["stock_board_industry_summary_ths"]),
        ("行业名称全表_THS_stock_board_industry_name_ths", ["stock_board_industry_name_ths"]),
    ]:
        fname, fn = get_func(ak, func_names)
        if fn is not None:
            tasks.append((label, fn, "full_sector_small"))
        else:
            tasks.append((label, lambda: pd.DataFrame(), f"missing_func:{func_names}"))

    # 指定板块接口：不是全量，用作对照
    sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
    fname, fn_spot_sector = get_func(ak, ["stock_board_industry_spot_em"])
    if fn_spot_sector is not None:
        for sec in sectors:
            tasks.append((
                f"指定板块_EM_stock_board_industry_spot_em({sec})",
                (lambda sec=sec: fn_spot_sector(symbol=sec)),
                "targeted_sector",
            ))

    # A+H 全表：不大，但仍是全表
    fname, fn_ah = get_func(ak, ["stock_zh_ah_spot"])
    if fn_ah is not None:
        tasks.append(("A+H全表_stock_zh_ah_spot", fn_ah, "full_ah_mid"))

    # 港股全量：14:55 关键路径禁止，只用于诊断
    if args.include_hk_full:
        for label, func_names in [
            ("港股全量_Sina_stock_hk_spot", ["stock_hk_spot"]),
            ("港股全量_EM_stock_hk_spot_em", ["stock_hk_spot_em"]),
        ]:
            fname, fn = get_func(ak, func_names)
            if fn is not None:
                tasks.append((label, fn, "full_hk_forbidden"))
            else:
                tasks.append((label, lambda: pd.DataFrame(), f"missing_func:{func_names}"))

    results: list[dict[str, Any]] = []
    for round_idx in range(1, args.rounds + 1):
        print(f"\n========== ROUND {round_idx}/{args.rounds} ==========", flush=True)
        for name, fn, category in tasks:
            print(f"[TEST] {name} ...", flush=True)
            r = run_with_timeout(name, fn, args.timeout)
            r["round"] = round_idx
            r["category"] = category
            results.append(r)
            save_result_csv(f"round{round_idx}_{name}", r)

            ok = r.get("ok") and (r.get("shape", [0, 0])[0] > 0)
            status = "OK" if ok else "FAIL"
            elapsed = r.get("elapsed_seconds")
            shape = r.get("shape")
            print(f"  {status} elapsed={elapsed} shape={shape} error={r.get('error', '')}", flush=True)

    # 聚合每个接口的耗时
    agg = {}
    for r in results:
        name = r["name"]
        a = agg.setdefault(name, {
            "category": r.get("category"),
            "rounds": 0,
            "ok_rounds": 0,
            "timeout_rounds": 0,
            "errors": [],
            "elapsed_seconds": [],
            "last_shape": None,
        })
        a["rounds"] += 1
        if r.get("ok") and r.get("shape", [0, 0])[0] > 0:
            a["ok_rounds"] += 1
        if "timeout" in str(r.get("error", "")):
            a["timeout_rounds"] += 1
        if r.get("error"):
            a["errors"].append(r.get("error"))
        if isinstance(r.get("elapsed_seconds"), (int, float)):
            a["elapsed_seconds"].append(r["elapsed_seconds"])
        if r.get("shape"):
            a["last_shape"] = r["shape"]

    for a in agg.values():
        xs = a["elapsed_seconds"]
        if xs:
            a["min_elapsed"] = min(xs)
            a["max_elapsed"] = max(xs)
            a["avg_elapsed"] = sum(xs) / len(xs)
        else:
            a["min_elapsed"] = a["max_elapsed"] = a["avg_elapsed"] = None

    summary = {
        "timeout_seconds": args.timeout,
        "rounds": args.rounds,
        "include_hk_full": args.include_hk_full,
        "notes": {
            "full_hk_forbidden": "港股全量接口不应进入 14:55 关键路径；应使用小批量指定代码 sina_hk_batch。",
            "full_a_stock": "A股全量接口可保留，但必须有源级硬超时和字段补齐短路。",
            "full_sector_small": "行业全表规模小，一般可接受；但指定板块接口更优。",
        },
        "aggregate": agg,
        "results": results,
    }

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== AGGREGATE ==========")
    for name, a in agg.items():
        print(
            f"{name}: ok={a['ok_rounds']}/{a['rounds']}, "
            f"timeout={a['timeout_rounds']}, "
            f"avg={a['avg_elapsed']}, max={a['max_elapsed']}, "
            f"shape={a['last_shape']}, category={a['category']}"
        )
    print(f"\nSaved: {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
