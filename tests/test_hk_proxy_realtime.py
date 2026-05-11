#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_hk_proxy_realtime.py

最小测试：港股 proxy 当天/盘中实时数据到底能不能拿。

测试对象：
  02714 牧原股份 H
  01610 中粮家佳康
  00288 万洲国际
  01068 雨润食品
  01117 现代牧业

测试接口：
  1) ak.stock_zh_ah_spot()
     - 已验证可拿 02714 牧原 H 股本体
     - 可能不覆盖所有 proxy，因为它主要是 A+H 表

  2) ak.stock_hk_spot()
     - 新浪港股实时快照，如果当前 AKShare 版本支持，优先作为 proxy 实时源

  3) ak.stock_hk_spot_em()
     - 东财港股实时全表；你服务器上之前容易 RemoteDisconnected，仅作为对照

  4) ak.stock_hk_daily(symbol=code)
     - 日线最新行，不等价于盘中实时；
     - 用于判断是否至少能拿到最新日线 fallback

输出：
  debug_hk_proxy_realtime/summary.json
  debug_hk_proxy_realtime/*.csv
"""

from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd


CODES = {
    "02714": "muyuan",
    "01610": "cofco_joycome",
    "00288": "wh_group",
    "01068": "yurun_food",
    "01117": "modern_dairy",
}

OUT = Path("debug_hk_proxy_realtime")
TIMEOUT = 30


def run_with_timeout(name: str, fn: Callable[[], pd.DataFrame], timeout: int = TIMEOUT) -> dict:
    def worker(q):
        try:
            df = fn()
            if not isinstance(df, pd.DataFrame):
                q.put({"ok": False, "error": f"not dataframe: {type(df)}"})
                return
            q.put({
                "ok": True,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "head": df.head(10).astype(str).to_dict("records"),
                "tail": df.tail(10).astype(str).to_dict("records"),
            })
        except Exception as e:
            q.put({
                "ok": False,
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
        return {"name": name, "ok": False, "error": f"timeout>{timeout}s"}

    if q.empty():
        return {"name": name, "ok": False, "error": "no child result"}

    r = q.get()
    r["name"] = name
    return r


def save_df(name: str, df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_").replace("(", "_").replace(")", "_")
    df.to_csv(OUT / f"{safe}.csv", index=False, encoding="utf-8-sig")


def code_variants(code: str) -> set[str]:
    raw = code.lstrip("0")
    return {
        code,
        raw,
        f"HK{raw}",
        f"HK{code}",
        f"{raw}.HK",
        f"{code}.HK",
    }


def find_candidates(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    mask = pd.Series(False, index=df.index)
    search_terms = set()
    for code in CODES:
        search_terms |= code_variants(code)

    # 名称兜底
    search_terms |= {
        "牧原",
        "中粮家佳康",
        "万洲",
        "雨润",
        "现代牧业",
        "COFCO",
        "WH GROUP",
        "YURUN",
        "MODERN DAIRY",
        "MUYUAN",
    }

    pattern = "|".join(sorted(search_terms, key=len, reverse=True))

    for col in df.columns:
        s = df[col].astype(str)
        mask |= s.str.contains(pattern, regex=True, case=False, na=False)

    hit = df[mask].copy()
    if not hit.empty:
        hit.insert(0, "source", source_name)
    return hit


def main():
    import akshare as ak

    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. A+H 实时全表
    if hasattr(ak, "stock_zh_ah_spot"):
        def fn():
            df = ak.stock_zh_ah_spot()
            hit = find_candidates(df, "stock_zh_ah_spot")
            save_df("stock_zh_ah_spot_hit", hit)
            return hit

        results.append(run_with_timeout("stock_zh_ah_spot_find_targets", fn))
    else:
        results.append({"name": "stock_zh_ah_spot_find_targets", "ok": False, "error": "akshare has no stock_zh_ah_spot"})

    # 2. 新浪港股实时快照
    if hasattr(ak, "stock_hk_spot"):
        def fn():
            df = ak.stock_hk_spot()
            hit = find_candidates(df, "stock_hk_spot")
            save_df("stock_hk_spot_hit", hit)
            return hit

        results.append(run_with_timeout("stock_hk_spot_find_targets", fn))
    else:
        results.append({"name": "stock_hk_spot_find_targets", "ok": False, "error": "akshare has no stock_hk_spot"})

    # 3. 东财港股实时全表
    if hasattr(ak, "stock_hk_spot_em"):
        def fn():
            df = ak.stock_hk_spot_em()
            hit = find_candidates(df, "stock_hk_spot_em")
            save_df("stock_hk_spot_em_hit", hit)
            return hit

        results.append(run_with_timeout("stock_hk_spot_em_find_targets", fn))
    else:
        results.append({"name": "stock_hk_spot_em_find_targets", "ok": False, "error": "akshare has no stock_hk_spot_em"})

    # 4. 单代码日线最新行：不是实时，但用于 fallback 可用性
    for code, label in CODES.items():
        if hasattr(ak, "stock_hk_daily"):
            def fn(code=code, label=label):
                df = ak.stock_hk_daily(symbol=code)
                if df is None or df.empty:
                    return pd.DataFrame()
                # 保存完整日线；返回末尾几行
                save_df(f"stock_hk_daily_full_{code}_{label}", df)
                out = df.tail(5).copy()
                out.insert(0, "target_code", code)
                out.insert(1, "target_label", label)
                out.insert(2, "source", "stock_hk_daily")
                return out

            results.append(run_with_timeout(f"stock_hk_daily_latest({code})/{label}", fn))
        else:
            results.append({"name": f"stock_hk_daily_latest({code})/{label}", "ok": False, "error": "akshare has no stock_hk_daily"})

    # 汇总判断：哪些代码在实时接口里出现
    realtime_hits = []
    for r in results:
        if not r.get("ok"):
            continue
        if not r["name"].startswith(("stock_zh_ah_spot", "stock_hk_spot_find", "stock_hk_spot_em")):
            continue
        rows = r.get("head", []) + r.get("tail", [])
        text = json.dumps(rows, ensure_ascii=False)
        found = []
        for code, label in CODES.items():
            variants = code_variants(code)
            if any(v in text for v in variants) or label.lower() in text.lower():
                found.append({"code": code, "label": label})
        realtime_hits.append({"source": r["name"], "found": found, "rows": r.get("shape", [0, 0])[0]})

    summary = {
        "created_date": str(date.today()),
        "codes": CODES,
        "realtime_hits": realtime_hits,
        "results": results,
        "decision_hint": {
            "if_stock_hk_spot_has_proxy_rows": "可把 stock_hk_spot 作为港股 proxy 14:55 实时源",
            "if_only_stock_zh_ah_spot_has_02714": "只能实时拿牧原 H 股本体；proxy 只能用 T-1/latest daily 或暂不作为实盘必需",
            "if_stock_hk_spot_em_fails": "不要把东财港股实时作为关键路径",
        },
    }

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 120)
    print("HK PROXY REALTIME TEST SUMMARY")
    print("=" * 120)

    for r in results:
        ok_rows = r.get("ok") and r.get("shape", [0, 0])[0] > 0
        status = "OK" if ok_rows else "FAIL"
        print(f"{status:4s} | {r['name']}")
        if r.get("shape"):
            print(f"     shape={r['shape']}")
            print(f"     columns={r.get('columns')}")
        if r.get("error"):
            print(f"     error={r['error']}")
        print()

    print("REALTIME HITS:")
    for h in realtime_hits:
        print(f"  {h['source']}: rows={h['rows']}, found={h['found']}")

    print()
    print(f"Saved: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
