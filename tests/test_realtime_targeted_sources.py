#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test targeted 14:55 realtime sources.

This test avoids full-market A-share/HK APIs.  It checks only the code paths
that should be used in production after the realtime source rewrite.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_collection.collect_akshare_l1_cache import fetch_sina_targeted_spot_map, normalize_symbol, load_akshare, fetch_ths_etf_spot_map
from data_collection.collect_realtime_context import fetch_sina_hk_realtime_batch, fetch_sector_snapshot

OUT = Path("debug_realtime_targeted_sources")
OUT.mkdir(exist_ok=True)


def run_with_timeout(name, fn, timeout=10):
    def worker(q):
        try:
            res = fn()
            q.put({"ok": True, "result": res})
        except Exception as e:
            q.put({"ok": False, "error": repr(e), "traceback": traceback.format_exc(limit=8)})
    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return {"name": name, "ok": False, "error": f"timeout>{timeout}s"}
    if q.empty():
        return {"name": name, "ok": False, "error": "no child result"}
    r = q.get(); r["name"] = name
    return r


def main():
    symbols = ["600312.SH", "601899.SH", "002714.SZ", "002311.SZ", "512890.SH", "159595.SZ"]
    hk_codes = ["02714", "01610", "00288", "01068", "01117", "02899"]
    sectors = ["电网设备", "贵金属", "工业金属", "养殖业", "农产品加工", "建筑材料", "化学制品"]

    results = []

    def test_a_sina():
        got = fetch_sina_targeted_spot_map(symbols, timeout=6)
        return {
            "requested": symbols,
            "got_symbols": sorted(got.keys()),
            "missing_symbols": [normalize_symbol(s) for s in symbols if normalize_symbol(s) not in got],
            "sample": {k: {kk: v.get(kk) for kk in ["名称", "最新价", "今开", "最高", "最低", "成交量", "成交额"]} for k, v in list(got.items())[:10]},
        }
    results.append(run_with_timeout("sina_targeted_a_stock_etf", test_a_sina, timeout=8))

    def test_hk_sina():
        df = fetch_sina_hk_realtime_batch(hk_codes, timeout=6)
        df.to_csv(OUT / "sina_hk_batch.csv", index=False, encoding="utf-8-sig")
        got = sorted(df.get("代码", pd.Series(dtype=str)).astype(str).tolist()) if not df.empty else []
        # Validate Sina HK amount/volume mapping.
        # Raw common format:
        #   en_name,cn_name,open,prev_close,high,low,last,change,pct,bid,ask,amount,volume,...
        # Previously amount and volume were reversed; keep this check here.
        hk_field_check = []
        for _, row in df.iterrows():
            raw = str(row.get("raw", ""))
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) >= 13:
                # Most current payloads have English name + Chinese name, so numeric fields start at index 2.
                offset = 2 if len(parts) >= 15 and not str(parts[1]).replace(".", "", 1).isdigit() else 1
                vals = parts[offset:]
                if len(vals) > 10:
                    expected_amount = vals[9]
                    expected_volume = vals[10]
                    hk_field_check.append({
                        "code": str(row.get("代码")),
                        "amount_ok": str(row.get("成交额")) == expected_amount,
                        "volume_ok": str(row.get("成交量")) == expected_volume,
                        "parsed_amount": str(row.get("成交额")),
                        "expected_amount": expected_amount,
                        "parsed_volume": str(row.get("成交量")),
                        "expected_volume": expected_volume,
                    })
        bad = [x for x in hk_field_check if not x["amount_ok"] or not x["volume_ok"]]
        return {
            "requested": hk_codes,
            "rows": int(len(df)),
            "got_codes": got,
            "missing_codes": [c for c in hk_codes if c not in got],
            "columns": list(df.columns),
            "amount_volume_field_check": hk_field_check,
            "amount_volume_field_check_ok": len(bad) == 0,
            "amount_volume_field_check_bad": bad,
            "sample": df.head(10).astype(str).to_dict("records"),
        }
    results.append(run_with_timeout("sina_targeted_hk_batch", test_hk_sina, timeout=8))

    def test_ths_etf():
        ak = load_akshare()
        got = fetch_ths_etf_spot_map(ak, symbols)
        return {"requested": symbols, "got_symbols": sorted(got.keys()), "rows": len(got)}
    results.append(run_with_timeout("ths_etf_targeted_filter", test_ths_etf, timeout=8))

    def test_sector_summary():
        ak = load_akshare()
        rows = []
        for sec in sectors:
            snap = fetch_sector_snapshot(ak, sec)
            rows.append({k: snap.get(k) for k in ["context_symbol", "provider", "status", "close", "pct_chg", "volume", "amount", "error"]})
        return {"sectors": rows}
    results.append(run_with_timeout("ths_sector_summary_exact_filter", test_sector_summary, timeout=10))

    summary = {"results": results}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Nonzero only if targeted A and HK both fail; ETF/sector may be partial depending on market day.
    hard_names = {"sina_targeted_a_stock_etf", "sina_targeted_hk_batch"}
    hard_fail = [r for r in results if r["name"] in hard_names and not r.get("ok")]
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
