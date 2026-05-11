#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test targeted HK realtime and daily sources without full-market HK calls.

This script intentionally does NOT call ak.stock_hk_spot() or
ak.stock_hk_spot_em(), because those pull the whole HK market.
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import pandas as pd
import requests
import akshare as ak

CODES = {
    "02714": "hog_hk_muyuan",
    "01610": "hog_hk_cofco_joycome",
    "00288": "hog_hk_wh_group",
    "01068": "hog_hk_yurun_food",
    "01117": "hog_hk_modern_dairy",
    "02899": "zijin_hk",
}
OUT = Path("debug_hk_targeted_sources")
OUT.mkdir(exist_ok=True)


def is_num(x):
    try:
        float(str(x).replace(',', '').replace('%', '').strip())
        return True
    except Exception:
        return False


def parse_line(line):
    m = re.search(r'hq_str_hk(\d{5})="(.*)";?', line)
    if not m:
        return None
    code, raw = m.group(1), m.group(2)
    parts = [p.strip() for p in raw.split(',')]
    if len(parts) >= 15 and not is_num(parts[1]) and is_num(parts[2]):
        name, off = parts[1] or parts[0], 2
    else:
        name, off = parts[0], 1
    vals = parts[off:]
    row = {"code": code, "name": name, "raw": raw, "n_fields": len(parts)}
    keys = ["open", "prev_close", "high", "low", "close", "change", "pct_chg", "bid", "ask", "volume", "amount"]
    for i, k in enumerate(keys):
        row[k] = vals[i] if i < len(vals) else None
    return row


def fetch_sina_batch(codes):
    symbols = ','.join('hk' + c for c in codes)
    url = 'https://hq.sinajs.cn/list=' + symbols
    r = requests.get(url, headers={"Referer": "https://finance.sina.com.cn/", "User-Agent": "Mozilla/5.0"}, timeout=5)
    r.raise_for_status()
    text = r.content.decode('gbk', errors='replace')
    rows = []
    for line in text.splitlines():
        row = parse_line(line)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    results = {}
    try:
        df = fetch_sina_batch(list(CODES))
        df.to_csv(OUT / 'sina_hk_batch.csv', index=False, encoding='utf-8-sig')
        got = set(df['code'].astype(str).str.zfill(5)) if not df.empty and 'code' in df else set()
        results['sina_hk_batch'] = {"ok": bool(got), "rows": len(df), "got_codes": sorted(got), "missing_codes": sorted(set(CODES)-got)}
    except Exception as e:
        results['sina_hk_batch'] = {"ok": False, "error": repr(e), "traceback": traceback.format_exc(limit=5)}

    daily = {}
    for code, label in CODES.items():
        try:
            df = ak.stock_hk_daily(symbol=code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.tail(5).to_csv(OUT / f'stock_hk_daily_{code}_{label}.csv', index=False, encoding='utf-8-sig')
                daily[code] = {"ok": True, "rows": len(df), "columns": list(df.columns), "last": df.tail(1).astype(str).to_dict('records')[0]}
            else:
                daily[code] = {"ok": False, "error": "empty"}
        except Exception as e:
            daily[code] = {"ok": False, "error": repr(e)}
    results['stock_hk_daily'] = daily

    (OUT / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
