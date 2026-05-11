#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_sina_hk_field_mapping.py

目的：
  只测试 Sina HK 小批量行情 raw 字段映射，避免再把成交额/成交量反过来。

运行：
  python3 tests/test_sina_hk_field_mapping.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_collection.collect_realtime_context import _parse_sina_hk_payload


def main():
    raw = "MUYUAN,牧原股份,41.900,41.780,42.500,41.100,41.300,-0.480,-1.149,41.30000,41.38000,134343826,3207517,0.000,0.000,50.700,37.580,2026/05/11,16:08"
    row = _parse_sina_hk_payload("02714", raw)

    assert row["代码"] == "02714"
    assert row["名称"] == "牧原股份"
    assert row["最新价"] == "41.300"

    # Critical check:
    # after bid/ask, Sina HK payload is amount first, volume second.
    assert row["成交额"] == "134343826", row
    assert row["成交量"] == "3207517", row

    raw2 = "WH GROUP,万洲国际,9.820,9.800,9.900,9.750,9.830,0.030,0.306,9.83000,9.84000,281953054,28703193,0.000,0.000,10.840,5.977,2026/05/11,16:08"
    row2 = _parse_sina_hk_payload("00288", raw2)
    assert row2["成交额"] == "281953054", row2
    assert row2["成交量"] == "28703193", row2

    print("[OK] Sina HK field mapping: amount/volume are correct.")


if __name__ == "__main__":
    main()
