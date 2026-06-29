#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field-contract self-test for AS1455 fast_v4.

This test uses synthetic old-schema 5m bars: trade_date + datetime, no date.
It verifies fast_v4 adds `date` before calling aggregate_as1455_from_5m.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import aggregate_as1455_from_5m  # noqa: E402
from pipelines.as1455_update_history_to_prevday_fast_v4 import add_aggregate_date_column  # noqa: E402


def main() -> None:
    times = [
        "2026-06-26 09:35:00",
        "2026-06-26 09:40:00",
        "2026-06-26 14:50:00",
        "2026-06-26 14:55:00",
    ]
    bars = pd.DataFrame({
        "symbol": ["000001.SZ"] * len(times),
        "trade_date": ["20260626"] * len(times),
        "datetime": pd.to_datetime(times),
        "open": [10.0, 10.1, 10.2, 10.3],
        "high": [10.2, 10.3, 10.4, 10.5],
        "low": [9.9, 10.0, 10.1, 10.2],
        "close": [10.1, 10.2, 10.3, 10.4],
        "volume": [1000, 2000, 3000, 4000],
        "amount": [10100, 20400, 30900, 41600],
        "source": ["synthetic"] * len(times),
        "bar_freq": ["5min"] * len(times),
        "bar_label": ["right"] * len(times),
    })
    assert "date" not in bars.columns
    fixed = add_aggregate_date_column(bars)
    assert "date" in fixed.columns
    out = aggregate_as1455_from_5m(fixed, symbol="000001.SZ", start_date="2026-06-26", end_date="2026-06-26")
    assert len(out) == 1, out
    r = out.iloc[0]
    assert r["date"] == "2026-06-26", r.to_dict()
    assert bool(r["has_14_55_bar"]) is True, r.to_dict()
    assert str(r["last_bar_time"]) == "14:55", r.to_dict()
    print("[OK] fast_v4 field contract self-test passed")


if __name__ == "__main__":
    main()
