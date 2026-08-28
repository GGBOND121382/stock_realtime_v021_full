#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Choose T or the previous completed trading day for the nightly BaoStock refresh.

BaoStock's calendar can mark T as a trading day before T's bars are published.
After 20:00 we prefer T only when a few liquid symbols already expose a normal
daily row. Otherwise this prints ``auto`` and the canonical updater resolves T-1.
Network/API failures are deliberately fail-soft and also return ``auto``.

The command's stdout contract is intentionally strict: exactly one line, either
``YYYY-MM-DD`` or ``auto``.  BaoStock sometimes prints login/logout diagnostics,
so all third-party stdout/stderr is suppressed while probing.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import baostock_code, load_universe, normalize_symbol  # noqa: E402


def trade_date(value: str, timezone: str) -> pd.Timestamp:
    if value.lower() == "today":
        return pd.Timestamp(datetime.now(ZoneInfo(timezone)).date())
    return pd.Timestamp(value).normalize()


def _quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return func(*args, **kwargs)


def probe(today: pd.Timestamp, universe_path: Path, probes: int) -> bool:
    try:
        import baostock as bs
    except Exception:
        return False
    try:
        universe = load_universe(str(universe_path), None)
    except Exception:
        return False
    if universe.empty or "symbol" not in universe.columns:
        return False

    symbols = [normalize_symbol(value) for value in universe["symbol"].astype(str)]
    symbols = sorted(
        dict.fromkeys(symbols),
        key=lambda s: (
            0 if str(s).startswith(("000", "001", "002", "600", "601", "603")) else 1,
            str(s),
        ),
    )[: max(1, int(probes))]

    login = None
    try:
        login = _quiet_call(bs.login)
        if login is None or getattr(login, "error_code", None) != "0":
            return False
        day = today.strftime("%Y-%m-%d")
        for symbol in symbols:
            rs = _quiet_call(
                bs.query_history_k_data_plus,
                baostock_code(symbol),
                "date,code,close,tradestatus",
                start_date=day,
                end_date=day,
                frequency="d",
                adjustflag="3",
            )
            rows = []
            while getattr(rs, "error_code", "1") == "0" and rs.next():
                rows.append(rs.get_row_data())
            if getattr(rs, "error_code", "1") != "0" or not rows:
                continue
            frame = pd.DataFrame(rows, columns=rs.fields)
            if not {"date", "close"}.issubset(frame.columns):
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            close = pd.to_numeric(frame["close"], errors="coerce")
            status = (
                pd.to_numeric(frame["tradestatus"], errors="coerce").fillna(1)
                if "tradestatus" in frame.columns
                else pd.Series(1, index=frame.index, dtype=float)
            )
            if bool((dates.eq(today) & close.gt(0) & status.eq(1)).any()):
                return True
    except Exception:
        return False
    finally:
        if login is not None:
            try:
                _quiet_call(bs.logout)
            except Exception:
                pass
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--timezone", default="Asia/Shanghai")
    ap.add_argument(
        "--universe",
        default="saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv",
    )
    ap.add_argument("--probes", type=int, default=8)
    args = ap.parse_args()
    day = trade_date(args.trade_date, args.timezone)
    print(day.strftime("%Y-%m-%d") if probe(day, Path(args.universe), args.probes) else "auto")


if __name__ == "__main__":
    main()
