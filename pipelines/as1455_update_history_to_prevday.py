#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incrementally update AS1455 historical caches to T-1 using the original AS1455 5m schema.

This script treats the Chapter-12 AS1455 cache directory as a canonical local
cache, not as a place to dump native BaoStock rows.  The 5m cache schema written
by this updater is exactly the schema used by the original AS1455 builder:

    symbol, trade_date, datetime, open, high, low, close, volume, amount,
    source, bar_freq, bar_label

BaoStock native fields such as date/time/code/adjustflag are parsed at the
boundary and are not written back to the AS1455 5m cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    aggregate_as1455_from_5m,
    as1455_daily_path,
    baostock_code,
    dash_to_yyyymmdd,
    ensure_dir,
    get_last_cached_date,
    load_universe,
    merge_dedup_csv,
    normalize_symbol,
    parse_trade_date,
    raw_5m_path,
    raw_daily_path,
    read_5m_csv,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)

DEFAULT_CH12_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455"
DEFAULT_RAW_5M_CACHE = DEFAULT_CH12_DIR / "baostock_5m_cache"
DEFAULT_RAW_DAILY_CACHE = DEFAULT_CH12_DIR / "baostock_raw_daily_cache"
DEFAULT_AS1455_DAILY_CACHE = DEFAULT_CH12_DIR / "as1455_daily_cache"
DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"

OLD_5M_COLUMNS = [
    "symbol",
    "trade_date",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "bar_freq",
    "bar_label",
]


def import_baostock():
    try:
        import baostock as bs
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("baostock is required for non-dry-run history update") from exc
    return bs


def fallback_prev_weekday(trade_date_yyyymmdd: str) -> str:
    d = datetime.strptime(trade_date_yyyymmdd, "%Y%m%d").date()
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def resolve_history_end_date(trade_date_yyyymmdd: str, history_end_date: str | None, use_baostock_calendar: bool) -> str:
    if history_end_date and str(history_end_date).lower() not in {"prev", "prev_trade_date", "auto"}:
        return dash_to_yyyymmdd(history_end_date)
    if use_baostock_calendar:
        try:
            bs = import_baostock()
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(lg.error_msg)
            start = (datetime.strptime(trade_date_yyyymmdd, "%Y%m%d") - timedelta(days=20)).strftime("%Y-%m-%d")
            end = yyyymmdd_to_dash(trade_date_yyyymmdd)
            rs = bs.query_trade_dates(start_date=start, end_date=end)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            bs.logout()
            cal = pd.DataFrame(rows, columns=rs.fields)
            cal = cal[(cal["is_trading_day"] == "1") & (cal["calendar_date"] < end)]
            if not cal.empty:
                return dash_to_yyyymmdd(cal["calendar_date"].max())
        except Exception as exc:
            print(f"[WARN] BaoStock calendar failed, fallback to previous weekday: {type(exc).__name__}: {exc}", flush=True)
    return fallback_prev_weekday(trade_date_yyyymmdd)


def parse_baostock_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse BaoStock minute date/time into pandas datetime.

    BaoStock minute ``time`` is often a compact full timestamp like
    ``20260623145500000``.  Some exports may split date and time.  The output is
    always the right-end bar timestamp used by the original AS1455 builder.
    """
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")

    if "time" not in df.columns:
        raise ValueError("5m data must contain either datetime or BaoStock time")

    vals = []
    date_values = df["date"].astype(str) if "date" in df.columns else pd.Series([""] * len(df), index=df.index)
    for date_v, time_v in zip(date_values, df["time"].astype(str)):
        date_digits = "".join(ch for ch in date_v if ch.isdigit())
        time_digits = "".join(ch for ch in time_v if ch.isdigit())
        if len(time_digits) >= 14:
            raw = time_digits[:14]
        else:
            raw = date_digits[:8] + time_digits[:6].zfill(6)
        vals.append(raw)
    return pd.to_datetime(vals, format="%Y%m%d%H%M%S", errors="coerce")


def standardize_5m_to_old_schema(df: pd.DataFrame, symbol: str, source: str = "baostock_5m_adjustflag_3") -> pd.DataFrame:
    """Convert a 5m frame to the original AS1455 cache schema.

    This is the single boundary where native BaoStock fields are allowed.  The
    returned frame never contains native BaoStock date/time/code/adjustflag
    columns, so the on-disk cache remains schema-stable.
    """
    symbol = normalize_symbol(symbol)
    if df is None or df.empty:
        return pd.DataFrame(columns=OLD_5M_COLUMNS)

    out = pd.DataFrame(index=df.index)
    if "symbol" in df.columns and not df["symbol"].isna().all():
        out["symbol"] = df["symbol"].map(normalize_symbol)
    else:
        out["symbol"] = symbol

    out["datetime"] = parse_baostock_datetime(df)
    if "trade_date" in df.columns:
        trade_date = df["trade_date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
        out["trade_date"] = trade_date.where(trade_date.str.len().eq(8), out["datetime"].dt.strftime("%Y%m%d"))
    else:
        out["trade_date"] = out["datetime"].dt.strftime("%Y%m%d")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA

    if "source" in df.columns:
        out["source"] = df["source"].astype(str).where(df["source"].notna(), source)
    else:
        out["source"] = source
    if "bar_freq" in df.columns:
        out["bar_freq"] = df["bar_freq"].astype(str).where(df["bar_freq"].notna(), "5min")
    else:
        out["bar_freq"] = "5min"
    if "bar_label" in df.columns:
        out["bar_label"] = df["bar_label"].astype(str).where(df["bar_label"].notna(), "right")
    else:
        out["bar_label"] = "right"

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    if out.empty:
        return pd.DataFrame(columns=OLD_5M_COLUMNS)
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    if out.empty:
        return pd.DataFrame(columns=OLD_5M_COLUMNS)

    t = out["datetime"].dt.time
    session = ((t >= dtime(9, 30)) & (t <= dtime(11, 30))) | ((t >= dtime(13, 0)) & (t <= dtime(15, 0)))
    out = out.loc[session, OLD_5M_COLUMNS].copy()
    out.sort_values(["symbol", "datetime"], inplace=True)
    out.drop_duplicates(["symbol", "datetime"], keep="last", inplace=True)
    return out.reset_index(drop=True)


def read_old_5m_cache(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=OLD_5M_COLUMNS)
    df = pd.read_csv(path, dtype={"symbol": str, "trade_date": str, "code": str}, encoding="utf-8-sig", low_memory=False)
    return standardize_5m_to_old_schema(df, symbol=symbol)


def write_old_5m_cache(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    out = standardize_5m_to_old_schema(df, symbol=df["symbol"].iloc[0] if "symbol" in df.columns and len(df) else "")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def latest_old_5m_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df.empty or "trade_date" not in df.columns:
        return None
    dates = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def merge_old_5m_cache(existing: pd.DataFrame, new_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    old = standardize_5m_to_old_schema(existing, symbol=symbol) if existing is not None and not existing.empty else pd.DataFrame(columns=OLD_5M_COLUMNS)
    new = standardize_5m_to_old_schema(new_df, symbol=symbol) if new_df is not None and not new_df.empty else pd.DataFrame(columns=OLD_5M_COLUMNS)
    if old.empty and new.empty:
        return pd.DataFrame(columns=OLD_5M_COLUMNS)
    merged = pd.concat([old, new], ignore_index=True, sort=False)
    merged = standardize_5m_to_old_schema(merged, symbol=symbol)
    return merged


def query_baostock(bs, symbol: str, start_date: str, end_date: str, frequency: str, fields: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        baostock_code(symbol), fields,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        adjustflag="3",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    return pd.DataFrame(rows, columns=rs.fields)


def fetch_raw_5m(bs, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    native = query_baostock(bs, symbol, start_date, end_date, "5", fields)
    if native.empty:
        return pd.DataFrame(columns=OLD_5M_COLUMNS)
    return standardize_5m_to_old_schema(native, symbol=symbol, source="baostock_5m_adjustflag_3")


def fetch_raw_daily(bs, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    df = query_baostock(bs, symbol, start_date, end_date, "d", fields)
    if df.empty:
        return df
    df.insert(0, "symbol", normalize_symbol(symbol))
    for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def next_query_start(last_cached: Optional[pd.Timestamp], default_start: str) -> str:
    if last_cached is None:
        return default_start
    return (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def update_one_symbol(symbol: str, args, history_end_dash: str, bs=None) -> dict:
    raw5m_p = raw_5m_path(Path(args.raw_5m_cache_dir), symbol)
    rawdaily_p = raw_daily_path(Path(args.raw_daily_cache_dir), symbol)
    as1455_p = as1455_daily_path(Path(args.as1455_daily_cache_dir), symbol)
    row = {
        "symbol": symbol,
        "history_end_date": history_end_dash,
        "raw_5m_path": str(raw5m_p),
        "raw_daily_path": str(rawdaily_p),
        "as1455_daily_path": str(as1455_p),
        "raw_5m_status": "skip",
        "raw_daily_status": "skip",
        "as1455_status": "skip",
        "raw_5m_new_rows": 0,
        "raw_daily_new_rows": 0,
        "as1455_new_rows": 0,
        "error": "",
    }

    default_start = args.history_start_date
    existing5 = read_old_5m_cache(raw5m_p, symbol)
    last5 = latest_old_5m_date(existing5)
    lastd = get_last_cached_date(rawdaily_p, "date")
    start5 = next_query_start(last5, default_start)
    startd = next_query_start(lastd, default_start)
    row["raw_5m_last_cached_date"] = "" if last5 is None else last5.strftime("%Y-%m-%d")
    row["raw_daily_last_cached_date"] = "" if lastd is None else lastd.strftime("%Y-%m-%d")
    row["raw_5m_query_start"] = start5
    row["raw_daily_query_start"] = startd

    if pd.Timestamp(start5) <= pd.Timestamp(history_end_dash):
        if args.dry_run:
            row["raw_5m_status"] = "would_fetch"
        else:
            if bs is None:
                raise RuntimeError("BaoStock session is not initialized")
            new5 = fetch_raw_5m(bs, symbol, start5, history_end_dash)
            row["raw_5m_new_rows"] = int(len(new5))
            if new5.empty:
                row["raw_5m_status"] = "empty"
                if not existing5.empty:
                    write_old_5m_cache(raw5m_p, existing5)
                    row["raw_5m_status"] = "cached_existing_only"
            else:
                merged5 = merge_old_5m_cache(existing5, new5, symbol)
                write_old_5m_cache(raw5m_p, merged5)
                row["raw_5m_status"] = "updated"
    else:
        row["raw_5m_status"] = "cached"
        if not existing5.empty:
            # Re-write only in canonical old schema. This is a deterministic
            # cleanup, not a schema compatibility layer for future writes.
            write_old_5m_cache(raw5m_p, existing5)

    if pd.Timestamp(startd) <= pd.Timestamp(history_end_dash):
        if args.dry_run:
            row["raw_daily_status"] = "would_fetch"
        else:
            if bs is None:
                raise RuntimeError("BaoStock session is not initialized")
            newd = fetch_raw_daily(bs, symbol, startd, history_end_dash)
            row["raw_daily_new_rows"] = int(len(newd))
            if newd.empty:
                row["raw_daily_status"] = "empty"
            else:
                merged = merge_dedup_csv(rawdaily_p, newd, subset=["symbol", "date"])
                write_csv(rawdaily_p, merged)
                row["raw_daily_status"] = "updated"
    else:
        row["raw_daily_status"] = "cached"

    if not args.skip_as1455_aggregate:
        agg_start = min(start5, history_end_dash)
        if args.dry_run:
            row["as1455_status"] = "would_aggregate"
        elif raw5m_p.exists() and raw5m_p.stat().st_size > 0:
            bars = read_5m_csv(raw5m_p, symbol=symbol)
            daily = aggregate_as1455_from_5m(bars, symbol=symbol, start_date=agg_start, end_date=history_end_dash)
            if daily.empty:
                row["as1455_status"] = "empty"
            else:
                before = len(daily)
                daily = daily[daily["has_14_55_bar"].astype(bool)].copy()
                row["as1455_missing_1455_rows"] = int(before - len(daily))
                row["as1455_new_rows"] = int(len(daily))
                merged = merge_dedup_csv(as1455_p, daily, subset=["symbol", "date"])
                write_csv(as1455_p, merged)
                row["as1455_status"] = "updated"
        else:
            row["as1455_status"] = "no_raw_5m_cache"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Incrementally update AS1455 historical raw caches to previous trading day")
    ap.add_argument("--trade-date", default="today", help="live trade date, YYYYMMDD/YYYY-MM-DD/today")
    ap.add_argument("--history-end-date", default="auto", help="target history end date; default=previous trading day")
    ap.add_argument("--history-start-date", default="2020-01-01", help="start date if a cache is missing")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--raw-5m-cache-dir", default=str(DEFAULT_RAW_5M_CACHE))
    ap.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY_CACHE))
    ap.add_argument("--as1455-daily-cache-dir", default=str(DEFAULT_AS1455_DAILY_CACHE))
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-as1455-aggregate", action="store_true")
    ap.add_argument("--sleep-seconds", type=float, default=0.05)
    ap.add_argument("--no-baostock-calendar", action="store_true")
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    history_end = resolve_history_end_date(trade_date, args.history_end_date, not args.no_baostock_calendar)
    history_end_dash = yyyymmdd_to_dash(history_end)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    for p in [args.raw_5m_cache_dir, args.raw_daily_cache_dir, args.as1455_daily_cache_dir]:
        ensure_dir(Path(p))

    universe = load_universe(args.universe, args.max_symbols)
    write_csv(live_dir / "01_universe.csv", universe)
    rows = []
    started = time.time()
    bs_session = None
    if not args.dry_run:
        bs_session = import_baostock()
        lg = bs_session.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        for i, symbol in enumerate(universe["symbol"].tolist(), 1):
            try:
                r = update_one_symbol(symbol, args, history_end_dash, bs=bs_session)
            except Exception as exc:
                r = {"symbol": symbol, "history_end_date": history_end_dash, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(r)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            if i % 50 == 0:
                print(f"[INFO] processed {i}/{len(universe)} symbols", flush=True)
    finally:
        if bs_session is not None:
            try:
                bs_session.logout()
            except Exception:
                pass

    report = pd.DataFrame(rows)
    write_csv(live_dir / "00_history_update_by_symbol.csv", report)
    summary = {
        "trade_date": trade_date,
        "history_end_date": history_end_dash,
        "n_symbols": int(len(universe)),
        "dry_run": bool(args.dry_run),
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": int(report.get("error", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()) if not report.empty else 0,
        "raw_5m_status_counts": report.get("raw_5m_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "raw_daily_status_counts": report.get("raw_daily_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "as1455_status_counts": report.get("as1455_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
    }
    write_json(live_dir / "00_history_update_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
