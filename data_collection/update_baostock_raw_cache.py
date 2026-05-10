#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incrementally update BaoStock daily and 5m raw cache for one symbol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_symbol(symbol: str) -> tuple[str, str]:
    s = str(symbol).strip().lower().replace("_", ".")
    if "." in s:
        market, code = s.split(".", 1) if s.startswith(("sh.", "sz.")) else (None, None)
        if market is None:
            code, market = s.split(".", 1)
        return code.zfill(6), f"{market}.{code.zfill(6)}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    market = "sh" if code.startswith(("6", "9")) else "sz"
    return code, f"{market}.{code}"


def result_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def numeric(df: pd.DataFrame, date_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in date_cols:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_daily(bs, code: str, start: str, end: str) -> pd.DataFrame:
    fields = "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(code, fields, start_date=start, end_date=end, frequency="d", adjustflag="2")
    if rs.error_code != "0":
        raise RuntimeError(f"daily query failed: {rs.error_code} {rs.error_msg}")
    df = result_to_df(rs)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "event_flag", "no_price_limit_flag"])
    df = numeric(df, {"date"})
    df["event_flag"] = 0
    df["no_price_limit_flag"] = 0
    return df[["date", "open", "high", "low", "close", "volume", "event_flag", "no_price_limit_flag"]]


def parse_baostock_time(value: str) -> pd.Timestamp:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 14:
        return pd.to_datetime(digits[:14], format="%Y%m%d%H%M%S", errors="coerce")
    return pd.NaT


def fetch_5m(bs, code: str, start: str, end: str) -> pd.DataFrame:
    fields = "date,time,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(code, fields, start_date=start, end_date=end, frequency="5", adjustflag="2")
    if rs.error_code != "0":
        raise RuntimeError(f"5m query failed: {rs.error_code} {rs.error_msg}")
    df = result_to_df(rs)
    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
    df["datetime"] = df["time"].map(parse_baostock_time)
    df = numeric(df.drop(columns=["date", "time"]), {"datetime"})
    return df[["datetime", "open", "high", "low", "close", "volume", "amount"]].dropna(subset=["datetime"])


def merge_cache(path: Path, new_df: pd.DataFrame, key: str) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df.copy()
    merged[key] = pd.to_datetime(merged[key], errors="coerce") if key == "datetime" else merged[key].astype(str)
    merged = merged.dropna(subset=[key]).drop_duplicates(subset=[key], keep="last").sort_values(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description="Update BaoStock raw cache for one symbol")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--raw-cache-dir", required=True)
    args = p.parse_args()

    import baostock as bs

    code6, bs_code = normalize_symbol(args.symbol)
    raw_dir = Path(args.raw_cache_dir)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
    try:
        daily_new = fetch_daily(bs, bs_code, args.start_date, args.end_date)
        intraday_new = fetch_5m(bs, bs_code, args.start_date, args.end_date)
    finally:
        bs.logout()

    daily_path = raw_dir / f"{code6}_daily_raw.csv"
    intraday_path = raw_dir / f"{code6}_5m_raw.csv"
    daily = merge_cache(daily_path, daily_new, "date")
    intraday = merge_cache(intraday_path, intraday_new, "datetime")
    report = {
        "symbol": bs_code,
        "daily_new_rows": int(len(daily_new)),
        "intraday_new_rows": int(len(intraday_new)),
        "daily_rows": int(len(daily)),
        "intraday_rows": int(len(intraday)),
        "daily_min": str(pd.to_datetime(daily["date"]).min().date()) if len(daily) else None,
        "daily_max": str(pd.to_datetime(daily["date"]).max().date()) if len(daily) else None,
        "intraday_min": str(pd.to_datetime(intraday["datetime"]).min()) if len(intraday) else None,
        "intraday_max": str(pd.to_datetime(intraday["datetime"]).max()) if len(intraday) else None,
        "daily_path": str(daily_path),
        "intraday_path": str(intraday_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
