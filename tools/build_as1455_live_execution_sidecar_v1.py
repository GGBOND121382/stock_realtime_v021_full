#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build execution sidecar for AS1455 live signal generation.

The sidecar makes live signal inputs closer to the backtest execution panel by
adding explicit up/down limit, tradable, ST and mainboard fields to the live
AS1455 raw row. It does not change model features or predictions.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_trade_date(value: str) -> str:
    if value.lower() == "today":
        return datetime.now().strftime("%Y%m%d")
    return value.replace("-", "")[:8]


def dash(value: str) -> str:
    s = value.replace("-", "")[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def normalize_symbol(value: object) -> str:
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        return ""
    m = re.search(r"(\d{6})", s)
    if not m:
        return s
    code = m.group(1)
    if ".SH" in s or code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def compact_symbol(symbol: str) -> str:
    m = re.search(r"(\d{6})", str(symbol))
    return m.group(1) if m else str(symbol)


def infer_mainboard(symbol: str) -> bool:
    code = compact_symbol(symbol)
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def raw_daily_candidates(cache_dir: Path, symbol: str) -> list[Path]:
    code = compact_symbol(symbol)
    return [cache_dir / f"{code}_daily_raw.csv", cache_dir / f"{code}_raw_daily.csv", cache_dir / f"{code}.csv"]


def read_raw_daily_row(cache_dir: Path, symbol: str, trade_date_dash: str) -> dict:
    for p in raw_daily_candidates(cache_dir, symbol):
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(p, dtype={"symbol": str, "code": str}, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        if "date" not in df.columns:
            continue
        df = df.copy()
        df["date_norm"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        exact = df[df["date_norm"].eq(trade_date_dash)]
        if exact.empty:
            hist = df[pd.to_datetime(df["date"], errors="coerce") < pd.Timestamp(trade_date_dash)].sort_values("date_norm")
            if hist.empty:
                return {}
            row = hist.iloc[-1].to_dict()
            row["_source_date"] = row.get("date_norm", "")
            row["_source_type"] = "prev_raw_daily"
            return row
        row = exact.iloc[-1].to_dict()
        row["_source_date"] = row.get("date_norm", "")
        row["_source_type"] = "same_day_raw_daily"
        return row
    return {}


def first_existing(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    return next((c for c in cols if c in df.columns), None)


def parse_bool(x, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t", "ok", "正常"}:
        return True
    if s in {"0", "false", "no", "n", "f", "停牌"}:
        return False
    try:
        return float(s) != 0.0
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AS1455 live execution sidecar")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--out-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--raw-daily-cache-dir", default="saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache")
    ap.add_argument("--universe", default="saved_data/ashare_ml4t/ch12_as1455/as1455_model_universe_from_h5.csv")
    ap.add_argument("--input-raw-row", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--st-limit-pct", type=float, default=0.05)
    ap.add_argument("--normal-limit-pct", type=float, default=0.10)
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    trade_date_dash = dash(trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    raw_path = Path(args.input_raw_row) if args.input_raw_row else live_dir / "08_live_raw_row_as1455.csv"
    out_path = Path(args.output) if args.output else live_dir / "08_live_execution_sidecar.csv"
    raw_daily_cache = Path(args.raw_daily_cache_dir)

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    live = pd.read_csv(raw_path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
    if "symbol" not in live.columns:
        if "code" in live.columns:
            live["symbol"] = live["code"]
        else:
            raise RuntimeError(f"{raw_path} has no symbol/code column")
    live = live.copy()
    live["symbol"] = live["symbol"].map(normalize_symbol)
    live = live[live["symbol"].astype(str).str.len() > 0].drop_duplicates("symbol", keep="last")

    # Optional universe enrichment for board/mainboard/name.
    uni = pd.DataFrame()
    up = Path(args.universe) if args.universe else None
    if up and up.exists():
        try:
            uni = pd.read_csv(up, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
            if "symbol" not in uni.columns and "code" in uni.columns:
                uni["symbol"] = uni["code"]
            if "symbol" in uni.columns:
                uni["symbol"] = uni["symbol"].map(normalize_symbol)
                uni = uni.drop_duplicates("symbol", keep="last")
        except Exception:
            uni = pd.DataFrame()
    if not uni.empty:
        keep_cols = [c for c in ["symbol", "name", "board", "industry", "is_mainboard", "trade_allowed_mainboard"] if c in uni.columns]
        live = live.merge(uni[keep_cols], on="symbol", how="left", suffixes=("", "_universe"))

    price_col = first_existing(live, ["raw_close_as1455", "close_as1455", "last_price", "price", "close"])
    pre_col = first_existing(live, ["live_preclose", "prev_close", "preclose", "raw_preclose"])
    open_col = first_existing(live, ["raw_open_as1455", "open_as1455", "open"])
    high_col = first_existing(live, ["raw_high_as1455", "high_as1455", "high"])
    low_col = first_existing(live, ["raw_low_as1455", "low_as1455", "low"])
    vol_col = first_existing(live, ["raw_volume_as1455", "volume_as1455", "volume", "volume_shares"])
    amt_col = first_existing(live, ["raw_amount_as1455", "amount_as1455", "amount", "amount_raw"])

    rows = []
    enrich_status = []
    for _, r in live.iterrows():
        sym = r["symbol"]
        rd = read_raw_daily_row(raw_daily_cache, sym, trade_date_dash)
        price = pd.to_numeric(pd.Series([r.get(price_col, np.nan)]), errors="coerce").iloc[0] if price_col else np.nan
        pre = pd.to_numeric(pd.Series([r.get(pre_col, np.nan)]), errors="coerce").iloc[0] if pre_col else np.nan
        if (not np.isfinite(pre) or pre <= 0) and rd:
            # For same-day raw daily rows, BaoStock preclose is exact. For T-1 row, close is today's preclose.
            if rd.get("_source_type") == "same_day_raw_daily" and "preclose" in rd:
                pre = pd.to_numeric(pd.Series([rd.get("preclose")]), errors="coerce").iloc[0]
            elif "close" in rd:
                pre = pd.to_numeric(pd.Series([rd.get("close")]), errors="coerce").iloc[0]
        name = str(r.get("name", r.get("name_universe", "")))
        is_st = False
        for st_col in ["isST", "is_st", "st"]:
            if st_col in live.columns:
                is_st = parse_bool(r.get(st_col), default=False)
                break
        if rd and "isST" in rd:
            is_st = parse_bool(rd.get("isST"), default=is_st)
        if "ST" in name.upper():
            is_st = True
        if "is_mainboard" in live.columns:
            is_mainboard = parse_bool(r.get("is_mainboard"), default=infer_mainboard(sym))
        elif "trade_allowed_mainboard" in live.columns:
            is_mainboard = parse_bool(r.get("trade_allowed_mainboard"), default=infer_mainboard(sym))
        elif "is_mainboard_universe" in live.columns:
            is_mainboard = parse_bool(r.get("is_mainboard_universe"), default=infer_mainboard(sym))
        else:
            is_mainboard = infer_mainboard(sym)
        tradestatus_val = None
        if rd and "tradestatus" in rd:
            tradestatus_val = pd.to_numeric(pd.Series([rd.get("tradestatus")]), errors="coerce").iloc[0]
        if tradestatus_val is None or pd.isna(tradestatus_val):
            core_complete = parse_bool(r.get("core_complete", True), default=True)
            tradestatus_val = 1 if (core_complete and np.isfinite(price) and price > 0) else 0
        tradable = bool(float(tradestatus_val) > 0 and np.isfinite(price) and price > 0)
        limit_pct = args.st_limit_pct if is_st else args.normal_limit_pct
        up_limit = round(float(pre) * (1.0 + limit_pct), 2) if np.isfinite(pre) and pre > 0 else np.nan
        down_limit = round(float(pre) * (1.0 - limit_pct), 2) if np.isfinite(pre) and pre > 0 else np.nan
        out = {
            "date": trade_date_dash,
            "symbol": sym,
            "name": name,
            "board": r.get("board", r.get("board_universe", "")),
            "industry": r.get("industry", r.get("industry_universe", "")),
            "raw_open_as1455": r.get(open_col, np.nan) if open_col else np.nan,
            "raw_high_as1455": r.get(high_col, np.nan) if high_col else np.nan,
            "raw_low_as1455": r.get(low_col, np.nan) if low_col else np.nan,
            "raw_close_as1455": price,
            "raw_volume_as1455": r.get(vol_col, np.nan) if vol_col else np.nan,
            "raw_amount_as1455": r.get(amt_col, np.nan) if amt_col else np.nan,
            "order_price": price,
            "raw_preclose": pre,
            "up_limit": up_limit,
            "down_limit": down_limit,
            "tradestatus": int(1 if tradable else 0),
            "tradable": tradable,
            "is_st": bool(is_st),
            "is_mainboard": bool(is_mainboard),
            "trade_allowed_mainboard": bool(is_mainboard),
            "last5_volume": np.nan,
            "last5_amount": np.nan,
            "execution_sidecar_source": "08_live_raw_row_as1455+raw_daily_cache",
            "raw_daily_source_date": rd.get("_source_date", "") if rd else "",
            "raw_daily_source_type": rd.get("_source_type", "missing_raw_daily") if rd else "missing_raw_daily",
        }
        rows.append(out)
        enrich_status.append(out["raw_daily_source_type"])

    sidecar = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(out_path, index=False, encoding="utf-8-sig")
    report = {
        "passed": bool(len(sidecar) > 0 and sidecar["raw_close_as1455"].notna().sum() >= max(1, int(len(sidecar) * 0.95))),
        "trade_date": trade_date,
        "input_raw_row": str(raw_path),
        "output": str(out_path),
        "rows": int(len(sidecar)),
        "symbols": int(sidecar["symbol"].nunique()) if not sidecar.empty else 0,
        "price_nonnull": int(sidecar["raw_close_as1455"].notna().sum()) if not sidecar.empty else 0,
        "preclose_nonnull": int(sidecar["raw_preclose"].notna().sum()) if not sidecar.empty else 0,
        "up_limit_nonnull": int(sidecar["up_limit"].notna().sum()) if not sidecar.empty else 0,
        "down_limit_nonnull": int(sidecar["down_limit"].notna().sum()) if not sidecar.empty else 0,
        "tradable_true": int(sidecar["tradable"].sum()) if not sidecar.empty else 0,
        "is_st_true": int(sidecar["is_st"].sum()) if not sidecar.empty else 0,
        "mainboard_true": int(sidecar["is_mainboard"].sum()) if not sidecar.empty else 0,
        "raw_daily_source_counts": pd.Series(enrich_status).value_counts(dropna=False).to_dict(),
        "last5_available": False,
        "note": "last5_volume/amount are left NaN at 14:55 unless supplied by a later data source; use capacity_mode=none for live planning before 15:00.",
    }
    report_path = out_path.with_name("08_live_execution_sidecar_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
