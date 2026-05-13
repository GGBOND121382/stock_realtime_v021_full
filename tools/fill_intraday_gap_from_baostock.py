#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill early intraday 5-minute bar gaps from BaoStock into realtime cache.

Purpose
-------
A live collection may start late (e.g. 10:38), which makes first_30m/first_60m
features unavailable. This script queries BaoStock 5-minute bars for the target
trade date and merges only the missing early bars into:

  saved_data/akshare_realtime_cache/pending/YYYYMMDD/<SYMBOL>/minute_bars_5min.csv

It also rebuilds/updates:

  saved_data/akshare_realtime_cache/feature_cache/<SYMBOL>_intraday_reversal_features.csv

Notes
-----
* BaoStock standard minute frequencies are 5/15/30/60; this script uses 5m.
* It does NOT fake 1-minute bars.
* Existing locally collected rows take priority for overlapping timestamps.
* The script is safe-by-default: it backs up files before overwriting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_DIR / "saved_data" / "akshare_realtime_cache"


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace("_", ".")
    if not s:
        return ""
    if "." in s:
        a, b = s.split(".", 1)
        if a in {"SH", "SZ"}:
            market, code = a, b
        else:
            code, market = a, b
        return f"{code.zfill(6)}.{market}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def baostock_code(symbol: str) -> str:
    s = normalize_symbol(symbol)
    code, market = s.split(".", 1)
    prefix = "sh" if market == "SH" else "sz"
    return f"{prefix}.{code}"


def date_compact_to_dash(value: str) -> str:
    value = str(value).strip()
    if "-" in value:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def discover_symbols(cache_dir: Path, date: str, symbols_file: Optional[Path]) -> list[str]:
    symbols: list[str] = []
    if symbols_file and symbols_file.exists():
        for line in symbols_file.read_text(encoding="utf-8-sig").splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                symbols.append(normalize_symbol(token))
    if not symbols:
        pending = cache_dir / "pending" / date
        if pending.exists():
            for p in sorted(pending.iterdir()):
                if p.is_dir():
                    sym = normalize_symbol(p.name)
                    if sym:
                        symbols.append(sym)
    return list(dict.fromkeys([s for s in symbols if s]))


def parse_bs_time(value) -> pd.Timestamp:
    text = str(value or "").strip()
    # Common BaoStock minute field: YYYYMMDDHHMMSSmmm, e.g. 20260513093500000.
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 14:
        return pd.to_datetime(digits[:14], format="%Y%m%d%H%M%S", errors="coerce")
    if len(digits) == 8:
        return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def query_baostock_5m(symbol: str, date: str) -> pd.DataFrame:
    try:
        import baostock as bs  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "BaoStock is not installed. Install locally with: pip install baostock"
        ) from exc

    code = baostock_code(symbol)
    day = date_compact_to_dash(date)
    lg = bs.login()
    if getattr(lg, "error_code", "") != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
    try:
        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=day,
            end_date=day,
            frequency="5",
            adjustflag="3",
        )
        if getattr(rs, "error_code", "") != "0":
            raise RuntimeError(f"BaoStock query failed for {code}: {rs.error_code} {rs.error_msg}")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "source"])

    out = pd.DataFrame()
    out["datetime"] = df["time"].map(parse_bs_time)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    out["source"] = "baostock_5m"
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    # Keep normal A-share session bars only.
    t = out["datetime"].dt.time
    mask = ((t >= dtime(9, 30)) & (t <= dtime(11, 30))) | ((t >= dtime(13, 0)) & (t <= dtime(15, 0)))
    out = out.loc[mask].reset_index(drop=True)
    return out


def read_existing_5m(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount", "source"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "datetime" not in df.columns:
        raise ValueError(f"missing datetime column in {path}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    if "source" not in df.columns:
        df["source"] = "local_existing"
    return df.dropna(subset=["datetime"]).sort_values("datetime")


def backup_file(path: Path, backup_root: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    dst = backup_root / path.name
    dst.write_bytes(path.read_bytes())
    return dst


def merge_bars(existing: pd.DataFrame, bs5: pd.DataFrame, only_before_existing: bool = True, before_time: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
    existing = existing.copy()
    bs5 = bs5.copy()
    if only_before_existing and not existing.empty:
        first_existing = existing["datetime"].min()
        bs5 = bs5[bs5["datetime"] < first_existing].copy()
    if before_time:
        hh, mm = [int(x) for x in before_time.split(":")[:2]]
        cutoff_t = dtime(hh, mm)
        bs5 = bs5[bs5["datetime"].dt.time < cutoff_t].copy()

    # Existing rows should win on overlap. concat bs first, existing second, keep last.
    merged = pd.concat([bs5, existing], ignore_index=True, sort=False)
    if merged.empty:
        return merged, {"added_rows": 0, "existing_rows": int(len(existing)), "baostock_candidate_rows": int(len(bs5))}
    merged = merged.sort_values("datetime").drop_duplicates("datetime", keep="last")
    cols = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    extra = [c for c in merged.columns if c not in cols]
    merged = merged[cols + extra]
    # count added datetimes that were not in existing
    existing_ts = set(existing["datetime"].astype("int64")) if not existing.empty else set()
    added = [x for x in bs5["datetime"].astype("int64") if x not in existing_ts]
    info = {
        "added_rows": int(len(added)),
        "existing_rows": int(len(existing)),
        "baostock_candidate_rows": int(len(bs5)),
        "merged_rows": int(len(merged)),
        "first_existing": str(existing["datetime"].min()) if not existing.empty else None,
        "first_merged": str(merged["datetime"].min()) if not merged.empty else None,
        "last_merged": str(merged["datetime"].max()) if not merged.empty else None,
    }
    return merged, info


def segment_ret(g: pd.DataFrame, start: str, end: str) -> float:
    part = g[(g["time_str"] >= start) & (g["time_str"] <= end)].sort_values("datetime")
    if part.empty:
        return np.nan
    first_open = pd.to_numeric(part["open"], errors="coerce").iloc[0]
    last_close = pd.to_numeric(part["close"], errors="coerce").iloc[-1]
    if not np.isfinite(first_open) or abs(first_open) < 1e-12:
        return np.nan
    return float(last_close / first_open - 1.0)


def segment_vwap(g: pd.DataFrame, start: str, end: str) -> float:
    part = g[(g["time_str"] >= start) & (g["time_str"] <= end)].copy()
    if part.empty or "amount" not in part.columns or "volume" not in part.columns:
        return np.nan
    vol = pd.to_numeric(part["volume"], errors="coerce").sum()
    amt = pd.to_numeric(part["amount"], errors="coerce").sum()
    return float(amt / vol) if vol > 0 else np.nan


def build_intraday_feature_row(bars: pd.DataFrame, date: str) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    g = bars.copy()
    g["datetime"] = pd.to_datetime(g["datetime"], errors="coerce")
    g = g.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime")
    g["date"] = g["datetime"].dt.normalize()
    g["time_str"] = g["datetime"].dt.strftime("%H:%M:%S")
    day_ts = pd.to_datetime(date_compact_to_dash(date)).normalize()
    g = g[g["date"] == day_ts]
    if g.empty:
        return pd.DataFrame()
    total_vol = max(pd.to_numeric(g.get("volume"), errors="coerce").sum(), 1e-12) if "volume" in g.columns else 1e-12
    first60 = g[(g["time_str"] >= "09:35:00") & (g["time_str"] <= "10:30:00")]
    last30 = g[(g["time_str"] >= "14:35:00") & (g["time_str"] <= "15:00:00")]
    row = {
        "date": day_ts,
        "bar_count": int(len(g)),
        "first_30m_ret": segment_ret(g, "09:35:00", "10:00:00"),
        "first_60m_ret": segment_ret(g, "09:35:00", "10:30:00"),
        "morning_ret": segment_ret(g, "09:35:00", "11:30:00"),
        "afternoon_ret": segment_ret(g, "13:05:00", "15:00:00"),
        "last_30m_ret": segment_ret(g, "14:35:00", "15:00:00"),
        "last_60m_ret": segment_ret(g, "14:05:00", "15:00:00"),
        "morning_vwap": segment_vwap(g, "09:35:00", "11:30:00"),
        "afternoon_vwap": segment_vwap(g, "13:05:00", "15:00:00"),
        "last_30m_vwap": segment_vwap(g, "14:35:00", "15:00:00"),
        "first_60m_volume_share": pd.to_numeric(first60.get("volume"), errors="coerce").sum() / total_vol if "volume" in first60.columns else np.nan,
        "last_30m_volume_share": pd.to_numeric(last30.get("volume"), errors="coerce").sum() / total_vol if "volume" in last30.columns else np.nan,
    }
    out = pd.DataFrame([row])
    return out


def update_feature_cache(cache_dir: Path, symbol: str, date: str, bars: pd.DataFrame, backup_root: Path) -> dict:
    feature_dir = cache_dir / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{normalize_symbol(symbol)}_intraday_reversal_features.csv"
    existing = pd.DataFrame()
    if path.exists():
        backup_file(path, backup_root / "feature_cache")
        existing = pd.read_csv(path, parse_dates=["date"], encoding="utf-8-sig")
    new = build_intraday_feature_row(bars, date)
    if new.empty:
        return {"feature_cache": str(path), "updated": False, "reason": "no feature row"}
    out = pd.concat([existing, new], ignore_index=True, sort=False) if not existing.empty else new
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return {"feature_cache": str(path), "updated": True, "row": new.iloc[0].to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill early same-day 5m bars from BaoStock and rebuild intraday feature cache")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="Trade date YYYYMMDD")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="saved_data/akshare_realtime_cache")
    ap.add_argument("--symbols-file", default=None, help="Optional watchlist file; otherwise discover pending/date subdirs")
    ap.add_argument("--symbols", default=None, help="Optional comma-separated symbols")
    ap.add_argument("--before-time", default="10:38", help="Only use BaoStock bars earlier than this HH:MM; set empty to disable")
    ap.add_argument("--merge-all-before-first-existing", action="store_true", default=True, help="Use BaoStock rows before first existing timestamp")
    ap.add_argument("--no-feature-cache", action="store_true", help="Do not update intraday feature_cache")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    date = str(args.date).replace("-", "")
    cache_dir = Path(args.cache_dir)
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(normalize_symbol(x) for x in args.symbols.split(",") if x.strip())
    else:
        symbols = discover_symbols(cache_dir, date, Path(args.symbols_file) if args.symbols_file else None)
    symbols = list(dict.fromkeys([s for s in symbols if s]))
    if not symbols:
        raise SystemExit(f"No symbols found. Check pending dir: {cache_dir / 'pending' / date}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = cache_dir / "backup_baostock_gap_fill" / f"{date}_{ts}"
    summary = {"date": date, "cache_dir": str(cache_dir), "symbols": symbols, "results": []}

    for sym in symbols:
        sym_dir = cache_dir / "pending" / date / sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        path5 = sym_dir / "minute_bars_5min.csv"
        result = {"symbol": sym, "path5": str(path5)}
        try:
            bs5 = query_baostock_5m(sym, date)
            existing = read_existing_5m(path5)
            merged, info = merge_bars(
                existing,
                bs5,
                only_before_existing=bool(args.merge_all_before_first_existing),
                before_time=args.before_time if args.before_time else None,
            )
            result.update(info)
            result["baostock_rows_total"] = int(len(bs5))
            if not args.dry_run:
                if path5.exists():
                    backup_file(path5, backup_root / sym)
                merged.to_csv(path5, index=False, encoding="utf-8-sig")
                if not args.no_feature_cache:
                    result["feature_cache_update"] = update_feature_cache(cache_dir, sym, date, merged, backup_root / sym)
            else:
                result["dry_run"] = True
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        summary["results"].append(result)

    out = cache_dir / "pending" / date / "baostock_gap_fill_summary.json"
    if not args.dry_run:
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
