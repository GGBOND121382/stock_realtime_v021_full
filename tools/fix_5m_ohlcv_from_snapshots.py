#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild local-collected 5m OHLCV bars from snapshots.

For realtime snapshots:
  - high/low fields are often day-to-date cumulative high/low, not 5m-bar high/low.
  - volume/amount fields are usually intraday cumulative values, not per-bar values.

Therefore 5m bars used for scoring should be:
  - open/high/low/close from sampled current/last prices within each 5m bucket
  - volume/amount from cumulative delta between bucket-end cumulative values

BaoStock pre-start gap rows are preserved. Only local-collected bars after the
first local timestamp are rebuilt from snapshot_5level.csv.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


EPS = 1e-12


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace("_", ".")
    if not s or s.startswith("#"):
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


def yyyymmdd_to_dash(value: str) -> str:
    value = str(value)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def parse_hhmm(value: str | None) -> Optional[dtime]:
    if not value:
        return None
    hh, mm = str(value).split(":", 1)
    return dtime(int(hh), int(mm))


def backup_file(path: Path, backup_root: Path) -> Optional[Path]:
    if not path.exists():
        return None
    dst = backup_root / path.as_posix().replace("/", "__")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst


def read_symbols_file(path: Path) -> list[str]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            sym = normalize_symbol(token)
            if sym:
                out.append(sym)
    return list(dict.fromkeys(out))


def discover_symbols(cache_dir: Path, date: str, symbols: str | None, symbols_file: str | None) -> list[str]:
    if symbols:
        out = []
        for x in symbols.replace(";", ",").split(","):
            sym = normalize_symbol(x)
            if sym:
                out.append(sym)
        return list(dict.fromkeys(out))
    if symbols_file:
        out = read_symbols_file(Path(symbols_file))
        if out:
            return out
    day_dir = cache_dir / "pending" / date
    if not day_dir.exists():
        raise FileNotFoundError(f"day cache dir not found: {day_dir}")
    out = []
    for p in day_dir.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        sym = normalize_symbol(p.name)
        if sym:
            out.append(sym)
    return sorted(set(out))


def ensure_datetime(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.floor("min")
    elif "time" in out.columns and "date" in out.columns:
        d = out["date"].astype(str).str.replace("-", "", regex=False)
        t = out["time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
        out["datetime"] = pd.to_datetime(d + t, format="%Y%m%d%H%M%S", errors="coerce").dt.floor("min")
    elif "trade_time" in out.columns:
        t = out["trade_time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
        out["datetime"] = pd.to_datetime(str(date) + t, format="%Y%m%d%H%M%S", errors="coerce").dt.floor("min")
    elif "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.floor("min")
    else:
        raise ValueError(f"cannot find datetime/time column in columns={list(out.columns)}")
    return out.dropna(subset=["datetime"]).copy()


def filter_cutoff(df: pd.DataFrame, cutoff_time: str | None) -> pd.DataFrame:
    t = parse_hhmm(cutoff_time)
    if t is None or df.empty:
        return df
    return df[df["datetime"].dt.time <= t].copy()


def normalize_existing_bars(df: pd.DataFrame, date: str, symbol: str) -> pd.DataFrame:
    out = ensure_datetime(df, date)
    rename = {}
    aliases = {
        "open": ["open", "开盘", "开盘价"],
        "high": ["high", "最高", "最高价"],
        "low": ["low", "最低", "最低价"],
        "close": ["close", "收盘", "收盘价"],
        "volume": ["volume", "成交量", "vol"],
        "amount": ["amount", "成交额"],
    }
    for canon, cands in aliases.items():
        if canon not in out.columns:
            for c in cands:
                if c in out.columns:
                    rename[c] = canon
                    break
    if rename:
        out = out.rename(columns=rename)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    out["symbol"] = normalize_symbol(symbol)
    if "source" not in out.columns:
        out["source"] = ""
    out["source"] = out["source"].fillna("").astype(str)
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    return out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def pick_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    lower_map = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None


def normalize_snapshots(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = ensure_datetime(df, date)
    price_col = pick_col(out, [
        "last_price", "close", "price", "最新价", "现价", "latest", "last", "current",
    ])
    vol_col = pick_col(out, ["volume", "成交量", "vol"])
    amt_col = pick_col(out, ["amount", "成交额"])
    if price_col is None:
        raise ValueError(f"snapshot has no usable current price column; columns={list(out.columns)}")
    out["last_price_for_bar"] = pd.to_numeric(out[price_col], errors="coerce")
    out["cum_volume_for_bar"] = pd.to_numeric(out[vol_col], errors="coerce") if vol_col else np.nan
    out["cum_amount_for_bar"] = pd.to_numeric(out[amt_col], errors="coerce") if amt_col else np.nan
    out = out.dropna(subset=["datetime", "last_price_for_bar"])
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    return out


def bar_bucket_floor(ts: pd.Series) -> pd.Series:
    # Keep current project convention: snapshots are assigned to the last completed 5m bucket.
    return ts.dt.floor("5min")


def rebuild_local_bars_from_snapshots(
    snapshots: pd.DataFrame,
    first_local_time: pd.Timestamp,
    prestart_bars: pd.DataFrame,
    symbol: str,
) -> tuple[pd.DataFrame, dict]:
    s = snapshots[snapshots["datetime"] >= first_local_time].copy()
    if s.empty:
        return pd.DataFrame(), {"snapshot_rows_used": 0, "rebuilt_local_bars": 0}
    s["bar_time"] = bar_bucket_floor(s["datetime"])
    s = s.sort_values("datetime")

    rows = []
    prev_cum_volume = pd.to_numeric(prestart_bars.get("volume", pd.Series(dtype=float)), errors="coerce").sum() if not prestart_bars.empty else 0.0
    prev_cum_amount = pd.to_numeric(prestart_bars.get("amount", pd.Series(dtype=float)), errors="coerce").sum() if not prestart_bars.empty else 0.0

    for bar_time, g in s.groupby("bar_time", sort=True):
        g = g.sort_values("datetime")
        px = pd.to_numeric(g["last_price_for_bar"], errors="coerce").dropna()
        if px.empty:
            continue
        last_vol = pd.to_numeric(g["cum_volume_for_bar"], errors="coerce").dropna()
        last_amt = pd.to_numeric(g["cum_amount_for_bar"], errors="coerce").dropna()
        cum_vol = float(last_vol.iloc[-1]) if len(last_vol) else np.nan
        cum_amt = float(last_amt.iloc[-1]) if len(last_amt) else np.nan
        vol_delta = cum_vol - prev_cum_volume if np.isfinite(cum_vol) and np.isfinite(prev_cum_volume) else np.nan
        amt_delta = cum_amt - prev_cum_amount if np.isfinite(cum_amt) and np.isfinite(prev_cum_amount) else np.nan
        if np.isfinite(cum_vol):
            prev_cum_volume = cum_vol
        if np.isfinite(cum_amt):
            prev_cum_amount = cum_amt
        if np.isfinite(vol_delta) and vol_delta < -1e-6:
            vol_delta = np.nan
        if np.isfinite(amt_delta) and amt_delta < -1e-6:
            amt_delta = np.nan
        rows.append({
            "datetime": bar_time,
            "symbol": normalize_symbol(symbol),
            "open": float(px.iloc[0]),
            "high": float(px.max()),
            "low": float(px.min()),
            "close": float(px.iloc[-1]),
            "volume": float(vol_delta) if np.isfinite(vol_delta) else np.nan,
            "amount": float(amt_delta) if np.isfinite(amt_delta) else np.nan,
            "source": "local_snapshot_ohlcv_fixed",
            "snapshot_count": int(len(g)),
            "first_snapshot_time": str(g["datetime"].iloc[0]),
            "last_snapshot_time": str(g["datetime"].iloc[-1]),
        })
    rebuilt = pd.DataFrame(rows)
    report = {
        "snapshot_rows_used": int(len(s)),
        "rebuilt_local_bars": int(len(rebuilt)),
        "rebuilt_start": str(rebuilt["datetime"].min()) if not rebuilt.empty else "",
        "rebuilt_end": str(rebuilt["datetime"].max()) if not rebuilt.empty else "",
    }
    return rebuilt, report


def segment_ret(g: pd.DataFrame, start: str, end: str) -> float:
    part = g[(g["time_str"] >= start) & (g["time_str"] <= end)].sort_values("datetime")
    if part.empty:
        return np.nan
    first_open = pd.to_numeric(part["open"], errors="coerce").iloc[0]
    last_close = pd.to_numeric(part["close"], errors="coerce").iloc[-1]
    if not np.isfinite(first_open) or abs(first_open) < EPS or not np.isfinite(last_close):
        return np.nan
    return float(last_close / first_open - 1.0)


def segment_vwap(g: pd.DataFrame, start: str, end: str) -> float:
    part = g[(g["time_str"] >= start) & (g["time_str"] <= end)]
    if part.empty:
        return np.nan
    vol = pd.to_numeric(part["volume"], errors="coerce").sum()
    amt = pd.to_numeric(part["amount"], errors="coerce").sum()
    return float(amt / vol) if np.isfinite(vol) and vol > 0 else np.nan


def build_intraday_feature_row(bars: pd.DataFrame, date: str) -> dict:
    g = bars.copy()
    g["time_str"] = g["datetime"].dt.strftime("%H:%M:%S")
    total_vol = pd.to_numeric(g["volume"], errors="coerce").sum()
    total_vol = max(float(total_vol), EPS)
    row = {
        "date": pd.to_datetime(yyyymmdd_to_dash(date)),
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
        "first_60m_volume_share": float(pd.to_numeric(g[(g["time_str"] >= "09:35:00") & (g["time_str"] <= "10:30:00")]["volume"], errors="coerce").sum() / total_vol),
        "last_30m_volume_share": float(pd.to_numeric(g[(g["time_str"] >= "14:35:00") & (g["time_str"] <= "15:00:00")]["volume"], errors="coerce").sum() / total_vol),
    }
    close = pd.to_numeric(g["close"], errors="coerce").dropna()
    last_close = float(close.iloc[-1]) if len(close) else np.nan
    for k in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
        v = row.get(k)
        row[f"{k}_to_close"] = float(last_close / v - 1.0) if np.isfinite(last_close) and np.isfinite(v) and abs(v) > EPS else np.nan
    row["morning_afternoon_reversal"] = -row["morning_ret"] * row["afternoon_ret"] if np.isfinite(row["morning_ret"]) and np.isfinite(row["afternoon_ret"]) else np.nan
    row["first60_last30_reversal"] = -row["first_60m_ret"] * row["last_30m_ret"] if np.isfinite(row["first_60m_ret"]) and np.isfinite(row["last_30m_ret"]) else np.nan
    return row


def update_feature_cache(cache_dir: Path, symbol: str, date: str, bars: pd.DataFrame, backup_root: Path, write: bool) -> str:
    feature_dir = cache_dir / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{normalize_symbol(symbol)}_intraday_reversal_features.csv"
    if write:
        backup_file(path, backup_root)
    new = pd.DataFrame([build_intraday_feature_row(bars, date)])
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        old = old[pd.to_datetime(old["date"]).dt.normalize() != pd.to_datetime(yyyymmdd_to_dash(date))]
        out = pd.concat([old, new], ignore_index=True).sort_values("date")
    else:
        out = new
    if write:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def rebuild_one_symbol(cache_dir: Path, date: str, symbol: str, cutoff_time: str | None, backup_root: Path, dry_run: bool) -> dict:
    sym = normalize_symbol(symbol)
    sym_dir = cache_dir / "pending" / date / sym
    bar_path = sym_dir / "minute_bars_5min.csv"
    snap_path = sym_dir / "snapshot_5level.csv"
    if not bar_path.exists():
        return {"symbol": sym, "status": "missing_bar_file", "bar_path": str(bar_path)}
    if not snap_path.exists():
        return {"symbol": sym, "status": "missing_snapshot_file", "snapshot_path": str(snap_path)}
    existing = normalize_existing_bars(pd.read_csv(bar_path), date, sym)
    existing = filter_cutoff(existing, cutoff_time)
    source = existing["source"].astype(str).str.lower()
    prestart = existing[source.str.contains("baostock", na=False)].copy()
    local_existing = existing[~source.str.contains("baostock", na=False)].copy()
    if local_existing.empty:
        return {"symbol": sym, "status": "no_local_rows_to_rebuild", "bar_path": str(bar_path)}
    first_local_time = local_existing["datetime"].min()
    snapshots = normalize_snapshots(pd.read_csv(snap_path), date)
    snapshots = filter_cutoff(snapshots, cutoff_time)
    rebuilt, rebuild_report = rebuild_local_bars_from_snapshots(snapshots, first_local_time, prestart, sym)
    if rebuilt.empty:
        return {"symbol": sym, "status": "no_rebuilt_rows", "first_local_time": str(first_local_time)}
    merged = pd.concat([
        prestart[["datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "source"]],
        rebuilt[["datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "source"]],
    ], ignore_index=True)
    merged = merged.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    merged = filter_cutoff(merged, cutoff_time)
    feature_cache_path = update_feature_cache(cache_dir, sym, date, merged, backup_root, write=not dry_run)
    if not dry_run:
        backup_file(bar_path, backup_root)
        merged.to_csv(bar_path, index=False, encoding="utf-8-sig")
    return {
        "symbol": sym,
        "status": "ok",
        "dry_run": bool(dry_run),
        "bar_path": str(bar_path),
        "snapshot_path": str(snap_path),
        "feature_cache_path": feature_cache_path,
        "backup_root": str(backup_root),
        "first_local_time": str(first_local_time),
        "prestart_rows_preserved": int(len(prestart)),
        "local_rows_before": int(len(local_existing)),
        "merged_rows_after": int(len(merged)),
        "merged_start": str(merged["datetime"].min()),
        "merged_end": str(merged["datetime"].max()),
        "volume_sum_after": float(pd.to_numeric(merged["volume"], errors="coerce").sum()),
        "amount_sum_after": float(pd.to_numeric(merged["amount"], errors="coerce").sum()),
        **rebuild_report,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--cache-dir", default="saved_data/akshare_realtime_cache")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--cutoff-time", default=None)
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    backup_root = Path(args.backup_dir or f"backup_5m_ohlc_snapshot_fix/{args.date}_{datetime.now():%Y%m%d_%H%M%S}")
    backup_root.mkdir(parents=True, exist_ok=True)
    symbols = discover_symbols(cache_dir, args.date, args.symbols, args.symbols_file)
    report_dir = cache_dir / "pending" / args.date / "_5m_ohlc_snapshot_fix_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in symbols:
        try:
            row = rebuild_one_symbol(cache_dir, args.date, sym, args.cutoff_time, backup_root, args.dry_run)
        except Exception as exc:
            row = {"symbol": normalize_symbol(sym), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(f"[{row.get('status')}] {row.get('symbol')}")
    summary = pd.DataFrame(rows)
    summary_csv = report_dir / "fix_summary.csv"
    summary_json = report_dir / "fix_summary.json"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps({
        "date": args.date,
        "dry_run": bool(args.dry_run),
        "cache_dir": str(cache_dir),
        "backup_root": str(backup_root),
        "summary_csv": str(summary_csv),
        "rows": rows,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": str(summary_csv), "backup": str(backup_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
