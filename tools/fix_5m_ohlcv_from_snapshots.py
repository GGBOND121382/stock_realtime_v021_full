#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild realtime 5m OHLCV bars from snapshots using BaoStock-compatible right endpoints."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace("_", ".")
    if "." in s:
        a, b = s.split(".", 1)
        if a in {"SH", "SZ"}:
            m, code = a, b
        else:
            code, m = a, b
        return f"{code.zfill(6)}.{m}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    m = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{m}"


def parse_hhmm(v: str | None) -> Optional[dtime]:
    if not v:
        return None
    h, m = str(v).split(":", 1)
    return dtime(int(h), int(m))


def backup_file(path: Path, backup_root: Path) -> Optional[str]:
    if not path.exists():
        return None
    dst = backup_root / path.as_posix().replace("/", "__")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return str(dst)


def read_symbols_file(path: Path) -> list[str]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.append(normalize_symbol(token))
    return list(dict.fromkeys(out))


def discover_symbols(cache_dir: Path, date: str, symbols: str | None, symbols_file: str | None) -> list[str]:
    if symbols:
        return list(dict.fromkeys(normalize_symbol(x) for x in symbols.replace(";", ",").split(",") if x.strip()))
    if symbols_file:
        got = read_symbols_file(Path(symbols_file))
        if got:
            return got
    day = cache_dir / "pending" / date
    if not day.exists():
        raise FileNotFoundError(day)
    return sorted(normalize_symbol(p.name) for p in day.iterdir() if p.is_dir() and not p.name.startswith("_"))


def ensure_dt(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif {"trade_date", "trade_time"}.issubset(out.columns):
        out["datetime"] = pd.to_datetime(
            out["trade_date"].astype(str) + out["trade_time"].astype(str).str.zfill(6),
            errors="coerce",
        )
    elif "trade_time" in out.columns:
        out["datetime"] = pd.to_datetime(
            str(date) + out["trade_time"].astype(str).str.replace(":", "", regex=False).str.zfill(6),
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
    else:
        raise ValueError(f"cannot infer datetime; columns={list(out.columns)}")
    return out.dropna(subset=["datetime"]).sort_values("datetime")


def filter_cutoff(df: pd.DataFrame, cutoff: str | None) -> pd.DataFrame:
    t = parse_hhmm(cutoff)
    if t is None or df.empty:
        return df
    return df[df["datetime"].dt.time <= t].copy()


def pick_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    cols = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in cols:
            return cols[n.lower()]
    return None


def normalize_snapshots(path: Path, date: str, cutoff: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = ensure_dt(df, date)
    df = filter_cutoff(df, cutoff)
    price = pick_col(df, ["last_price", "close", "price", "最新价", "现价"])
    vol = pick_col(df, ["volume", "成交量", "vol"])
    amt = pick_col(df, ["amount", "成交额"])
    if price is None:
        raise ValueError(f"no price column in {path}")
    out = pd.DataFrame({
        "datetime": df["datetime"],
        "price": pd.to_numeric(df[price], errors="coerce"),
        "cum_volume": pd.to_numeric(df[vol], errors="coerce") if vol else np.nan,
        "cum_amount": pd.to_numeric(df[amt], errors="coerce") if amt else np.nan,
    }).dropna(subset=["datetime", "price"])
    ratio = (
        pd.to_numeric(out["cum_amount"], errors="coerce")
        / pd.to_numeric(out["cum_volume"], errors="coerce").replace(0, np.nan)
        / pd.to_numeric(out["price"], errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if not ratio.empty and 50.0 <= float(ratio.median()) <= 150.0:
        out["cum_volume"] = pd.to_numeric(out["cum_volume"], errors="coerce") * 100.0
    return out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def bar_bucket_right(ts: pd.Series) -> pd.Series:
    floored = ts.dt.floor("5min")
    exact = ts.eq(floored)
    right = floored.where(exact, floored + pd.Timedelta(minutes=5))
    date_part = right.dt.normalize()
    times = right.dt.strftime("%H:%M:%S")
    right = right.mask(times < "09:35:00", date_part + pd.Timedelta(hours=9, minutes=35))
    right = right.mask((times >= "11:35:00") & (times < "13:05:00"), date_part + pd.Timedelta(hours=13, minutes=5))
    return right


def trading_endpoint_mask(dt: pd.Series, cutoff: str | None) -> pd.Series:
    t = dt.dt.strftime("%H:%M:%S")
    mask = ((t >= "09:35:00") & (t <= "11:30:00")) | ((t >= "13:05:00") & (t <= "15:00:00"))
    if cutoff:
        hh, mm = cutoff.split(":", 1)
        cutoff_s = f"{int(hh):02d}:{int(mm):02d}:00"
        mask &= t <= cutoff_s
    return mask


def rebuild_bars(snap: pd.DataFrame, symbol: str, cutoff: str | None) -> pd.DataFrame:
    s = snap.copy()
    s["bar_time"] = bar_bucket_right(s["datetime"])
    s = s[trading_endpoint_mask(s["bar_time"], cutoff)].copy()
    rows = []
    prev_vol = np.nan
    prev_amt = np.nan
    for bt, g in s.groupby("bar_time", sort=True):
        g = g.sort_values("datetime")
        px = pd.to_numeric(g["price"], errors="coerce").dropna()
        if px.empty:
            continue
        cv = pd.to_numeric(g["cum_volume"], errors="coerce").dropna()
        ca = pd.to_numeric(g["cum_amount"], errors="coerce").dropna()
        cur_vol = float(cv.iloc[-1]) if len(cv) else np.nan
        cur_amt = float(ca.iloc[-1]) if len(ca) else np.nan
        vol_delta = cur_vol - prev_vol if np.isfinite(cur_vol) and np.isfinite(prev_vol) else np.nan
        amt_delta = cur_amt - prev_amt if np.isfinite(cur_amt) and np.isfinite(prev_amt) else np.nan
        if np.isfinite(cur_vol):
            prev_vol = cur_vol
        if np.isfinite(cur_amt):
            prev_amt = cur_amt
        if np.isfinite(vol_delta) and vol_delta < 0:
            vol_delta = np.nan
        if np.isfinite(amt_delta) and amt_delta < 0:
            amt_delta = np.nan
        rows.append({
            "datetime": bt,
            "symbol": normalize_symbol(symbol),
            "open": float(px.iloc[0]),
            "high": float(px.max()),
            "low": float(px.min()),
            "close": float(px.iloc[-1]),
            "volume": float(vol_delta) if np.isfinite(vol_delta) else np.nan,
            "amount": float(amt_delta) if np.isfinite(amt_delta) else np.nan,
            "source": "local_snapshot_5m_right_endpoint",
            "snapshot_count": int(len(g)),
            "first_snapshot_time": str(g["datetime"].iloc[0]),
            "last_snapshot_time": str(g["datetime"].iloc[-1]),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("datetime").reset_index(drop=True) if not out.empty else out


def rebuild_one(cache_dir: Path, date: str, symbol: str, cutoff: str | None, backup_root: Path, dry_run: bool) -> dict:
    sym = normalize_symbol(symbol)
    d = cache_dir / "pending" / date / sym
    snap_path = d / "snapshot_5level.csv"
    bar_path = d / "minute_bars_5min.csv"
    if not snap_path.exists():
        return {"symbol": sym, "status": "missing_snapshot", "snapshot_path": str(snap_path)}
    snap = normalize_snapshots(snap_path, date, cutoff)
    bars = rebuild_bars(snap, sym, cutoff)
    if bars.empty:
        return {"symbol": sym, "status": "no_bars_rebuilt"}
    backup = None
    if not dry_run:
        backup = backup_file(bar_path, backup_root)
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(bar_path, index=False, encoding="utf-8-sig")
    times = bars["datetime"].dt.strftime("%H:%M:%S").tolist()
    invalid = [x for x in times if x in {"09:15:00", "09:20:00", "09:25:00", "09:30:00", "13:00:00"}]
    return {
        "symbol": sym,
        "status": "ok",
        "dry_run": dry_run,
        "bar_path": str(bar_path),
        "backup": backup,
        "rows": int(len(bars)),
        "first_time": times[0] if times else "",
        "last_time": times[-1] if times else "",
        "invalid_times": ",".join(invalid),
        "volume_sum": float(pd.to_numeric(bars.get("volume"), errors="coerce").sum()),
        "amount_sum": float(pd.to_numeric(bars.get("amount"), errors="coerce").sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--cache-dir", default="saved_data/akshare_realtime_cache")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--cutoff-time", default="14:55")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    backup_root = Path(args.backup_dir or f"saved_data/patch_backups/5m_right_endpoint_{args.date}_{datetime.now():%Y%m%d_%H%M%S}")
    backup_root.mkdir(parents=True, exist_ok=True)
    symbols = discover_symbols(cache_dir, args.date, args.symbols, args.symbols_file)
    rows = []
    for sym in symbols:
        try:
            row = rebuild_one(cache_dir, args.date, sym, args.cutoff_time, backup_root, args.dry_run)
        except Exception as exc:
            row = {"symbol": normalize_symbol(sym), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(f"[{row.get('status')}] {row.get('symbol')}")
    report_dir = cache_dir / "pending" / args.date / "_5m_right_endpoint_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "fix_summary.csv", index=False, encoding="utf-8-sig")
    (report_dir / "fix_summary.json").write_text(json.dumps({
        "date": args.date,
        "dry_run": args.dry_run,
        "backup_root": str(backup_root),
        "rows": rows,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": str(report_dir / "fix_summary.csv"), "backup": str(backup_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
