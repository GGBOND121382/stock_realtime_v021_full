#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix 5-minute bars whose volume/amount were built from cumulative snapshots.

Realtime snapshots usually provide cumulative intraday volume/amount.  A 5m bar
must contain per-bar volume/amount.  If build-bars sums snapshot cumulative
fields, segment VWAP and volume-share features are polluted.

This tool preserves BaoStock pre-start gap rows (source contains "baostock") and
converts only local-collected rows from cumulative to per-bar delta.
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
BAR_COLS = ["open", "high", "low", "close", "volume", "amount"]


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
    elif "trade_time" in out.columns:
        t = out["trade_time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
        out["datetime"] = pd.to_datetime(str(date) + t, format="%Y%m%d%H%M%S", errors="coerce").dt.floor("min")
    else:
        raise ValueError(f"cannot find datetime/trade_time in columns={list(out.columns)}")
    return out.dropna(subset=["datetime"]).copy()


def normalize_bars(df: pd.DataFrame, date: str, symbol: str) -> pd.DataFrame:
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
    for canon, cand in aliases.items():
        if canon not in out.columns:
            for c in cand:
                if c in out.columns:
                    rename[c] = canon
                    break
    if rename:
        out = out.rename(columns=rename)
    for c in BAR_COLS:
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


def filter_cutoff(df: pd.DataFrame, cutoff_time: str | None) -> pd.DataFrame:
    t = parse_hhmm(cutoff_time)
    if t is None or df.empty:
        return df
    return df[df["datetime"].dt.time <= t].copy()


def looks_cumulative(series: pd.Series) -> bool:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 4:
        return False
    diffs = x.diff().dropna()
    nonneg_ratio = float((diffs >= -1e-9).mean()) if len(diffs) else 0.0
    if nonneg_ratio >= 0.90:
        return True
    last = float(x.iloc[-1])
    total = float(x.sum())
    if last > 0 and total / last > 8:
        return True
    return False


def compute_delta_for_local_rows(df: pd.DataFrame, col: str, local_mask: pd.Series) -> tuple[pd.Series, dict]:
    values = pd.to_numeric(df[col], errors="coerce").copy()
    out = values.copy()
    idxs = list(df.index[local_mask])
    report = {
        f"{col}_local_rows": int(len(idxs)),
        f"{col}_converted": False,
        f"{col}_negative_deltas": 0,
        f"{col}_first_local_delta": np.nan,
        f"{col}_last_local_delta": np.nan,
    }
    if not idxs:
        return out, report
    local_values = values.loc[idxs]
    if not looks_cumulative(local_values):
        return out, report

    first_idx = idxs[0]
    before = df.index[df.index < first_idx]
    prev_base = pd.to_numeric(df.loc[before, col], errors="coerce").sum() if len(before) else 0.0
    prev_cum = float(prev_base) if np.isfinite(prev_base) else 0.0

    deltas = []
    neg = 0
    for idx in idxs:
        cur = values.loc[idx]
        if not np.isfinite(cur):
            delta = np.nan
        else:
            delta = float(cur) - float(prev_cum)
            if delta < -1e-6:
                neg += 1
                delta = np.nan
            prev_cum = float(cur) if np.isfinite(cur) else prev_cum
        deltas.append(delta)

    out.loc[idxs] = deltas
    report[f"{col}_converted"] = True
    report[f"{col}_negative_deltas"] = int(neg)
    if deltas:
        report[f"{col}_first_local_delta"] = float(deltas[0]) if np.isfinite(deltas[0]) else np.nan
        report[f"{col}_last_local_delta"] = float(deltas[-1]) if np.isfinite(deltas[-1]) else np.nan
    return out, report


def fix_one_bars(df: pd.DataFrame, date: str, symbol: str) -> tuple[pd.DataFrame, dict]:
    bars = normalize_bars(df, date, symbol)
    if bars.empty:
        return bars, {"status": "empty"}

    source = bars["source"].fillna("").astype(str).str.lower()
    trusted_mask = source.str.contains("baostock", na=False)
    local_mask = ~trusted_mask

    report = {
        "status": "ok",
        "rows_before": int(len(bars)),
        "trusted_baostock_rows": int(trusted_mask.sum()),
        "local_rows": int(local_mask.sum()),
        "start": str(bars["datetime"].min()),
        "end": str(bars["datetime"].max()),
    }

    fixed = bars.copy()
    for col in ["volume", "amount"]:
        fixed_col, r = compute_delta_for_local_rows(fixed, col, local_mask)
        fixed[col] = fixed_col
        report.update(r)

    fixed["source"] = np.where(
        trusted_mask,
        fixed["source"],
        fixed["source"].replace("", "local_collected_delta_fixed").fillna("local_collected_delta_fixed"),
    )
    fixed["volume_amount_delta_fixed"] = local_mask.astype(int)
    report["volume_sum_after"] = float(pd.to_numeric(fixed["volume"], errors="coerce").sum())
    report["amount_sum_after"] = float(pd.to_numeric(fixed["amount"], errors="coerce").sum())
    return fixed, report


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
    g["date"] = g["datetime"].dt.normalize()
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
    row = build_intraday_feature_row(bars, date)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        old = old[pd.to_datetime(old["date"]).dt.normalize() != pd.to_datetime(yyyymmdd_to_dash(date))]
        out = pd.concat([old, new], ignore_index=True).sort_values("date")
    else:
        out = new
    if write:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--cache-dir", default="saved_data/akshare_realtime_cache")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols")
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--cutoff-time", default=None)
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    backup_root = Path(args.backup_dir or f"backup_5m_cumulative_delta_fix/{args.date}_{datetime.now():%Y%m%d_%H%M%S}")
    backup_root.mkdir(parents=True, exist_ok=True)
    symbols = discover_symbols(cache_dir, args.date, args.symbols, args.symbols_file)

    report_dir = cache_dir / "pending" / args.date / "_5m_cumulative_delta_fix_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sym in symbols:
        sym = normalize_symbol(sym)
        if not sym:
            continue
        sym_dir = cache_dir / "pending" / args.date / sym
        bar_path = sym_dir / "minute_bars_5min.csv"
        if not bar_path.exists():
            rows.append({"symbol": sym, "status": "missing_bar_file", "bar_path": str(bar_path)})
            continue
        try:
            raw = pd.read_csv(bar_path)
            fixed, report = fix_one_bars(raw, args.date, sym)
            fixed = filter_cutoff(fixed, args.cutoff_time)
            fc_path = update_feature_cache(cache_dir, sym, args.date, fixed, backup_root, write=not args.dry_run)

            if not args.dry_run:
                backup_file(bar_path, backup_root)
                fixed.to_csv(bar_path, index=False, encoding="utf-8-sig")

            fixed.to_csv(report_dir / f"{sym}_fixed_5m_preview.csv", index=False, encoding="utf-8-sig")
            row = {
                "symbol": sym,
                "dry_run": bool(args.dry_run),
                "bar_path": str(bar_path),
                "feature_cache_path": fc_path,
                "backup_root": str(backup_root),
                **report,
            }
            rows.append(row)
            print(f"[{report.get('status')}] {sym}: local_rows={report.get('local_rows')} volume_converted={report.get('volume_converted')} amount_converted={report.get('amount_converted')}")
        except Exception as exc:
            rows.append({"symbol": sym, "status": "error", "error": f"{type(exc).__name__}: {exc}", "bar_path": str(bar_path)})
            print(f"[ERROR] {sym}: {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows)
    summary_csv = report_dir / "fix_summary.csv"
    summary_json = report_dir / "fix_summary.json"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps({
        "date": args.date,
        "cache_dir": str(cache_dir),
        "dry_run": bool(args.dry_run),
        "n_symbols": len(symbols),
        "backup_root": str(backup_root),
        "summary_csv": str(summary_csv),
        "rows": rows,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": str(summary_csv), "backup": str(backup_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
