#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill ONLY the pre-start intraday 5m gap with BaoStock, preserving local collected data after start.

This is NOT a full BaoStock overwrite.

Goal:
  If realtime collection started late (e.g. 10:38), use BaoStock only for bars
  strictly before the first locally collected 5m timestamp. Keep all local bars
  from the first local timestamp onward so collection-code issues remain visible.

Outputs:
  - Updates saved_data/akshare_realtime_cache/pending/<DATE>/<SYMBOL>/minute_bars_5min.csv
    by prepending BaoStock gap bars only.
  - Rebuilds the DATE row in feature_cache/<SYMBOL>_intraday_reversal_features.csv
    from the mixed bars, so first_30m/first_60m features are no longer missing.
  - Writes diagnostics comparing preserved local post-start bars against BaoStock.
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
FEATURE_COLS = [
    "bar_count",
    "first_30m_ret",
    "first_60m_ret",
    "morning_ret",
    "afternoon_ret",
    "last_30m_ret",
    "last_60m_ret",
    "morning_vwap",
    "afternoon_vwap",
    "last_30m_vwap",
    "first_60m_volume_share",
    "last_30m_volume_share",
    "morning_vwap_to_close",
    "afternoon_vwap_to_close",
    "last_30m_vwap_to_close",
    "morning_afternoon_reversal",
    "first60_last30_reversal",
]


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
    return f"{market.lower()}.{code}"


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


def discover_symbols(cache_dir: Path, date: str, explicit: str | None) -> list[str]:
    if explicit:
        out = []
        for x in explicit.replace(";", ",").split(","):
            sym = normalize_symbol(x)
            if sym:
                out.append(sym)
        return list(dict.fromkeys(out))
    day_dir = cache_dir / "pending" / date
    if not day_dir.exists():
        raise FileNotFoundError(f"day cache dir not found: {day_dir}")
    return sorted(normalize_symbol(p.name) for p in day_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))


def parse_baostock_datetime(df: pd.DataFrame) -> pd.Series:
    vals = []
    for date_v, time_v in zip(df["date"].astype(str), df["time"].astype(str)):
        t = str(time_v)
        d = str(date_v).replace("-", "")
        if len(t) >= 14 and t[:14].isdigit():
            raw = t[:14]
        elif len(t) >= 6:
            raw = d[:8] + t[:6]
        else:
            raw = d[:8] + t.zfill(6)
        vals.append(raw)
    return pd.Series(pd.to_datetime(vals, format="%Y%m%d%H%M%S", errors="coerce")).dt.floor("min")


def query_baostock_5m_logged_in(bs, symbol: str, date: str, adjustflag: str = "3") -> pd.DataFrame:
    bs_code = baostock_code(symbol)
    date_dash = yyyymmdd_to_dash(date)
    fields = "date,time,code,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(
        bs_code,
        fields,
        start_date=date_dash,
        end_date=date_dash,
        frequency="5",
        adjustflag=str(adjustflag),
    )
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock query failed for {bs_code}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
    df["datetime"] = parse_baostock_datetime(df)
    out = df[["datetime", "open", "high", "low", "close", "volume", "amount"]].copy()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    return out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def normalize_local_bars(df: pd.DataFrame, date: str, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns:
        if "trade_time" in out.columns:
            t = out["trade_time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
            out["datetime"] = pd.to_datetime(str(date) + t, format="%Y%m%d%H%M%S", errors="coerce")
        else:
            raise ValueError(f"local bars have no datetime/trade_time columns: {list(out.columns)}")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.floor("min")
    rename = {}
    for canon, aliases in {
        "open": ["open", "开盘", "开盘价"],
        "high": ["high", "最高", "最高价"],
        "low": ["low", "最低", "最低价"],
        "close": ["close", "收盘", "收盘价"],
        "volume": ["volume", "成交量", "vol"],
        "amount": ["amount", "成交额"],
    }.items():
        if canon not in out.columns:
            for a in aliases:
                if a in out.columns:
                    rename[a] = canon
                    break
    if rename:
        out = out.rename(columns=rename)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    if "symbol" not in out.columns:
        out["symbol"] = normalize_symbol(symbol)
    if "source" not in out.columns:
        out["source"] = "local_collected"
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    return out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def filter_cutoff(df: pd.DataFrame, cutoff_time: str | None) -> pd.DataFrame:
    t = parse_hhmm(cutoff_time)
    if t is None or df.empty:
        return df
    return df[df["datetime"].dt.time <= t].copy()


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
    # Derived *_to_close ratios
    close = pd.to_numeric(g["close"], errors="coerce").dropna()
    last_close = float(close.iloc[-1]) if len(close) else np.nan
    for k in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
        v = row.get(k)
        row[f"{k}_to_close"] = float(last_close / v - 1.0) if np.isfinite(last_close) and np.isfinite(v) and abs(v) > EPS else np.nan
    row["morning_afternoon_reversal"] = -row["morning_ret"] * row["afternoon_ret"] if np.isfinite(row["morning_ret"]) and np.isfinite(row["afternoon_ret"]) else np.nan
    row["first60_last30_reversal"] = -row["first_60m_ret"] * row["last_30m_ret"] if np.isfinite(row["first_60m_ret"]) and np.isfinite(row["last_30m_ret"]) else np.nan
    return row


def update_feature_cache(cache_dir: Path, symbol: str, date: str, bars: pd.DataFrame, backup_root: Path) -> Path:
    feature_dir = cache_dir / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{normalize_symbol(symbol)}_intraday_reversal_features.csv"
    backup_file(path, backup_root)
    row = build_intraday_feature_row(bars, date)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        old = old[pd.to_datetime(old["date"]).dt.normalize() != pd.to_datetime(yyyymmdd_to_dash(date))]
        out = pd.concat([old, new], ignore_index=True).sort_values("date")
    else:
        out = new
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def post_start_compare(local: pd.DataFrame, bao: pd.DataFrame, first_local: pd.Timestamp) -> dict:
    lp = local[local["datetime"] >= first_local].copy()
    bp = bao[bao["datetime"] >= first_local].copy()
    m = lp.merge(bp, on="datetime", how="inner", suffixes=("_local", "_bao"))
    row = {
        "post_start_local_bars": int(len(lp)),
        "post_start_baostock_bars": int(len(bp)),
        "post_start_aligned": int(len(m)),
    }
    if not m.empty:
        for c in ["open", "high", "low", "close"]:
            diff_bps = (pd.to_numeric(m[f"{c}_local"], errors="coerce") / pd.to_numeric(m[f"{c}_bao"], errors="coerce") - 1.0) * 10000.0
            row[f"post_start_{c}_max_abs_diff_bps"] = float(diff_bps.abs().max())
            row[f"post_start_{c}_mean_diff_bps"] = float(diff_bps.mean())
        for c in ["volume", "amount"]:
            lv = pd.to_numeric(m[f"{c}_local"], errors="coerce").sum()
            bv = pd.to_numeric(m[f"{c}_bao"], errors="coerce").sum()
            row[f"post_start_{c}_rel_diff"] = float((lv - bv) / bv) if bv else np.nan
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols; default auto-discover")
    ap.add_argument("--cache-dir", default="saved_data/akshare_realtime_cache")
    ap.add_argument("--cutoff-time", default="14:55")
    ap.add_argument("--adjustflag", default="3")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--min-gap-bars", type=int, default=1)
    ap.add_argument("--write-in-place", action="store_true", help="Actually update minute_bars_5min.csv and feature_cache. Without this, dry-run only.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    backup_root = Path(args.backup_dir or f"backup_baostock_prestart_gap_only/{args.date}_{datetime.now():%Y%m%d_%H%M%S}")
    backup_root.mkdir(parents=True, exist_ok=True)
    report_dir = cache_dir / "pending" / args.date / "_baostock_prestart_gap_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        import baostock as bs
    except Exception as exc:
        raise SystemExit("[ERROR] baostock not installed. Run: python3 -m pip install baostock") from exc

    symbols = discover_symbols(cache_dir, args.date, args.symbols)
    print(f"[INFO] date={args.date} symbols={len(symbols)} write_in_place={args.write_in_place}")

    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        raise SystemExit(f"[ERROR] BaoStock login failed: {lg.error_code} {lg.error_msg}")

    summary = []
    try:
        for sym in symbols:
            sym = normalize_symbol(sym)
            sym_dir = cache_dir / "pending" / args.date / sym
            bar_path = sym_dir / "minute_bars_5min.csv"
            if not bar_path.exists():
                summary.append({"symbol": sym, "status": "missing_local_bar_file"})
                continue

            local_raw = pd.read_csv(bar_path)
            local = normalize_local_bars(local_raw, args.date, sym)
            local = filter_cutoff(local, args.cutoff_time)
            if local.empty:
                summary.append({"symbol": sym, "status": "empty_local_bars"})
                continue

            bao = query_baostock_5m_logged_in(bs, sym, args.date, adjustflag=args.adjustflag)
            bao = filter_cutoff(bao, args.cutoff_time)
            if bao.empty:
                summary.append({"symbol": sym, "status": "empty_baostock"})
                continue

            first_local = local["datetime"].min()
            gap = bao[bao["datetime"] < first_local].copy()
            gap["symbol"] = sym
            gap["source"] = "baostock_prestart_gap_only"

            local_keep = local.copy()
            local_keep["symbol"] = sym
            if "source" not in local_keep.columns:
                local_keep["source"] = "local_collected_preserved"
            else:
                local_keep["source"] = local_keep["source"].fillna("local_collected_preserved")

            merged = pd.concat([gap[["datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "source"]], 
                                local_keep[["datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "source"]]],
                               ignore_index=True)
            # If overlap somehow exists, preserve local for duplicate timestamps.
            merged["_is_local"] = merged["source"].ne("baostock_prestart_gap_only").astype(int)
            merged = merged.sort_values(["datetime", "_is_local"]).drop_duplicates("datetime", keep="last")
            merged = merged.drop(columns=["_is_local"]).sort_values("datetime").reset_index(drop=True)

            diag = post_start_compare(local, bao, first_local)

            status = "ok_gap_added" if len(gap) >= args.min_gap_bars else "no_gap_needed_or_gap_empty"
            fc_path = ""
            if args.write_in_place:
                backup_file(bar_path, backup_root)
                merged.to_csv(bar_path, index=False, encoding="utf-8-sig")
                fc_path = str(update_feature_cache(cache_dir, sym, args.date, merged, backup_root))

            # Save per-symbol diagnostic copies.
            gap.to_csv(report_dir / f"{sym}_baostock_gap_only_rows.csv", index=False, encoding="utf-8-sig")
            local.to_csv(report_dir / f"{sym}_local_preserved_after_start.csv", index=False, encoding="utf-8-sig")
            merged.to_csv(report_dir / f"{sym}_merged_gap_only_preview.csv", index=False, encoding="utf-8-sig")

            feat = build_intraday_feature_row(merged, args.date)
            row = {
                "symbol": sym,
                "status": status,
                "write_in_place": bool(args.write_in_place),
                "first_local_time": str(first_local),
                "local_bars_before": int(len(local)),
                "baostock_bars": int(len(bao)),
                "gap_baostock_rows_added": int(len(gap)),
                "merged_bars_after": int(len(merged)),
                "merged_start": str(merged["datetime"].min()),
                "merged_end": str(merged["datetime"].max()),
                "bar_path": str(bar_path),
                "feature_cache_path": fc_path,
                "backup_root": str(backup_root),
                **{f"feature_{k}": feat.get(k) for k in FEATURE_COLS},
                **diag,
            }
            summary.append(row)
            print(f"[{status}] {sym}: first_local={first_local}, gap_added={len(gap)}, merged={len(merged)}")
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    summary_df = pd.DataFrame(summary)
    summary_csv = report_dir / "prestart_gap_only_summary.csv"
    summary_json = report_dir / "prestart_gap_only_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps({
            "date": args.date,
            "cutoff_time": args.cutoff_time,
            "write_in_place": bool(args.write_in_place),
            "n_symbols": len(symbols),
            "backup_root": str(backup_root),
            "summary_csv": str(summary_csv),
            "rows": summary,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_csv), "backup": str(backup_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
