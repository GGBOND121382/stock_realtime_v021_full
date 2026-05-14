#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare today's local realtime collected bars with BaoStock 5-minute bars.

Inputs:
  saved_data/akshare_realtime_cache/pending/<DATE>/<SYMBOL>/minute_bars_5min.csv
  saved_data/akshare_realtime_cache/pending/<DATE>/<SYMBOL>/daily_features.csv

Outputs:
  saved_data/baostock_compare/<DATE>/comparison_summary.csv
  saved_data/baostock_compare/<DATE>/comparison_summary.json
  saved_data/baostock_compare/<DATE>/<SYMBOL>_aligned_diff.csv
  saved_data/baostock_compare/<DATE>/<SYMBOL>_baostock_5m.csv
  saved_data/baostock_compare/<DATE>/<SYMBOL>_missing_times.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS = ["volume", "amount"]
ALL_COMPARE_COLS = PRICE_COLS + VOL_COLS


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


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_datetime_col(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif {"date", "time"}.issubset(out.columns):
        # BaoStock time sometimes looks like 20260513093500000.
        date_s = out["date"].astype(str).str.replace("-", "", regex=False)
        time_s = out["time"].astype(str).str.replace(":", "", regex=False)
        dt_vals = []
        for d, t in zip(date_s, time_s):
            t = str(t)
            if len(t) >= 14 and t[:8].isdigit():
                raw = t[:14]
            elif len(t) >= 6:
                raw = str(d)[:8] + t[:6]
            else:
                raw = str(d)[:8] + t.zfill(6)
            dt_vals.append(raw)
        out["datetime"] = pd.to_datetime(dt_vals, format="%Y%m%d%H%M%S", errors="coerce")
    elif "trade_time" in out.columns:
        t = out["trade_time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
        out["datetime"] = pd.to_datetime(str(date) + t, format="%Y%m%d%H%M%S", errors="coerce")
    else:
        raise ValueError(f"cannot infer datetime column; columns={list(out.columns)}")

    out = out.dropna(subset=["datetime"]).copy()
    # Normalize to minute resolution.
    out["datetime"] = out["datetime"].dt.floor("min")
    return out


def normalize_bar_df(df: pd.DataFrame, date: str, source: str) -> pd.DataFrame:
    out = ensure_datetime_col(df, date)
    rename_map = {}
    aliases = {
        "open": ["open", "开盘", "开盘价"],
        "high": ["high", "最高", "最高价"],
        "low": ["low", "最低", "最低价"],
        "close": ["close", "收盘", "收盘价"],
        "volume": ["volume", "成交量", "vol"],
        "amount": ["amount", "成交额"],
    }
    for canon, cands in aliases.items():
        if canon in out.columns:
            continue
        for c in cands:
            if c in out.columns:
                rename_map[c] = canon
                break
    if rename_map:
        out = out.rename(columns=rename_map)

    keep = ["datetime"] + [c for c in ALL_COMPARE_COLS if c in out.columns]
    out = out[keep].copy()
    for c in ALL_COMPARE_COLS:
        if c in out.columns:
            out[c] = safe_num(out[c])
        else:
            out[c] = np.nan
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    out["source"] = source
    return out


def filter_cutoff(df: pd.DataFrame, cutoff: Optional[str]) -> pd.DataFrame:
    if not cutoff:
        return df
    t = parse_hhmm(cutoff)
    if t is None:
        return df
    return df[df["datetime"].dt.time <= t].copy()


def query_baostock_5m(symbol: str, date: str, adjustflag: str = "3") -> pd.DataFrame:
    try:
        import baostock as bs
    except Exception as exc:
        raise RuntimeError(
            "baostock is not installed. Install it in your project environment: "
            "python3 -m pip install baostock"
        ) from exc

    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")

    try:
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
            return pd.DataFrame(columns=["datetime"] + ALL_COMPARE_COLS)
        return normalize_bar_df(df, date, "baostock")
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def read_local_collected(cache_dir: Path, date: str, symbol: str) -> tuple[pd.DataFrame, Optional[pd.Series], Path]:
    sym = normalize_symbol(symbol)
    sym_dir = cache_dir / "pending" / date / sym
    bar_path = sym_dir / "minute_bars_5min.csv"
    daily_path = sym_dir / "daily_features.csv"

    if not bar_path.exists():
        return pd.DataFrame(columns=["datetime"] + ALL_COMPARE_COLS), None, sym_dir

    bars = pd.read_csv(bar_path)
    bars = normalize_bar_df(bars, date, "collected")

    daily_row = None
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        if not daily.empty:
            daily_row = daily.iloc[-1]
    return bars, daily_row, sym_dir


def discover_symbols(cache_dir: Path, date: str, explicit: str | None) -> list[str]:
    if explicit:
        out = []
        for x in explicit.replace(";", ",").split(","):
            x = normalize_symbol(x)
            if x:
                out.append(x)
        return list(dict.fromkeys(out))
    day_dir = cache_dir / "pending" / date
    if not day_dir.exists():
        raise FileNotFoundError(f"day cache dir not found: {day_dir}")
    syms = [p.name for p in day_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    return sorted(normalize_symbol(s) for s in syms)


def rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.maximum(np.abs(b), 1e-12)
    return (a - b) / den


def price_bps(a, b):
    return rel_diff(a, b) * 10000.0


def calc_summary(symbol: str, collected: pd.DataFrame, bao: pd.DataFrame, daily_row: Optional[pd.Series], cutoff: Optional[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    c = filter_cutoff(collected, cutoff)
    b = filter_cutoff(bao, cutoff)

    merged = c.merge(b, on="datetime", how="outer", suffixes=("_collected", "_baostock"), indicator=True)
    exact = merged[merged["_merge"] == "both"].copy()

    for col in ALL_COMPARE_COLS:
        lc = f"{col}_collected"
        rb = f"{col}_baostock"
        if lc in exact.columns and rb in exact.columns:
            exact[f"{col}_abs_diff"] = exact[lc] - exact[rb]
            exact[f"{col}_rel_diff"] = rel_diff(exact[lc], exact[rb])
            if col in PRICE_COLS:
                exact[f"{col}_diff_bps"] = price_bps(exact[lc], exact[rb])

    missing_times = merged.loc[merged["_merge"] != "both", ["datetime", "_merge"]].copy()
    missing_times["_merge"] = missing_times["_merge"].map({
        "left_only": "only_in_collected",
        "right_only": "only_in_baostock",
        "both": "both",
    })

    row: dict = {
        "symbol": symbol,
        "cutoff": cutoff or "",
        "collected_bars": int(len(c)),
        "baostock_bars": int(len(b)),
        "aligned_bars": int(len(exact)),
        "only_in_collected": int((merged["_merge"] == "left_only").sum()),
        "only_in_baostock": int((merged["_merge"] == "right_only").sum()),
        "collected_start": str(c["datetime"].min()) if not c.empty else "",
        "collected_end": str(c["datetime"].max()) if not c.empty else "",
        "baostock_start": str(b["datetime"].min()) if not b.empty else "",
        "baostock_end": str(b["datetime"].max()) if not b.empty else "",
    }

    for col in PRICE_COLS:
        diff_col = f"{col}_diff_bps"
        if diff_col in exact.columns and not exact[diff_col].dropna().empty:
            s = exact[diff_col].dropna()
            row[f"{col}_mean_diff_bps"] = float(s.mean())
            row[f"{col}_max_abs_diff_bps"] = float(s.abs().max())
            row[f"{col}_p95_abs_diff_bps"] = float(s.abs().quantile(0.95))
        else:
            row[f"{col}_mean_diff_bps"] = np.nan
            row[f"{col}_max_abs_diff_bps"] = np.nan
            row[f"{col}_p95_abs_diff_bps"] = np.nan

    for col in VOL_COLS:
        lc = f"{col}_collected"
        rb = f"{col}_baostock"
        if lc in exact.columns and rb in exact.columns:
            row[f"{col}_sum_collected_aligned"] = float(pd.to_numeric(exact[lc], errors="coerce").sum())
            row[f"{col}_sum_baostock_aligned"] = float(pd.to_numeric(exact[rb], errors="coerce").sum())
            den = row[f"{col}_sum_baostock_aligned"]
            row[f"{col}_sum_rel_diff"] = float((row[f"{col}_sum_collected_aligned"] - den) / den) if abs(den) > 1e-12 else np.nan

    # Compare daily_features cumulative amount/volume/close with BaoStock 5m aggregate before cutoff.
    if daily_row is not None:
        for col in ["close", "volume", "amount", "daily_vwap"]:
            row[f"daily_{col}"] = float(pd.to_numeric(pd.Series([daily_row.get(col)]), errors="coerce").iloc[0]) if col in daily_row.index else np.nan
        if not b.empty:
            row["baostock_last_close_before_cutoff"] = float(b["close"].dropna().iloc[-1]) if b["close"].dropna().size else np.nan
            row["baostock_sum_volume_before_cutoff"] = float(b["volume"].sum(skipna=True))
            row["baostock_sum_amount_before_cutoff"] = float(b["amount"].sum(skipna=True))
            if row["baostock_sum_volume_before_cutoff"] and np.isfinite(row["baostock_sum_volume_before_cutoff"]):
                row["baostock_vwap_before_cutoff"] = row["baostock_sum_amount_before_cutoff"] / row["baostock_sum_volume_before_cutoff"]
            for col in ["close"]:
                dv = row.get(f"daily_{col}", np.nan)
                bv = row.get(f"baostock_last_{col}_before_cutoff", np.nan)
                row[f"daily_vs_baostock_{col}_diff_bps"] = float((dv / bv - 1) * 10000) if np.isfinite(dv) and np.isfinite(bv) and bv != 0 else np.nan
            for col in ["volume", "amount"]:
                dv = row.get(f"daily_{col}", np.nan)
                bv = row.get(f"baostock_sum_{col}_before_cutoff", np.nan)
                row[f"daily_vs_baostock_{col}_rel_diff"] = float((dv - bv) / bv) if np.isfinite(dv) and np.isfinite(bv) and bv != 0 else np.nan
            dvwap = row.get("daily_daily_vwap", np.nan)
            bvwap = row.get("baostock_vwap_before_cutoff", np.nan)
            row["daily_vs_baostock_vwap_diff_bps"] = float((dvwap / bvwap - 1) * 10000) if np.isfinite(dvwap) and np.isfinite(bvwap) and bvwap != 0 else np.nan

    # Simple severity label.
    max_close_bps = row.get("close_max_abs_diff_bps", np.nan)
    missing_bao = row.get("only_in_baostock", 0)
    if missing_bao > 0:
        row["severity"] = "missing_collected_bars"
    elif np.isfinite(max_close_bps) and max_close_bps > 10:
        row["severity"] = "large_price_diff"
    else:
        row["severity"] = "ok"

    return row, exact.sort_values("datetime"), missing_times.sort_values("datetime")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD, e.g. 20260513")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols; default: auto-discover under cache")
    ap.add_argument("--cache-dir", default="saved_data/akshare_realtime_cache")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cutoff-time", default="14:55")
    ap.add_argument("--adjustflag", default="3", help="BaoStock adjustflag: 3=none")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir or f"saved_data/baostock_compare/{args.date}")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = discover_symbols(cache_dir, args.date, args.symbols)
    print(f"[INFO] date={args.date} symbols={len(symbols)} out={out_dir}")

    rows = []
    errors = []
    for sym in symbols:
        print(f"[COMPARE] {sym}", flush=True)
        try:
            collected, daily_row, sym_dir = read_local_collected(cache_dir, args.date, sym)
            bao = query_baostock_5m(sym, args.date, adjustflag=args.adjustflag)

            bao.to_csv(out_dir / f"{sym}_baostock_5m.csv", index=False, encoding="utf-8-sig")
            collected.to_csv(out_dir / f"{sym}_collected_5m_normalized.csv", index=False, encoding="utf-8-sig")

            row, aligned, missing = calc_summary(sym, collected, bao, daily_row, args.cutoff_time)
            aligned.to_csv(out_dir / f"{sym}_aligned_diff.csv", index=False, encoding="utf-8-sig")
            missing.to_csv(out_dir / f"{sym}_missing_times.csv", index=False, encoding="utf-8-sig")
            rows.append(row)
        except Exception as exc:
            err = {"symbol": sym, "error": f"{type(exc).__name__}: {exc}"}
            print("[ERROR]", err, file=sys.stderr)
            errors.append(err)
            if args.fail_fast:
                raise

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["severity", "symbol"])
    summary.to_csv(out_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "date": args.date,
        "cutoff_time": args.cutoff_time,
        "symbols": symbols,
        "n_symbols": len(symbols),
        "n_ok": int(len(rows)),
        "n_error": int(len(errors)),
        "errors": errors,
        "summary_rows": rows,
        "outputs": {
            "comparison_summary_csv": str(out_dir / "comparison_summary.csv"),
            "out_dir": str(out_dir),
        },
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"n_ok": len(rows), "n_error": len(errors), "summary": str(out_dir / "comparison_summary.csv")}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
