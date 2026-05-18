#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare realtime pending 5m bars against BaoStock/raw-cache full 5m bars."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING_TIMES = set(
    pd.date_range("09:35", "11:30", freq="5min").strftime("%H:%M:%S").tolist()
    + pd.date_range("13:05", "15:00", freq="5min").strftime("%H:%M:%S").tolist()
)


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace("_", ".")
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


def raw_path_for(symbol: str) -> Path | None:
    code = symbol.split(".", 1)[0]
    candidates = [
        ROOT / "saved_data" / f"{code}_pipeline_out" / "00_base" / "raw_cache" / f"{code}_5m_raw.csv",
        ROOT / "saved_data" / f"{code}_pipeline_out_v2_all14" / "00_base" / "raw_cache" / f"{code}_5m_raw.csv",
        ROOT / "saved_data" / f"{code}_pipeline_out_v2_new27_full" / "00_base" / "raw_cache" / f"{code}_5m_raw.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted((ROOT / "saved_data").glob(f"**/{code}_5m_raw.csv"))
    return hits[0] if hits else None


def normalize_realtime_bars(path: Path, date: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return df
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(
            df["trade_date"].astype(str) + df["trade_time"].astype(str).str.zfill(6),
            errors="coerce",
        )
    df = df.dropna(subset=["datetime"]).copy()
    df = df[df["datetime"].dt.strftime("%Y%m%d") == date].copy()
    for col in ["open", "high", "low", "close", "volume", "amount", "bar_volume", "bar_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "bar_volume" in df.columns and df["bar_volume"].notna().any():
        df["volume"] = df["bar_volume"].where(df["bar_volume"].notna(), df.get("volume"))
    if "bar_amount" in df.columns and df["bar_amount"].notna().any():
        df["amount"] = df["bar_amount"].where(df["bar_amount"].notna(), df.get("amount"))
    ratio = (df["amount"] / df["volume"].replace(0, np.nan)) / df["close"].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if not ratio.empty and 50 <= float(ratio.median()) <= 150:
        df["volume"] = df["volume"] * 100.0
    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")
    df = df[df["time_str"].isin(TRAINING_TIMES)].copy()
    return df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def load_raw(path: Path, date: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df[pd.to_datetime(df["datetime"]).dt.strftime("%Y%m%d") == date].copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")
    df = df[df["time_str"].isin(TRAINING_TIMES)].copy()
    return df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def safe_vwap(df: pd.DataFrame) -> float:
    vol = pd.to_numeric(df.get("volume"), errors="coerce").sum()
    amt = pd.to_numeric(df.get("amount"), errors="coerce").sum()
    return float(amt / vol) if vol and np.isfinite(vol) else np.nan


def compare_one(cache_dir: Path, date: str, symbol: str) -> tuple[dict, pd.DataFrame]:
    sym = normalize_symbol(symbol)
    rt_path = cache_dir / "pending" / date / sym / "minute_bars_5min.csv"
    rp = raw_path_for(sym)
    base = {
        "trade_date": date,
        "stock_code": sym,
        "realtime_path": str(rt_path) if rt_path.exists() else "",
        "baostock_path": str(rp) if rp else "",
    }
    if not rt_path.exists() or rp is None:
        base.update({"status": "missing_file"})
        return base, pd.DataFrame()
    rt = normalize_realtime_bars(rt_path, date)
    raw = load_raw(rp, date)
    if rt.empty or raw.empty:
        base.update({"status": "empty"})
        return base, pd.DataFrame()
    merged = raw.merge(rt, on="datetime", how="outer", suffixes=("_bao", "_rt"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    missing_rt = merged[merged["_merge"] == "left_only"].copy()
    extra_rt = merged[merged["_merge"] == "right_only"].copy()

    def col_abs(col: str) -> pd.Series:
        return (pd.to_numeric(matched[f"{col}_rt"], errors="coerce") - pd.to_numeric(matched[f"{col}_bao"], errors="coerce")).abs()

    def col_rel(col: str) -> pd.Series:
        return col_abs(col) / pd.to_numeric(matched[f"{col}_bao"], errors="coerce").abs().replace(0, np.nan)

    summary = dict(base)
    summary.update(
        {
            "status": "ok",
            "rt_bars": int(len(rt)),
            "baostock_bars": int(len(raw)),
            "matched_bars": int(len(matched)),
            "missing_rt_bars": int(len(missing_rt)),
            "extra_rt_bars": int(len(extra_rt)),
            "first_rt_time": str(rt["datetime"].min()),
            "last_rt_time": str(rt["datetime"].max()),
            "first_bao_time": str(raw["datetime"].min()),
            "last_bao_time": str(raw["datetime"].max()),
            "missing_rt_times": ",".join(missing_rt["datetime"].dt.strftime("%H:%M").tolist()),
            "extra_rt_times": ",".join(extra_rt["datetime"].dt.strftime("%H:%M").tolist()),
            "close_rt_last": float(rt["close"].iloc[-1]),
            "close_bao_last": float(raw["close"].iloc[-1]),
            "close_last_diff": float(rt["close"].iloc[-1] - raw["close"].iloc[-1]),
            "day_vwap_rt": safe_vwap(rt),
            "day_vwap_bao": safe_vwap(raw),
            "day_vwap_diff": safe_vwap(rt) - safe_vwap(raw),
            "amount_rt_sum": float(rt["amount"].sum()),
            "amount_bao_sum": float(raw["amount"].sum()),
            "amount_rel_diff": float((rt["amount"].sum() - raw["amount"].sum()) / raw["amount"].sum()) if raw["amount"].sum() else np.nan,
            "volume_rt_sum": float(rt["volume"].sum()),
            "volume_bao_sum": float(raw["volume"].sum()),
            "volume_rel_diff": float((rt["volume"].sum() - raw["volume"].sum()) / raw["volume"].sum()) if raw["volume"].sum() else np.nan,
        }
    )
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if not matched.empty:
            summary[f"{col}_mae"] = float(col_abs(col).mean())
            summary[f"{col}_median_rel_abs"] = float(col_rel(col).median())
            summary[f"{col}_p90_rel_abs"] = float(col_rel(col).quantile(0.9))
        else:
            summary[f"{col}_mae"] = np.nan
            summary[f"{col}_median_rel_abs"] = np.nan
            summary[f"{col}_p90_rel_abs"] = np.nan
    return summary, merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "saved_data" / "feature_reconstruction_audit" / "zip_cache_extract" / "akshare_realtime_cache"))
    ap.add_argument("--out-dir", default=str(ROOT / "saved_data" / "feature_reconstruction_audit" / "zip_baostock_compare"))
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []
    pending = cache_dir / "pending"
    for day_dir in sorted(p for p in pending.iterdir() if p.is_dir()):
        date = day_dir.name
        for sym_dir in sorted(p for p in day_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            summary, detail = compare_one(cache_dir, date, sym_dir.name)
            rows.append(summary)
            if not detail.empty:
                detail["stock_code_cmp"] = normalize_symbol(sym_dir.name)
                detail["trade_date_cmp"] = date
                details.append(detail)
    summary_df = pd.DataFrame(rows)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_df.to_csv(out_dir / "bar_compare_summary.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(out_dir / "bar_compare_detail.csv", index=False, encoding="utf-8-sig")
    print(f"WROTE {out_dir / 'bar_compare_summary.csv'} rows={len(summary_df)}")
    print(f"WROTE {out_dir / 'bar_compare_detail.csv'} rows={len(detail_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
