#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch and merge sector index features for the current daily samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_ths_sector_index(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_board_industry_index_ths(
        symbol=symbol,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    rename = {
        "日期": "date",
        "开盘价": "sector_open",
        "最高价": "sector_high",
        "最低价": "sector_low",
        "收盘价": "sector_close",
        "成交量": "sector_volume",
        "成交额": "sector_amount",
    }
    out = raw.rename(columns=rename).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def add_sector_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    if "sector_close" in out.columns:
        out["sector_ret1"] = out["sector_close"].pct_change()
        out["sector_ret5"] = out["sector_close"] / out["sector_close"].shift(5) - 1.0
        out["sector_ret20"] = out["sector_close"] / out["sector_close"].shift(20) - 1.0
        out["sector_ret60"] = out["sector_close"] / out["sector_close"].shift(60) - 1.0
        out["sector_ma20_gap"] = out["sector_close"] / out["sector_close"].shift(1).rolling(20, min_periods=10).mean() - 1.0
        out["sector_ma60_gap"] = out["sector_close"] / out["sector_close"].shift(1).rolling(60, min_periods=30).mean() - 1.0
        out["sector_vol20"] = out["sector_ret1"].shift(1).rolling(20, min_periods=10).std()
    if {"sector_high", "sector_low"}.issubset(out.columns):
        out["sector_range_pct"] = out["sector_high"] / out["sector_low"].replace(0, np.nan) - 1.0
        out["sector_range_z20"] = (
            out["sector_range_pct"] - out["sector_range_pct"].shift(1).rolling(20, min_periods=10).mean()
        ) / out["sector_range_pct"].shift(1).rolling(20, min_periods=10).std().replace(0, np.nan)
    for col in ["sector_volume", "sector_amount"]:
        if col in out.columns:
            prev_mean = out[col].shift(1).rolling(20, min_periods=10).mean()
            prev_std = out[col].shift(1).rolling(20, min_periods=10).std()
            out[f"{col}_shock20"] = out[col] / prev_mean.replace(0, np.nan) - 1.0
            out[f"{col}_z20"] = (out[col] - prev_mean) / prev_std.replace(0, np.nan)
    if {"sector_ret5", "bench_ret5"}.issubset(out.columns):
        out["sector_vs_bench_ret5"] = out["sector_ret5"] - out["bench_ret5"]
    if {"sector_ret20", "bench_ret20"}.issubset(out.columns):
        out["sector_vs_bench_ret20"] = out["sector_ret20"] - out["bench_ret20"]
    if {"sector_ret1", "close"}.issubset(out.columns):
        stock_ret1 = out["close"].pct_change()
        out["stock_vs_sector_ret1"] = stock_ret1 - out["sector_ret1"]
    if {"sector_ret5", "close"}.issubset(out.columns):
        stock_ret5 = out["close"] / out["close"].shift(5) - 1.0
        out["stock_vs_sector_ret5"] = stock_ret5 - out["sector_ret5"]
    if {"sector_ret20", "close"}.issubset(out.columns):
        stock_ret20 = out["close"] / out["close"].shift(20) - 1.0
        out["stock_vs_sector_ret20"] = stock_ret20 - out["sector_ret20"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build sector features and merge into samples")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "603308_pipeline_out" / "02_fundamental" / "training_samples_with_fundamentals.csv"))
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "603308_pipeline_out" / "03_sector"))
    p.add_argument("--sector-symbol", default="养殖业")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = pd.read_csv(args.samples, parse_dates=["date"]).sort_values("date")
    start_date = args.start_date or str(samples["date"].min().date())
    end_date = args.end_date or str(max(samples["date"].max().date(), pd.Timestamp.today().date()))

    sector = fetch_ths_sector_index(args.sector_symbol, start_date, end_date)
    context_cols = [c for c in ["date", "close", "bench_ret5", "bench_ret20"] if c in samples.columns]
    sector_features = add_sector_derived_features(
        samples[context_cols].merge(sector, on="date", how="left")
    )
    keep_cols = ["date"] + [c for c in sector_features.columns if c.startswith("sector_") or c.startswith("stock_vs_sector_")]
    merged = samples.merge(sector_features[keep_cols], on="date", how="left")
    sector.to_csv(out_dir / "sector_index_ths.csv", index=False, encoding="utf-8-sig")
    sector_features[keep_cols].to_csv(out_dir / "sector_features.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "training_samples_with_sector.csv", index=False, encoding="utf-8-sig")

    report: Dict[str, object] = {
        "sector_symbol": args.sector_symbol,
        "sector_rows": int(len(sector)),
        "sector_date_min": str(sector["date"].min().date()) if len(sector) else None,
        "sector_date_max": str(sector["date"].max().date()) if len(sector) else None,
        "sample_rows": int(len(samples)),
        "merged_rows": int(len(merged)),
        "sector_feature_cols": int(len(keep_cols) - 1),
        "top_missing": {k: float(v) for k, v in merged[keep_cols].isna().mean().sort_values(ascending=False).head(20).items()},
        "outputs": {
            "sector_index": str(out_dir / "sector_index_ths.csv"),
            "sector_features": str(out_dir / "sector_features.csv"),
            "merged_samples": str(out_dir / "training_samples_with_sector.csv"),
        },
    }
    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
