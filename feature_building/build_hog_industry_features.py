#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch AKShare hog-industry data and merge lagged features into samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def numeric_series(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[name] = pd.to_numeric(out["value"], errors="coerce")
    return out[["date", name]].dropna(subset=["date"]).sort_values("date")


def fetch_akshare_hog_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    frames: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}

    jobs = [
        ("hog_spot", ak.futures_hog_core, {"symbol": "外三元"}),
        ("hog_corn", ak.futures_hog_cost, {"symbol": "玉米"}),
        ("hog_soymeal", ak.futures_hog_cost, {"symbol": "豆粕"}),
        ("hog_piglet", ak.futures_hog_cost, {"symbol": "仔猪价格"}),
        ("hog_sow", ak.futures_hog_cost, {"symbol": "二元母猪价格"}),
        ("hog_pig_corn_ratio", ak.futures_hog_supply, {"symbol": "猪粮比价"}),
        ("hog_pork_wholesale", ak.futures_hog_supply, {"symbol": "猪肉批发价"}),
    ]
    for name, fn, kwargs in jobs:
        try:
            raw = fn(**kwargs)
            if {"date", "value"}.issubset(raw.columns):
                frames.append(numeric_series(raw, name))
            else:
                errors[name] = f"unexpected columns: {raw.columns.tolist()}"
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    try:
        fut = ak.futures_zh_daily_sina(symbol="LH0")
        fut = fut.rename(columns={
            "open": "hog_fut_open",
            "high": "hog_fut_high",
            "low": "hog_fut_low",
            "close": "hog_fut_close",
            "volume": "hog_fut_volume",
            "hold": "hog_fut_hold",
            "settle": "hog_fut_settle",
        })
        fut["date"] = pd.to_datetime(fut["date"], errors="coerce")
        for c in [c for c in fut.columns if c != "date"]:
            fut[c] = pd.to_numeric(fut[c], errors="coerce")
        frames.append(fut.dropna(subset=["date"]).sort_values("date"))
    except Exception as exc:
        errors["hog_futures_lh0"] = f"{type(exc).__name__}: {exc}"

    if not frames:
        raise RuntimeError(f"no hog industry data fetched: {errors}")
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out, errors


def add_derived_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.sort_values("date").copy()
    base_cols = [
        "hog_spot", "hog_corn", "hog_soymeal", "hog_piglet", "hog_sow",
        "hog_pig_corn_ratio", "hog_pork_wholesale", "hog_fut_close",
        "hog_fut_volume", "hog_fut_hold",
    ]
    for col in [c for c in base_cols if c in out.columns]:
        out[f"{col}_ret1"] = out[col].pct_change()
        out[f"{col}_ret5"] = out[col] / out[col].shift(5) - 1.0
        out[f"{col}_ret20"] = out[col] / out[col].shift(20) - 1.0
        ma20 = out[col].shift(1).rolling(20, min_periods=10).mean()
        sd20 = out[col].shift(1).rolling(20, min_periods=10).std()
        ma60 = out[col].shift(1).rolling(60, min_periods=20).mean()
        sd60 = out[col].shift(1).rolling(60, min_periods=20).std()
        out[f"{col}_ma20_gap"] = out[col] / ma20.replace(0, np.nan) - 1.0
        out[f"{col}_z20"] = (out[col] - ma20) / sd20.replace(0, np.nan)
        out[f"{col}_z60"] = (out[col] - ma60) / sd60.replace(0, np.nan)
    if "hog_fut_close_ret1" in out.columns:
        out["hog_fut_vol20"] = out["hog_fut_close_ret1"].shift(1).rolling(20, min_periods=10).std()
    if {"hog_fut_close", "hog_spot"}.issubset(out.columns):
        # Spot is yuan/kg, LH futures is yuan/ton.
        out["hog_fut_spot_basis"] = out["hog_fut_close"] / (out["hog_spot"] * 1000).replace(0, np.nan) - 1.0
        basis_ma = out["hog_fut_spot_basis"].shift(1).rolling(60, min_periods=20).mean()
        basis_sd = out["hog_fut_spot_basis"].shift(1).rolling(60, min_periods=20).std()
        out["hog_fut_spot_basis_z60"] = (out["hog_fut_spot_basis"] - basis_ma) / basis_sd.replace(0, np.nan)
    if {"hog_corn", "hog_soymeal"}.issubset(out.columns):
        out["hog_feed_cost_index"] = 0.65 * out["hog_corn"] + 0.35 * out["hog_soymeal"]
        out["hog_feed_cost_ret20"] = out["hog_feed_cost_index"] / out["hog_feed_cost_index"].shift(20) - 1.0
    return out


def merge_lagged(samples: pd.DataFrame, features: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    s = samples.sort_values("date").copy()
    s["hog_asof_date"] = pd.to_datetime(s["date"]) - pd.to_timedelta(lag_days, unit="D")
    f = features.sort_values("date").copy()
    merged = pd.merge_asof(
        s.sort_values("hog_asof_date"),
        f.rename(columns={"date": "hog_feature_date"}).sort_values("hog_feature_date"),
        left_on="hog_asof_date",
        right_on="hog_feature_date",
        direction="backward",
    )
    return merged.sort_values("date").drop(columns=["hog_asof_date"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build AKShare hog industry features for 002714 samples")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "002311_pipeline_out" / "03_sector" / "training_samples_with_sector.csv"))
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "002311_pipeline_out" / "04_external" / "hog"))
    p.add_argument("--lag-days", type=int, default=1)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = pd.read_csv(args.samples, parse_dates=["date"]).sort_values("date")
    raw, errors = fetch_akshare_hog_data()
    feats = add_derived_features(raw)
    merged = merge_lagged(samples, feats, args.lag_days)

    raw.to_csv(out_dir / "hog_industry_raw.csv", index=False, encoding="utf-8-sig")
    feats.to_csv(out_dir / "hog_industry_features.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "training_samples_with_hog_industry.csv", index=False, encoding="utf-8-sig")

    hog_cols = [c for c in merged.columns if c.startswith("hog_")]
    report = {
        "sample_rows": int(len(samples)),
        "raw_rows": int(len(raw)),
        "raw_date_min": str(raw["date"].min().date()) if len(raw) else None,
        "raw_date_max": str(raw["date"].max().date()) if len(raw) else None,
        "feature_cols": int(len(hog_cols)),
        "lag_days": args.lag_days,
        "errors": errors,
        "top_missing": {k: float(v) for k, v in merged[hog_cols].isna().mean().sort_values(ascending=False).head(30).items()},
        "outputs": {
            "raw": str(out_dir / "hog_industry_raw.csv"),
            "features": str(out_dir / "hog_industry_features.csv"),
            "merged_samples": str(out_dir / "training_samples_with_hog_industry.csv"),
        },
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
