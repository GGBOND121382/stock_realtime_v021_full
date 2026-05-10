#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build feed/agri futures features for 002311 next-day samples."""
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


def add_ts_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    for col in cols:
        if col not in out.columns:
            continue
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
    return out


def fetch_feed_futures() -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    mapping = {
        "M0": "feed_soymeal",
        "C0": "feed_corn",
        "RM0": "feed_rapeseed_meal",
        "Y0": "feed_soyoil",
        "A0": "feed_soybean",
        "LH0": "feed_hog",
    }
    frames = []
    errors: Dict[str, str] = {}
    for symbol, prefix in mapping.items():
        try:
            raw = ak.futures_zh_daily_sina(symbol=symbol)
            if raw is None or raw.empty:
                errors[symbol] = "empty"
                continue
            out = raw.rename(columns={
                "open": f"{prefix}_open",
                "high": f"{prefix}_high",
                "low": f"{prefix}_low",
                "close": f"{prefix}_close",
                "volume": f"{prefix}_volume",
                "hold": f"{prefix}_hold",
                "settle": f"{prefix}_settle",
            }).copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            for c in out.columns:
                if c != "date":
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            if f"{prefix}_high" in out.columns and f"{prefix}_low" in out.columns:
                out[f"{prefix}_range_pct"] = out[f"{prefix}_high"] / out[f"{prefix}_low"].replace(0, np.nan) - 1.0
            value_cols = [f"{prefix}_close", f"{prefix}_volume", f"{prefix}_hold", f"{prefix}_range_pct"]
            frames.append(add_ts_features(out.dropna(subset=["date"]).sort_values("date"), value_cols))
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"
    if not frames:
        raise RuntimeError(f"no feed futures data fetched: {errors}")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    if {"feed_soymeal_close", "feed_corn_close"}.issubset(merged.columns):
        merged["feed_cost_index"] = 0.45 * merged["feed_soymeal_close"] + 0.35 * merged["feed_corn_close"]
        if "feed_rapeseed_meal_close" in merged.columns:
            merged["feed_cost_index"] += 0.20 * merged["feed_rapeseed_meal_close"]
        merged = add_ts_features(merged, ["feed_cost_index"])
    if {"feed_soymeal_close", "feed_corn_close"}.issubset(merged.columns):
        merged["feed_soymeal_corn_ratio"] = merged["feed_soymeal_close"] / merged["feed_corn_close"].replace(0, np.nan)
        merged = add_ts_features(merged, ["feed_soymeal_corn_ratio"])
    if {"feed_hog_close", "feed_cost_index"}.issubset(merged.columns):
        merged["feed_hog_cost_ratio"] = merged["feed_hog_close"] / merged["feed_cost_index"].replace(0, np.nan)
        merged = add_ts_features(merged, ["feed_hog_cost_ratio"])
    return merged, errors


def merge_asof_lag(samples: pd.DataFrame, features: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    s = samples.sort_values("date").copy()
    s["feed_asof_date"] = pd.to_datetime(s["date"]) - pd.to_timedelta(lag_days, unit="D")
    f = features.sort_values("date").copy().rename(columns={"date": "feed_feature_date"})
    merged = pd.merge_asof(
        s.sort_values("feed_asof_date"),
        f.sort_values("feed_feature_date"),
        left_on="feed_asof_date",
        right_on="feed_feature_date",
        direction="backward",
    )
    return merged.sort_values("date").drop(columns=["feed_asof_date"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build feed/agri futures features for 002311")
    p.add_argument("--samples", default="002311_fundamental_features_out/training_samples_with_fundamentals.csv")
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "002311_feed_external_features_out"))
    p.add_argument("--lag-days", type=int, default=1)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = pd.read_csv(args.samples, parse_dates=["date"]).sort_values("date")
    raw, errors = fetch_feed_futures()
    merged = merge_asof_lag(samples, raw, args.lag_days)

    raw.to_csv(out_dir / "feed_futures_features.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "training_samples_with_feed_external.csv", index=False, encoding="utf-8-sig")
    feed_cols = [c for c in merged.columns if c.startswith("feed_")]
    report = {
        "sample_rows": int(len(samples)),
        "raw_rows": int(len(raw)),
        "raw_date_min": str(raw["date"].min().date()) if len(raw) else None,
        "raw_date_max": str(raw["date"].max().date()) if len(raw) else None,
        "feature_cols": int(len(feed_cols)),
        "lag_days": args.lag_days,
        "errors": errors,
        "top_missing": {k: float(v) for k, v in merged[feed_cols].isna().mean().sort_values(ascending=False).head(30).items()},
        "outputs": {
            "features": str(out_dir / "feed_futures_features.csv"),
            "merged_samples": str(out_dir / "training_samples_with_feed_external.csv"),
        },
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
