#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch Muyuan HK and HK pork/food-chain proxy features for 002714 samples.

The direct HK listing is fetched with AKShare symbol 02714.  Related HK-listed
pork/food-chain proxies are kept as broader context.  Features are lagged
before merging, so an A-share close decision only sees previously available HK
data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


HK_PROXIES = {
    "02714": "hog_hk_muyuan",
    "01610": "hog_hk_cofco_joycome",
    "00288": "hog_hk_wh_group",
    "01068": "hog_hk_yurun_food",
    "01117": "hog_hk_modern_dairy",
}


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
        out[f"{col}_ma20_gap"] = out[col] / ma20.replace(0, np.nan) - 1.0
        out[f"{col}_z20"] = (out[col] - ma20) / sd20.replace(0, np.nan)
    return out


def fetch_hk_proxy(symbol: str, prefix: str, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    try:
        raw = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), {f"hk_{symbol}": f"{type(exc).__name__}: {exc}"}
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date"]), {f"hk_{symbol}": "empty"}

    rename = {
        "日期": "date",
        "开盘": f"{prefix}_open",
        "收盘": f"{prefix}_close",
        "最高": f"{prefix}_high",
        "最低": f"{prefix}_low",
        "成交量": f"{prefix}_volume",
        "成交额": f"{prefix}_amount",
        "振幅": f"{prefix}_amplitude",
        "涨跌幅": f"{prefix}_pct_chg",
        "换手率": f"{prefix}_turnover",
    }
    out = raw.rename(columns=rename).copy()
    keep = ["date"] + [v for k, v in rename.items() if k != "日期" and v in out.columns]
    out = out[keep]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out = add_ts_features(out, [f"{prefix}_close", f"{prefix}_volume", f"{prefix}_amount"])
    return out, {}


def merge_asof_lag(samples: pd.DataFrame, features: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    s = samples.sort_values("date").copy()
    s["hk_asof_date"] = (
        pd.to_datetime(s["date"]) - pd.to_timedelta(lag_days, unit="D")
    ).astype("datetime64[ns]")
    f = features.sort_values("date").copy().rename(columns={"date": "hog_hk_feature_date"})
    f["hog_hk_feature_date"] = pd.to_datetime(f["hog_hk_feature_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        s.sort_values("hk_asof_date"),
        f.sort_values("hog_hk_feature_date"),
        left_on="hk_asof_date",
        right_on="hog_hk_feature_date",
        direction="backward",
    )
    return merged.sort_values("date").drop(columns=["hk_asof_date"]).reset_index(drop=True)


def add_cross_features(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.sort_values("date").copy()
    prefixes = list(HK_PROXIES.values())
    close_cols = [f"{p}_close" for p in prefixes if f"{p}_close" in out.columns]
    ret1_cols = [f"{p}_close_ret1" for p in prefixes if f"{p}_close_ret1" in out.columns]
    ret5_cols = [f"{p}_close_ret5" for p in prefixes if f"{p}_close_ret5" in out.columns]
    ret20_cols = [f"{p}_close_ret20" for p in prefixes if f"{p}_close_ret20" in out.columns]
    if close_cols:
        out["hog_hk_proxy_close_mean"] = out[close_cols].mean(axis=1)
        out = add_ts_features(out, ["hog_hk_proxy_close_mean"])
    if ret1_cols:
        out["hog_hk_proxy_ret1_mean"] = out[ret1_cols].mean(axis=1)
    if ret5_cols:
        out["hog_hk_proxy_ret5_mean"] = out[ret5_cols].mean(axis=1)
    if ret20_cols:
        out["hog_hk_proxy_ret20_mean"] = out[ret20_cols].mean(axis=1)
    if {"close", "hog_hk_proxy_ret20_mean"}.issubset(out.columns):
        stock_ret20 = out["close"] / out["close"].shift(20) - 1.0
        out["stock_vs_hog_hk_proxy_ret20"] = stock_ret20 - out["hog_hk_proxy_ret20_mean"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build HK proxy features for 002714")
    p.add_argument("--samples", default="002714_hog_industry_current_out/training_samples_with_hog_industry.csv")
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "002714_hk_external_current_out"))
    p.add_argument("--lag-days", type=int, default=1)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = pd.read_csv(args.samples, parse_dates=["date"]).sort_values("date")
    start = samples["date"].min().strftime("%Y%m%d")
    end = max(samples["date"].max(), pd.Timestamp.today()).strftime("%Y%m%d")

    frames = []
    errors: Dict[str, str] = {}
    for symbol, prefix in HK_PROXIES.items():
        frame, err = fetch_hk_proxy(symbol, prefix, start, end)
        errors.update(err)
        if not frame.empty:
            frame.to_csv(out_dir / f"{prefix}_raw_features.csv", index=False, encoding="utf-8-sig")
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"no HK proxy data fetched: {errors}")

    features = frames[0]
    for frame in frames[1:]:
        features = features.merge(frame, on="date", how="outer")
    features = features.sort_values("date").reset_index(drop=True)
    merged = add_cross_features(merge_asof_lag(samples, features, args.lag_days))

    features.to_csv(out_dir / "hog_hk_external_features.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "training_samples_with_hk_external.csv", index=False, encoding="utf-8-sig")
    hk_cols = [c for c in merged.columns if c.startswith("hog_hk_") or c.startswith("stock_vs_hog_hk_")]
    report = {
        "sample_rows": int(len(samples)),
        "external_rows": int(len(features)),
        "external_date_min": str(features["date"].min().date()) if len(features) else None,
        "external_date_max": str(features["date"].max().date()) if len(features) else None,
        "feature_cols": int(len(hk_cols)),
        "lag_days": int(args.lag_days),
        "symbols": HK_PROXIES,
        "errors": errors,
        "top_missing": {k: float(v) for k, v in merged[hk_cols].isna().mean().sort_values(ascending=False).head(30).items()},
        "outputs": {
            "features": str(out_dir / "hog_hk_external_features.csv"),
            "merged_samples": str(out_dir / "training_samples_with_hk_external.csv"),
        },
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
