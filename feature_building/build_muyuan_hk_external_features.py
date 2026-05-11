#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch Muyuan HK and HK pork/food-chain proxy features for 002714 samples.

Confirmed data-source policy after AKShare diagnostics:

Historical training:
  1. Prefer ak.stock_hk_daily(symbol="02714")
     - confirmed available for Muyuan H shares in the user's server test
     - returns date/open/high/low/close/volume/amount
  2. Fallback ak.stock_zh_ah_daily(symbol="02714") for Muyuan only
     - confirmed available, but usually lacks amount
  3. Do NOT use stock_hk_hist as primary path
     - repeatedly failed with RemoteDisconnected in the user's server test

For HK pork/food-chain proxies, this builder also tries stock_hk_daily first.
Features are lagged before merging, so an A-share close decision only sees
previously available HK data.
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


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _normalize_hk_daily_frame(raw: pd.DataFrame, prefix: str, start: str, end: str) -> pd.DataFrame:
    """Normalize HK daily data from different AKShare interfaces.

    Supported schemas:
      stock_hk_daily: date/open/high/low/close/volume/amount
      stock_zh_ah_daily: 日期/开盘/收盘/最高/最低/成交量
      stock_hk_hist: 日期/开盘/收盘/最高/最低/成交量/成交额/...
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date"])

    df = raw.copy()
    date_col = _first_existing_col(df, ["date", "日期"])
    open_col = _first_existing_col(df, ["open", "开盘", "开盘价"])
    high_col = _first_existing_col(df, ["high", "最高", "最高价"])
    low_col = _first_existing_col(df, ["low", "最低", "最低价"])
    close_col = _first_existing_col(df, ["close", "收盘", "收盘价"])
    volume_col = _first_existing_col(df, ["volume", "成交量"])
    amount_col = _first_existing_col(df, ["amount", "成交额"])
    amplitude_col = _first_existing_col(df, ["振幅", "amplitude"])
    pct_chg_col = _first_existing_col(df, ["涨跌幅", "pct_chg", "changepercent"])
    turnover_col = _first_existing_col(df, ["换手率", "turnover"])

    if date_col is None or close_col is None:
        return pd.DataFrame(columns=["date"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    mapping = {
        f"{prefix}_open": open_col,
        f"{prefix}_high": high_col,
        f"{prefix}_low": low_col,
        f"{prefix}_close": close_col,
        f"{prefix}_volume": volume_col,
        f"{prefix}_amount": amount_col,
        f"{prefix}_amplitude": amplitude_col,
        f"{prefix}_pct_chg": pct_chg_col,
        f"{prefix}_turnover": turnover_col,
    }
    for dst, src in mapping.items():
        if src is None:
            out[dst] = np.nan
        else:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if start:
        out = out[out["date"] >= pd.to_datetime(start, format="%Y%m%d", errors="coerce")]
    if end:
        out = out[out["date"] <= pd.to_datetime(end, format="%Y%m%d", errors="coerce")]
    return out.reset_index(drop=True)


def fetch_hk_proxy(symbol: str, prefix: str, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    errors: Dict[str, str] = {}
    attempts: list[tuple[str, callable]] = []

    # Confirmed-good path for 02714 on user's server, and usually works for HK proxies too.
    attempts.append(("stock_hk_daily", lambda: ak.stock_hk_daily(symbol=symbol)))

    # Confirmed-good fallback for Muyuan 02714, but usually A+H only and lacks amount.
    if symbol == "02714":
        attempts.append((
            "stock_zh_ah_daily",
            lambda: ak.stock_zh_ah_daily(
                symbol="02714",
                start_year=str(pd.to_datetime(start, format="%Y%m%d").year),
                end_year=str(pd.to_datetime(end, format="%Y%m%d").year),
                adjust="",
            ),
        ))

    # Keep stock_hk_hist only as last-resort fallback. It is known to be unstable on some servers.
    attempts.append((
        "stock_hk_hist_last_resort",
        lambda: ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust=""),
    ))

    for provider_name, fn in attempts:
        try:
            raw = fn()
            out = _normalize_hk_daily_frame(raw, prefix, start, end)
            if out is not None and not out.empty:
                out = add_ts_features(out, [f"{prefix}_close", f"{prefix}_volume", f"{prefix}_amount"])
                out["hk_provider"] = provider_name
                return out, {}
            errors[f"hk_{symbol}_{provider_name}"] = "empty_or_unusable_schema"
        except Exception as exc:
            errors[f"hk_{symbol}_{provider_name}"] = f"{type(exc).__name__}: {exc}"

    return pd.DataFrame(columns=["date"]), errors


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
    providers: Dict[str, str] = {}
    for symbol, prefix in HK_PROXIES.items():
        frame, err = fetch_hk_proxy(symbol, prefix, start, end)
        errors.update(err)
        if not frame.empty:
            if "hk_provider" in frame.columns:
                providers[prefix] = str(frame["hk_provider"].dropna().iloc[-1]) if not frame["hk_provider"].dropna().empty else "unknown"
                frame = frame.drop(columns=["hk_provider"])
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
        "providers": providers,
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
