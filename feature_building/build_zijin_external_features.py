#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch AKShare commodity / HK / sector features for Zijin Mining (601899)."""
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


def to_numeric_frame(df: pd.DataFrame, date_col: str, rename: Dict[str, str]) -> pd.DataFrame:
    out = df.rename(columns=rename).copy()
    out["date"] = pd.to_datetime(out[rename.get(date_col, date_col)], errors="coerce")
    keep = ["date"] + [v for k, v in rename.items() if k != date_col and v in out.columns]
    out = out[keep].dropna(subset=["date"]).sort_values("date")
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def add_ts_features(df: pd.DataFrame, cols: List[str], prefix: str = "") -> pd.DataFrame:
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


def fetch_main_futures() -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    errors: Dict[str, str] = {}
    frames = []
    mapping = {"AU0": "gold", "CU0": "copper", "AG0": "silver"}
    for symbol, name in mapping.items():
        try:
            raw = ak.futures_main_sina(symbol=symbol)
            frame = to_numeric_frame(
                raw,
                "日期",
                {
                    "日期": "date",
                    "开盘价": f"{name}_open",
                    "最高价": f"{name}_high",
                    "最低价": f"{name}_low",
                    "收盘价": f"{name}_close",
                    "成交量": f"{name}_volume",
                    "持仓量": f"{name}_hold",
                    "动态结算价": f"{name}_settle",
                },
            )
            if f"{name}_high" in frame.columns and f"{name}_low" in frame.columns:
                frame[f"{name}_range_pct"] = frame[f"{name}_high"] / frame[f"{name}_low"].replace(0, np.nan) - 1.0
            frame = add_ts_features(frame, [f"{name}_close", f"{name}_volume", f"{name}_hold"])
            frames.append(frame)
        except Exception as exc:
            errors[f"futures_{symbol}"] = f"{type(exc).__name__}: {exc}"
    out = frames[0] if frames else pd.DataFrame(columns=["date"])
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    if {"gold_close", "silver_close"}.issubset(out.columns):
        out["gold_silver_ratio"] = out["gold_close"] / out["silver_close"].replace(0, np.nan)
        out = add_ts_features(out, ["gold_silver_ratio"])
    if {"gold_close", "copper_close"}.issubset(out.columns):
        out["gold_copper_ratio"] = out["gold_close"] / out["copper_close"].replace(0, np.nan)
        out = add_ts_features(out, ["gold_copper_ratio"])
    return out.sort_values("date").reset_index(drop=True), errors


def fetch_basis(start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    errors: Dict[str, str] = {}
    try:
        raw = ak.futures_spot_price_daily(start_day=start_date, end_day=end_date, vars_list=["CU", "AU", "AG"])
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), {"basis": f"{type(exc).__name__}: {exc}"}
    if raw.empty:
        return pd.DataFrame(columns=["date"]), errors
    raw["date"] = pd.to_datetime(raw["date"].astype(str), errors="coerce")
    frames = []
    for sym, name in {"CU": "copper", "AU": "gold", "AG": "silver"}.items():
        part = raw[raw["symbol"].astype(str).str.upper() == sym].copy()
        if part.empty:
            continue
        keep = part[["date", "spot_price", "near_basis", "dom_basis", "near_basis_rate", "dom_basis_rate"]].copy()
        keep = keep.rename(columns={
            "spot_price": f"{name}_spot",
            "near_basis": f"{name}_near_basis",
            "dom_basis": f"{name}_dom_basis",
            "near_basis_rate": f"{name}_near_basis_rate",
            "dom_basis_rate": f"{name}_dom_basis_rate",
        })
        for c in keep.columns:
            if c != "date":
                keep[c] = pd.to_numeric(keep[c], errors="coerce")
        frames.append(add_ts_features(keep, [f"{name}_spot", f"{name}_dom_basis_rate"]))
    out = frames[0] if frames else pd.DataFrame(columns=["date"])
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True), errors


def fetch_hk(start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    try:
        raw = ak.stock_hk_hist(symbol="02899", period="daily", start_date=start_date, end_date=end_date, adjust="")
        out = to_numeric_frame(
            raw,
            "日期",
            {
                "日期": "date",
                "开盘": "zijin_hk_open",
                "收盘": "zijin_hk_close",
                "最高": "zijin_hk_high",
                "最低": "zijin_hk_low",
                "成交量": "zijin_hk_volume",
                "成交额": "zijin_hk_amount",
                "涨跌幅": "zijin_hk_pct_chg",
                "换手率": "zijin_hk_turnover",
            },
        )
        out = add_ts_features(out, ["zijin_hk_close", "zijin_hk_volume"])
        return out.reset_index(drop=True), {}
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), {"hk_02899": f"{type(exc).__name__}: {exc}"}


def fetch_sector_indices(start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    import akshare as ak

    errors: Dict[str, str] = {}
    frames = []
    for symbol, name in [("贵金属", "precious"), ("工业金属", "industrial_metal"), ("小金属", "minor_metal")]:
        try:
            raw = ak.stock_board_industry_index_ths(symbol=symbol, start_date=start_date, end_date=end_date)
            frame = to_numeric_frame(
                raw,
                "日期",
                {
                    "日期": "date",
                    "开盘价": f"{name}_sector_open",
                    "最高价": f"{name}_sector_high",
                    "最低价": f"{name}_sector_low",
                    "收盘价": f"{name}_sector_close",
                    "成交量": f"{name}_sector_volume",
                    "成交额": f"{name}_sector_amount",
                },
            )
            frame = add_ts_features(frame, [f"{name}_sector_close", f"{name}_sector_volume", f"{name}_sector_amount"])
            frames.append(frame)
        except Exception as exc:
            errors[f"sector_{symbol}"] = f"{type(exc).__name__}: {exc}"
    out = frames[0] if frames else pd.DataFrame(columns=["date"])
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True), errors


def merge_asof_lag(samples: pd.DataFrame, features: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    s = samples.sort_values("date").copy()
    s["external_asof_date"] = (pd.to_datetime(s["date"]) - pd.to_timedelta(lag_days, unit="D")).astype("datetime64[ns]")
    f = features.sort_values("date").copy().rename(columns={"date": "external_feature_date"})
    f["external_feature_date"] = pd.to_datetime(f["external_feature_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        s.sort_values("external_asof_date"),
        f.sort_values("external_feature_date"),
        left_on="external_asof_date",
        right_on="external_feature_date",
        direction="backward",
    )
    return merged.sort_values("date").drop(columns=["external_asof_date"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build Zijin commodity/HK/sector features")
    p.add_argument("--samples", default="zijin_601899_samples_out/training_samples.csv")
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "zijin_601899_external_features_out"))
    p.add_argument("--lag-days", type=int, default=1)
    p.add_argument("--skip-basis", action="store_true", help="Skip slower 100ppi spot/basis source")
    p.add_argument("--skip-sector", action="store_true")
    p.add_argument("--skip-hk", action="store_true")
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = pd.read_csv(args.samples, parse_dates=["date"]).sort_values("date")
    start = samples["date"].min().strftime("%Y%m%d")
    end = max(samples["date"].max(), pd.Timestamp.today()).strftime("%Y%m%d")

    frames = []
    errors: Dict[str, str] = {}
    jobs = [("main_futures", lambda: fetch_main_futures())]
    if not args.skip_basis:
        jobs.append(("basis", lambda: fetch_basis(start, end)))
    if not args.skip_hk:
        jobs.append(("hk", lambda: fetch_hk(start, end)))
    if not args.skip_sector:
        jobs.append(("sector", lambda: fetch_sector_indices(start, end)))
    for name, fn in jobs:
        frame, err = fn()
        errors.update(err)
        if not frame.empty:
            frame.to_csv(out_dir / f"{name}_raw_features.csv", index=False, encoding="utf-8-sig")
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"no external frames fetched: {errors}")
    external = frames[0]
    for frame in frames[1:]:
        external = external.merge(frame, on="date", how="outer")
    external = external.sort_values("date").reset_index(drop=True)

    merged = merge_asof_lag(samples, external, args.lag_days)
    # Relative strength features after lag merge.
    if {"close", "gold_close_ret20"}.issubset(merged.columns):
        merged["stock_vs_gold_ret20"] = merged["close"] / merged["close"].shift(20) - 1.0 - merged["gold_close_ret20"]
    if {"close", "copper_close_ret20"}.issubset(merged.columns):
        merged["stock_vs_copper_ret20"] = merged["close"] / merged["close"].shift(20) - 1.0 - merged["copper_close_ret20"]
    if {"close", "precious_sector_close_ret20"}.issubset(merged.columns):
        merged["stock_vs_precious_sector_ret20"] = merged["close"] / merged["close"].shift(20) - 1.0 - merged["precious_sector_close_ret20"]
    if {"close", "industrial_metal_sector_close_ret20"}.issubset(merged.columns):
        merged["stock_vs_industrial_metal_sector_ret20"] = merged["close"] / merged["close"].shift(20) - 1.0 - merged["industrial_metal_sector_close_ret20"]
    if {"close", "zijin_hk_close"}.issubset(merged.columns):
        merged["zijin_a_h_close_ratio"] = merged["close"] / merged["zijin_hk_close"].replace(0, np.nan)
        merged["zijin_a_h_close_ratio_z20"] = (
            merged["zijin_a_h_close_ratio"] - merged["zijin_a_h_close_ratio"].shift(1).rolling(20, min_periods=10).mean()
        ) / merged["zijin_a_h_close_ratio"].shift(1).rolling(20, min_periods=10).std().replace(0, np.nan)

    external.to_csv(out_dir / "zijin_external_features.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "training_samples_with_zijin_external.csv", index=False, encoding="utf-8-sig")
    ext_cols = [c for c in merged.columns if any(k in c for k in [
        "gold", "copper", "silver", "zijin_hk", "precious", "industrial_metal", "minor_metal", "a_h",
    ])]
    report = {
        "sample_rows": int(len(samples)),
        "external_rows": int(len(external)),
        "external_date_min": str(external["date"].min().date()),
        "external_date_max": str(external["date"].max().date()),
        "feature_cols": int(len(ext_cols)),
        "lag_days": args.lag_days,
        "errors": errors,
        "top_missing": {k: float(v) for k, v in merged[ext_cols].isna().mean().sort_values(ascending=False).head(30).items()},
        "outputs": {
            "external": str(out_dir / "zijin_external_features.csv"),
            "merged_samples": str(out_dir / "training_samples_with_zijin_external.csv"),
        },
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
