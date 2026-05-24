#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add 14:55 as-of daily features to next-day training samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_cutoff(value: str) -> tuple[int, int]:
    hh, mm = str(value).split(":", 1)
    return int(hh), int(mm)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_intraday_asof(intraday_bars: Path, cutoff_time: str, min_bars: int) -> pd.DataFrame:
    bars = pd.read_csv(intraday_bars, parse_dates=["datetime"])
    if bars.empty:
        return pd.DataFrame()
    hh, mm = parse_cutoff(cutoff_time)
    bars = bars.dropna(subset=["datetime"]).copy()
    bars["date"] = bars["datetime"].dt.normalize()
    cutoff_minutes = hh * 60 + mm
    bar_minutes = bars["datetime"].dt.hour * 60 + bars["datetime"].dt.minute
    bars = bars[bar_minutes <= cutoff_minutes].copy()
    if bars.empty:
        return pd.DataFrame()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in bars.columns:
            bars[col] = pd.to_numeric(bars[col], errors="coerce")
    grouped = bars.groupby("date", sort=True)
    out = grouped.agg(
        open_asof1455=("open", "first"),
        high_asof1455=("high", "max"),
        low_asof1455=("low", "min"),
        close_asof1455=("close", "last"),
        volume_asof1455=("volume", "sum"),
        amount_asof1455=("amount", "sum"),
        n_intraday_bars_asof1455=("close", "size"),
        asof_last_bar_time=("datetime", "max"),
    ).reset_index()
    out["vwap_asof1455"] = out["amount_asof1455"] / out["volume_asof1455"].replace(0, np.nan)
    out = out[out["n_intraday_bars_asof1455"] >= int(min_bars)].copy()
    return out


def add_asof_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    prev_close = pd.to_numeric(out["close"], errors="coerce").shift(1)
    close_asof = pd.to_numeric(out["close_asof1455"], errors="coerce")
    for n in [1, 3, 5, 20, 60]:
        ref = pd.to_numeric(out["close"], errors="coerce").shift(n)
        out[f"close_ret{n}_asof"] = close_asof / ref.replace(0, np.nan) - 1.0
    roll20 = prev_close.rolling(20, min_periods=20)
    roll60 = prev_close.rolling(60, min_periods=60)
    ma20 = roll20.mean()
    ma60 = roll60.mean()
    std20 = roll20.std(ddof=0)
    std60 = roll60.std(ddof=0)
    out["close_ma20_gap_asof"] = close_asof / ma20.replace(0, np.nan) - 1.0
    out["close_ma60_gap_asof"] = close_asof / ma60.replace(0, np.nan) - 1.0
    out["close_z20_asof"] = (close_asof - ma20) / std20.replace(0, np.nan)
    out["close_z60_asof"] = (close_asof - ma60) / std60.replace(0, np.nan)
    out["range_pct_asof1455"] = (pd.to_numeric(out["high_asof1455"], errors="coerce") - pd.to_numeric(out["low_asof1455"], errors="coerce")) / close_asof.replace(0, np.nan)
    out["feature_time_mode"] = "asof1455"
    out["feature_cutoff_time"] = "14:55"
    return out


def build_asof_samples(samples: Path, intraday_bars: Path, cutoff_time: str, min_bars: int) -> pd.DataFrame:
    base = pd.read_csv(samples, parse_dates=["date"]).sort_values("date")
    asof = build_intraday_asof(intraday_bars, cutoff_time, min_bars)
    if asof.empty:
        raise ValueError(f"no as-of bars from {intraday_bars} cutoff={cutoff_time}")
    merged = base.merge(asof, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged = add_asof_rolling_features(merged)
    merged["feature_cutoff_time"] = cutoff_time
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description="Build 14:55 as-of training samples from daily samples and 5m bars")
    p.add_argument("--samples", required=True)
    p.add_argument("--intraday-bars", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cutoff-time", default="14:55")
    p.add_argument("--min-bars", type=int, default=40)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = build_asof_samples(Path(args.samples), Path(args.intraday_bars), args.cutoff_time, args.min_bars)
    out_path = out_dir / "training_samples_asof1455.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    report = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "cutoff_time": args.cutoff_time,
        "min_bars": int(args.min_bars),
        "out": str(out_path),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
