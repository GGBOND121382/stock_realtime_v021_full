#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build next-day VWAP/high/close samples from cached BaoStock daily and 5m bars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_daily_vwap(intraday_path: Path) -> pd.DataFrame:
    bars = pd.read_csv(intraday_path, parse_dates=["datetime"])
    bars["date"] = bars["datetime"].dt.normalize()
    bars["pv"] = pd.to_numeric(bars["amount"], errors="coerce")
    bars["vol"] = pd.to_numeric(bars["volume"], errors="coerce")
    daily = bars.groupby("date").agg(
        daily_vwap_pv=("pv", "sum"),
        daily_vwap_volume=("vol", "sum"),
        n_intraday_bars=("close", "size"),
    )
    daily["daily_vwap"] = daily["daily_vwap_pv"] / daily["daily_vwap_volume"].replace(0, pd.NA)
    return daily.reset_index()


def build_samples(daily_features: Path, intraday_bars: Path, min_bars: int, keep_unlabeled_tail: bool = False) -> pd.DataFrame:
    daily = pd.read_csv(daily_features, parse_dates=["date"]).sort_values("date")
    vwap = build_daily_vwap(intraday_bars)
    df = daily.merge(vwap, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df = df[df["n_intraday_bars"] >= min_bars].copy()
    df["next_date"] = df["date"].shift(-1)
    df["next_day_vwap"] = df["daily_vwap"].shift(-1)
    df["next_day_close"] = df["close"].shift(-1)
    df["next_day_high"] = df["high"].shift(-1)
    df["next_day_low"] = df["low"].shift(-1)
    df["next_day_vwap_ret_close"] = df["next_day_vwap"] / df["close"] - 1.0
    df["next_day_vwap_ret_vwap"] = df["next_day_vwap"] / df["daily_vwap"] - 1.0
    df["next_day_close_ret_close"] = df["next_day_close"] / df["close"] - 1.0
    if keep_unlabeled_tail:
        return df.reset_index(drop=True)
    return df.dropna(subset=["next_day_vwap", "next_day_close", "next_day_high"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build next-day samples from daily_features and 5m raw bars")
    p.add_argument("--daily-features", required=True)
    p.add_argument("--intraday-bars", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-bars", type=int, default=40)
    p.add_argument("--keep-unlabeled-tail", action="store_true", help="Keep the latest row even when next-day labels are not available")
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = build_samples(Path(args.daily_features), Path(args.intraday_bars), args.min_bars, args.keep_unlabeled_tail)
    out_path = out_dir / "training_samples.csv"
    samples.to_csv(out_path, index=False, encoding="utf-8-sig")
    report = {
        "rows": int(len(samples)),
        "date_min": str(samples["date"].min().date()) if len(samples) else None,
        "date_max": str(samples["date"].max().date()) if len(samples) else None,
        "out": str(out_path),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
