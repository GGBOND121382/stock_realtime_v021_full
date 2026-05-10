#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backtest close-near/below-daily-VWAP next-day exit rules.

Entry:
    Buy at the signal day's last 5-minute close when daily close is below
    or near the signal day's daily VWAP.

Exit variants:
    1. trail_or_close: next day, after price first rises enough, sell when it
       pulls back from the intraday high; if no trigger, sell at next close.
    2. close_only: always sell at the next day's close.

All prices are adjusted prices from signal_samples.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_intraday(signal_samples: Path) -> pd.DataFrame:
    cols = [
        "symbol", "signal_time", "trade_date", "bar_open", "bar_high", "bar_low", "bar_close",
        "bar_bar_vwap", "bar_bar_volume", "bar_bar_pv",
    ]
    df = pd.read_csv(signal_samples, usecols=cols, parse_dates=["signal_time", "trade_date"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["signal_time", "trade_date", "bar_close", "bar_bar_vwap"])
    return df.sort_values("signal_time").reset_index(drop=True)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "bar_bar_pv" not in work.columns or work["bar_bar_pv"].isna().all():
        work["bar_bar_pv"] = work["bar_bar_vwap"] * work["bar_bar_volume"]
    g = work.groupby("trade_date", sort=True)
    daily = g.agg(
        symbol=("symbol", "last"),
        open=("bar_open", "first"),
        high=("bar_high", "max"),
        low=("bar_low", "min"),
        close=("bar_close", "last"),
        volume=("bar_bar_volume", "sum"),
        pv=("bar_bar_pv", "sum"),
        n_bars=("bar_close", "size"),
    ).reset_index()
    daily["daily_vwap"] = daily["pv"] / daily["volume"]
    daily["next_trade_date"] = daily["trade_date"].shift(-1)
    return daily


def calc_metrics(trades: pd.DataFrame, ret_col: str) -> Dict[str, float]:
    r = trades[ret_col].dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return {}
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return {
        "trades": int(len(r)),
        "win_rate": float(np.mean(r > 0)),
        "avg_return": float(np.mean(r)),
        "median_return": float(np.median(r)),
        "p10_return": float(np.quantile(r, 0.10)),
        "p25_return": float(np.quantile(r, 0.25)),
        "p75_return": float(np.quantile(r, 0.75)),
        "p90_return": float(np.quantile(r, 0.90)),
        "total_compound_return": float(equity[-1] - 1.0),
        "max_drawdown": float(np.min(drawdown)),
        "avg_win": float(np.mean(r[r > 0])) if np.any(r > 0) else 0.0,
        "avg_loss": float(np.mean(r[r <= 0])) if np.any(r <= 0) else 0.0,
        "profit_factor": float(r[r > 0].sum() / abs(r[r < 0].sum())) if np.any(r < 0) else np.inf,
    }


def backtest(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    near_bps: float,
    min_rise_bps: float,
    trail_bps: float,
    round_trip_cost_bps: float,
    min_bars: int,
) -> pd.DataFrame:
    by_date = {d: x.sort_values("signal_time").copy() for d, x in intraday.groupby("trade_date", sort=True)}
    rows: List[Dict] = []
    near = near_bps / 10000.0
    min_rise = min_rise_bps / 10000.0
    trail = trail_bps / 10000.0
    cost = round_trip_cost_bps / 10000.0

    candidates = daily[daily["n_bars"] >= min_bars].copy()
    candidates["entry_signal"] = candidates["close"] <= candidates["daily_vwap"] * (1.0 + near)
    candidates["close_vs_vwap"] = candidates["close"] / candidates["daily_vwap"] - 1.0
    for _, day in candidates[candidates["entry_signal"]].iterrows():
        next_date = day["next_trade_date"]
        if pd.isna(next_date) or next_date not in by_date:
            continue
        next_bars = by_date[next_date]
        if len(next_bars) < min_bars:
            continue

        buy_price = float(day["close"])
        close_sell_price = float(next_bars.iloc[-1]["bar_close"])
        close_sell_time = next_bars.iloc[-1]["signal_time"]
        high_water = -np.inf
        trigger_price = close_sell_price
        trigger_time = close_sell_time
        trigger_reason = "next_close"
        max_runup = np.nan
        max_drawdown_from_high = np.nan

        for _, bar in next_bars.iterrows():
            high_water = max(high_water, float(bar["bar_high"]))
            runup = high_water / buy_price - 1.0
            close_from_high = float(bar["bar_close"]) / high_water - 1.0 if high_water > 0 else 0.0
            max_runup = runup if not np.isfinite(max_runup) else max(max_runup, runup)
            max_drawdown_from_high = close_from_high if not np.isfinite(max_drawdown_from_high) else min(max_drawdown_from_high, close_from_high)
            if runup >= min_rise and close_from_high <= -trail:
                trigger_price = float(bar["bar_close"])
                trigger_time = bar["signal_time"]
                trigger_reason = "trail_pullback"
                break

        row = {
            "signal_date": day["trade_date"],
            "next_trade_date": next_date,
            "symbol": str(day["symbol"]).zfill(6),
            "daily_vwap": float(day["daily_vwap"]),
            "signal_close": buy_price,
            "close_vs_vwap": float(day["close_vs_vwap"]),
            "next_open": float(next_bars.iloc[0]["bar_open"]),
            "next_high": float(next_bars["bar_high"].max()),
            "next_low": float(next_bars["bar_low"].min()),
            "next_close": close_sell_price,
            "max_runup": float(max_runup),
            "max_drawdown_from_high": float(max_drawdown_from_high),
            "trail_sell_time": trigger_time,
            "trail_sell_price": trigger_price,
            "trail_reason": trigger_reason,
            "gross_return_trail_or_close": trigger_price / buy_price - 1.0,
            "gross_return_close_only": close_sell_price / buy_price - 1.0,
        }
        row["net_return_trail_or_close"] = row["gross_return_trail_or_close"] - cost
        row["net_return_close_only"] = row["gross_return_close_only"] - cost
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest next-day rebound sell after close near/below daily VWAP")
    p.add_argument("--signal-samples", default="dual_opp_out_002714_v12/signal_samples.csv")
    p.add_argument("--out-dir", default="vwap_nextday_rebound_backtest_out")
    p.add_argument("--near-bps", type=float, default=20.0, help="Entry if close <= daily_vwap * (1 + near_bps/10000).")
    p.add_argument("--min-rise-bps", type=float, default=30.0, help="Trail can trigger only after next-day high is this far above entry.")
    p.add_argument("--trail-bps", type=float, default=30.0, help="Sell when next-day close pulls back this far from intraday high.")
    p.add_argument("--round-trip-cost-bps", type=float, default=21.0)
    p.add_argument("--min-bars", type=int, default=40)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    intraday = load_intraday(Path(args.signal_samples))
    daily = build_daily(intraday)
    trades = backtest(
        intraday,
        daily,
        near_bps=args.near_bps,
        min_rise_bps=args.min_rise_bps,
        trail_bps=args.trail_bps,
        round_trip_cost_bps=args.round_trip_cost_bps,
        min_bars=args.min_bars,
    )
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "daily_vwap.csv", index=False, encoding="utf-8-sig")

    summary = {
        "params": vars(args),
        "date_min": str(daily["trade_date"].min().date()),
        "date_max": str(daily["trade_date"].max().date()),
        "daily_rows": int(len(daily)),
        "entry_days": int(len(trades)),
        "trigger_counts": trades["trail_reason"].value_counts().to_dict() if len(trades) else {},
        "gross_trail_or_close": calc_metrics(trades, "gross_return_trail_or_close"),
        "net_trail_or_close": calc_metrics(trades, "net_return_trail_or_close"),
        "gross_close_only": calc_metrics(trades, "gross_return_close_only"),
        "net_close_only": calc_metrics(trades, "net_return_close_only"),
        "outputs": {
            "trades": str(out_dir / "trades.csv"),
            "daily_vwap": str(out_dir / "daily_vwap.csv"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
