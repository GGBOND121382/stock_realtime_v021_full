#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot saved-model next-day scores, daily price, buy/sell points, and equity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_daily(path: str | Path) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    for col in ["open", "high", "low", "close"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
    return daily


def build_trades(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = load_daily(args.daily)
    samples = pd.read_csv(args.samples, parse_dates=["date", "next_date"]).sort_values("date")
    scores = pd.read_csv(args.scores, parse_dates=["date"]).sort_values("date")
    target_ret = args.target_hit_bps / 10000.0
    cost = args.round_trip_cost_bps / 10000.0

    cols = [
        "date", "close", "next_date", "next_day_close", "next_day_high",
    ]
    scored = scores.merge(samples[[c for c in cols if c in samples.columns]], on="date", how="left")
    scored["signal"] = scored["signal"].astype(bool)
    trades = scored[scored["signal"]].copy()
    trades["buy_date"] = trades["date"]
    trades["buy_price"] = trades["close"]
    trades["hit"] = trades["next_day_high"] >= trades["buy_price"] * (1.0 + target_ret)
    trades["sell_date"] = trades["next_date"]
    if args.sell_mode == "close":
        trades["sell_price"] = trades["next_day_close"]
    elif args.sell_mode == "target_or_close":
        trades["sell_price"] = np.where(
            trades["hit"],
            trades["buy_price"] * (1.0 + target_ret),
            trades["next_day_close"],
        )
    else:
        raise ValueError(f"unknown sell_mode={args.sell_mode}")
    trades["net_return"] = trades["sell_price"] / trades["buy_price"] - 1.0 - cost
    trades["equity"] = (1.0 + trades["net_return"].fillna(0.0)).cumprod()
    return daily, scored, trades


def plot(args: argparse.Namespace) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    out_dir = ensure_dir(args.out_dir)
    daily, scored, trades = build_trades(args)
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date) if args.end_date else pd.Timestamp.today().normalize()

    daily_view = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    scored_view = scored[(scored["date"] >= start) & (scored["date"] <= end)].copy()
    trades_view = trades[(trades["buy_date"] >= start) & (trades["buy_date"] <= end)].copy()
    if not trades_view.empty:
        trades_view["equity"] = (1.0 + trades_view["net_return"].fillna(0.0)).cumprod()
    if daily_view.empty:
        raise ValueError(f"no daily rows from {start.date()} to {end.date()}")

    fig, (ax_price, ax_score, ax_equity) = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
        constrained_layout=True,
    )
    ax_price.plot(daily_view["date"], daily_view["close"], color="#111827", linewidth=1.5, label="Close")
    ax_price.fill_between(daily_view["date"], daily_view["low"], daily_view["high"], color="#cbd5e1", alpha=0.4, label="High-low")
    if not trades_view.empty:
        ax_price.scatter(trades_view["buy_date"], trades_view["buy_price"], marker="^", s=80, color="#16a34a", edgecolor="white", linewidth=0.7, label="Buy close", zorder=5)
        ax_price.scatter(trades_view["sell_date"], trades_view["sell_price"], marker="v", s=80, color="#dc2626", edgecolor="white", linewidth=0.7, label="Sell next day", zorder=5)
        for _, row in trades_view.iterrows():
            if pd.notna(row["sell_date"]):
                ax_price.plot([row["buy_date"], row["sell_date"]], [row["buy_price"], row["sell_price"]], color="#64748b", linewidth=0.9, alpha=0.65)
    ax_price.set_title(f"{args.stock_code} saved next-day model signals | {args.artifact_name}", fontsize=13)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", ncol=4, fontsize=9)

    if not scored_view.empty:
        ax_score.plot(scored_view["date"], scored_view["hit_score"], color="#2563eb", linewidth=1.2, label="Hit score")
        threshold = float(scored_view["threshold"].dropna().iloc[-1])
        ax_score.axhline(threshold, color="#ef4444", linestyle="--", linewidth=1.0, label="Threshold")
    ax_score.set_ylabel("Score")
    ax_score.set_ylim(0, 1.02)
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="upper left", fontsize=9)

    if trades_view.empty:
        ax_equity.axhline(1.0, color="#0f766e", linewidth=1.4)
    else:
        equity_view = pd.concat(
            [pd.DataFrame({"buy_date": [start], "equity": [1.0]}), trades_view[["buy_date", "equity"]]],
            ignore_index=True,
        )
        ax_equity.step(equity_view["buy_date"], equity_view["equity"], where="post", color="#0f766e", linewidth=1.6)
    ret = trades_view["net_return"].dropna()
    summary = {
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "daily_last_date": str(daily_view["date"].max().date()),
        "score_last_date": str(scored_view["date"].max().date()) if not scored_view.empty else None,
        "trades": int(len(ret)),
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "avg_return": float(ret.mean()) if len(ret) else np.nan,
        "compound_return": float((1.0 + ret).prod() - 1.0) if len(ret) else 0.0,
    }
    text = f"trades={summary['trades']}"
    if len(ret):
        text += f"  win={summary['win_rate']:.1%}  avg={summary['avg_return']:.3%}  compound={summary['compound_return']:.2%}"
    ax_equity.text(0.01, 0.95, text, transform=ax_equity.transAxes, va="top", fontsize=9)
    ax_equity.set_ylabel("Equity")
    ax_equity.set_xlabel("Date")
    ax_equity.grid(True, alpha=0.25)
    ax_equity.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_equity.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_equity.xaxis.get_major_locator()))

    out_png = out_dir / args.output_name
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    out_csv = out_dir / args.output_name.replace(".png", "_trades.csv")
    keep = [
        "buy_date", "buy_price", "sell_date", "sell_price", "net_return", "hit",
        "hit_score", "threshold", "equity",
    ]
    trades_view[[c for c in keep if c in trades_view.columns]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print({"plot": str(out_png), "trades_csv": str(out_csv), **summary})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot saved-model next-day signals")
    p.add_argument("--stock-code", default="002311.SZ")
    p.add_argument("--artifact-name", default="nextday_hit_50bps_lgbm_sector_v2")
    p.add_argument("--daily", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "trade_plots"))
    p.add_argument("--target-hit-bps", type=float, default=50.0)
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--sell-mode", choices=["target_or_close", "close"], default="target_or_close")
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-05-07")
    p.add_argument("--output-name", default="002311_nextday_trades_20260401_20260507.png")
    return p.parse_args()


if __name__ == "__main__":
    plot(parse_args())
