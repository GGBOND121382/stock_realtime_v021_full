#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot daily price, next-day trade signals, sell points, and equity curve."""
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


def load_best_trades(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(args.samples, parse_dates=["date", "next_date"]).sort_values("date")
    pred = pd.read_csv(args.predictions, parse_dates=["date"])
    pred = pred[
        (pred["feature_group"] == args.feature_group)
        & (pred["model_name"] == args.model_name)
        & (pred["split"] == args.split)
        & (pred["selected"] == 1)
    ].copy()
    if pred.empty:
        raise ValueError("no selected trades matched the requested model/group/split")

    cols = ["date", "open", "high", "low", "close", "next_date", "next_day_close"]
    trades = pred.merge(daily[[c for c in cols if c in daily.columns]], on="date", how="left")
    target_ret = args.target_hit_bps / 10000.0
    trades["buy_date"] = trades["date"]
    trades["buy_price"] = trades["close"]
    trades["sell_date"] = trades["next_date"]
    trades["sell_price"] = np.where(
        trades["trade_hit_label"].astype(int) == 1,
        trades["buy_price"] * (1.0 + target_ret),
        trades["next_day_close"],
    )
    trades["equity"] = (1.0 + trades["selected_return"].fillna(0.0)).cumprod()
    return daily, trades


def plot(args: argparse.Namespace) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    out_dir = ensure_dir(args.out_dir)
    daily, trades = load_best_trades(args)
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date) if args.end_date else pd.Timestamp.today().normalize()

    daily_view = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    trades_view = trades[(trades["buy_date"] >= start) & (trades["buy_date"] <= end)].copy()
    if daily_view.empty:
        raise ValueError(f"no daily rows in range {start.date()} to {end.date()}")

    fig, (ax_price, ax_equity) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    ax_price.plot(daily_view["date"], daily_view["close"], color="#1f2937", linewidth=1.4, label="Close")
    ax_price.fill_between(
        daily_view["date"],
        daily_view["low"],
        daily_view["high"],
        color="#cbd5e1",
        alpha=0.35,
        label="Daily high-low",
    )
    if not trades_view.empty:
        ax_price.scatter(
            trades_view["buy_date"],
            trades_view["buy_price"],
            marker="^",
            s=70,
            color="#16a34a",
            edgecolor="white",
            linewidth=0.6,
            label="Buy at close",
            zorder=5,
        )
        ax_price.scatter(
            trades_view["sell_date"],
            trades_view["sell_price"],
            marker="v",
            s=70,
            color="#dc2626",
            edgecolor="white",
            linewidth=0.6,
            label="Sell next day",
            zorder=5,
        )
        for _, row in trades_view.iterrows():
            if pd.notna(row["sell_date"]):
                ax_price.plot(
                    [row["buy_date"], row["sell_date"]],
                    [row["buy_price"], row["sell_price"]],
                    color="#64748b",
                    linewidth=0.8,
                    alpha=0.55,
                )

    ax_price.set_title(
        f"{args.stock_code} next-day trades | {args.feature_group} | {args.model_name}",
        fontsize=13,
    )
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", ncol=4, fontsize=9)

    all_trades = trades.copy()
    all_trades = all_trades[all_trades["buy_date"] <= end]
    before_start = all_trades[all_trades["buy_date"] < start]
    base_equity = float(before_start["equity"].iloc[-1]) if not before_start.empty else 1.0
    equity_view = trades_view[["buy_date", "equity"]].copy()
    if equity_view.empty:
        ax_equity.axhline(base_equity, color="#2563eb", linewidth=1.4)
    else:
        equity_view = pd.concat(
            [
                pd.DataFrame({"buy_date": [start], "equity": [base_equity]}),
                equity_view,
            ],
            ignore_index=True,
        )
        ax_equity.step(equity_view["buy_date"], equity_view["equity"], where="post", color="#2563eb", linewidth=1.6)
    ax_equity.set_ylabel("Equity")
    ax_equity.set_xlabel("Date")
    ax_equity.grid(True, alpha=0.25)
    ax_equity.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_equity.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_equity.xaxis.get_major_locator()))

    ret = trades_view["selected_return"].dropna()
    summary = {
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "trades": int(len(ret)),
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "avg_return": float(ret.mean()) if len(ret) else np.nan,
        "compound_return": float((1.0 + ret).prod() - 1.0) if len(ret) else 0.0,
    }
    ax_equity.text(
        0.01,
        0.95,
        f"trades={summary['trades']}  win={summary['win_rate']:.1%}  "
        f"avg={summary['avg_return']:.3%}  compound={summary['compound_return']:.2%}",
        transform=ax_equity.transAxes,
        va="top",
        fontsize=9,
    )

    out_png = out_dir / args.output_name
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_csv = out_dir / args.output_name.replace(".png", "_trades.csv")
    keep = [
        "buy_date", "buy_price", "sell_date", "sell_price", "selected_return",
        "trade_hit_label", "score", "chosen_threshold", "chosen_quantile", "equity",
    ]
    trades_view[[c for c in keep if c in trades_view.columns]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print({"plot": str(out_png), "trades_csv": str(out_csv), **summary})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot next-day trade signals and equity curve")
    p.add_argument("--stock-code", default="002714.SZ")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "hog_industry_features_out" / "training_samples_with_hog_industry.csv"))
    p.add_argument("--predictions", default=str(SAVED_DATA_DIR / "walk_forward_hog_refine_50bps" / "predictions_50bps.csv"))
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "trade_plots"))
    p.add_argument("--feature-group", default="all_no_ak")
    p.add_argument("--model-name", default="xgb_d4_500_lr002_mcw5")
    p.add_argument("--split", default="test")
    p.add_argument("--target-hit-bps", type=float, default=50.0)
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--output-name", default="002714_nextday_trades_2026.png")
    return p.parse_args()


if __name__ == "__main__":
    plot(parse_args())
