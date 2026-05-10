#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate next-day exit variants for model-filtered VWAP rebound trades."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import vwap_nextday_rebound_backtest as bt
from vwap_nextday_model_filter_eval import load_predictions


def build_entries(intraday: pd.DataFrame, daily: pd.DataFrame, near_bps: float, min_bars: int) -> pd.DataFrame:
    by_date = {d: x.sort_values("signal_time").copy() for d, x in intraday.groupby("trade_date", sort=True)}
    near = near_bps / 10000.0
    rows = []
    candidates = daily[daily["n_bars"] >= min_bars].copy()
    candidates["entry_signal"] = candidates["close"] <= candidates["daily_vwap"] * (1.0 + near)
    candidates["close_vs_vwap"] = candidates["close"] / candidates["daily_vwap"] - 1.0
    for _, day in candidates[candidates["entry_signal"]].iterrows():
        next_date = day["next_trade_date"]
        if pd.isna(next_date) or next_date not in by_date:
            continue
        nb = by_date[next_date]
        if len(nb) < min_bars:
            continue
        rows.append({
            "signal_date": day["trade_date"],
            "next_trade_date": next_date,
            "symbol": str(day["symbol"]).zfill(6),
            "daily_vwap": float(day["daily_vwap"]),
            "signal_close": float(day["close"]),
            "close_vs_vwap": float(day["close_vs_vwap"]),
        })
    return pd.DataFrame(rows)


def simulate_exit(next_bars: pd.DataFrame, entry_price: float, variant: Dict) -> Dict:
    stop = variant.get("stop_bps")
    take = variant.get("take_bps")
    time_exit = variant.get("time_exit")
    stop_price = entry_price * (1.0 - stop / 10000.0) if stop is not None else None
    take_price = entry_price * (1.0 + take / 10000.0) if take is not None else None

    for _, bar in next_bars.iterrows():
        t = pd.to_datetime(bar["signal_time"]).time()
        if stop_price is not None and float(bar["bar_low"]) <= stop_price:
            return {"sell_time": bar["signal_time"], "sell_price": stop_price, "reason": "stop"}
        if take_price is not None and float(bar["bar_high"]) >= take_price:
            return {"sell_time": bar["signal_time"], "sell_price": take_price, "reason": "take"}
        if time_exit is not None and t >= pd.Timestamp(time_exit).time():
            return {"sell_time": bar["signal_time"], "sell_price": float(bar["bar_close"]), "reason": f"time_{time_exit}"}

    last = next_bars.iloc[-1]
    return {"sell_time": last["signal_time"], "sell_price": float(last["bar_close"]), "reason": "next_close"}


def calc_variant_trades(entries: pd.DataFrame, intraday: pd.DataFrame, variant: Dict, cost_bps: float) -> pd.DataFrame:
    by_date = {d: x.sort_values("signal_time").copy() for d, x in intraday.groupby("trade_date", sort=True)}
    rows = []
    cost = cost_bps / 10000.0
    for _, e in entries.iterrows():
        nb = by_date[e["next_trade_date"]]
        ex = simulate_exit(nb, float(e["signal_close"]), variant)
        row = e.to_dict()
        row.update(ex)
        row["gross_return"] = row["sell_price"] / row["signal_close"] - 1.0
        row["net_return"] = row["gross_return"] - cost
        rows.append(row)
    return pd.DataFrame(rows)


def metric_prefix(trades: pd.DataFrame, prefix: str) -> Dict:
    m = bt.calc_metrics(trades, "net_return")
    return {f"{prefix}_{k}": v for k, v in m.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate exit variants for model-filtered VWAP trades")
    p.add_argument("--signal-samples", default="dual_opp_out_002714_v12/signal_samples.csv")
    p.add_argument("--valid-predictions", default="nextday_vwap_return_vwap_ref_noleak_out/valid_predictions.csv")
    p.add_argument("--test-predictions", default="nextday_vwap_return_vwap_ref_noleak_out/test_predictions.csv")
    p.add_argument("--out-dir", default="vwap_nextday_exit_variants_eval_out")
    p.add_argument("--near-bps", type=float, default=50.0)
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--filter-quantile", type=float, default=0.50, help="Use valid pred quantile threshold; 0.50 means top 50%.")
    args = p.parse_args()

    out_dir = bt.ensure_dir(args.out_dir)
    intraday = bt.load_intraday(Path(args.signal_samples))
    daily = bt.build_daily(intraday)
    entries = build_entries(intraday, daily, near_bps=args.near_bps, min_bars=40)
    entries["signal_date"] = pd.to_datetime(entries["signal_date"])
    preds = load_predictions(Path(args.valid_predictions), Path(args.test_predictions))
    merged = entries.merge(preds, on="signal_date", how="inner")
    threshold = float(merged.loc[merged["split"] == "valid", "model_pred"].quantile(args.filter_quantile))
    merged = merged[merged["model_pred"] >= threshold].copy()

    variants: List[Dict] = [
        {"name": "close_only"},
        {"name": "stop_50_close", "stop_bps": 50},
        {"name": "stop_80_close", "stop_bps": 80},
        {"name": "stop_100_close", "stop_bps": 100},
        {"name": "take_80_close", "take_bps": 80},
        {"name": "take_100_close", "take_bps": 100},
        {"name": "stop_80_take_120", "stop_bps": 80, "take_bps": 120},
        {"name": "stop_100_take_150", "stop_bps": 100, "take_bps": 150},
        {"name": "time_1030", "time_exit": "10:30"},
        {"name": "time_1130", "time_exit": "11:30"},
        {"name": "stop_80_time_1130", "stop_bps": 80, "time_exit": "11:30"},
    ]

    rows = []
    all_trades = []
    for variant in variants:
        trades = calc_variant_trades(merged, intraday, variant, args.round_trip_cost_bps)
        trades["variant"] = variant["name"]
        all_trades.append(trades)
        for split in ["valid", "test"]:
            part = trades[trades["split"] == split].copy()
            row = {
                "variant": variant["name"],
                "split": split,
                "trades": int(len(part)),
                "sell_reason_counts": part["reason"].value_counts().to_dict() if len(part) else {},
                "pred_mean": float(part["model_pred"].mean()) if len(part) else np.nan,
                "target_mean": float(part["model_target"].mean()) if len(part) else np.nan,
            }
            row.update(metric_prefix(part, "net"))
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "exit_variant_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_trades, ignore_index=True).to_csv(out_dir / "exit_variant_trades.csv", index=False, encoding="utf-8-sig")
    summary = {
        "near_bps": args.near_bps,
        "round_trip_cost_bps": args.round_trip_cost_bps,
        "filter_quantile": args.filter_quantile,
        "valid_threshold": threshold,
        "top_test_by_avg": result[result["split"] == "test"].sort_values("net_avg_return", ascending=False).head(10).to_dict(orient="records"),
        "outputs": {
            "summary": str(out_dir / "exit_variant_summary.csv"),
            "trades": str(out_dir / "exit_variant_trades.csv"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
