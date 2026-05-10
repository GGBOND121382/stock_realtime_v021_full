#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate model-filtered VWAP next-day rebound strategy.

Thresholds are chosen from validation predictions and applied to test
predictions to reduce look-ahead bias when selecting model filters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import vwap_nextday_rebound_backtest as bt


def load_predictions(valid_path: Path, test_path: Path) -> pd.DataFrame:
    usecols = ["date", "pred", "target"]
    valid = pd.read_csv(valid_path, usecols=usecols, parse_dates=["date"])
    test = pd.read_csv(test_path, usecols=usecols, parse_dates=["date"])
    valid["split"] = "valid"
    test["split"] = "test"
    return pd.concat([valid, test], ignore_index=True).rename(columns={"date": "signal_date", "pred": "model_pred", "target": "model_target"})


def metrics_row(trades: pd.DataFrame, ret_col: str, prefix: str) -> Dict[str, float]:
    m = bt.calc_metrics(trades, ret_col)
    return {f"{prefix}_{k}": v for k, v in m.items()}


def apply_filter(df: pd.DataFrame, spec: Dict, valid_pred: pd.Series) -> pd.DataFrame:
    out = df.copy()
    if spec["kind"] == "all":
        return out
    if spec["kind"] == "pred_gt":
        return out[out["model_pred"] > spec["threshold"]].copy()
    if spec["kind"] == "top_quantile":
        threshold = float(valid_pred.quantile(spec["quantile"]))
        return out[out["model_pred"] >= threshold].copy()
    if spec["kind"] == "top_quantile_vwap_band":
        threshold = float(valid_pred.quantile(spec["quantile"]))
        lo = spec["close_vs_vwap_min"]
        hi = spec["close_vs_vwap_max"]
        return out[(out["model_pred"] >= threshold) & (out["close_vs_vwap"] >= lo) & (out["close_vs_vwap"] <= hi)].copy()
    raise ValueError(spec)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate model-filtered VWAP rebound strategy")
    p.add_argument("--signal-samples", default="dual_opp_out_002714_v12/signal_samples.csv")
    p.add_argument("--valid-predictions", default="nextday_vwap_return_vwap_ref_noleak_out/valid_predictions.csv")
    p.add_argument("--test-predictions", default="nextday_vwap_return_vwap_ref_noleak_out/test_predictions.csv")
    p.add_argument("--out-dir", default="vwap_nextday_model_filter_eval_out")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    args = p.parse_args()

    out_dir = bt.ensure_dir(args.out_dir)
    intraday = bt.load_intraday(Path(args.signal_samples))
    daily = bt.build_daily(intraday)
    preds = load_predictions(Path(args.valid_predictions), Path(args.test_predictions))

    strategy_params = [
        {"name": "default_near20_rise30_trail30", "near_bps": 20, "min_rise_bps": 30, "trail_bps": 30},
        {"name": "grid_best_near50_rise30_trail50", "near_bps": 50, "min_rise_bps": 30, "trail_bps": 50},
    ]
    filter_specs: List[Dict] = [
        {"name": "all", "kind": "all"},
        {"name": "pred_gt_0", "kind": "pred_gt", "threshold": 0.0},
        {"name": "top50", "kind": "top_quantile", "quantile": 0.50},
        {"name": "top40", "kind": "top_quantile", "quantile": 0.60},
        {"name": "top30", "kind": "top_quantile", "quantile": 0.70},
        {"name": "top20", "kind": "top_quantile", "quantile": 0.80},
        {
            "name": "top40_close_vs_vwap_-1pct_to_0.5pct",
            "kind": "top_quantile_vwap_band",
            "quantile": 0.60,
            "close_vs_vwap_min": -0.010,
            "close_vs_vwap_max": 0.005,
        },
        {
            "name": "top30_close_vs_vwap_-1pct_to_0.5pct",
            "kind": "top_quantile_vwap_band",
            "quantile": 0.70,
            "close_vs_vwap_min": -0.010,
            "close_vs_vwap_max": 0.005,
        },
    ]

    rows = []
    all_trades_out = []
    for sp in strategy_params:
        trades = bt.backtest(
            intraday,
            daily,
            near_bps=sp["near_bps"],
            min_rise_bps=sp["min_rise_bps"],
            trail_bps=sp["trail_bps"],
            round_trip_cost_bps=args.round_trip_cost_bps,
            min_bars=40,
        )
        trades["signal_date"] = pd.to_datetime(trades["signal_date"])
        merged = trades.merge(preds, on="signal_date", how="inner")
        valid_pred = merged.loc[merged["split"] == "valid", "model_pred"]
        for spec in filter_specs:
            for split in ["valid", "test"]:
                base = merged[merged["split"] == split].copy()
                filt = apply_filter(base, spec, valid_pred)
                row = {
                    "strategy": sp["name"],
                    "filter": spec["name"],
                    "split": split,
                    "trades": int(len(filt)),
                    "pred_mean": float(filt["model_pred"].mean()) if len(filt) else np.nan,
                    "target_mean": float(filt["model_target"].mean()) if len(filt) else np.nan,
                    "close_vs_vwap_mean": float(filt["close_vs_vwap"].mean()) if len(filt) else np.nan,
                }
                row.update(metrics_row(filt, "net_return_trail_or_close", "net_trail"))
                row.update(metrics_row(filt, "net_return_close_only", "net_close"))
                rows.append(row)
            tagged = apply_filter(merged.copy(), spec, valid_pred)
            tagged["strategy"] = sp["name"]
            tagged["filter"] = spec["name"]
            all_trades_out.append(tagged)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "model_filter_summary.csv", index=False, encoding="utf-8-sig")
    if all_trades_out:
        pd.concat(all_trades_out, ignore_index=True).to_csv(out_dir / "model_filtered_trades.csv", index=False, encoding="utf-8-sig")
    summary = {
        "round_trip_cost_bps": args.round_trip_cost_bps,
        "valid_predictions": args.valid_predictions,
        "test_predictions": args.test_predictions,
        "outputs": {
            "summary": str(out_dir / "model_filter_summary.csv"),
            "trades": str(out_dir / "model_filtered_trades.csv"),
        },
        "top_test_by_net_trail_avg": result[result["split"] == "test"].sort_values("net_trail_avg_return", ascending=False).head(10).to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
