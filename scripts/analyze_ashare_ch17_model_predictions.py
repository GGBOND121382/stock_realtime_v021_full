#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze individual Chapter 17 prediction columns and selected ensembles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy.stats import spearmanr

from run_ashare_ch17_backtest_profiles import (
    CostConfig,
    TOP_NS,
    build_execution_panel,
    ensure_dir,
    load_predictions,
    load_universe,
    run_open_rebalance_profile,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREDS = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_reproduce" / "results" / "test_preds.h5"
DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce" / "model_data.h5"
DEFAULT_BEST_PARAMS = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_reproduce" / "results" / "best_params.csv"
DEFAULT_OUT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_reproduce" / "model_diagnostics"
DEFAULT_CACHE = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce" / "baostock_qfq_daily_cache"
DEFAULT_UNIVERSE = PROJECT_DIR / "saved_data" / "ashare_static_universe" / "07_universe_allA_top1000_static.csv"
EXPECTED_OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]


def signal_panel(predictions: pd.DataFrame, columns: Iterable[int]) -> pd.DataFrame:
    signal = predictions.loc[:, list(columns)].mean(axis=1).rename("signal")
    panel = signal.unstack("symbol").sort_index()
    panel.index.name = "date"
    return panel


def load_actuals(model_data_path: Path, prediction_index: pd.MultiIndex) -> pd.Series:
    data = pd.read_hdf(model_data_path, "model_data")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")
    actual = data["r01_fwd"].dropna()
    symbols = pd.Series(actual.index.get_level_values("symbol").astype(str))
    normalized = symbols.str.extract(r"(\d{6})", expand=False).fillna(symbols.str[-6:])
    actual.index = pd.MultiIndex.from_arrays(
        [
            normalized,
            pd.to_datetime(actual.index.get_level_values("date")).normalize(),
        ],
        names=["symbol", "date"],
    )
    return actual.reindex(prediction_index)


def compute_daily_ic(predictions: pd.DataFrame, actual: pd.Series, out_dir: Path) -> pd.DataFrame:
    rows = []
    daily_frames = []
    for col in predictions.columns:
        pair = pd.concat([predictions[col].rename("prediction"), actual.rename("actual")], axis=1).dropna()
        daily = pair.groupby(level="date", group_keys=False).apply(lambda x: spearmanr(x["actual"], x["prediction"])[0]).rename("daily_ic")
        daily_frames.append(daily.to_frame().assign(model=str(col)).reset_index())
        rows.append(
            {
                "model": str(col),
                "n_days": int(daily.notna().sum()),
                "mean_daily_ic": float(daily.mean()),
                "median_daily_ic": float(daily.median()),
                "positive_ic_rate": float((daily > 0).mean()),
            }
        )
    by_day = pd.concat(daily_frames, ignore_index=True)
    by_day.to_csv(out_dir / "single_model_daily_ic.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows).sort_values("median_daily_ic", ascending=False)
    summary.to_csv(out_dir / "single_model_ic_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def enrich_with_params(summary: pd.DataFrame, best_params_path: Path, out_dir: Path) -> pd.DataFrame:
    if not best_params_path.exists():
        return summary
    params = pd.read_csv(best_params_path).reset_index().rename(columns={"index": "model"})
    params["model"] = params["model"].astype(str)
    enriched = summary.merge(params, on="model", how="left")
    enriched.to_csv(out_dir / "single_model_ic_summary_with_params.csv", index=False, encoding="utf-8-sig")
    return enriched


def run_single_model_backtests(predictions: pd.DataFrame, cache_dir: Path, universe_path: Path, out_dir: Path) -> pd.DataFrame:
    universe = load_universe(universe_path)
    execution = build_execution_panel(cache_dir, universe, sorted(predictions.index.get_level_values("symbol").unique()))
    execution.to_hdf(out_dir / "execution_panel_for_diagnostics.h5", "execution", mode="w", format="table")
    leaderboard = []
    for col in predictions.columns:
        panel = signal_panel(predictions, [col])
        for n in TOP_NS:
            leaderboard.append(
                run_open_rebalance_profile(
                    f"single_model_{col}_open_rebalance_mainboard_limit_hold_cost",
                    panel,
                    execution,
                    out_dir / "single_model_backtests" / f"model_{col}" / f"top{n}",
                    n,
                    enforce_limits=True,
                    cost=CostConfig(),
                )
            )
    for name, cols in [("ensemble_first3", list(predictions.columns[:3])), ("ensemble_all5", list(predictions.columns[:5]))]:
        panel = signal_panel(predictions, cols)
        for n in TOP_NS:
            leaderboard.append(
                run_open_rebalance_profile(
                    f"{name}_open_rebalance_mainboard_limit_hold_cost",
                    panel,
                    execution,
                    out_dir / "ensemble_backtests" / name / f"top{n}",
                    n,
                    enforce_limits=True,
                    cost=CostConfig(),
                )
            )
    if 0 in predictions.columns:
        panel = signal_panel(predictions, [0])
        for sell_rank_gt in [50, 75]:
            leaderboard.append(
                run_open_rebalance_profile(
                    f"model_0_rank25_sell_gt_{sell_rank_gt}_open_rebalance_mainboard_limit_hold_cost",
                    panel,
                    execution,
                    out_dir / "rank_band_backtests" / "model_0" / f"buy_top25_sell_gt_{sell_rank_gt}",
                    25,
                    enforce_limits=True,
                    cost=CostConfig(),
                    sell_rank_gt=sell_rank_gt,
                )
            )
    board = pd.DataFrame(leaderboard)
    board.to_csv(out_dir / "single_model_backtest_leaderboard.csv", index=False, encoding="utf-8-sig")
    return board


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze individual Chapter 17 NN prediction columns")
    p.add_argument("--predictions", default=str(DEFAULT_PREDS))
    p.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    p.add_argument("--best-params", default=str(DEFAULT_BEST_PARAMS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--skip-backtests", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    predictions = load_predictions(Path(args.predictions))
    actual = load_actuals(Path(args.model_data), predictions.index)
    ic_summary = compute_daily_ic(predictions, actual, out_dir)
    enriched = enrich_with_params(ic_summary, Path(args.best_params), out_dir)
    backtest = pd.DataFrame()
    if not args.skip_backtests:
        backtest = run_single_model_backtests(predictions, Path(args.cache_dir), Path(args.universe), out_dir)
    report = {
        "out_dir": str(out_dir.resolve()),
        "prediction_columns": [int(c) if isinstance(c, int) else str(c) for c in predictions.columns],
        "ic_rows": int(len(enriched)),
        "backtest_profiles": int(len(backtest)),
    }
    (out_dir / "diagnostics_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
