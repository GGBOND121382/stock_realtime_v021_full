#!/usr/bin/env python3
"""Run a dependency-light Chapter 17 long/short backtest.

This is not a Zipline replacement. It is a deterministic local backtest of the
same prediction-to-portfolio data path used by the Chapter 17 notebook:

- read results/test_preds.h5::predictions
- use the mean of the first three model columns as the signal
- read data/assets.h5::quandl/wiki/prices
- trade prediction symbols using adjusted open prices
- hold one day, long top positive signals and short bottom negative signals

It intentionally avoids Zipline, pyfolio, alphalens, and trading_calendars so
the core data path can be run even when those packages are not installable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    strategy: str
    repo_dir: str
    prediction_rows: int
    prediction_symbols: int
    trading_days: int
    rebalance_days: int
    average_longs: float
    average_shorts: float
    win_rate: float
    mean_daily_return: float
    median_daily_return: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    best_day: str
    best_day_return: float
    worst_day: str
    worst_day_return: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _annualized_return(cumulative_return: float, days: int) -> float:
    if days <= 0:
        return float("nan")
    if cumulative_return <= -1:
        return -1.0
    return float((1.0 + cumulative_return) ** (252.0 / days) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    return float(drawdown.min())


def _run_signal_backtest(
    strategy: str,
    repo_dir: Path,
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    prediction_rows: int,
    prediction_symbols: int,
    n_longs: int,
    n_shorts: int,
    min_positions: int,
    output_dir: Path,
) -> BacktestSummary:
    common_dates = signal_panel.index.intersection(forward_returns.index)
    signal_panel = signal_panel.loc[common_dates]
    forward_returns = forward_returns.loc[common_dates]

    rows = []
    for date, scores in signal_panel.iterrows():
        scores = scores.dropna()
        if scores.empty:
            continue
        longs = scores[scores > 0].nlargest(min(n_longs, int((scores > 0).sum()))).index
        shorts = scores[scores < 0].nsmallest(min(n_shorts, int((scores < 0).sum()))).index
        if len(longs) <= min_positions or len(shorts) <= min_positions:
            continue
        day_returns = forward_returns.loc[date]
        long_ret = day_returns.reindex(longs).dropna()
        short_ret = day_returns.reindex(shorts).dropna()
        if len(long_ret) <= min_positions or len(short_ret) <= min_positions:
            continue
        portfolio_return = float(long_ret.mean() - short_ret.mean())
        rows.append(
            {
                "date": pd.Timestamp(date),
                "portfolio_return": portfolio_return,
                "long_return": float(long_ret.mean()),
                "short_return": float(short_ret.mean()),
                "long_count": int(len(long_ret)),
                "short_count": int(len(short_ret)),
            }
        )

    _require(rows, f"{strategy}: local backtest produced no rebalance days")
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    returns = daily["portfolio_return"].replace([np.inf, -np.inf], np.nan).dropna()
    _require(not returns.empty, f"{strategy}: local backtest returns are empty after cleaning")

    cumulative_return = float((1.0 + returns).prod() - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else float("nan")
    sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252.0)) if len(returns) > 1 and returns.std(ddof=1) else float("nan")
    best_day = returns.idxmax()
    worst_day = returns.idxmin()
    daily.to_csv(output_dir / f"ch17_local_backtest_daily_returns_{strategy}.csv")
    return BacktestSummary(
        strategy=strategy,
        repo_dir=str(repo_dir),
        prediction_rows=int(prediction_rows),
        prediction_symbols=int(prediction_symbols),
        trading_days=int(len(common_dates)),
        rebalance_days=int(len(returns)),
        average_longs=float(daily["long_count"].mean()),
        average_shorts=float(daily["short_count"].mean()),
        win_rate=float((returns > 0).mean()),
        mean_daily_return=float(returns.mean()),
        median_daily_return=float(returns.median()),
        cumulative_return=cumulative_return,
        annualized_return=_annualized_return(cumulative_return, len(returns)),
        annualized_volatility=vol,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(returns),
        best_day=pd.Timestamp(best_day).strftime("%Y-%m-%d"),
        best_day_return=float(returns.loc[best_day]),
        worst_day=pd.Timestamp(worst_day).strftime("%Y-%m-%d"),
        worst_day_return=float(returns.loc[worst_day]),
    )


def run_local_backtest(
    repo_dir: Path,
    output_dir: Path,
    n_longs: int,
    n_shorts: int,
    min_positions: int,
) -> list[BacktestSummary]:
    assets_path = repo_dir / "data" / "assets.h5"
    predictions_path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
    _require(assets_path.exists(), f"missing assets file: {assets_path}")
    _require(predictions_path.exists(), f"missing predictions file: {predictions_path}")

    predictions_raw = pd.read_hdf(predictions_path, "predictions")
    _require(list(predictions_raw.index.names) == ["symbol", "date"], f"unexpected predictions index: {predictions_raw.index.names}")
    _require(predictions_raw.shape[1] >= 3, f"predictions needs at least 3 columns; got {predictions_raw.shape[1]}")
    _require(not predictions_raw.index.has_duplicates, "predictions index has duplicate symbol/date rows")

    first_signal = predictions_raw.iloc[:, :3].mean(axis=1).to_frame("prediction")
    first_signal_panel = first_signal.unstack("symbol").prediction.sort_index()
    symbols = pd.Index(first_signal_panel.columns)

    prices = pd.read_hdf(assets_path, "quandl/wiki/prices")
    _require(list(prices.index.names) == ["date", "ticker"], f"unexpected prices index: {prices.index.names}")
    _require("adj_open" in prices.columns, "prices missing adj_open column")
    price_symbols = pd.Index(prices.index.get_level_values("ticker").unique())
    missing_symbols = sorted(set(symbols) - set(price_symbols))
    _require(not missing_symbols, f"prediction symbols missing from prices: {missing_symbols[:20]}")

    prices = prices.swaplevel().sort_index()
    prices.index.names = ["symbol", "date"]
    trade_prices = (
        prices.loc[(symbols, slice("2015", "2018")), "adj_open"]
        .unstack("symbol")
        .sort_index()
        .shift(-1)
    )
    _require(not trade_prices.empty, "trade prices are empty")
    missing_price_columns = sorted(set(symbols) - set(trade_prices.columns))
    _require(not missing_price_columns, f"trade price columns missing symbols: {missing_price_columns[:20]}")

    forward_returns = trade_prices.shift(-1).divide(trade_prices).subtract(1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies: dict[str, pd.DataFrame] = {}
    for col in predictions_raw.columns:
        label = f"model_{col}"
        strategies[label] = predictions_raw[col].to_frame("prediction").unstack("symbol").prediction.sort_index()
    strategies["ensemble_first3"] = first_signal_panel
    strategies["ensemble_all5"] = predictions_raw.mean(axis=1).to_frame("prediction").unstack("symbol").prediction.sort_index()

    summaries = [
        _run_signal_backtest(
            strategy=strategy,
            repo_dir=repo_dir,
            signal_panel=signal_panel,
            forward_returns=forward_returns,
            prediction_rows=len(predictions_raw),
            prediction_symbols=len(symbols),
            n_longs=n_longs,
            n_shorts=n_shorts,
            min_positions=min_positions,
            output_dir=output_dir,
        )
        for strategy, signal_panel in strategies.items()
    ]
    leaderboard = pd.DataFrame([asdict(item) for item in summaries]).sort_values(
        ["sharpe", "cumulative_return"],
        ascending=False,
    )
    leaderboard.to_csv(output_dir / "ch17_local_backtest_leaderboard.csv", index=False)
    (output_dir / "ch17_local_backtest_summary.json").write_text(
        json.dumps([asdict(item) for item in summaries], indent=2),
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parent / "machine-learning-for-trading")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--n-longs", type=int, default=25)
    parser.add_argument("--n-shorts", type=int, default=25)
    parser.add_argument("--min-positions", type=int, default=10)
    args = parser.parse_args()

    summaries = run_local_backtest(
        repo_dir=args.repo_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        n_longs=args.n_longs,
        n_shorts=args.n_shorts,
        min_positions=args.min_positions,
    )
    print(json.dumps([asdict(item) for item in summaries], indent=2))
    print("CH17 local dependency-light backtest passed")


if __name__ == "__main__":
    main()
