#!/usr/bin/env python3
"""Run a dependency-light Chapter 17 backtest.

This is not a Zipline replacement. It is a deterministic local backtest of the
same prediction-to-portfolio data path used by the Chapter 17 notebook:

- read results/test_preds.h5::predictions
- use the mean of the first three model columns as the signal
- read data/assets.h5::quandl/wiki/prices
- trade prediction symbols using adjusted open prices
- hold one day, long top positive signals and optionally short bottom signals

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
    trade_timing: str
    period: str
    repo_dir: str
    prediction_rows: int
    prediction_symbols: int
    trading_days: int
    rebalance_days: int
    average_longs: float
    average_shorts: float
    average_blocked_long_entries: float
    average_blocked_long_exits: float
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
    trade_timing: str,
    portfolio_mode: str,
    execution_model: str,
    repo_dir: Path,
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    blocked_long_entries: pd.DataFrame | None,
    blocked_long_exits: pd.DataFrame | None,
    prediction_rows: int,
    prediction_symbols: int,
    n_longs: int,
    n_shorts: int,
    min_positions: int,
    output_dir: Path,
) -> BacktestSummary:
    _require(portfolio_mode in {"long_short", "long_only"}, f"unsupported portfolio mode: {portfolio_mode}")
    _require(execution_model in {"ideal_open", "skip_open_limit"}, f"unsupported execution model: {execution_model}")
    common_dates = signal_panel.index.intersection(forward_returns.index)
    signal_panel = signal_panel.loc[common_dates]
    forward_returns = forward_returns.loc[common_dates]
    if blocked_long_entries is not None:
        blocked_long_entries = blocked_long_entries.reindex(index=common_dates, columns=signal_panel.columns).fillna(False).astype(bool)
    if blocked_long_exits is not None:
        blocked_long_exits = blocked_long_exits.reindex(index=common_dates, columns=signal_panel.columns).fillna(False).astype(bool)

    rows = []
    for date, scores in signal_panel.iterrows():
        scores = scores.dropna()
        if scores.empty:
            continue
        longs = scores[scores > 0].nlargest(min(n_longs, int((scores > 0).sum()))).index
        shorts = scores[scores < 0].nsmallest(min(n_shorts, int((scores < 0).sum()))).index
        if len(longs) < min_positions:
            continue
        if portfolio_mode == "long_short" and len(shorts) < min_positions:
            continue
        day_returns = forward_returns.loc[date]
        long_candidates = pd.Index(longs)
        blocked_entry_count = 0
        blocked_exit_count = 0
        if execution_model == "skip_open_limit":
            entry_blocked = blocked_long_entries.loc[date, long_candidates].astype(bool) if blocked_long_entries is not None else pd.Series(False, index=long_candidates)
            exit_blocked = blocked_long_exits.loc[date, long_candidates].astype(bool) if blocked_long_exits is not None else pd.Series(False, index=long_candidates)
            blocked_entry_count = int(entry_blocked.sum())
            blocked_exit_count = int(exit_blocked.sum())
            long_candidates = long_candidates[~(entry_blocked | exit_blocked).to_numpy()]
        long_ret = day_returns.reindex(long_candidates).dropna()
        short_ret = day_returns.reindex(shorts).dropna() if portfolio_mode == "long_short" else pd.Series(dtype=float)
        if len(long_ret) < min_positions:
            continue
        if portfolio_mode == "long_short" and len(short_ret) < min_positions:
            continue
        short_mean = float(short_ret.mean()) if portfolio_mode == "long_short" else 0.0
        portfolio_return = float(long_ret.mean() - short_mean)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "portfolio_return": portfolio_return,
                "long_return": float(long_ret.mean()),
                "short_return": short_mean,
                "long_count": int(len(long_ret)),
                "short_count": int(len(short_ret)),
                "blocked_long_entry_count": blocked_entry_count,
                "blocked_long_exit_count": blocked_exit_count,
            }
        )

    _require(rows, f"{strategy}: local backtest produced no rebalance days")
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    returns = daily["portfolio_return"].replace([np.inf, -np.inf], np.nan).dropna()
    _require(not returns.empty, f"{strategy}: local backtest returns are empty after cleaning")
    daily.to_csv(output_dir / f"ch17_local_backtest_daily_returns_{portfolio_mode}_{execution_model}_L{n_longs}_S{n_shorts}_{trade_timing}_{strategy}.csv")

    return _summarize_returns(
        strategy=strategy,
        trade_timing=f"{portfolio_mode}_{execution_model}_{trade_timing}",
        period="all",
        repo_dir=repo_dir,
        prediction_rows=prediction_rows,
        prediction_symbols=prediction_symbols,
        trading_days=len(common_dates),
        daily=daily,
        returns=returns,
    )


def _summarize_returns(
    strategy: str,
    trade_timing: str,
    period: str,
    repo_dir: Path,
    prediction_rows: int,
    prediction_symbols: int,
    trading_days: int,
    daily: pd.DataFrame,
    returns: pd.Series,
) -> BacktestSummary:
    _require(not returns.empty, f"{strategy}/{trade_timing}/{period}: no returns to summarize")
    daily = daily.reindex(returns.index)
    cumulative_return = float((1.0 + returns).prod() - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else float("nan")
    sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252.0)) if len(returns) > 1 and returns.std(ddof=1) else float("nan")
    best_day = returns.idxmax()
    worst_day = returns.idxmin()
    return BacktestSummary(
        strategy=strategy,
        trade_timing=trade_timing,
        period=period,
        repo_dir=str(repo_dir),
        prediction_rows=int(prediction_rows),
        prediction_symbols=int(prediction_symbols),
        trading_days=int(trading_days),
        rebalance_days=int(len(returns)),
        average_longs=float(daily["long_count"].mean()),
        average_shorts=float(daily["short_count"].mean()),
        average_blocked_long_entries=float(daily.get("blocked_long_entry_count", pd.Series(0, index=daily.index)).mean()),
        average_blocked_long_exits=float(daily.get("blocked_long_exit_count", pd.Series(0, index=daily.index)).mean()),
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


def _period_summaries(
    summary_all: BacktestSummary,
    repo_dir: Path,
    daily_path: Path,
    live_start_date: pd.Timestamp,
) -> list[BacktestSummary]:
    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    returns = daily["portfolio_return"].replace([np.inf, -np.inf], np.nan).dropna()
    out = [summary_all]
    period_specs = {
        "is_before_live": returns.index < live_start_date,
        "oos_from_live": returns.index >= live_start_date,
    }
    for period, mask in period_specs.items():
        period_returns = returns.loc[mask]
        if period_returns.empty:
            continue
        out.append(
            _summarize_returns(
                strategy=summary_all.strategy,
                trade_timing=summary_all.trade_timing,
                period=period,
                repo_dir=repo_dir,
                prediction_rows=summary_all.prediction_rows,
                prediction_symbols=summary_all.prediction_symbols,
                trading_days=int(mask.sum()),
                daily=daily,
                returns=period_returns,
            )
        )
    return out


def run_local_backtest(
    repo_dir: Path,
    output_dir: Path,
    portfolio_mode: str,
    execution_model: str,
    n_longs: int,
    n_shorts: int,
    min_positions: int,
    live_start_date: pd.Timestamp,
    limit_pct: float,
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
    _require("adj_close" in prices.columns, "prices missing adj_close column")
    price_symbols = pd.Index(prices.index.get_level_values("ticker").unique())
    missing_symbols = sorted(set(symbols) - set(price_symbols))
    _require(not missing_symbols, f"prediction symbols missing from prices: {missing_symbols[:20]}")

    prices = prices.swaplevel().sort_index()
    prices.index.names = ["symbol", "date"]
    raw_open_prices = (
        prices.loc[(symbols, slice("2015", "2018")), "adj_open"]
        .unstack("symbol")
        .sort_index()
    )
    raw_close_prices = (
        prices.loc[(symbols, slice("2015", "2018")), "adj_close"]
        .unstack("symbol")
        .sort_index()
    )
    _require(not raw_open_prices.empty, "trade prices are empty")
    missing_price_columns = sorted(set(symbols) - set(raw_open_prices.columns))
    _require(not missing_price_columns, f"trade price columns missing symbols: {missing_price_columns[:20]}")

    forward_returns_by_timing = {
        # Dangerous baseline: signal dated t trades at t open and exits t+1 open.
        "leaky_same_day_open": raw_open_prices.shift(-1).divide(raw_open_prices).subtract(1.0),
        # Safe path matching the Alphalens next-open convention: signal dated t
        # can only trade at t+1 open and exits at t+2 open.
        "safe_next_open": raw_open_prices.shift(-2).divide(raw_open_prices.shift(-1)).subtract(1.0),
        # Equivalent Zipline-style stress test: shift the signal one date later,
        # then use same-day open-to-next-open returns.
        "shifted_signal_same_day_open": raw_open_prices.shift(-1).divide(raw_open_prices).subtract(1.0),
    }
    previous_close = raw_close_prices.shift(1)
    open_return_from_previous_close = raw_open_prices.divide(previous_close).subtract(1.0)
    open_limit_up = open_return_from_previous_close >= limit_pct
    open_limit_down = open_return_from_previous_close <= -limit_pct
    blocked_long_entries_by_timing = {
        "leaky_same_day_open": open_limit_up,
        "safe_next_open": open_limit_up.shift(-1),
        "shifted_signal_same_day_open": open_limit_up,
    }
    blocked_long_exits_by_timing = {
        "leaky_same_day_open": open_limit_down.shift(-1),
        "safe_next_open": open_limit_down.shift(-2),
        "shifted_signal_same_day_open": open_limit_down.shift(-1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies: dict[str, pd.DataFrame] = {}
    for col in predictions_raw.columns:
        label = f"model_{col}"
        strategies[label] = predictions_raw[col].to_frame("prediction").unstack("symbol").prediction.sort_index()
    strategies["ensemble_first3"] = first_signal_panel
    strategies["ensemble_all5"] = predictions_raw.mean(axis=1).to_frame("prediction").unstack("symbol").prediction.sort_index()

    summaries = []
    for trade_timing, forward_returns in forward_returns_by_timing.items():
        for strategy, signal_panel in strategies.items():
            if trade_timing == "shifted_signal_same_day_open":
                signal_panel = signal_panel.shift(1)
            summary_all = _run_signal_backtest(
                    strategy=strategy,
                    trade_timing=trade_timing,
                    portfolio_mode=portfolio_mode,
                    execution_model=execution_model,
                    repo_dir=repo_dir,
                    signal_panel=signal_panel,
                    forward_returns=forward_returns,
                    blocked_long_entries=blocked_long_entries_by_timing[trade_timing] if execution_model == "skip_open_limit" else None,
                    blocked_long_exits=blocked_long_exits_by_timing[trade_timing] if execution_model == "skip_open_limit" else None,
                    prediction_rows=len(predictions_raw),
                    prediction_symbols=len(symbols),
                    n_longs=n_longs,
                    n_shorts=n_shorts,
                    min_positions=min_positions,
                    output_dir=output_dir,
                )
            daily_path = output_dir / f"ch17_local_backtest_daily_returns_{portfolio_mode}_{execution_model}_L{n_longs}_S{n_shorts}_{trade_timing}_{strategy}.csv"
            summaries.extend(_period_summaries(summary_all, repo_dir, daily_path, live_start_date))
    leaderboard = pd.DataFrame([asdict(item) for item in summaries]).sort_values(
        ["sharpe", "cumulative_return"],
        ascending=False,
    )
    output_stem = f"ch17_local_backtest_{portfolio_mode}_{execution_model}_L{n_longs}_S{n_shorts}"
    summary_json = json.dumps([asdict(item) for item in summaries], indent=2)
    leaderboard.to_csv(output_dir / f"{output_stem}_leaderboard.csv", index=False)
    (output_dir / f"{output_stem}_summary.json").write_text(summary_json, encoding="utf-8")
    if execution_model == "ideal_open":
        legacy_output_stem = f"ch17_local_backtest_{portfolio_mode}_L{n_longs}_S{n_shorts}"
        legacy_leaderboard = leaderboard.copy()
        legacy_prefix = f"{portfolio_mode}_ideal_open_"
        legacy_replacement = f"{portfolio_mode}_"
        legacy_leaderboard["trade_timing"] = legacy_leaderboard["trade_timing"].str.replace(
            legacy_prefix,
            legacy_replacement,
            regex=False,
        )
        legacy_summaries = []
        for item in summaries:
            row = asdict(item)
            row["trade_timing"] = row["trade_timing"].replace(legacy_prefix, legacy_replacement)
            legacy_summaries.append(row)
        legacy_leaderboard.to_csv(output_dir / f"{legacy_output_stem}_leaderboard.csv", index=False)
        (output_dir / f"{legacy_output_stem}_summary.json").write_text(
            json.dumps(legacy_summaries, indent=2),
            encoding="utf-8",
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parent / "machine-learning-for-trading")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--portfolio-mode", choices=["long_short", "long_only"], default="long_short")
    parser.add_argument("--execution-model", choices=["ideal_open", "skip_open_limit"], default="ideal_open")
    parser.add_argument("--n-longs", type=int, default=25)
    parser.add_argument("--n-shorts", type=int, default=25)
    parser.add_argument("--min-positions", type=int, default=10)
    parser.add_argument("--live-start-date", type=pd.Timestamp, default=pd.Timestamp("2016-11-30"))
    parser.add_argument("--limit-pct", type=float, default=0.095)
    args = parser.parse_args()

    summaries = run_local_backtest(
        repo_dir=args.repo_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        portfolio_mode=args.portfolio_mode,
        execution_model=args.execution_model,
        n_longs=args.n_longs,
        n_shorts=args.n_shorts,
        min_positions=args.min_positions,
        live_start_date=args.live_start_date,
        limit_pct=args.limit_pct,
    )
    print(json.dumps([asdict(item) for item in summaries], indent=2))
    print("CH17 local dependency-light backtest passed")


if __name__ == "__main__":
    main()
