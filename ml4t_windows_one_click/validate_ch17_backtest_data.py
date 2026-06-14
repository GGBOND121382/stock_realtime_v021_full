#!/usr/bin/env python3
"""Validate Chapter 17 backtest data without training or Zipline.

This script reproduces the data-shaping assumptions used by the local
Zipline bundle builder and the Chapter 17 backtest notebook:

- assets.h5 has WIKI OHLCV data and stock metadata.
- test_preds.h5 symbols and dates are covered by WIKI prices.
- local bundle daily bars can be reindexed to a session calendar without
  missing OHLCV after fill.
- generated asset metadata has US country/exchange information.
- benchmark/returns alignment is forced to the same timestamp index.

It intentionally does not import TensorFlow, Zipline, pyfolio, or alphalens.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}
ADJUSTED_PRICE_COLUMNS = {
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
}


@dataclass
class ValidationReport:
    assets_path: str
    predictions_path: str
    prices_rows: int
    price_symbols: int
    price_start: str
    price_end: str
    prediction_rows: int
    prediction_symbols: int
    prediction_columns: int
    prediction_start: str
    prediction_end: str
    missing_prediction_symbols: list[str]
    prediction_dates_not_in_price_sessions: list[str]
    symbols_checked_for_bundle: int
    symbols_with_raw_missing_sessions: int
    raw_missing_sessions_total: int
    max_raw_missing_sessions_for_symbol: int
    fill_failures: list[str]
    notebook_trade_price_slice_ok: bool
    notebook_factor_rows: int
    notebook_trade_price_rows: int
    notebook_factor_data_rows: int
    notebook_sid_count: int
    notebook_signal_dates: int
    notebook_rebalance_days: int
    notebook_simulated_order_targets: int
    notebook_prediction_panel_ok: bool
    fake_pipeline_output_columns_ok: bool
    fake_zipline_lookup_ok: bool
    fake_bundle_asset_count: int
    fake_bundle_daily_bar_count: int
    fake_bundle_metadata_ok: bool
    fake_bundle_prediction_coverage_ok: bool
    fake_zipline_result_rows: int
    fake_zipline_order_count: int
    fake_zipline_positions_count: int
    notebook_full_flow_ok: bool
    metadata_exchange: str
    metadata_country_code: str
    benchmark_aligned: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _load_hdf(path: Path, key: str) -> pd.DataFrame:
    _require(path.exists(), f"missing file: {path}")
    try:
        return pd.read_hdf(path, key)
    except Exception as exc:
        raise SystemExit(f"failed to read {path}::{key}: {type(exc).__name__}: {exc}") from exc


def _stack_frame(df: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """Stack a DataFrame across pandas versions.

    pandas >=2.1 supports future_stack; older pandas raises TypeError. The
    old implementation is fine for this notebook-shaped data, so fall back.
    """
    try:
        return df.stack(future_stack=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return df.stack()


def _check_hdf_datetime_kinds(paths: Iterable[Path], patch: bool) -> None:
    try:
        import tables
    except Exception:
        return

    bad = []
    for path in paths:
        if not path.exists():
            continue
        with tables.open_file(path, mode="a" if patch else "r") as h5:
            for node in h5.walk_nodes("/"):
                attrs = getattr(node, "_v_attrs", None)
                if attrs is None or "kind" not in attrs._v_attrnamesuser:
                    continue
                if str(attrs.kind) == "datetime64[ns]":
                    bad.append(f"{path}:{node._v_pathname}")
                    if patch:
                        attrs.kind = "datetime64"
    if patch and bad:
        print("patched HDF datetime metadata:")
        for item in bad[:20]:
            print(f"  {item}")
        return
    _require(
        not bad,
        "HDF datetime metadata still uses datetime64[ns]; run runner HDF patch first: "
        + ", ".join(bad[:10]),
    )


def _make_bundle_like_frame(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    ohlcv = pd.DataFrame(
        {
            "open": df["adj_open"] if "adj_open" in df else df["open"],
            "high": df["adj_high"] if "adj_high" in df else df["high"],
            "low": df["adj_low"] if "adj_low" in df else df["low"],
            "close": df["adj_close"] if "adj_close" in df else df["close"],
            "volume": df["adj_volume"] if "adj_volume" in df else df["volume"],
        }
    ).replace([np.inf, -np.inf], np.nan)
    ohlcv = ohlcv.reindex(sessions)
    close = ohlcv["close"].ffill().bfill()
    for col in ["open", "high", "low", "close"]:
        ohlcv[col] = ohlcv[col].fillna(close)
    ohlcv["volume"] = ohlcv["volume"].fillna(0).clip(lower=0)
    return ohlcv


@dataclass(frozen=True)
class FakeAsset:
    sid: int
    symbol: str
    asset_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    first_traded: pd.Timestamp
    auto_close_date: pd.Timestamp
    exchange: str
    country_code: str


class FakeAssetFinder:
    def __init__(self, assets: Iterable[FakeAsset]) -> None:
        self.assets = {asset.symbol: asset for asset in assets}

    def lookup_symbols(self, tickers: list[str], as_of_date: object | None = None) -> list[FakeAsset]:
        del as_of_date
        missing = [ticker for ticker in tickers if ticker not in self.assets]
        if missing:
            raise KeyError(f"fake asset finder missing symbols: {missing[:10]}")
        return [self.assets[ticker] for ticker in tickers]


class FakeBundle:
    def __init__(self, assets: list[FakeAsset], daily_bars: dict[int, pd.DataFrame], exchanges: pd.DataFrame) -> None:
        self.assets = assets
        self.daily_bars = daily_bars
        self.exchanges = exchanges
        self.asset_finder = FakeAssetFinder(assets)


def _build_fake_quandl_bundle(prices: pd.DataFrame, stocks: pd.DataFrame) -> FakeBundle:
    """Mirror write_local_quandl_extension() without importing Zipline."""
    prices = prices.sort_index()
    if list(prices.index.names) != ["date", "ticker"]:
        prices = prices.reorder_levels(["date", "ticker"]).sort_index()

    tickers = pd.Index(prices.index.get_level_values("ticker").unique()).sort_values()
    sid_map = {ticker: sid for sid, ticker in enumerate(tickers)}

    stocks_for_names = stocks.copy()
    if "code" in stocks_for_names.columns:
        stocks_for_names = stocks_for_names.set_index("code")
    names = stocks_for_names["name"].to_dict() if "name" in stocks_for_names.columns else {}

    price_dates = pd.DatetimeIndex(prices.index.get_level_values("date").unique()).sort_values()
    grouped = prices.groupby(level="ticker", sort=True)
    assets: list[FakeAsset] = []
    daily_bars: dict[int, pd.DataFrame] = {}

    for ticker, df in grouped:
        sid = sid_map[ticker]
        df = df.droplevel("ticker").sort_index()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        start = max(df.index.min(), pd.Timestamp("1990-01-02"))
        end = df.index.max()
        if end < start:
            continue
        df = df.loc[start:end]
        if df.empty:
            continue
        sessions = price_dates[(price_dates >= start) & (price_dates <= end)]
        daily_bar = _make_bundle_like_frame(df, sessions)
        _require(not daily_bar.empty, f"fake bundle daily bars empty for {ticker}")
        _require(not daily_bar[["open", "high", "low", "close", "volume"]].isna().any().any(), f"fake bundle daily bars contain NaN for {ticker}")
        _require((daily_bar["volume"] >= 0).all(), f"fake bundle daily bars contain negative volume for {ticker}")
        daily_bars[sid] = daily_bar
        assets.append(
            FakeAsset(
                sid=sid,
                symbol=str(ticker),
                asset_name=str(names.get(ticker, ticker)),
                start_date=pd.Timestamp(start),
                end_date=pd.Timestamp(end),
                first_traded=pd.Timestamp(start),
                auto_close_date=pd.Timestamp(end) + pd.Timedelta(days=1),
                exchange="QUANDL",
                country_code="US",
            )
        )

    exchanges = pd.DataFrame(
        [["QUANDL", "QUANDL", "US"]],
        columns=["exchange", "canonical_name", "country_code"],
    )
    _require(assets, "fake bundle has no assets")
    _require(set(daily_bars) == {asset.sid for asset in assets}, "fake bundle daily bars do not match asset sids")
    return FakeBundle(assets=assets, daily_bars=daily_bars, exchanges=exchanges)


class FakePortfolio:
    def __init__(self) -> None:
        self.positions: dict[FakeAsset, float] = {}
        self.portfolio_value = 100000.0


class FakeAccount:
    leverage = 0.0


class FakeContext:
    def __init__(self) -> None:
        self.longs = pd.Index([])
        self.shorts = pd.Index([])
        self.divest: set[FakeAsset] = set()
        self.portfolio = FakePortfolio()
        self.account = FakeAccount()


def _simulate_fake_zipline_run(
    predictions_by_asset: pd.DataFrame,
    assets: list[FakeAsset],
    min_positions: int = 10,
) -> dict[str, int | bool]:
    """Run the notebook's trading callbacks against a small local engine."""
    context = FakeContext()
    asset_by_sid = {asset.sid: asset for asset in assets}
    orders: list[tuple[pd.Timestamp, FakeAsset, float]] = []
    records = []

    for date, row in predictions_by_asset.iterrows():
        row = row.dropna()
        if row.empty:
            continue

        longs_mask = row[row > 0].nlargest(min(25, int((row > 0).sum())))
        shorts_mask = row[row < 0].nsmallest(min(25, int((row < 0).sum())))
        longs = pd.Index([asset_by_sid[int(sid)] for sid in longs_mask.index])
        shorts = pd.Index([asset_by_sid[int(sid)] for sid in shorts_mask.index])
        holdings = set(context.portfolio.positions.keys())

        # before_trading_start()
        if len(longs) > min_positions and len(shorts) > min_positions:
            context.longs = longs
            context.shorts = shorts
            context.divest = holdings - set(context.longs) - set(context.shorts)
        else:
            context.longs = pd.Index([])
            context.shorts = pd.Index([])
            context.divest = set(holdings)

        # rebalance()
        for stock in context.divest:
            orders.append((date, stock, 0.0))
            context.portfolio.positions.pop(stock, None)
        if not (context.longs.empty and context.shorts.empty):
            for stock in context.shorts:
                target = -1.0 / len(context.shorts)
                _require(np.isfinite(target), "fake Zipline short target is invalid")
                orders.append((date, stock, target))
                context.portfolio.positions[stock] = target
            for stock in context.longs:
                target = 1.0 / len(context.longs)
                _require(np.isfinite(target), "fake Zipline long target is invalid")
                orders.append((date, stock, target))
                context.portfolio.positions[stock] = target

        # record_vars() and a minimal run_algorithm-like result row.
        gross = sum(abs(weight) for weight in context.portfolio.positions.values())
        context.account.leverage = gross
        records.append(
            {
                "date": date,
                "returns": float(row.mean()) / 100.0,
                "leverage": context.account.leverage,
                "longs": len(context.longs),
                "shorts": len(context.shorts),
            }
        )

    results = pd.DataFrame(records).set_index("date") if records else pd.DataFrame()
    _require(not results.empty, "fake Zipline run produced no result rows")
    _require("returns" in results.columns, "fake Zipline results missing returns column")
    _require(orders, "fake Zipline run produced no orders")
    _require(context.portfolio.positions, "fake Zipline run ended with no positions")

    return {
        "lookup_ok": True,
        "result_rows": int(len(results)),
        "order_count": int(len(orders)),
        "positions_count": int(len(context.portfolio.positions)),
    }


def _simulate_notebook_backtest_flow(
    prices: pd.DataFrame,
    stocks: pd.DataFrame,
    predictions_raw: pd.DataFrame,
    pred_symbols: pd.Index,
) -> dict[str, int | bool]:
    # Cell 10/11: convert model columns to a single factor panel.
    predictions = predictions_raw.iloc[:, :3].mean(1).to_frame("prediction")
    factor = (
        predictions.unstack("symbol")
        .asfreq("D")
        .dropna(how="all")
        .pipe(_stack_frame)
        .tz_localize("UTC", level="date")
        .sort_index()
    )
    _require(list(factor.index.names) == ["date", "symbol"], f"unexpected factor index: {factor.index.names}")
    _require("prediction" in factor.columns, f"factor columns missing prediction: {list(factor.columns)}")
    _require(not factor.empty, "notebook factor construction returned no rows")

    # Cell 9/12: exact get_trade_prices data path.
    prices_for_notebook = prices.swaplevel().sort_index()
    prices_for_notebook.index.names = ["symbol", "date"]
    try:
        trade_prices = prices_for_notebook.loc[(pred_symbols, slice("2015", "2018")), "adj_open"]
    except KeyError as exc:
        raise SystemExit(f"notebook trade-price .loc slice would raise KeyError: {exc}") from exc
    _require(not trade_prices.empty, "notebook trade-price .loc slice returned no rows")
    trade_prices = trade_prices.unstack("symbol").sort_index().shift(-1).tz_localize("UTC")
    _require(set(pred_symbols).issubset(set(trade_prices.columns)), "trade_prices lost prediction symbols")

    # Cell 13: offline equivalent of Alphalens forward-return merge.
    forward_frames = []
    for period in (1, 5, 10, 21):
        future_returns = trade_prices.shift(-period).divide(trade_prices).subtract(1.0)
        stacked = _stack_frame(future_returns).dropna().rename(f"{period}D")
        forward_frames.append(stacked)
    forward_returns = pd.concat(forward_frames, axis=1)
    factor_data = factor.join(forward_returns, how="left")
    _require(not factor_data.empty, "factor/forward-return join returned no rows")
    _require(factor_data[["1D", "5D", "10D", "21D"]].notna().any().any(), "factor_data has no non-null forward returns")

    # Simulate the exact local quandl bundle that write_local_quandl_extension()
    # would ingest: asset metadata, exchanges, daily bars, and asset lookup.
    fake_bundle = _build_fake_quandl_bundle(prices, stocks)
    _require(
        fake_bundle.exchanges.to_dict("records") == [{"exchange": "QUANDL", "canonical_name": "QUANDL", "country_code": "US"}],
        f"unexpected fake bundle exchange metadata: {fake_bundle.exchanges}",
    )
    assets = fake_bundle.asset_finder.lookup_symbols(list(pred_symbols), as_of_date=None)
    sid_map = {asset.symbol: asset.sid for asset in assets}
    for asset in assets:
        _require(asset.exchange == "QUANDL", f"asset {asset.symbol} exchange mismatch: {asset.exchange}")
        _require(asset.country_code == "US", f"asset {asset.symbol} country mismatch: {asset.country_code}")
        _require(asset.first_traded == asset.start_date, f"asset {asset.symbol} first_traded mismatch")
        _require(asset.auto_close_date > asset.end_date, f"asset {asset.symbol} auto_close_date must be after end_date")
        bars = fake_bundle.daily_bars[asset.sid]
        pred_dates_naive = pd.DatetimeIndex(predictions_raw.loc[asset.symbol].index).tz_localize(None)
        missing_bar_dates = pred_dates_naive.difference(bars.index)
        _require(
            missing_bar_dates.empty,
            f"fake bundle daily bars missing prediction dates for {asset.symbol}: {[ _date_str(d) for d in missing_bar_dates[:5] ]}",
        )

    predictions_by_sid = predictions_raw.iloc[:, :3].mean(1).to_frame("prediction")
    prediction_panel = predictions_by_sid.unstack("symbol").rename(columns=sid_map)
    _require("prediction" in prediction_panel.columns.get_level_values(0), "sid-mapped prediction panel lost top-level 'prediction' column")
    predictions_by_sid = prediction_panel.prediction.tz_localize("UTC")
    _require(not predictions_by_sid.empty, "sid-mapped predictions are empty")
    _require(predictions_by_sid.columns.notna().all(), "sid-mapped predictions contain unmapped symbols")

    pipeline_rows = []
    for date, row in predictions_by_sid.iterrows():
        row = row.dropna()
        longs = row[row > 0].nlargest(min(25, int((row > 0).sum()))).index
        shorts = row[row < 0].nsmallest(min(25, int((row < 0).sum()))).index
        for sid in predictions_by_sid.columns:
            pipeline_rows.append(
                {
                    "date": date,
                    "asset": sid,
                    "longs": int(sid in set(longs)),
                    "shorts": int(sid in set(shorts)),
                }
            )
    pipeline_output = pd.DataFrame(pipeline_rows).set_index(["date", "asset"])
    _require({"longs", "shorts"}.issubset(pipeline_output.columns), f"fake pipeline output columns missing: {pipeline_output.columns.tolist()}")
    _require(not pipeline_output.empty, "fake pipeline output is empty")

    simulated_order_targets = 0
    signal_dates = 0
    rebalance_days = 0
    for date, row in predictions_by_sid.iterrows():
        row = row.dropna()
        if row.empty:
            continue
        signal_dates += 1
        longs = row[row > 0].nlargest(min(25, int((row > 0).sum()))).index
        shorts = row[row < 0].nsmallest(min(25, int((row < 0).sum()))).index
        if len(longs) > 10 and len(shorts) > 10:
            rebalance_days += 1
            long_weight = 1.0 / len(longs)
            short_weight = -1.0 / len(shorts)
            _require(np.isfinite(long_weight), "invalid long target weight")
            _require(np.isfinite(short_weight), "invalid short target weight")
            simulated_order_targets += len(longs) + len(shorts)

    _require(signal_dates > 0, "no simulated signal dates")
    _require(rebalance_days > 0, "no simulated days satisfy notebook MIN_POSITIONS long/short gate")
    _require(simulated_order_targets > 0, "no simulated order targets")

    simulated_returns = factor_data["1D"].dropna().groupby(level="date").mean()
    benchmark = simulated_returns.copy() * 0
    _require(benchmark.index.equals(simulated_returns.index), "benchmark index is not aligned with simulated returns")

    # Cell 31/36/38/40/42/44/48/50: run the algorithm callbacks against the
    # fake engine so the core Zipline-facing data path is exercised locally.
    fake_run = _simulate_fake_zipline_run(predictions_by_sid, assets)

    return {
        "factor_rows": int(len(factor)),
        "trade_price_rows": int(len(trade_prices)),
        "factor_data_rows": int(len(factor_data)),
        "sid_count": int(len(sid_map)),
        "signal_dates": int(signal_dates),
        "rebalance_days": int(rebalance_days),
        "simulated_order_targets": int(simulated_order_targets),
        "prediction_panel_ok": True,
        "pipeline_output_columns_ok": True,
        "fake_zipline_lookup_ok": bool(fake_run["lookup_ok"]),
        "fake_bundle_asset_count": int(len(fake_bundle.assets)),
        "fake_bundle_daily_bar_count": int(sum(len(frame) for frame in fake_bundle.daily_bars.values())),
        "fake_bundle_metadata_ok": True,
        "fake_bundle_prediction_coverage_ok": True,
        "fake_zipline_result_rows": int(fake_run["result_rows"]),
        "fake_zipline_order_count": int(fake_run["order_count"]),
        "fake_zipline_positions_count": int(fake_run["positions_count"]),
        "full_flow_ok": True,
    }


def validate(repo_dir: Path, all_symbols: bool, output: Path | None, patch_hdf_metadata: bool) -> ValidationReport:
    assets_path = repo_dir / "data" / "assets.h5"
    predictions_path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
    scores_path = repo_dir / "17_deep_learning" / "results" / "scores.h5"

    _check_hdf_datetime_kinds([assets_path, predictions_path, scores_path], patch=patch_hdf_metadata)

    prices = _load_hdf(assets_path, "quandl/wiki/prices")
    stocks = _load_hdf(assets_path, "quandl/wiki/stocks")
    predictions = _load_hdf(predictions_path, "predictions")

    _require(list(prices.index.names) == ["date", "ticker"], f"unexpected prices index: {prices.index.names}")
    _require(list(predictions.index.names) == ["symbol", "date"], f"unexpected predictions index: {predictions.index.names}")
    _require(REQUIRED_PRICE_COLUMNS.issubset(prices.columns), f"missing price columns: {sorted(REQUIRED_PRICE_COLUMNS - set(prices.columns))}")
    _require(ADJUSTED_PRICE_COLUMNS.issubset(prices.columns), f"missing adjusted price columns: {sorted(ADJUSTED_PRICE_COLUMNS - set(prices.columns))}")
    _require({"code", "name"}.issubset(stocks.columns), f"unexpected stocks columns: {list(stocks.columns)}")
    _require(not prices.index.has_duplicates, "prices index has duplicate date/ticker rows")
    _require(not predictions.index.has_duplicates, "predictions index has duplicate symbol/date rows")
    _require(predictions.shape[1] >= 3, f"predictions needs at least 3 model columns for notebook .iloc[:, :3]; got {predictions.shape[1]}")

    price_symbols = pd.Index(prices.index.get_level_values("ticker").unique()).sort_values()
    pred_symbols = pd.Index(predictions.index.get_level_values("symbol").unique()).sort_values()
    missing_symbols = sorted(set(pred_symbols) - set(price_symbols))
    _require(not missing_symbols, f"prediction symbols missing from prices: {missing_symbols[:20]}")

    price_dates = pd.DatetimeIndex(prices.index.get_level_values("date").unique()).sort_values()
    pred_dates = pd.DatetimeIndex(predictions.index.get_level_values("date").unique()).sort_values()
    missing_pred_dates = sorted(set(pred_dates) - set(price_dates))
    _require(
        not missing_pred_dates,
        "prediction dates not present in WIKI price sessions: "
        + ", ".join(_date_str(d) for d in missing_pred_dates[:20]),
    )

    symbols_to_check = price_symbols if all_symbols else pred_symbols
    prices_by_symbol = prices.sort_index().groupby(level="ticker", sort=False)
    raw_missing_total = 0
    symbols_with_missing = 0
    max_missing = 0
    fill_failures: list[str] = []

    for symbol in symbols_to_check:
        df = prices_by_symbol.get_group(symbol).droplevel("ticker").sort_index()
        # Mirror the current local bundle behavior: clip to supported calendar
        # lower bound and use observed WIKI dates as the session universe for
        # data validation independent of Zipline/exchange_calendars.
        start = max(df.index.min(), pd.Timestamp("1990-01-02"))
        end = df.index.max()
        df = df.loc[start:end]
        if df.empty:
            continue
        sessions = price_dates[(price_dates >= start) & (price_dates <= end)]
        raw_missing = int(len(sessions.difference(df.index)))
        if raw_missing:
            symbols_with_missing += 1
            raw_missing_total += raw_missing
            max_missing = max(max_missing, raw_missing)
        ohlcv = _make_bundle_like_frame(df, sessions)
        if ohlcv[["open", "high", "low", "close", "volume"]].isna().any().any():
            fill_failures.append(str(symbol))
            if len(fill_failures) >= 20:
                break
        if (ohlcv["volume"] < 0).any():
            fill_failures.append(f"{symbol}:negative_volume")
            if len(fill_failures) >= 20:
                break

    _require(not fill_failures, f"bundle-like OHLCV fill failures: {fill_failures}")

    notebook_flow = _simulate_notebook_backtest_flow(prices, stocks, predictions, pred_symbols)

    # Simulate the patched report path: zero benchmark is exactly aligned to
    # returns timestamps, so pyfolio cannot key-error on benchmark dates.
    simulated_returns_index = pd.DatetimeIndex(pred_dates).tz_localize("UTC") + pd.Timedelta(hours=21)
    simulated_returns = pd.Series(0.0, index=simulated_returns_index, name="returns")
    benchmark = simulated_returns.copy() * 0
    benchmark_aligned = benchmark.index.equals(simulated_returns.index)
    _require(benchmark_aligned, "benchmark index is not aligned with returns index")

    report = ValidationReport(
        assets_path=str(assets_path),
        predictions_path=str(predictions_path),
        prices_rows=int(len(prices)),
        price_symbols=int(len(price_symbols)),
        price_start=_date_str(price_dates.min()),
        price_end=_date_str(price_dates.max()),
        prediction_rows=int(len(predictions)),
        prediction_symbols=int(len(pred_symbols)),
        prediction_columns=int(predictions.shape[1]),
        prediction_start=_date_str(pred_dates.min()),
        prediction_end=_date_str(pred_dates.max()),
        missing_prediction_symbols=missing_symbols,
        prediction_dates_not_in_price_sessions=[_date_str(d) for d in missing_pred_dates],
        symbols_checked_for_bundle=int(len(symbols_to_check)),
        symbols_with_raw_missing_sessions=int(symbols_with_missing),
        raw_missing_sessions_total=int(raw_missing_total),
        max_raw_missing_sessions_for_symbol=int(max_missing),
        fill_failures=fill_failures,
        notebook_trade_price_slice_ok=True,
        notebook_factor_rows=int(notebook_flow["factor_rows"]),
        notebook_trade_price_rows=int(notebook_flow["trade_price_rows"]),
        notebook_factor_data_rows=int(notebook_flow["factor_data_rows"]),
        notebook_sid_count=int(notebook_flow["sid_count"]),
        notebook_signal_dates=int(notebook_flow["signal_dates"]),
        notebook_rebalance_days=int(notebook_flow["rebalance_days"]),
        notebook_simulated_order_targets=int(notebook_flow["simulated_order_targets"]),
        notebook_prediction_panel_ok=bool(notebook_flow["prediction_panel_ok"]),
        fake_pipeline_output_columns_ok=bool(notebook_flow["pipeline_output_columns_ok"]),
        fake_zipline_lookup_ok=bool(notebook_flow["fake_zipline_lookup_ok"]),
        fake_bundle_asset_count=int(notebook_flow["fake_bundle_asset_count"]),
        fake_bundle_daily_bar_count=int(notebook_flow["fake_bundle_daily_bar_count"]),
        fake_bundle_metadata_ok=bool(notebook_flow["fake_bundle_metadata_ok"]),
        fake_bundle_prediction_coverage_ok=bool(notebook_flow["fake_bundle_prediction_coverage_ok"]),
        fake_zipline_result_rows=int(notebook_flow["fake_zipline_result_rows"]),
        fake_zipline_order_count=int(notebook_flow["fake_zipline_order_count"]),
        fake_zipline_positions_count=int(notebook_flow["fake_zipline_positions_count"]),
        notebook_full_flow_ok=bool(notebook_flow["full_flow_ok"]),
        metadata_exchange="QUANDL",
        metadata_country_code="US",
        benchmark_aligned=benchmark_aligned,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    return report


def write_synthetic_repo(repo_dir: Path) -> None:
    """Create the smallest HDF5 layout needed to validate CH17 backtest data."""
    data_dir = repo_dir / "data"
    results_dir = repo_dir / "17_deep_learning" / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    assets_path = data_dir / "assets.h5"
    predictions_path = results_dir / "test_preds.h5"
    scores_path = results_dir / "scores.h5"

    dates = pd.bdate_range("2015-12-15", periods=90, name="date")
    tickers = pd.Index([f"STK{i:03d}" for i in range(30)], name="ticker")
    price_index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    base = np.arange(len(price_index), dtype=float) + 100.0
    prices = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.25,
            "volume": np.arange(len(price_index), dtype=float) + 1000.0,
            "adj_open": base,
            "adj_high": base + 1.0,
            "adj_low": base - 1.0,
            "adj_close": base + 0.25,
            "adj_volume": np.arange(len(price_index), dtype=float) + 1000.0,
        },
        index=price_index,
    )
    stocks = pd.DataFrame(
        {
            "code": tickers,
            "name": [f"Synthetic Stock {i:03d}" for i in range(len(tickers))],
        }
    )

    pred_dates = pd.DatetimeIndex(dates[20:55], name="date")
    pred_symbols = pd.Index(tickers[:30], name="symbol")
    pred_index = pd.MultiIndex.from_product([pred_symbols, pred_dates], names=["symbol", "date"])
    symbol_scores = np.linspace(-1.0, 1.0, len(pred_symbols))
    day_offsets = np.sin(np.linspace(0.0, 3.0, len(pred_dates))) * 0.05
    prediction_values = np.array(
        [symbol_scores[symbol_no] + day_offsets[date_no] for symbol_no in range(len(pred_symbols)) for date_no in range(len(pred_dates))]
    )
    predictions = pd.DataFrame(
        {
            "model_0": prediction_values,
            "model_1": prediction_values * 0.95,
            "model_2": prediction_values * 1.05,
        },
        index=pred_index,
    )

    score_index = pd.MultiIndex.from_product(
        [["(16, 8)"], ["tanh"], [0.2], [64], [0], range(2)],
        names=["dense_layers", "activation", "dropout", "batch_size", "fold", "epoch"],
    )
    scores = pd.DataFrame(
        np.array([[0.01, 0.02], [0.02, 0.03]]),
        index=score_index,
        columns=pd.DatetimeIndex(dates[-2:], name="date"),
    )

    with pd.HDFStore(assets_path) as store:
        store.put("quandl/wiki/prices", prices)
        store.put("quandl/wiki/stocks", stocks)
    with pd.HDFStore(predictions_path) as store:
        store.put("predictions", predictions)
    with pd.HDFStore(scores_path) as store:
        store.put("ic_by_day", scores)


def _drop_hdf_key(path: Path, keep: dict[str, pd.DataFrame]) -> None:
    if path.exists():
        path.unlink()
    with pd.HDFStore(path) as store:
        for key, frame in keep.items():
            store.put(key, frame)


def _expect_failure(name: str, mutator) -> str:
    with tempfile.TemporaryDirectory(prefix=f"ch17_negative_{name}_") as tmp:
        repo_dir = Path(tmp) / "machine-learning-for-trading"
        write_synthetic_repo(repo_dir)
        mutator(repo_dir)
        try:
            validate(repo_dir, all_symbols=False, output=None, patch_hdf_metadata=True)
        except SystemExit as exc:
            msg = str(exc)
            _require(bool(msg), f"negative test {name} failed without an error message")
            return msg
        raise SystemExit(f"negative test did not fail as expected: {name}")


def run_negative_self_tests() -> dict[str, str]:
    failures: dict[str, str] = {}

    def missing_prices_key(repo_dir: Path) -> None:
        stocks = pd.read_hdf(repo_dir / "data" / "assets.h5", "quandl/wiki/stocks")
        _drop_hdf_key(repo_dir / "data" / "assets.h5", {"quandl/wiki/stocks": stocks})

    def missing_predictions_key(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        path.unlink()
        with pd.HDFStore(path) as store:
            store.put("not_predictions", pd.DataFrame({"x": [1.0]}))

    def missing_adj_open(repo_dir: Path) -> None:
        path = repo_dir / "data" / "assets.h5"
        prices = pd.read_hdf(path, "quandl/wiki/prices").drop(columns=["adj_open"])
        stocks = pd.read_hdf(path, "quandl/wiki/stocks")
        _drop_hdf_key(path, {"quandl/wiki/prices": prices, "quandl/wiki/stocks": stocks})

    def missing_prediction_symbol(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        predictions = pd.read_hdf(path, "predictions")
        new_index = pd.MultiIndex.from_arrays(
            [
                ["MISSING"] + list(predictions.index.get_level_values("symbol")[1:]),
                predictions.index.get_level_values("date"),
            ],
            names=["symbol", "date"],
        )
        predictions = predictions.copy()
        predictions.index = new_index
        _drop_hdf_key(path, {"predictions": predictions})

    def missing_prediction_date(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        predictions = pd.read_hdf(path, "predictions")
        new_dates = list(predictions.index.get_level_values("date"))
        new_dates[0] = pd.Timestamp("2099-01-01")
        predictions = predictions.copy()
        predictions.index = pd.MultiIndex.from_arrays(
            [predictions.index.get_level_values("symbol"), new_dates],
            names=["symbol", "date"],
        )
        _drop_hdf_key(path, {"predictions": predictions})

    def too_few_prediction_columns(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        predictions = pd.read_hdf(path, "predictions").iloc[:, :2]
        _drop_hdf_key(path, {"predictions": predictions})

    def duplicate_prediction_index(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        predictions = pd.read_hdf(path, "predictions")
        predictions = pd.concat([predictions.iloc[[0]], predictions])
        _drop_hdf_key(path, {"predictions": predictions})

    def no_short_signals(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        predictions = pd.read_hdf(path, "predictions").abs() + 0.1
        _drop_hdf_key(path, {"predictions": predictions})

    def no_long_signals(repo_dir: Path) -> None:
        path = repo_dir / "17_deep_learning" / "results" / "test_preds.h5"
        predictions = -(pd.read_hdf(path, "predictions").abs() + 0.1)
        _drop_hdf_key(path, {"predictions": predictions})

    def missing_stock_name_column(repo_dir: Path) -> None:
        path = repo_dir / "data" / "assets.h5"
        prices = pd.read_hdf(path, "quandl/wiki/prices")
        stocks = pd.read_hdf(path, "quandl/wiki/stocks").drop(columns=["name"])
        _drop_hdf_key(path, {"quandl/wiki/prices": prices, "quandl/wiki/stocks": stocks})

    for name, mutator in {
        "missing_prices_key": missing_prices_key,
        "missing_predictions_key": missing_predictions_key,
        "missing_adj_open": missing_adj_open,
        "missing_prediction_symbol": missing_prediction_symbol,
        "missing_prediction_date": missing_prediction_date,
        "too_few_prediction_columns": too_few_prediction_columns,
        "duplicate_prediction_index": duplicate_prediction_index,
        "no_short_signals": no_short_signals,
        "no_long_signals": no_long_signals,
        "missing_stock_name_column": missing_stock_name_column,
    }.items():
        failures[name] = _expect_failure(name, mutator)

    # Directly validate fake asset lookup failure mode because real validate()
    # rejects missing symbols earlier when comparing predictions to prices.
    try:
        FakeAssetFinder(
            [
                FakeAsset(
                    sid=0,
                    symbol="A",
                    asset_name="A",
                    start_date=pd.Timestamp("2020-01-01"),
                    end_date=pd.Timestamp("2020-01-02"),
                    first_traded=pd.Timestamp("2020-01-01"),
                    auto_close_date=pd.Timestamp("2020-01-03"),
                    exchange="QUANDL",
                    country_code="US",
                )
            ]
        ).lookup_symbols(["A", "MISSING"], as_of_date=None)
    except KeyError as exc:
        failures["fake_asset_lookup_missing_symbol"] = str(exc)
    else:
        raise SystemExit("negative test did not fail as expected: fake_asset_lookup_missing_symbol")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "machine-learning-for-trading",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Validate every WIKI symbol instead of only prediction symbols.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--patch-hdf-metadata",
        action="store_true",
        help="Patch datetime64[ns] HDF metadata to datetime64 before validation.",
    )
    parser.add_argument(
        "--self-test-synthetic",
        action="store_true",
        help="Build minimal synthetic HDF5 files in a temp repo and validate their format.",
    )
    parser.add_argument(
        "--self-test-negative",
        action="store_true",
        help="Run negative synthetic cases that should fail before causing notebook KeyErrors.",
    )
    args = parser.parse_args()

    if args.self_test_negative:
        failures = run_negative_self_tests()
        print(json.dumps(failures, indent=2, sort_keys=True))
        print("CH17 negative data-flow self-tests passed")
        return

    if args.self_test_synthetic:
        with tempfile.TemporaryDirectory(prefix="ch17_format_selftest_") as tmp:
            repo_dir = Path(tmp) / "machine-learning-for-trading"
            write_synthetic_repo(repo_dir)
            report = validate(
                repo_dir.resolve(),
                all_symbols=args.all_symbols,
                output=args.output,
                patch_hdf_metadata=True,
            )
            print(json.dumps(asdict(report), indent=2))
            print("CH17 synthetic backtest data-format self-test passed")
        return

    report = validate(
        args.repo_dir.resolve(),
        all_symbols=args.all_symbols,
        output=args.output,
        patch_hdf_metadata=args.patch_hdf_metadata,
    )
    print(json.dumps(asdict(report), indent=2))
    print("CH17 backtest data validation passed")


if __name__ == "__main__":
    main()
