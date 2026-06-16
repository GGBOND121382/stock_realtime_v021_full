#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Chapter 17 prediction backtest profiles for A-share execution rules.

This script does not retrain models. It consumes only
test_preds.h5::/predictions for signal construction and builds execution
panels from the local BaoStock qfq daily cache plus static universe metadata.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREDS = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_reproduce" / "results" / "test_preds.h5"
DEFAULT_OUT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_reproduce" / "backtest"
DEFAULT_CACHE = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce" / "baostock_qfq_daily_cache"
DEFAULT_UNIVERSE = PROJECT_DIR / "saved_data" / "ashare_static_universe" / "07_universe_allA_top1000_static.csv"

TOP_NS = [5, 10, 20, 25]
MAINBOARD = {"sh_mainboard", "sz_mainboard"}


@dataclass
class CostConfig:
    buy_commission_rate: float = 0.0003
    sell_commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage_bps: float = 0.0
    min_commission: float = 5.0
    capital: float = 1_000_000.0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 6:
        digits = digits.zfill(6)
    return digits[-6:]


def annual_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    total = float((1.0 + returns).prod() - 1.0)
    if total <= -1:
        return -1.0
    return float((1.0 + total) ** (252.0 / len(returns)) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity.div(equity.cummax()).sub(1.0)
    return float(dd.min()) if not dd.empty else float("nan")


def summarize(
    profile: str,
    top_n: int | str,
    returns: pd.Series,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    blocked: pd.DataFrame | None = None,
    total_cost: float = 0.0,
) -> dict[str, Any]:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    blocked = pd.DataFrame() if blocked is None else blocked
    turnover = trades.groupby("date").size().reindex(returns.index, fill_value=0) if not trades.empty else pd.Series(0, index=returns.index)
    avg_positions = positions.groupby("date")["symbol"].nunique().mean() if not positions.empty else 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else float("nan")
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 and returns.std(ddof=1) else float("nan")
    final_nav = float((1.0 + returns).prod()) if not returns.empty else float("nan")
    return {
        "profile": profile,
        "top_n": top_n,
        "start_date": returns.index.min().strftime("%Y-%m-%d") if not returns.empty else "",
        "end_date": returns.index.max().strftime("%Y-%m-%d") if not returns.empty else "",
        "n_days": int(len(returns)),
        "annual_return": annual_return(returns),
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "win_rate": float((returns > 0).mean()) if not returns.empty else float("nan"),
        "average_daily_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "average_positions": float(avg_positions) if pd.notna(avg_positions) else 0.0,
        "total_trades": int(len(trades)),
        "blocked_buy_count": int((blocked.get("action", pd.Series(dtype=str)) == "buy").sum()) if not blocked.empty else 0,
        "blocked_sell_count": int((blocked.get("action", pd.Series(dtype=str)) == "sell").sum()) if not blocked.empty else 0,
        "total_cost": float(total_cost),
        "final_nav": final_nav,
    }


def write_profile_outputs(
    out_dir: Path,
    profile: str,
    top_n: int | str,
    returns: pd.Series,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    orders: pd.DataFrame | None = None,
    skipped: pd.DataFrame | None = None,
    extra: dict[str, pd.DataFrame] | None = None,
    metrics_extra: dict[str, Any] | None = None,
    total_cost: float = 0.0,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    returns = returns.rename("daily_return").sort_index()
    equity = (1.0 + returns.fillna(0.0)).cumprod().rename("nav")
    orders = pd.DataFrame() if orders is None else orders
    skipped = pd.DataFrame() if skipped is None else skipped
    metrics = summarize(profile, top_n, returns, positions, trades, skipped, total_cost=total_cost)
    if metrics_extra:
        metrics.update(metrics_extra)

    returns.to_csv(out_dir / "daily_returns.csv", encoding="utf-8-sig")
    returns.to_csv(out_dir / "returns.csv", encoding="utf-8-sig")
    equity.to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")
    positions.to_csv(out_dir / "positions.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(out_dir / "transactions.csv", index=False, encoding="utf-8-sig")
    orders.to_csv(out_dir / "orders.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(out_dir / "skipped_orders.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    results_path = out_dir / "results.h5"
    if results_path.exists():
        results_path.unlink()
    returns.to_frame("daily_return").to_hdf(results_path, "daily_returns", mode="w")
    equity.to_frame("nav").to_hdf(results_path, "equity_curve")
    positions.to_hdf(results_path, "positions")
    trades.to_hdf(results_path, "trades")
    orders.to_hdf(results_path, "orders")
    skipped.to_hdf(results_path, "skipped_orders")
    if extra:
        for name, df in extra.items():
            df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    return metrics


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing predictions file: {predictions_path}")
    predictions = pd.read_hdf(predictions_path, "predictions")
    if list(predictions.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected predictions index: {predictions.index.names}")
    if predictions.shape[1] < 3:
        raise RuntimeError(f"predictions must have at least 3 columns; got {predictions.shape[1]}")
    predictions = predictions.sort_index()
    predictions.index = pd.MultiIndex.from_arrays(
        [
            predictions.index.get_level_values("symbol").map(normalize_code),
            pd.to_datetime(predictions.index.get_level_values("date")).normalize(),
        ],
        names=["symbol", "date"],
    )
    return predictions


def build_signal_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    signal = predictions.iloc[:, :3].mean(axis=1).rename("signal")
    panel = signal.unstack("symbol").sort_index()
    panel.index.name = "date"
    return panel


def load_universe(path: Path) -> pd.DataFrame:
    universe = pd.read_csv(path, dtype={"code": str})
    universe["symbol"] = universe["code"].map(normalize_code)
    universe["board"] = universe["board"].fillna("").astype(str)
    universe["is_mainboard"] = universe["board"].isin(MAINBOARD)
    universe["isST"] = universe.get("is_current_st", False).fillna(False).astype(bool) if "is_current_st" in universe else False
    return universe.set_index("symbol", drop=False)


def read_cache_symbol(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]].dropna()


def build_execution_panel(cache_dir: Path, universe: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        path = cache_dir / f"{symbol}_qfq_daily.csv"
        if path.exists():
            frames.append(read_cache_symbol(path, symbol))
    if not frames:
        raise RuntimeError(f"no cache files found under {cache_dir}")
    data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    meta_cols = ["board", "is_mainboard", "isST"]
    data = data.merge(universe[meta_cols], left_on="symbol", right_index=True, how="left")
    data["board"] = data["board"].fillna("")
    data["is_mainboard"] = data["is_mainboard"].fillna(False).astype(bool)
    data["isST"] = data["isST"].fillna(False).astype(bool)
    data["preclose"] = data.groupby("symbol")["close"].shift(1)
    data["adj_open"] = data["open"]
    data["adj_close"] = data["close"]
    data["amount"] = data["close"].mul(data["volume"])
    data["tradestatus"] = np.where(data["volume"] > 0, 1, 0)
    limit_pct = np.select(
        [data["board"].eq("chinext"), data["board"].eq("star")],
        [0.20, 0.20],
        default=0.10,
    )
    up = data["preclose"].mul(1.0 + limit_pct)
    down = data["preclose"].mul(1.0 - limit_pct)
    data["open_limit_up"] = data["preclose"].notna() & (data["open"] >= up.mul(0.999))
    data["open_limit_down"] = data["preclose"].notna() & (data["open"] <= down.mul(1.001))
    data["close_limit_up"] = data["preclose"].notna() & (data["close"] >= up.mul(0.999))
    data["close_limit_down"] = data["preclose"].notna() & (data["close"] <= down.mul(1.001))
    data = data.set_index(["date", "symbol"]).sort_index()
    return data


def panel_from_exec(execution: pd.DataFrame, column: str) -> pd.DataFrame:
    return execution[column].unstack("symbol").sort_index()


def next_date_maps(dates: pd.Index) -> tuple[dict[pd.Timestamp, pd.Timestamp], dict[pd.Timestamp, pd.Timestamp]]:
    dates = pd.Index(pd.to_datetime(dates).sort_values().unique())
    next1 = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    next2 = {dates[i]: dates[i + 2] for i in range(len(dates) - 2)}
    return next1, next2


def select_top(scores: pd.Series, top_n: int, eligible: pd.Series | None = None) -> pd.Index:
    scores = scores.dropna()
    scores = scores[scores > 0]
    if eligible is not None:
        eligible = eligible.reindex(scores.index).fillna(False).astype(bool)
        scores = scores[eligible]
    return scores.nlargest(min(top_n, len(scores))).index


def select_bottom(scores: pd.Series, top_n: int) -> pd.Index:
    scores = scores.dropna()
    scores = scores[scores < 0]
    return scores.nsmallest(min(top_n, len(scores))).index


def run_independent_profile(
    profile: str,
    signal_panel: pd.DataFrame,
    execution: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    universe_mask: pd.DataFrame | None,
    mode: str,
    timing: str,
    skip_open_limit: bool = False,
) -> dict[str, Any]:
    close = panel_from_exec(execution, "adj_close")
    open_ = panel_from_exec(execution, "adj_open")
    open_up = panel_from_exec(execution, "open_limit_up").fillna(False).astype(bool)
    open_down = panel_from_exec(execution, "open_limit_down").fillna(False).astype(bool)
    dates = signal_panel.index.intersection(close.index).intersection(open_.index)
    next1, next2 = next_date_maps(open_.index)
    rows = []
    pos_rows = []
    trades = []
    skipped = []
    align_rows = []
    for date in dates:
        date = pd.Timestamp(date)
        if timing == "close_to_close":
            exit_date = next1.get(date)
            if exit_date is None:
                continue
            rets = close.loc[exit_date].div(close.loc[date]).sub(1.0)
        else:
            buy_date = next1.get(date)
            sell_date = next2.get(date)
            if buy_date is None or sell_date is None:
                continue
            rets = open_.loc[sell_date].div(open_.loc[buy_date]).sub(1.0)
        eligible = universe_mask.loc[date] if universe_mask is not None and date in universe_mask.index else None
        longs = select_top(signal_panel.loc[date], top_n, eligible)
        shorts = select_bottom(signal_panel.loc[date], top_n) if mode == "long_short" else pd.Index([])
        if skip_open_limit and timing == "next_open":
            buy_date = next1[date]
            sell_date = next2[date]
            keep = []
            for sym in longs:
                if bool(open_up.reindex(index=[buy_date], columns=[sym]).iloc[0, 0]):
                    skipped.append({"date": buy_date, "signal_date": date, "symbol": sym, "action": "buy", "reason": "open_limit_up", "signal": signal_panel.at[date, sym]})
                elif bool(open_down.reindex(index=[sell_date], columns=[sym]).iloc[0, 0]):
                    skipped.append({"date": sell_date, "signal_date": date, "symbol": sym, "action": "sell", "reason": "open_limit_down", "signal": signal_panel.at[date, sym]})
                else:
                    keep.append(sym)
            longs = pd.Index(keep)
        long_ret = rets.reindex(longs).dropna()
        short_ret = rets.reindex(shorts).dropna()
        if len(long_ret) == 0 or (mode == "long_short" and len(short_ret) == 0):
            continue
        daily_ret = float(long_ret.mean() - (short_ret.mean() if mode == "long_short" else 0.0))
        rows.append({"date": date, "daily_return": daily_ret, "long_count": len(long_ret), "short_count": len(short_ret)})
        for sym in long_ret.index:
            pos_rows.append({"date": date, "symbol": sym, "weight": 1.0 / len(long_ret), "side": "long", "signal": signal_panel.at[date, sym]})
            trades.append({"date": next1.get(date, date), "signal_date": date, "symbol": sym, "side": "buy", "return": float(long_ret.at[sym])})
        for sym in short_ret.index:
            pos_rows.append({"date": date, "symbol": sym, "weight": -1.0 / len(short_ret), "side": "short", "signal": signal_panel.at[date, sym]})
            trades.append({"date": next1.get(date, date), "signal_date": date, "symbol": sym, "side": "short", "return": float(short_ret.at[sym])})
        if timing == "next_open" and len(align_rows) < 200:
            buy_date = next1[date]
            sell_date = next2[date]
            for sym in long_ret.index[:5]:
                manual = float(open_.at[sell_date, sym] / open_.at[buy_date, sym] - 1.0)
                align_rows.append({"signal_date": date, "buy_date": buy_date, "sell_date": sell_date, "symbol": sym, "adj_open_buy": open_.at[buy_date, sym], "adj_open_sell": open_.at[sell_date, sym], "manual_ret": manual, "recorded_ret": float(long_ret.at[sym])})
    returns = pd.DataFrame(rows).set_index("date")["daily_return"] if rows else pd.Series(dtype=float)
    positions = pd.DataFrame(pos_rows)
    trades_df = pd.DataFrame(trades)
    skipped_df = pd.DataFrame(skipped)
    extra = {}
    if align_rows:
        extra["execution_alignment_check.csv"] = pd.DataFrame(align_rows)
    if skip_open_limit:
        extra["skipped_buy_orders.csv"] = skipped_df[skipped_df.get("action", "") == "buy"] if not skipped_df.empty else skipped_df
        extra["blocked_sell_orders.csv"] = skipped_df[skipped_df.get("action", "") == "sell"] if not skipped_df.empty else skipped_df
    return write_profile_outputs(out_dir, profile, top_n, returns, positions, trades_df, orders=trades_df, skipped=skipped_df, extra=extra)


def run_realistic_hold(
    profile: str,
    signal_panel: pd.DataFrame,
    execution: pd.DataFrame,
    out_dir: Path,
    top_n: int = 25,
    cost: CostConfig | None = None,
) -> dict[str, Any]:
    cost = cost or CostConfig(0, 0, 0, 0, 0)
    open_ = panel_from_exec(execution, "adj_open")
    is_main = panel_from_exec(execution, "is_mainboard").fillna(False).astype(bool)
    is_st = panel_from_exec(execution, "isST").fillna(False).astype(bool)
    trade = panel_from_exec(execution, "tradestatus").fillna(0).astype(int).eq(1)
    open_up = panel_from_exec(execution, "open_limit_up").fillna(False).astype(bool)
    open_down = panel_from_exec(execution, "open_limit_down").fillna(False).astype(bool)
    dates = pd.Index(signal_panel.index.intersection(open_.index)).sort_values()
    next1, _next2 = next_date_maps(open_.index)
    nav = 1.0
    holdings: dict[str, float] = {}
    rows = []
    pos_rows = []
    orders = []
    fills = []
    blocked = []
    costs = []
    turnover_rows = []
    for signal_date in dates:
        exec_date = next1.get(pd.Timestamp(signal_date))
        if exec_date is None or exec_date not in open_.index:
            continue
        before_nav = nav
        if holdings:
            gross = 0.0
            for sym, weight in holdings.items():
                if sym in open_.columns and signal_date in open_.index and pd.notna(open_.at[signal_date, sym]) and pd.notna(open_.at[exec_date, sym]):
                    gross += weight * (open_.at[exec_date, sym] / open_.at[signal_date, sym] - 1.0)
            nav *= 1.0 + gross
        scores = signal_panel.loc[signal_date].dropna()
        eligible = is_main.loc[signal_date].reindex(scores.index).fillna(False) & (~is_st.loc[signal_date].reindex(scores.index).fillna(True)) & trade.loc[exec_date].reindex(scores.index).fillna(False)
        target = set(select_top(scores, top_n, eligible))
        current = set(holdings)
        executed_sell = []
        executed_buy = []
        day_cost = 0.0
        for sym in sorted(current - target):
            orders.append({"date": exec_date, "signal_date": signal_date, "symbol": sym, "action": "sell", "target_weight": 0.0})
            if not bool(trade.at[exec_date, sym]) if sym in trade.columns else True:
                blocked.append({"date": exec_date, "symbol": sym, "action": "sell", "reason": "suspended", "signal": scores.get(sym, np.nan)})
                continue
            if bool(open_down.at[exec_date, sym]) if sym in open_down.columns else False:
                blocked.append({"date": exec_date, "symbol": sym, "action": "sell", "reason": "open_limit_down", "signal": scores.get(sym, np.nan)})
                continue
            notional = nav * abs(holdings.get(sym, 0.0))
            fee = max(cost.min_commission, notional * cost.sell_commission_rate) + notional * cost.stamp_tax_rate + notional * cost.slippage_bps / 10000.0
            day_cost += fee / max(cost.capital, 1.0)
            holdings.pop(sym, None)
            executed_sell.append(sym)
            fills.append({"date": exec_date, "signal_date": signal_date, "symbol": sym, "action": "sell", "notional": notional, "cost": fee})
        buy_candidates = sorted(target - set(holdings))
        for sym in buy_candidates:
            orders.append({"date": exec_date, "signal_date": signal_date, "symbol": sym, "action": "buy", "target_weight": 1.0 / max(len(target), 1)})
            reason = None
            if sym not in trade.columns or not bool(trade.at[exec_date, sym]):
                reason = "suspended"
            elif bool(open_up.at[exec_date, sym]):
                reason = "open_limit_up"
            elif sym in is_st.columns and bool(is_st.at[signal_date, sym]):
                reason = "st"
            elif sym in is_main.columns and not bool(is_main.at[signal_date, sym]):
                reason = "non_mainboard"
            if reason:
                blocked.append({"date": exec_date, "symbol": sym, "action": "buy", "reason": reason, "signal": scores.get(sym, np.nan)})
                continue
            executed_buy.append(sym)
        if target:
            new_symbols = sorted(set(holdings).union(executed_buy))
            if new_symbols:
                equal = 1.0 / len(new_symbols)
                holdings = {sym: equal for sym in new_symbols}
                for sym in executed_buy:
                    notional = nav * equal
                    fee = max(cost.min_commission, notional * cost.buy_commission_rate) + notional * cost.slippage_bps / 10000.0
                    day_cost += fee / max(cost.capital, 1.0)
                    fills.append({"date": exec_date, "signal_date": signal_date, "symbol": sym, "action": "buy", "notional": notional, "cost": fee})
        nav_after_cost = nav - day_cost
        daily_ret = nav_after_cost / before_nav - 1.0 if before_nav else 0.0
        nav = nav_after_cost
        rows.append({"date": exec_date, "daily_return": daily_ret, "nav": nav, "gross_nav": nav + day_cost, "cost": day_cost})
        turnover_rows.append({"date": exec_date, "turnover": len(executed_buy) + len(executed_sell)})
        for sym, weight in holdings.items():
            pos_rows.append({"date": exec_date, "symbol": sym, "weight": weight, "side": "long", "signal": scores.get(sym, np.nan)})
    daily = pd.DataFrame(rows).drop_duplicates("date", keep="last").set_index("date") if rows else pd.DataFrame(columns=["daily_return"])
    returns = daily["daily_return"] if not daily.empty else pd.Series(dtype=float)
    positions = pd.DataFrame(pos_rows)
    fills_df = pd.DataFrame(fills)
    orders_df = pd.DataFrame(orders)
    blocked_df = pd.DataFrame(blocked)
    extra = {
        "fills.csv": fills_df,
        "blocked_orders.csv": blocked_df,
        "daily_nav.csv": daily.reset_index(),
        "daily_holdings.csv": positions,
        "turnover.csv": pd.DataFrame(turnover_rows),
    }
    if cost.buy_commission_rate or cost.sell_commission_rate or cost.stamp_tax_rate or cost.slippage_bps or cost.min_commission:
        extra["daily_returns_gross.csv"] = (daily["gross_nav"].pct_change().fillna(daily["gross_nav"].sub(1.0))).rename("daily_return_gross").reset_index()
        extra["daily_returns_net.csv"] = returns.rename("daily_return_net").reset_index()
        extra["cost_breakdown.csv"] = daily[["cost"]].reset_index()
    metrics_extra = {"cost_config": asdict(cost)}
    return write_profile_outputs(out_dir, profile, top_n, returns, positions, fills_df, orders=orders_df, skipped=blocked_df, extra=extra, metrics_extra=metrics_extra, total_cost=float(daily["cost"].sum()) if "cost" in daily else 0.0)


def write_common(predictions_path: Path, cache_dir: Path, universe_path: Path, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = ensure_dir(out_root / "common")
    predictions = load_predictions(predictions_path)
    signal_panel = build_signal_panel(predictions)
    signal_panel.to_hdf(common / "signal_panel.h5", "signal", mode="w")
    universe = load_universe(universe_path)
    execution = build_execution_panel(cache_dir, universe, sorted(signal_panel.columns))
    execution.to_hdf(common / "execution_panel.h5", "execution", mode="w", format="table")
    check = {
        "prediction_rows": int(len(predictions)),
        "signal_dates": int(signal_panel.shape[0]),
        "signal_symbols": int(signal_panel.shape[1]),
        "signal_missing": int(signal_panel.isna().sum().sum()),
        "execution_rows": int(len(execution)),
        "execution_symbols": int(execution.index.get_level_values("symbol").nunique()),
        "date_start": signal_panel.index.min().strftime("%Y-%m-%d"),
        "date_end": signal_panel.index.max().strftime("%Y-%m-%d"),
    }
    (common / "input_checks.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    return signal_panel, execution


def try_ch17_original_zipline(out_dir: Path) -> None:
    try:
        import zipline  # noqa: F401
    except Exception as exc:
        (out_dir / "zipline_status.json").write_text(
            json.dumps({"status": "skipped", "reason": f"zipline unavailable: {type(exc).__name__}: {exc}"}, indent=2),
            encoding="utf-8",
        )


def run_all(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    signal_panel, execution = write_common(Path(args.predictions), Path(args.cache_dir), Path(args.universe), out_root)
    is_main = panel_from_exec(execution, "is_mainboard").reindex(index=signal_panel.index, columns=signal_panel.columns).fillna(False).astype(bool)
    leaderboard = []

    ch17_dir = ensure_dir(out_root / "ch17_original")
    try_ch17_original_zipline(ch17_dir)
    leaderboard.append(run_independent_profile("ch17_original", signal_panel, execution, ch17_dir, 25, None, "long_short", "close_to_close"))

    for n in TOP_NS:
        leaderboard.append(run_independent_profile("ashare_long_only_allA", signal_panel, execution, out_root / "ashare_long_only_allA" / f"top{n}", n, None, "long_only", "close_to_close"))
        leaderboard.append(run_independent_profile("ashare_long_only_mainboard", signal_panel, execution, out_root / "ashare_long_only_mainboard" / f"top{n}", n, is_main, "long_only", "close_to_close"))

    leaderboard.append(run_independent_profile("ashare_mainboard_safe_next_open", signal_panel, execution, out_root / "ashare_mainboard_safe_next_open", 25, is_main, "long_only", "next_open"))
    leaderboard.append(run_independent_profile("ashare_mainboard_skip_open_limit", signal_panel, execution, out_root / "ashare_mainboard_skip_open_limit", 25, is_main, "long_only", "next_open", skip_open_limit=True))
    leaderboard.append(run_realistic_hold("ashare_mainboard_realistic_hold", signal_panel, execution, out_root / "ashare_mainboard_realistic_hold", top_n=25, cost=CostConfig(0, 0, 0, 0, 0)))
    leaderboard.append(run_realistic_hold("ashare_mainboard_realistic_hold_cost", signal_panel, execution, out_root / "ashare_mainboard_realistic_hold_cost", top_n=25, cost=CostConfig()))

    board = pd.DataFrame(leaderboard)
    board.to_csv(out_root / "leaderboard.csv", index=False, encoding="utf-8-sig")
    (out_root / "comparison_report.json").write_text(json.dumps({"profiles": board.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_root.resolve()), "profiles": int(len(board))}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A-share Chapter 17 backtest profiles from test_preds.h5")
    p.add_argument("--predictions", default=str(DEFAULT_PREDS), help="Path to test_preds.h5 containing /predictions")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="BaoStock qfq daily cache directory")
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE), help="Static universe CSV with board metadata")
    return p.parse_args()


def main() -> None:
    run_all(parse_args())


if __name__ == "__main__":
    main()
