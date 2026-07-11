#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 in-process close-auction grid with shared prepared data.

This runner keeps the original v7 trading semantics while eliminating repeated
data preparation across grid configurations:

- the execution panel is built once for the whole prediction file;
- each signal is normalized and ranked once per trading date;
- all max_positions/sell_rank/rebalance_offset combinations reuse those caches;
- non-rebalance dates do not touch ranking maps unless full position-detail
  output is requested.

The portfolio loop below is an explicit, reviewable implementation. It does not
rewrite source code dynamically and does not use inspect()/exec().
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


legacy = load_module("as1455_grid_legacy", HERE / "run_as1455_close_auction_grid_v1.py")
bt = load_module("as1455_bt_v7", HERE / "run_as1455_close_auction_backtest_v7_maxpos_grid.py")


def prepare_signal(preds: pd.DataFrame, exec_panel: pd.DataFrame) -> dict[str, Any]:
    """Rank one signal once per overlapping trading date."""
    pred_dates = pd.DatetimeIndex(preds["date"].unique()).sort_values()
    exec_dates = pd.DatetimeIndex(exec_panel["date"].unique()).sort_values()
    exec_date_set = set(exec_dates)
    dates = [d for d in pred_dates if d in exec_date_set]
    if len(dates) < 2:
        raise ValueError(
            f"not enough overlapping dates: pred={len(pred_dates)} "
            f"exec={len(exec_dates)} overlap={len(dates)}"
        )

    ranked_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    rank_map_by_date: dict[pd.Timestamp, dict[str, int]] = {}
    score_map_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    date_set = set(dates)
    for date, g in preds[preds["date"].isin(date_set)].groupby("date", sort=True):
        ranked = g.sort_values("score", ascending=False).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked_by_date[date] = ranked
        rank_map_by_date[date] = ranked.set_index("symbol")["rank"].to_dict()
        score_map_by_date[date] = ranked.set_index("symbol")["score"].to_dict()

    missing = [d for d in dates if d not in ranked_by_date]
    if missing:
        raise RuntimeError(f"prepared ranking missing dates: {missing[:5]}")

    return {
        "dates": dates,
        "ranked_by_date": ranked_by_date,
        "rank_map_by_date": rank_map_by_date,
        "score_map_by_date": score_map_by_date,
    }


def backtest_prepared(
    prepared: dict[str, Any],
    exec_by_date: dict[pd.Timestamp, pd.DataFrame],
    cfg,
    corporate_actions: pd.DataFrame | None = None,
    *,
    collect_position_details: bool = False,
) -> dict[str, pd.DataFrame | dict]:
    """Run one portfolio configuration using shared rankings/execution data."""
    dates = prepared["dates"]
    ranked_by_date = prepared["ranked_by_date"]
    rank_map_by_date = prepared["rank_map_by_date"]
    score_map_by_date = prepared["score_map_by_date"]
    if len(dates) < 2:
        raise ValueError(f"not enough prepared dates: {len(dates)}")
    corporate_actions = corporate_actions if corporate_actions is not None else pd.DataFrame()

    cash = float(cfg.initial_cash)
    positions: dict[str, dict[str, object]] = {}
    nav_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    reject_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    round_trip_rows: list[dict[str, object]] = []

    last_nav = float(cfg.initial_cash)
    round_trip_id = 0

    for day_index, date in enumerate(dates):
        exec_t = exec_by_date[date]
        is_reb = bt.is_rebalance_day_index(day_index, cfg)

        if is_reb:
            pred_t = ranked_by_date[date]
            rank_map = rank_map_by_date[date]
            score_map = score_map_by_date[date]
        elif collect_position_details:
            pred_t = None
            rank_map = rank_map_by_date[date]
            score_map = score_map_by_date[date]
        else:
            pred_t = None
            rank_map = {}
            score_map = {}

        order_start = len(order_rows)
        reject_start = len(reject_rows)

        cash_delta, action_log = bt.apply_corporate_actions_for_date(
            date, positions, exec_t, corporate_actions, cfg
        )
        if cash_delta:
            cash += cash_delta
        action_rows.extend(action_log)

        holding_values, missing_marks = bt._portfolio_holding_values(positions, exec_t)
        nav_before_trade = cash + sum(holding_values.values())

        if is_reb:
            # 1) Sell positions whose current rank falls beyond sell_rank.
            for sym in list(positions.keys()):
                rank = rank_map.get(sym, math.inf)
                should_sell = bool(rank > cfg.sell_rank)
                if not should_sell:
                    continue
                buy_date = pd.Timestamp(positions[sym]["buy_date"])
                if date <= buy_date:
                    bt._append_rejection(
                        reject_rows,
                        date=date,
                        symbol=sym,
                        side="sell",
                        reason="t_plus_1_restriction",
                        rank=rank,
                        score=score_map.get(sym, np.nan),
                        cfg=cfg,
                        is_rebalance_day=is_reb,
                    )
                    continue
                row = exec_t.loc[sym] if sym in exec_t.index else pd.Series(dtype=object)
                ok, reason = bt.can_sell(row, cfg)
                if not ok:
                    bt._append_rejection(
                        reject_rows,
                        date=date,
                        symbol=sym,
                        side="sell",
                        reason=reason,
                        rank=rank,
                        score=score_map.get(sym, np.nan),
                        cfg=cfg,
                        is_rebalance_day=is_reb,
                    )
                    continue

                held_shares = float(positions[sym].get("shares", 0.0))
                shares_before = held_shares
                cap_shares, cap_reason = bt.sell_capacity_shares(row, cfg)
                if cap_shares is None:
                    sell_shares = held_shares
                elif cap_shares >= held_shares:
                    sell_shares = held_shares
                else:
                    sell_shares = float(bt.floor_to_lot(cap_shares, cfg.lot_size))
                if sell_shares <= 1e-12:
                    bt._append_rejection(
                        reject_rows,
                        date=date,
                        symbol=sym,
                        side="sell",
                        reason=f"capacity_zero_{cap_reason}",
                        rank=rank,
                        score=score_map.get(sym, np.nan),
                        cfg=cfg,
                        is_rebalance_day=is_reb,
                    )
                    continue

                raw_price = float(row["raw_close_1500"]) * (
                    1.0 - cfg.slippage_bps / 10000.0
                )
                notional = sell_shares * raw_price
                fees = bt.trade_fee_components(notional, "sell", cfg)
                cash_before = cash
                position_before = held_shares
                cash += notional - fees["total_fee"]

                pos = positions[sym]
                ratio = (
                    min(1.0, sell_shares / shares_before)
                    if shares_before > 0
                    else 1.0
                )
                entry_notional_alloc = (
                    float(pos.get("cost_basis_notional", 0.0)) * ratio
                )
                entry_fee_alloc = float(pos.get("cost_basis_fee", 0.0)) * ratio
                gross_pnl = notional - entry_notional_alloc
                net_pnl = (
                    notional
                    - fees["total_fee"]
                    - entry_notional_alloc
                    - entry_fee_alloc
                )
                holding_days = int(
                    (pd.Timestamp(date) - pd.Timestamp(pos.get("buy_date"))).days
                )
                round_trip_id += 1
                round_trip_rows.append(
                    {
                        "round_trip_id": round_trip_id,
                        "symbol": sym,
                        "entry_date": pos.get("buy_date"),
                        "exit_date": date,
                        "holding_days": holding_days,
                        "entry_price": pos.get("avg_entry_price", np.nan),
                        "exit_price": raw_price,
                        "entry_amount": entry_notional_alloc,
                        "exit_amount": notional,
                        "shares": sell_shares,
                        "gross_pnl": gross_pnl,
                        "entry_fee_alloc": entry_fee_alloc,
                        "exit_commission": fees["commission"],
                        "exit_stamp_tax": fees["stamp_tax"],
                        "exit_transfer_fee": fees["transfer_fee"],
                        "exit_total_fee": fees["total_fee"],
                        "total_fee": entry_fee_alloc + fees["total_fee"],
                        "net_pnl": net_pnl,
                        "net_return": net_pnl
                        / max(entry_notional_alloc + entry_fee_alloc, 1e-12),
                        "exit_reason": "rank_gt_sell_rank",
                        "entry_rank": pos.get("entry_rank", np.nan),
                        "exit_rank": rank,
                        "entry_score": pos.get("entry_score", np.nan),
                        "exit_score": score_map.get(sym, np.nan),
                        "partial_exit": bool(
                            sell_shares < shares_before - 1e-12
                        ),
                        "max_positions": cfg.max_positions,
                        "buy_candidate_rank": cfg.buy_candidate_rank,
                        "sell_rank": cfg.sell_rank,
                        "rebalance_every": cfg.rebalance_every,
                        "rebalance_offset": cfg.rebalance_offset,
                    }
                )

                positions[sym]["shares"] = shares_before - sell_shares
                positions[sym]["cost_basis_notional"] = float(
                    pos.get("cost_basis_notional", 0.0)
                ) * (1.0 - ratio)
                positions[sym]["cost_basis_fee"] = float(
                    pos.get("cost_basis_fee", 0.0)
                ) * (1.0 - ratio)
                partial = positions[sym]["shares"] > 1e-12
                if positions[sym]["shares"] <= 1e-12:
                    positions.pop(sym, None)
                    holding_values.pop(sym, None)
                else:
                    holding_values[sym] = positions[sym]["shares"] * float(
                        row["raw_close_1500"]
                    )
                order_rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "side": "sell",
                        "rank": rank,
                        "score": score_map.get(sym, np.nan),
                        "raw_exec_price": raw_price,
                        "raw_close_1500": float(row["raw_close_1500"]),
                        "qfq_exec_price": float(row["qfq_close_1500"]),
                        "intended_shares": shares_before,
                        "filled_shares": sell_shares,
                        "shares": sell_shares,
                        "intended_amount": shares_before * raw_price,
                        "filled_amount": notional,
                        "notional": notional,
                        "commission": fees["commission"],
                        "stamp_tax": fees["stamp_tax"],
                        "transfer_fee": fees["transfer_fee"],
                        "cost": fees["total_fee"],
                        "total_fee": fees["total_fee"],
                        "cash_before": cash_before,
                        "cash_after": cash,
                        "position_before": position_before,
                        "position_after": float(
                            positions.get(sym, {}).get("shares", 0.0)
                        ),
                        "order_status": "filled_partial" if partial else "filled",
                        "capacity_reason": cap_reason,
                        "partial_fill": partial,
                        "reason": "rank_gt_sell_rank",
                        "is_rebalance_day": bool(is_reb),
                        "day_index": day_index,
                        "max_positions": cfg.max_positions,
                        "buy_candidate_rank": cfg.buy_candidate_rank,
                        "sell_rank": cfg.sell_rank,
                        "rebalance_every": cfg.rebalance_every,
                        "rebalance_offset": cfg.rebalance_offset,
                    }
                )

            holding_values, _ = bt._portfolio_holding_values(positions, exec_t)
            nav_after_sells = cash + sum(holding_values.values())

            # 2) Fill empty slots. Existing holdings are not replaced merely
            # because a new symbol has a better current rank.
            if len(positions) < cfg.max_positions and cash > 0:
                if pred_t is None:
                    raise RuntimeError("rebalance date missing prepared ranking")
                candidate_symbols = pred_t.loc[
                    pred_t["rank"] <= cfg.buy_candidate_rank, "symbol"
                ].tolist()
                for sym in candidate_symbols:
                    if len(positions) >= cfg.max_positions:
                        break
                    if sym in positions:
                        continue
                    row = (
                        exec_t.loc[sym]
                        if sym in exec_t.index
                        else pd.Series(dtype=object)
                    )
                    ok, reason = bt.can_buy(row, cfg)
                    rank = rank_map.get(sym, math.inf)
                    score = score_map.get(sym, np.nan)
                    if not ok:
                        bt._append_rejection(
                            reject_rows,
                            date=date,
                            symbol=sym,
                            side="buy",
                            reason=reason,
                            rank=rank,
                            score=score,
                            cfg=cfg,
                            is_rebalance_day=is_reb,
                        )
                        continue
                    slots = max(1, cfg.max_positions - len(positions))
                    base_target = min(
                        nav_after_sells / cfg.max_positions,
                        cash / slots if slots > 1 else cash,
                    )
                    cap_notional, cap_reason = bt.buy_capacity_notional(row, cfg)
                    target_notional = min(base_target, cap_notional)
                    raw_price = float(row["raw_close_1500"]) * (
                        1.0 + cfg.slippage_bps / 10000.0
                    )
                    if not np.isfinite(raw_price) or raw_price <= 0:
                        bt._append_rejection(
                            reject_rows,
                            date=date,
                            symbol=sym,
                            side="buy",
                            reason="bad_raw_price",
                            rank=rank,
                            score=score,
                            cfg=cfg,
                            is_rebalance_day=is_reb,
                        )
                        continue
                    intended_shares = bt.floor_to_lot(
                        target_notional / raw_price, cfg.lot_size
                    )
                    shares = intended_shares
                    while shares > 0:
                        notional = shares * raw_price
                        fees = bt.trade_fee_components(notional, "buy", cfg)
                        total_cash_needed = notional + fees["total_fee"]
                        if total_cash_needed <= cash + 1e-9:
                            break
                        shares -= cfg.lot_size
                    if shares <= 0:
                        reason2 = (
                            f"capacity_zero_{cap_reason}"
                            if np.isfinite(cap_notional) and cap_notional <= 0
                            else "cash_or_lot_too_small"
                        )
                        bt._append_rejection(
                            reject_rows,
                            date=date,
                            symbol=sym,
                            side="buy",
                            reason=reason2,
                            rank=rank,
                            score=score,
                            cfg=cfg,
                            is_rebalance_day=is_reb,
                            extra={"intended_amount": target_notional},
                        )
                        continue
                    notional = shares * raw_price
                    fees = bt.trade_fee_components(notional, "buy", cfg)
                    cash_before = cash
                    cash -= notional + fees["total_fee"]
                    positions[sym] = {
                        "shares": float(shares),
                        "buy_date": date,
                        "avg_entry_price": raw_price,
                        "entry_rank": rank,
                        "entry_score": score,
                        "cost_basis_notional": float(notional),
                        "cost_basis_fee": float(fees["total_fee"]),
                    }
                    holding_values[sym] = float(shares) * float(
                        row["raw_close_1500"]
                    )
                    partial = bool(
                        np.isfinite(cap_notional)
                        and cap_notional < base_target - 1e-9
                    )
                    order_rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "side": "buy",
                            "rank": rank,
                            "score": score,
                            "raw_exec_price": raw_price,
                            "raw_close_1500": float(row["raw_close_1500"]),
                            "qfq_exec_price": float(row["qfq_close_1500"]),
                            "intended_shares": intended_shares,
                            "filled_shares": int(shares),
                            "shares": int(shares),
                            "intended_amount": target_notional,
                            "filled_amount": notional,
                            "notional": notional,
                            "commission": fees["commission"],
                            "stamp_tax": fees["stamp_tax"],
                            "transfer_fee": fees["transfer_fee"],
                            "cost": fees["total_fee"],
                            "total_fee": fees["total_fee"],
                            "cash_before": cash_before,
                            "cash_after": cash,
                            "position_before": 0.0,
                            "position_after": float(shares),
                            "order_status": (
                                "filled_partial" if partial else "filled"
                            ),
                            "capacity_reason": cap_reason,
                            "partial_fill": partial,
                            "reason": "fill_empty_slot_rank_le_buy_candidate_rank",
                            "is_rebalance_day": bool(is_reb),
                            "day_index": day_index,
                            "max_positions": cfg.max_positions,
                            "buy_candidate_rank": cfg.buy_candidate_rank,
                            "sell_rank": cfg.sell_rank,
                            "rebalance_every": cfg.rebalance_every,
                            "rebalance_offset": cfg.rebalance_offset,
                        }
                    )

        holding_values, missing_marks = bt._portfolio_holding_values(
            positions, exec_t
        )
        nav = cash + sum(holding_values.values())
        daily_return = nav / last_nav - 1.0 if last_nav > 0 else np.nan
        todays_orders = order_rows[order_start:]
        todays_rejects = reject_rows[reject_start:]
        gross_trade_amount = sum(
            abs(float(r.get("notional", 0.0))) for r in todays_orders
        )
        turnover = gross_trade_amount / max(nav_before_trade, 1e-12)
        total_fee_today = sum(
            float(r.get("total_fee", r.get("cost", 0.0)))
            for r in todays_orders
        )
        nav_rows.append(
            {
                "date": date,
                "day_index": day_index,
                "is_rebalance_day": bool(is_reb),
                "nav": nav,
                "daily_return": daily_return,
                "cash": cash,
                "cash_ratio": cash / nav if nav > 0 else np.nan,
                "holding_value": sum(holding_values.values()),
                "gross_exposure": (
                    sum(holding_values.values()) / nav if nav > 0 else np.nan
                ),
                "n_positions": len(positions),
                "turnover": turnover,
                "gross_trade_amount": gross_trade_amount,
                "total_fee": total_fee_today,
                "nav_before_trade": nav_before_trade,
                "corporate_action_cash_delta": cash_delta,
                "orders": len(todays_orders),
                "buy_orders": sum(
                    1 for r in todays_orders if r.get("side") == "buy"
                ),
                "sell_orders": sum(
                    1 for r in todays_orders if r.get("side") == "sell"
                ),
                "partial_fill_orders": int(
                    sum(bool(r.get("partial_fill", False)) for r in todays_orders)
                ),
                "rejections": len(todays_rejects),
                "missing_marks": ";".join(missing_marks),
                "max_positions": cfg.max_positions,
                "buy_candidate_rank": cfg.buy_candidate_rank,
                "sell_rank": cfg.sell_rank,
                "rebalance_every": cfg.rebalance_every,
                "rebalance_offset": cfg.rebalance_offset,
            }
        )

        if collect_position_details:
            for sym, value in sorted(holding_values.items()):
                pos = positions[sym]
                position_rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "rank": rank_map.get(sym, np.nan),
                        "score": score_map.get(sym, np.nan),
                        "shares": float(pos.get("shares", 0.0)),
                        "raw_close_1500": (
                            float(exec_t.loc[sym, "raw_close_1500"])
                            if sym in exec_t.index
                            else np.nan
                        ),
                        "value": value,
                        "weight": value / nav if nav > 0 else np.nan,
                        "buy_date": pos.get("buy_date"),
                        "holding_days": (
                            int(
                                (
                                    pd.Timestamp(date)
                                    - pd.Timestamp(pos.get("buy_date"))
                                ).days
                            )
                            if pos.get("buy_date") is not None
                            else np.nan
                        ),
                        "entry_rank": pos.get("entry_rank", np.nan),
                        "entry_score": pos.get("entry_score", np.nan),
                        "avg_entry_price": pos.get("avg_entry_price", np.nan),
                        "cost_basis_notional": pos.get(
                            "cost_basis_notional", np.nan
                        ),
                        "cost_basis_fee": pos.get("cost_basis_fee", np.nan),
                        "max_positions": cfg.max_positions,
                        "buy_candidate_rank": cfg.buy_candidate_rank,
                        "sell_rank": cfg.sell_rank,
                        "rebalance_every": cfg.rebalance_every,
                        "rebalance_offset": cfg.rebalance_offset,
                    }
                )
        last_nav = nav

    nav_df = pd.DataFrame(nav_rows)
    orders_df = pd.DataFrame(order_rows)
    rejects_df = pd.DataFrame(reject_rows)
    positions_df = pd.DataFrame(position_rows)
    actions_df = pd.DataFrame(action_rows)
    round_trips_df = pd.DataFrame(round_trip_rows)
    daily_drawdown_df = bt.build_daily_drawdown(nav_df)
    monthly_summary_df = bt.build_period_summary(nav_df, "M")
    yearly_summary_df = bt.build_period_summary(nav_df, "Y")
    fee_summary_df = bt.build_fee_summary(orders_df, cfg)
    turnover_summary_df = bt.build_turnover_summary(nav_df, orders_df)
    summary = bt.summarize_nav(
        nav_df,
        orders_df,
        rejects_df,
        cfg,
        actions_df,
        round_trips_df,
        daily_drawdown_df,
    )
    return {
        "nav": nav_df,
        "orders": orders_df,
        "trades": orders_df.copy(),
        "rejections": rejects_df,
        "positions": positions_df,
        "corporate_actions": actions_df,
        "round_trips": round_trips_df,
        "daily_drawdown": daily_drawdown_df,
        "monthly_summary": monthly_summary_df,
        "yearly_summary": yearly_summary_df,
        "fee_summary": fee_summary_df,
        "turnover_summary": turnover_summary_df,
        "summary": summary,
    }


def output_maps(result: dict[str, Any], mode: str):
    full = {
        "close_auction_nav.csv": result["nav"],
        "close_auction_orders.csv": result["orders"],
        "close_auction_trades.csv": result["trades"],
        "close_auction_rejections.csv": result["rejections"],
        "close_auction_positions.csv": result["positions"],
        "close_auction_corporate_actions.csv": result["corporate_actions"],
        "daily_drawdown.csv": result["daily_drawdown"],
        "round_trips.csv": result["round_trips"],
        "monthly_summary.csv": result["monthly_summary"],
        "yearly_summary.csv": result["yearly_summary"],
        "fee_summary.csv": result["fee_summary"],
        "turnover_summary.csv": result["turnover_summary"],
    }
    compact = {
        "close_auction_nav.csv": result["nav"],
        "daily_drawdown.csv": result["daily_drawdown"],
        "monthly_summary.csv": result["monthly_summary"],
        "yearly_summary.csv": result["yearly_summary"],
        "fee_summary.csv": result["fee_summary"],
        "turnover_summary.csv": result["turnover_summary"],
    }
    return full if mode == "full" else compact if mode == "compact" else {}, full


def write_run(
    run_dir: Path,
    result: dict[str, Any],
    cfg,
    signal_meta: dict[str, Any],
    args: argparse.Namespace,
    prediction_sha: str,
    exec_panel: pd.DataFrame,
    capacity_precheck: dict[str, Any],
    model_run: str | None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    selected, full = output_maps(result, args.run_output_mode)
    # Remove stale detailed files left by an earlier full-output run.
    for name in set(full) - set(selected):
        stale = run_dir / name
        if stale.exists():
            stale.unlink()
    for name, df in selected.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(run_dir / name, index=False, encoding="utf-8-sig")

    model_meta = {
        "model_family": str(args.model_family),
        "model_run": str(model_run) if model_run else None,
    }
    summary = dict(result["summary"])
    summary.update(model_meta)
    summary.update(
        {k: v for k, v in signal_meta.items() if not isinstance(v, (list, dict))}
    )

    config = dict(cfg.__dict__)
    config.update(model_meta)
    config.update(signal_meta)
    config["output_mode"] = args.run_output_mode
    config["grid_engine"] = "inprocess_shared_rank_v2"

    run_meta = {
        "predictions": str(args.predictions),
        "prediction_file_sha256": prediction_sha,
        "model_meta": model_meta,
        "signal_meta": signal_meta,
        "capacity_precheck": capacity_precheck,
        "n_execution_rows": int(len(exec_panel)),
        "n_execution_symbols": int(exec_panel["symbol"].nunique()),
        "n_execution_dates": int(exec_panel["date"].nunique()),
        "config": cfg.__dict__,
        "summary": summary,
        "output_mode": args.run_output_mode,
        "output_files": sorted(selected),
        "suppressed_output_files": sorted(set(full) - set(selected)),
        "grid_engine": "inprocess_shared_rank_v2",
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )
    (run_dir / "close_auction_summary.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )


def assert_frame_equivalent(
    name: str, left: pd.DataFrame, right: pd.DataFrame
) -> None:
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        obj=name,
    )


def run_parity_check(
    payload: dict[str, Any],
    exec_panel: pd.DataFrame,
    exec_by_date: dict[pd.Timestamp, pd.DataFrame],
    cfg,
    corporate_actions: pd.DataFrame,
    collect_position_details: bool,
) -> dict[str, Any]:
    """Compare one prepared run against the original v7 implementation."""
    reference = bt.backtest(
        payload["preds"], exec_panel, cfg, corporate_actions=corporate_actions
    )
    optimized = backtest_prepared(
        payload["prepared"],
        exec_by_date,
        cfg,
        corporate_actions,
        collect_position_details=collect_position_details,
    )

    keys = ["nav", "orders", "rejections", "round_trips"]
    if collect_position_details:
        keys.append("positions")
    for key in keys:
        assert_frame_equivalent(key, reference[key], optimized[key])

    summary_keys = [
        "final_nav",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "calmar",
        "max_drawdown",
        "avg_turnover",
        "annualized_turnover",
        "n_orders",
        "n_rejections",
        "n_round_trips",
        "total_fee",
    ]
    for key in summary_keys:
        lv = reference["summary"].get(key)
        rv = optimized["summary"].get(key)
        if lv is None and rv is None:
            continue
        if isinstance(lv, (int, float, np.integer, np.floating)) and isinstance(
            rv, (int, float, np.integer, np.floating)
        ):
            if pd.isna(lv) and pd.isna(rv):
                continue
            if not np.isclose(float(lv), float(rv), rtol=1e-12, atol=1e-12):
                raise AssertionError(
                    f"summary mismatch {key}: reference={lv} optimized={rv}"
                )
        elif lv != rv:
            raise AssertionError(
                f"summary mismatch {key}: reference={lv!r} optimized={rv!r}"
            )

    return {
        "passed": True,
        "signal_name": payload["meta"].get("signal_name"),
        "config": dict(cfg.__dict__),
        "compared_frames": keys,
        "compared_summary_keys": summary_keys,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="AS1455 shared-ranking in-process grid"
    )
    p.add_argument("--out-root", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--prediction-key", default=None)
    p.add_argument("--raw-daily-cache-dir", required=True)
    p.add_argument("--raw-5m-cache-dir", default=None)
    p.add_argument("--last5-panel", default=None)
    p.add_argument("--universe", default=None)
    p.add_argument("--st-symbols", default=None)
    p.add_argument("--st-status", default=None)
    p.add_argument("--corporate-actions", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--profile",
        default="close_auction_skip_limit",
        choices=["close_auction_simple", "close_auction_skip_limit"],
    )
    p.add_argument(
        "--capacity-mode",
        default="none",
        choices=["none", "last5_amount", "last5_volume", "last5_both"],
    )
    p.add_argument(
        "--capacity-missing-policy",
        default="fail",
        choices=["fail", "reject", "disable"],
    )
    p.add_argument("--min-last5-coverage", type=float, default=0.95)
    p.add_argument("--participation-rate", type=float, default=0.05)
    p.add_argument("--initial-cash", type=float, default=200000)
    p.add_argument("--commission-rate", type=float, default=0.000085)
    p.add_argument("--min-commission", type=float, default=5)
    p.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    p.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    p.add_argument("--slippage-bps", type=float, default=0)
    p.add_argument("--lot-size", type=int, default=100)
    p.add_argument("--allow-non-mainboard", action="store_true")
    p.add_argument("--allow-st", action="store_true")
    p.add_argument(
        "--corporate-action-mode",
        default="synthetic_share_factor_from_preclose",
        choices=[
            "none",
            "synthetic_share_factor_from_preclose",
            "synthetic_cash_from_preclose",
        ],
    )
    p.add_argument("--corporate-action-threshold", type=float, default=1e-3)
    p.add_argument("--min-price", type=float, default=0)
    p.add_argument("--limit-eps", type=float, default=1e-6)
    p.add_argument(
        "--max-positions-list",
        type=legacy.parse_int_list,
        default=legacy.DEFAULT_MAX_POSITIONS,
    )
    p.add_argument(
        "--sell-rank-list",
        type=legacy.parse_int_list,
        default=legacy.DEFAULT_SELL_RANKS,
    )
    p.add_argument(
        "--rebalance-every-list",
        type=legacy.parse_int_list,
        default=legacy.DEFAULT_REBALANCE_EVERY,
    )
    p.add_argument(
        "--signal-spec",
        dest="signal_specs",
        action="append",
        type=legacy.parse_signal_spec,
        default=None,
    )
    p.add_argument("--offset-mode", choices=["zero", "full"], default="zero")
    p.add_argument(
        "--run-output-mode",
        choices=["summary", "compact", "full"],
        default="compact",
        help=(
            "File-retention level only; it does not change trading logic. "
            "summary=JSON only, compact=core NAV/summary CSVs, "
            "full=also write orders/rejections/positions/round trips."
        ),
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-parity-check", action="store_true")
    p.add_argument("--model-family", default="ML4T Ch17 NN")
    p.add_argument("--model-run", default=None)
    p.add_argument("--model-params-file", default=None)
    p.add_argument("--prediction-file-sha256", default=None)
    args = p.parse_args()
    if args.signal_specs is None:
        args.signal_specs = [
            legacy.parse_signal_spec(x) for x in legacy.DEFAULT_SIGNAL_SPECS
        ]
    return args


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"prediction file not found: {pred_path}")

    out_root = Path(args.out_root)
    runs_root = out_root / "01_runs"
    logs_root = out_root / "04_logs"
    summary_root = out_root / "02_summary"
    for p in [runs_root, logs_root, summary_root]:
        p.mkdir(parents=True, exist_ok=True)

    configs = legacy.build_configs(args)
    legacy.write_grid_config(out_root / "00_grid_config.csv", configs)
    if args.dry_run:
        print(
            f"[DRY RUN] configs={len(configs)} "
            "engine=inprocess_shared_rank_v2"
        )
        return

    prediction_sha = (
        args.prediction_file_sha256 or legacy.sha256_file(pred_path)
    )
    model_run = args.model_run or legacy.infer_model_run(args.predictions)
    model_params_file = (
        args.model_params_file
        or legacy.infer_model_params_file(args.predictions)
    )

    signals: dict[str, dict[str, Any]] = {}
    all_symbols: set[str] = set()
    for spec in args.signal_specs:
        preds, meta = bt.load_predictions(
            pred_path,
            args.prediction_key,
            None,
            signal_cols=bt.parse_csv_tokens(spec["signal_cols"]),
            signal_mode=spec["signal_mode"],
            signal_name=spec["signal_name"],
            prediction_file_sha256=prediction_sha,
            model_params_file=(
                Path(model_params_file) if model_params_file else None
            ),
        )
        preds = bt.apply_date_filters(
            preds, args.start_date, args.end_date
        )
        if preds.empty:
            raise SystemExit(
                f"empty predictions for {spec['signal_name']}"
            )
        signals[spec["signal_name"]] = {
            "preds": preds,
            "meta": meta,
        }
        all_symbols.update(preds["symbol"].unique())

    universe = bt.read_universe(
        Path(args.universe) if args.universe else None
    )
    st_symbols = bt.load_st_symbols(
        Path(args.st_symbols) if args.st_symbols else None
    )
    st_status = bt.load_st_status(
        Path(args.st_status) if args.st_status else None
    )
    last5_panel = bt.load_last5_panel(
        Path(args.last5_panel) if args.last5_panel else None
    )
    corporate_actions = bt.load_corporate_actions(
        Path(args.corporate_actions)
        if args.corporate_actions
        else None
    )
    exec_panel, exec_report = bt.build_execution_panel(
        all_symbols,
        Path(args.raw_daily_cache_dir),
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=last5_panel,
        raw_5m_cache_dir=(
            Path(args.raw_5m_cache_dir)
            if args.raw_5m_cache_dir
            else None
        ),
    )
    exec_panel = bt.apply_date_filters(
        exec_panel, args.start_date, args.end_date
    )
    if exec_panel.empty:
        raise SystemExit("empty execution panel")

    exec_by_date = {
        d: g.set_index("symbol", drop=False)
        for d, g in exec_panel.groupby("date", sort=True)
    }

    for name, payload in signals.items():
        payload["prepared"] = prepare_signal(
            payload["preds"], exec_panel
        )
        effective = args.capacity_mode
        precheck = bt.build_capacity_precheck(
            exec_panel, payload["preds"], effective
        )
        if effective != "none":
            coverage = float(precheck.get("coverage_rate", 0))
            positive = float(precheck.get("positive_rate", 0))
            if (
                coverage < args.min_last5_coverage
                or positive <= 0
            ):
                if args.capacity_missing_policy == "fail":
                    raise SystemExit(
                        f"capacity data insufficient for {name}: "
                        f"coverage={coverage:.6f} "
                        f"positive={positive:.6f}"
                    )
                if args.capacity_missing_policy == "disable":
                    effective = "none"
                    precheck["policy_action"] = (
                        "disabled_capacity_mode"
                    )
                else:
                    precheck["policy_action"] = (
                        "reject_on_missing_capacity"
                    )
        payload["capacity_mode"] = effective
        payload["capacity_precheck"] = precheck

    parity_result: dict[str, Any] = {
        "passed": None,
        "skipped": bool(args.skip_parity_check),
    }
    if not args.skip_parity_check and configs:
        spec, max_pos, sell_rank, reb_every, off = configs[0]
        payload = signals[spec["signal_name"]]
        parity_cfg = bt.TradeConfig(
            max_positions=max_pos,
            buy_candidate_rank=sell_rank,
            sell_rank=sell_rank,
            rebalance_every=reb_every,
            rebalance_offset=off,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            transfer_fee_rate=args.transfer_fee_rate,
            slippage_bps=args.slippage_bps,
            profile=args.profile,
            mainboard_only=not args.allow_non_mainboard,
            min_price=args.min_price,
            limit_eps=args.limit_eps,
            lot_size=args.lot_size,
            min_commission=args.min_commission,
            exclude_st=not args.allow_st,
            capacity_mode=payload["capacity_mode"],
            participation_rate=args.participation_rate,
            corporate_action_mode=args.corporate_action_mode,
            corporate_action_threshold=args.corporate_action_threshold,
        )
        print(
            "[PARITY] comparing optimized engine with original v7 "
            f"for {spec['signal_name']} "
            f"max={max_pos} sell={sell_rank} "
            f"reb={reb_every} off={off}"
        )
        parity_result = run_parity_check(
            payload,
            exec_panel,
            exec_by_date,
            parity_cfg,
            corporate_actions,
            collect_position_details=(
                args.run_output_mode == "full"
            ),
        )
        parity_result["skipped"] = False
        print("[PARITY] PASS")

    rows: list[dict[str, Any]] = []
    for i, cfg_tuple in enumerate(configs, 1):
        spec, max_pos, sell_rank, reb_every, off = cfg_tuple
        run_name = legacy.run_name(
            spec["signal_name"],
            max_pos,
            sell_rank,
            reb_every,
            off,
        )
        run_dir = runs_root / run_name
        log_path = logs_root / f"{run_name}.log"

        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[{i}/{len(configs)}] SKIP existing {run_name}")
            rows.append(
                legacy.flatten_summary(
                    run_dir, cfg_tuple, "ok", returncode=0
                )
            )
            continue

        payload = signals[spec["signal_name"]]
        cfg = bt.TradeConfig(
            max_positions=max_pos,
            buy_candidate_rank=sell_rank,
            sell_rank=sell_rank,
            rebalance_every=reb_every,
            rebalance_offset=off,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            transfer_fee_rate=args.transfer_fee_rate,
            slippage_bps=args.slippage_bps,
            profile=args.profile,
            mainboard_only=not args.allow_non_mainboard,
            min_price=args.min_price,
            limit_eps=args.limit_eps,
            lot_size=args.lot_size,
            min_commission=args.min_commission,
            exclude_st=not args.allow_st,
            capacity_mode=payload["capacity_mode"],
            participation_rate=args.participation_rate,
            corporate_action_mode=args.corporate_action_mode,
            corporate_action_threshold=args.corporate_action_threshold,
        )

        print(f"[{i}/{len(configs)}] RUN {run_name}")
        try:
            result = backtest_prepared(
                payload["prepared"],
                exec_by_date,
                cfg,
                corporate_actions,
                collect_position_details=(
                    args.run_output_mode == "full"
                ),
            )
            write_run(
                run_dir,
                result,
                cfg,
                payload["meta"],
                args,
                prediction_sha,
                exec_panel,
                payload["capacity_precheck"],
                model_run,
            )
            log_path.write_text(
                "[OK] inprocess_shared_rank_v2\n",
                encoding="utf-8",
            )
            rows.append(
                legacy.flatten_summary(
                    run_dir, cfg_tuple, "ok", returncode=0
                )
            )
        except Exception as exc:
            log_path.write_text(
                f"[FAILED] {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            rows.append(
                legacy.flatten_summary(
                    run_dir, cfg_tuple, "failed", returncode=1
                )
            )
            print(f"    FAILED {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "grid_summary.csv"
    summary.to_csv(
        summary_csv, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        out_root / "grid_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    legacy.write_leaderboards(summary_csv, out_root)

    manifest = {
        "engine": "inprocess_shared_rank_v2",
        "configs": len(configs),
        "signals": [
            x["signal_name"] for x in args.signal_specs
        ],
        "execution_panel_built_once": True,
        "daily_rankings_built_once_per_signal": True,
        "rankings_reused_across_all_grid_configs": True,
        "non_rebalance_rank_maps_accessed_only_for_full_output": True,
        "dynamic_source_rewrite": False,
        "prediction_file_sha256": prediction_sha,
        "output_mode": args.run_output_mode,
        "parity_check": parity_result,
    }
    (out_root / "grid_engine_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.run_output_mode == "full":
        exec_report.to_csv(
            out_root / "execution_panel_build_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(
        f"[OK] configs={len(configs)} summary={summary_csv}"
    )


if __name__ == "__main__":
    main()
