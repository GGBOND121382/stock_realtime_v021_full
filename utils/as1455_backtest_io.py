#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration and output helpers for AS1455 backtests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_trade_config(
    bt: Any,
    args: Any,
    payload: dict[str, Any],
    *,
    max_positions: int,
    sell_rank: int,
    rebalance_every: int,
    rebalance_offset: int,
) -> Any:
    """Create the v7 TradeConfig from one grid tuple.

    All grid runners use this function so fee, board, capacity, and corporate
    action defaults cannot drift between subprocess and in-process entry points.
    """
    return bt.TradeConfig(
        max_positions=max_positions,
        buy_candidate_rank=sell_rank,
        sell_rank=sell_rank,
        rebalance_every=rebalance_every,
        rebalance_offset=rebalance_offset,
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


def output_frames(
    result: dict[str, Any], mode: str
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
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
    selected = full if mode == "full" else compact if mode == "compact" else {}
    return selected, full


def write_run(
    *,
    run_dir: Path,
    result: dict[str, Any],
    cfg: Any,
    signal_meta: dict[str, Any],
    args: Any,
    prediction_sha: str,
    exec_panel: pd.DataFrame,
    capacity_precheck: dict[str, Any],
    model_run: str | None,
    json_default: Any,
    engine_name: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    selected, full = output_frames(result, args.run_output_mode)
    for name in set(full) - set(selected):
        stale = run_dir / name
        if stale.exists():
            stale.unlink()
    for name, frame in selected.items():
        frame.to_csv(run_dir / name, index=False, encoding="utf-8-sig")

    model_meta = {
        "model_family": str(args.model_family),
        "model_run": str(model_run) if model_run else None,
    }
    summary = dict(result["summary"])
    summary.update(model_meta)
    summary.update(
        {
            key: value
            for key, value in signal_meta.items()
            if not isinstance(value, (list, dict))
        }
    )

    config = dict(cfg.__dict__)
    config.update(model_meta)
    config.update(signal_meta)
    config["output_mode"] = args.run_output_mode
    config["grid_engine"] = engine_name

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
        "grid_engine": engine_name,
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (run_dir / "close_auction_summary.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
