#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebalance-phase alignment across historical and fold0-forward windows."""
from __future__ import annotations

from typing import Iterable, Any

import pandas as pd


def _normalize_dates(values: Iterable[Any]) -> pd.DatetimeIndex:
    dates = pd.to_datetime(list(values), errors="coerce")
    dates = pd.DatetimeIndex(dates).dropna().normalize().unique().sort_values()
    return dates


def align_forward_rebalance_phase(
    *,
    rebalance_every: int,
    historical_offset: int,
    historical_n_days: int,
    historical_first_date: str,
    historical_last_date: str,
    forward_prediction_dates: Iterable[Any],
    execution_calendar_dates: Iterable[Any],
) -> dict[str, Any]:
    """Translate a historical local offset into the forward window's local offset.

    The historical v7 run defines ``historical_offset`` relative to the first
    historical overlap date.  The forward run starts a new local ``day_index`` at
    zero, so reusing the same integer is generally wrong.  We continue the phase
    by preserving the historical number of effective backtest dates and then
    counting executable market dates in any bridge between the historical end
    and the first forward overlap date.
    """
    every = int(rebalance_every)
    offset = int(historical_offset)
    history_days = int(historical_n_days)
    if every <= 0:
        raise ValueError(f"rebalance_every must be positive: {every}")
    if not 0 <= offset < every:
        raise ValueError(
            f"historical_offset must be in [0,{every - 1}]: {offset}"
        )
    if history_days <= 0:
        raise ValueError(f"historical_n_days must be positive: {history_days}")

    history_first = pd.Timestamp(historical_first_date).normalize()
    history_last = pd.Timestamp(historical_last_date).normalize()
    if history_first > history_last:
        raise ValueError(
            f"historical window is reversed: {history_first} > {history_last}"
        )

    prediction_dates = _normalize_dates(forward_prediction_dates)
    calendar_dates = _normalize_dates(execution_calendar_dates)
    overlap = prediction_dates.intersection(calendar_dates).sort_values()
    if overlap.empty:
        raise ValueError("forward predictions and execution calendar do not overlap")
    forward_first = pd.Timestamp(overlap[0]).normalize()
    if forward_first <= history_last:
        raise ValueError(
            "forward window must begin after the historical window: "
            f"historical_last={history_last:%Y-%m-%d} "
            f"forward_first={forward_first:%Y-%m-%d}"
        )

    bridge = calendar_dates[
        (calendar_dates > history_last) & (calendar_dates < forward_first)
    ]
    forward_global_index = history_days + int(len(bridge))
    effective_offset = (offset - forward_global_index) % every

    historical_last_index = history_days - 1
    next_historical_phase_index = historical_last_index + 1 + int(len(bridge))
    if next_historical_phase_index != forward_global_index:
        raise AssertionError(
            (next_historical_phase_index, forward_global_index)
        )

    return {
        "method": "continue_historical_effective_trading_day_phase",
        "rebalance_every": every,
        "historical_offset": offset,
        "historical_first_date": history_first.strftime("%Y-%m-%d"),
        "historical_last_date": history_last.strftime("%Y-%m-%d"),
        "historical_n_days": history_days,
        "historical_last_local_index": historical_last_index,
        "bridge_execution_days": int(len(bridge)),
        "bridge_first_date": (
            pd.Timestamp(bridge[0]).strftime("%Y-%m-%d") if len(bridge) else None
        ),
        "bridge_last_date": (
            pd.Timestamp(bridge[-1]).strftime("%Y-%m-%d") if len(bridge) else None
        ),
        "forward_first_prediction_date": pd.Timestamp(
            prediction_dates[0]
        ).strftime("%Y-%m-%d"),
        "forward_first_overlap_date": forward_first.strftime("%Y-%m-%d"),
        "forward_global_index": int(forward_global_index),
        "effective_forward_offset": int(effective_offset),
        "historical_offset_numeric_reused": bool(effective_offset == offset),
        "formula": "(historical_offset - forward_global_index) mod rebalance_every",
    }
