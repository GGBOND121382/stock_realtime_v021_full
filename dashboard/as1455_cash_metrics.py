from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def order_amount_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize buy/sell cash demand from planned or simulated order rows.

    ``notional`` is authoritative when present. Older artifacts can be
    reconstructed from filled/planned shares and the first available execution
    price. All monetary values are positive magnitudes except ``net_buy_amount``.
    ``conservative_cash_required`` deliberately does not offset same-auction sell
    proceeds, so it is the useful reserve-cash upper bound for simultaneous orders.
    """
    if frame is None or frame.empty:
        return {
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "gross_trade_amount": 0.0,
            "net_buy_amount": 0.0,
            "conservative_cash_required": 0.0,
        }

    work = frame.copy()
    side = (
        work["side"].astype(str).str.strip().str.lower()
        if "side" in work.columns
        else pd.Series("", index=work.index, dtype="object")
    )
    if "notional" in work.columns:
        notional = pd.to_numeric(work["notional"], errors="coerce").abs()
    else:
        shares = _numeric_column(work, ("filled_shares", "shares")).abs()
        price = _numeric_column(
            work,
            ("raw_exec_price", "raw_close_1500", "price", "exec_price"),
        ).abs()
        notional = shares * price
    notional = notional.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    buy_amount = float(notional.loc[side.eq("buy")].sum())
    sell_amount = float(notional.loc[side.eq("sell")].sum())
    return {
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "gross_trade_amount": buy_amount + sell_amount,
        "net_buy_amount": buy_amount - sell_amount,
        "conservative_cash_required": buy_amount,
    }


def aggregate_cash_requirements(summary: pd.DataFrame) -> dict[str, float]:
    if summary is None or summary.empty:
        return {
            "max_single_strategy_cash_required": 0.0,
            "sum_cash_required": 0.0,
            "sum_gross_trade_amount": 0.0,
        }
    cash = pd.to_numeric(
        summary.get("conservative_cash_required", pd.Series(0.0, index=summary.index)),
        errors="coerce",
    ).fillna(0.0)
    gross = pd.to_numeric(
        summary.get("gross_trade_amount", pd.Series(0.0, index=summary.index)),
        errors="coerce",
    ).fillna(0.0)
    return {
        "max_single_strategy_cash_required": float(cash.max()) if len(cash) else 0.0,
        "sum_cash_required": float(cash.sum()),
        "sum_gross_trade_amount": float(gross.sum()),
    }
