from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_ROOT = PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "live_as1455"


def _coalesce_numeric(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _code6(value: object) -> str | None:
    match = re.search(r"(\d{6})", str(value))
    return match.group(1) if match else None


@lru_cache(maxsize=64)
def _sidecar_upper_limits(live_root_text: str, date_token: str) -> dict[str, float]:
    path = Path(live_root_text) / date_token / "08_live_execution_sidecar.csv"
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return {}
    if not {"symbol", "up_limit"}.issubset(frame.columns):
        return {}
    frame = frame[["symbol", "up_limit"]].copy()
    frame["code6"] = frame["symbol"].map(_code6)
    frame["up_limit"] = pd.to_numeric(frame["up_limit"], errors="coerce")
    frame = frame.dropna(subset=["code6", "up_limit"])
    frame = frame.loc[frame["up_limit"].gt(0)].drop_duplicates("code6", keep="last")
    return {str(row["code6"]): float(row["up_limit"]) for _, row in frame.iterrows()}


def _upper_limit_prices(work: pd.DataFrame, live_root: Path) -> pd.Series:
    upper = _coalesce_numeric(
        work,
        ("up_limit", "buy_limit_price", "upper_limit_price"),
    )
    missing = upper.isna() | upper.le(0)
    if not missing.any() or "date" not in work.columns or "symbol" not in work.columns:
        return upper

    dates = pd.to_datetime(work["date"], errors="coerce")
    codes = work["symbol"].map(_code6)
    for idx in work.index[missing]:
        date = dates.loc[idx]
        code = codes.loc[idx]
        if pd.isna(date) or not code:
            continue
        lookup = _sidecar_upper_limits(str(live_root), pd.Timestamp(date).strftime("%Y%m%d"))
        value = lookup.get(code)
        if value is not None and np.isfinite(value) and value > 0:
            upper.loc[idx] = float(value)
    return upper


def _fee_reserve(
    work: pd.DataFrame,
    simulated_notional: pd.Series,
    limit_notional: pd.Series,
) -> pd.Series:
    """Conservatively scale saved buy fees to the upper-limit order notional.

    Existing order rows already contain the strategy's configured commission and
    transfer-fee outcome at the simulated close-auction fill. Scaling that fee by
    the notional ratio preserves proportional fees and slightly over-reserves a
    minimum commission, which is appropriate for a cash-availability estimate.
    """
    actual_fee = _coalesce_numeric(work, ("total_fee", "cost")).abs()
    if actual_fee.isna().all():
        components = []
        for column in ("commission", "transfer_fee"):
            if column in work.columns:
                components.append(pd.to_numeric(work[column], errors="coerce").abs())
        if components:
            actual_fee = sum(components[1:], components[0].copy())
    actual_fee = actual_fee.fillna(0.0)
    ratio = pd.Series(1.0, index=work.index, dtype="float64")
    valid = simulated_notional.gt(0) & limit_notional.notna() & limit_notional.ge(0)
    ratio.loc[valid] = (limit_notional.loc[valid] / simulated_notional.loc[valid]).clip(lower=1.0)
    return actual_fee * ratio


def order_amount_metrics(
    frame: pd.DataFrame,
    *,
    live_root: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize planned trade amounts and true conservative buy cash demand.

    ``buy_amount``/``sell_amount`` keep the strategy/backtest notional semantics.
    ``conservative_cash_required`` is different: every planned buy is valued at
    that stock's actual exchange ``up_limit`` for the order date, plus a
    conservative buy-fee reserve, and same-auction sell proceeds are not netted.

    For old dates whose orders do not store ``up_limit``, the function looks up
    the saved 14:55 execution sidecar for that date. If a buy still has no exact
    upper-limit price, the conservative requirement is returned as NaN rather
    than silently falling back to the simulated close price.
    """
    if frame is None or frame.empty:
        return {
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "gross_trade_amount": 0.0,
            "net_buy_amount": 0.0,
            "limit_buy_notional": 0.0,
            "limit_buy_fee_reserve": 0.0,
            "conservative_cash_required": 0.0,
            "cash_requirement_complete": True,
        }

    work = frame.copy()
    side = (
        work["side"].astype(str).str.strip().str.lower()
        if "side" in work.columns
        else pd.Series("", index=work.index, dtype="object")
    )
    shares = _coalesce_numeric(work, ("filled_shares", "shares")).abs()
    price = _coalesce_numeric(
        work,
        ("raw_exec_price", "raw_close_1500", "price", "exec_price"),
    ).abs()
    fallback_notional = shares * price
    if "notional" in work.columns:
        notional = pd.to_numeric(work["notional"], errors="coerce").abs()
        notional = notional.where(notional.notna(), fallback_notional)
    else:
        notional = fallback_notional
    notional = notional.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    buy_mask = side.eq("buy")
    sell_mask = side.eq("sell")
    buy_amount = float(notional.loc[buy_mask].sum())
    sell_amount = float(notional.loc[sell_mask].sum())

    if not buy_mask.any():
        limit_buy_notional = 0.0
        limit_buy_fee_reserve = 0.0
        conservative_cash_required = 0.0
        complete = True
    else:
        root = Path(live_root).expanduser().resolve() if live_root is not None else DEFAULT_LIVE_ROOT.resolve()
        upper = _upper_limit_prices(work, root)
        limit_notional_rows = shares * upper
        precise = (
            shares.loc[buy_mask].notna()
            & shares.loc[buy_mask].gt(0)
            & upper.loc[buy_mask].notna()
            & upper.loc[buy_mask].gt(0)
        )
        complete = bool(precise.all())
        if complete:
            fee_rows = _fee_reserve(work, notional, limit_notional_rows)
            limit_buy_notional = float(limit_notional_rows.loc[buy_mask].sum())
            limit_buy_fee_reserve = float(fee_rows.loc[buy_mask].sum())
            conservative_cash_required = limit_buy_notional + limit_buy_fee_reserve
        else:
            limit_buy_notional = np.nan
            limit_buy_fee_reserve = np.nan
            conservative_cash_required = np.nan

    return {
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "gross_trade_amount": buy_amount + sell_amount,
        "net_buy_amount": buy_amount - sell_amount,
        "limit_buy_notional": limit_buy_notional,
        "limit_buy_fee_reserve": limit_buy_fee_reserve,
        "conservative_cash_required": conservative_cash_required,
        "cash_requirement_complete": complete,
    }


def aggregate_cash_requirements(summary: pd.DataFrame) -> dict[str, float]:
    if summary is None or summary.empty:
        return {
            "max_single_strategy_cash_required": 0.0,
            "sum_cash_required": 0.0,
            "sum_gross_trade_amount": 0.0,
        }

    gross = pd.to_numeric(
        summary.get("gross_trade_amount", pd.Series(0.0, index=summary.index)),
        errors="coerce",
    ).fillna(0.0)
    buy_amount = pd.to_numeric(
        summary.get("buy_amount", pd.Series(0.0, index=summary.index)),
        errors="coerce",
    ).fillna(0.0)
    cash = pd.to_numeric(
        summary.get("conservative_cash_required", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    incomplete = buy_amount.gt(0) & cash.isna()
    if incomplete.any():
        max_cash = np.nan
        sum_cash = np.nan
    else:
        cash = cash.fillna(0.0)
        max_cash = float(cash.max()) if len(cash) else 0.0
        sum_cash = float(cash.sum())
    return {
        "max_single_strategy_cash_required": max_cash,
        "sum_cash_required": sum_cash,
        "sum_gross_trade_amount": float(gross.sum()),
    }
