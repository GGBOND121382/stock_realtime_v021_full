from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dashboard.as1455_cash_metrics import order_amount_metrics


_ORDER_FILENAMES = (
    "close_auction_orders.csv",
    "orders.csv",
    "16_live_orders.csv",
    "close_auction_trades.csv",
    "trades.csv",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _run_orders(run_dir: Path | None) -> tuple[pd.DataFrame, Path | None]:
    if run_dir is None:
        return pd.DataFrame(), None
    root = Path(run_dir)
    for name in _ORDER_FILENAMES:
        path = root / name
        frame = _read_csv(path)
        if not frame.empty or path.is_file():
            return frame, path
    return pd.DataFrame(), None


def _normalize_orders(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "date" not in out.columns:
        for candidate in ("trade_date", "datetime", "dt"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "date"})
                break
    if "symbol" not in out.columns:
        for candidate in ("code", "ticker", "asset"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "symbol"})
                break
    if "date" not in out.columns or "symbol" not in out.columns:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date", "symbol"]).reset_index(drop=True)


def daily_cash_requirements(
    orders: pd.DataFrame,
    *,
    live_root: Path,
    raw_daily_cache_dir: Path,
) -> pd.DataFrame:
    work = _normalize_orders(orders)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "buy_orders",
                "buy_amount",
                "limit_buy_notional",
                "limit_buy_fee_reserve",
                "conservative_cash_required",
                "cash_requirement_complete",
            ]
        )
    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("date", sort=True):
        metrics = order_amount_metrics(
            group,
            live_root=live_root,
            raw_daily_cache_dir=raw_daily_cache_dir,
        )
        side = (
            group["side"].astype(str).str.strip().str.lower()
            if "side" in group.columns
            else pd.Series("", index=group.index)
        )
        rows.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "buy_orders": int(side.eq("buy").sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _pick_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _segment_row(
    label: str,
    daily: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    part = daily.copy()
    if start is not None:
        part = part.loc[part["date"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        part = part.loc[part["date"] <= pd.Timestamp(end).normalize()]
    if part.empty:
        return {
            "segment": label,
            "start_date": start,
            "end_date": end,
            "n_days": 0,
            "n_buy_days": 0,
            "peak_cash_required": np.nan,
            "peak_date": pd.NaT,
            "complete": False,
        }
    cash = pd.to_numeric(part["conservative_cash_required"], errors="coerce")
    complete = bool(
        part.loc[part["buy_orders"].gt(0), "cash_requirement_complete"].fillna(False).all()
    )
    if cash.notna().any():
        idx = cash.idxmax()
        peak = float(cash.loc[idx])
        peak_date = pd.Timestamp(part.loc[idx, "date"]).normalize()
    else:
        peak = np.nan
        peak_date = pd.NaT
    return {
        "segment": label,
        "start_date": pd.Timestamp(part["date"].min()).normalize(),
        "end_date": pd.Timestamp(part["date"].max()).normalize(),
        "n_days": int(part["date"].nunique()),
        "n_buy_days": int(part.loc[part["buy_orders"].gt(0), "date"].nunique()),
        "peak_cash_required": peak,
        "peak_date": peak_date,
        "complete": complete,
    }


def strategy_cash_peak_report(
    item: dict[str, Any],
    *,
    live_root: Path,
    raw_daily_cache_dir: Path,
) -> dict[str, Any]:
    historical_orders, historical_orders_file = _run_orders(item.get("historical_run"))
    forward_orders, forward_orders_file = _run_orders(item.get("forward_run"))
    historical_daily = daily_cash_requirements(
        historical_orders,
        live_root=live_root,
        raw_daily_cache_dir=raw_daily_cache_dir,
    )
    forward_daily = daily_cash_requirements(
        forward_orders,
        live_root=live_root,
        raw_daily_cache_dir=raw_daily_cache_dir,
    )

    rows: list[dict[str, Any]] = []
    folds = item.get("fold_returns")
    if isinstance(folds, pd.DataFrame) and not folds.empty and not historical_daily.empty:
        fold_col = _pick_column(folds, ("fold", "target_fold", "fold_id"))
        start_col = _pick_column(
            folds,
            ("test_start", "start_date", "date_min", "segment_start", "forward_start"),
        )
        end_col = _pick_column(
            folds,
            ("test_end", "end_date", "date_max", "segment_end", "forward_end"),
        )
        if fold_col and start_col and end_col:
            for _, fold in folds.iterrows():
                start = pd.to_datetime(fold[start_col], errors="coerce")
                end = pd.to_datetime(fold[end_col], errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                label = str(fold[fold_col])
                if label.isdigit():
                    label = f"fold{label}"
                rows.append(
                    _segment_row(
                        label,
                        historical_daily,
                        start=pd.Timestamp(start),
                        end=pd.Timestamp(end),
                    )
                )
    if not rows:
        rows.append(_segment_row("historical", historical_daily))
    rows.append(_segment_row("strict_forward", forward_daily))
    segments = pd.DataFrame(rows)

    hist_row = _segment_row("historical_all", historical_daily)
    fwd_row = _segment_row("strict_forward", forward_daily)
    candidates = [
        value
        for value in (
            hist_row.get("peak_cash_required"),
            fwd_row.get("peak_cash_required"),
        )
        if pd.notna(value)
    ]
    overall_peak = max(map(float, candidates)) if candidates else np.nan

    return {
        "segments": segments,
        "historical_daily": historical_daily,
        "forward_daily": forward_daily,
        "historical_peak": hist_row.get("peak_cash_required"),
        "historical_peak_date": hist_row.get("peak_date"),
        "forward_peak": fwd_row.get("peak_cash_required"),
        "forward_peak_date": fwd_row.get("peak_date"),
        "overall_peak": overall_peak,
        "historical_complete": bool(hist_row.get("complete", False)),
        "forward_complete": bool(fwd_row.get("complete", False)),
        "historical_orders_file": str(historical_orders_file) if historical_orders_file else None,
        "forward_orders_file": str(forward_orders_file) if forward_orders_file else None,
        "historical_orders": historical_orders,
        "forward_orders": forward_orders,
    }


def _max_observed_buy_fee_rate(frames: Iterable[pd.DataFrame]) -> float:
    rates: list[float] = []
    for frame in frames:
        work = _normalize_orders(frame)
        if work.empty or "side" not in work.columns:
            continue
        buys = work.loc[work["side"].astype(str).str.lower().eq("buy")].copy()
        if buys.empty:
            continue
        notional = pd.to_numeric(
            buys.get("notional", pd.Series(np.nan, index=buys.index)), errors="coerce"
        ).abs()
        fee = pd.to_numeric(
            buys.get("total_fee", buys.get("cost", pd.Series(np.nan, index=buys.index))),
            errors="coerce",
        ).abs()
        valid = notional.gt(0) & fee.notna()
        if valid.any():
            rates.extend((fee.loc[valid] / notional.loc[valid]).tolist())
    finite = [float(rate) for rate in rates if np.isfinite(rate) and rate >= 0]
    return max(finite) if finite else 0.0


def current_position_cash_estimate(
    item: dict[str, Any],
    *,
    live_root: Path,
    raw_daily_cache_dir: Path,
    historical_orders: pd.DataFrame | None = None,
    forward_orders: pd.DataFrame | None = None,
) -> dict[str, Any]:
    positions = item.get("tracking_latest_positions")
    state = item.get("tracking_latest_state") or {}
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return {
            "asof_date": state.get("asof_date"),
            "n_positions": 0,
            "upper_limit_notional": 0.0,
            "estimated_fee_reserve": 0.0,
            "estimated_cash_required": 0.0,
            "observed_fee_rate": 0.0,
            "complete": True,
        }
    asof = pd.to_datetime(state.get("asof_date"), errors="coerce")
    if pd.isna(asof):
        nav = item.get("tracking_nav")
        if isinstance(nav, pd.DataFrame) and not nav.empty and "date" in nav.columns:
            asof = pd.to_datetime(nav["date"], errors="coerce").max()
    if pd.isna(asof):
        return {
            "asof_date": None,
            "n_positions": int(len(positions)),
            "upper_limit_notional": np.nan,
            "estimated_fee_reserve": np.nan,
            "estimated_cash_required": np.nan,
            "observed_fee_rate": np.nan,
            "complete": False,
        }

    synthetic = positions.copy()
    synthetic["date"] = pd.Timestamp(asof).normalize()
    synthetic["side"] = "buy"
    if "filled_shares" not in synthetic.columns and "shares" in synthetic.columns:
        synthetic["filled_shares"] = synthetic["shares"]
    metrics = order_amount_metrics(
        synthetic,
        live_root=live_root,
        raw_daily_cache_dir=raw_daily_cache_dir,
    )
    rate = _max_observed_buy_fee_rate(
        [
            historical_orders if isinstance(historical_orders, pd.DataFrame) else pd.DataFrame(),
            forward_orders if isinstance(forward_orders, pd.DataFrame) else pd.DataFrame(),
            item.get("tracking_orders")
            if isinstance(item.get("tracking_orders"), pd.DataFrame)
            else pd.DataFrame(),
        ]
    )
    notional = pd.to_numeric(
        pd.Series([metrics.get("limit_buy_notional")]), errors="coerce"
    ).iloc[0]
    if pd.isna(notional):
        fee = np.nan
        total = np.nan
    else:
        fee = float(notional) * float(rate)
        total = float(notional) + fee
    return {
        "asof_date": pd.Timestamp(asof).strftime("%Y-%m-%d"),
        "n_positions": int(len(positions)),
        "upper_limit_notional": float(notional) if pd.notna(notional) else np.nan,
        "estimated_fee_reserve": fee,
        "estimated_cash_required": total,
        "observed_fee_rate": float(rate),
        "complete": bool(metrics.get("cash_requirement_complete", False)),
        "semantics": (
            "same current share quantities repriced at the latest available daily upper limits; "
            "fee reserve uses the maximum observed effective buy-fee rate from this strategy"
        ),
    }
