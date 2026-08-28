#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only rebalance calendar helpers for the AS1455 dashboard.

The schedule is anchored to each tracking account's latest persisted V7 phase:
``day_index``, ``rebalance_every`` and ``rebalance_offset``. Future market dates
come from BaoStock's trading calendar. We intentionally do not substitute a
weekday-only calendar because Chinese exchange holidays would shift the phase.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dashboard.as1455_backtest_data import (
    discover_experiment_names,
    load_experiment,
    parse_experiment_name,
)

DEFAULT_PRODUCTION_EXPERIMENT = "r21_best_reb21_fold0_4_forward"


def _dates(values: Iterable[Any]) -> pd.DatetimeIndex:
    return (
        pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise RuntimeError(f"invalid persisted boolean value: {value!r}")


def query_baostock_trade_dates(start: date | str, end: date | str) -> pd.DatetimeIndex:
    """Return authoritative A-share trading dates from BaoStock.

    This function raises on login/query failure. Callers should surface the
    failure instead of silently replacing the exchange calendar with weekdays.
    """
    import baostock as bs

    start_text = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end).strftime("%Y-%m-%d")
    login = bs.login()
    if str(login.error_code) != "0":
        raise RuntimeError(
            f"BaoStock login failed: code={login.error_code} msg={login.error_msg}"
        )
    try:
        result = bs.query_trade_dates(start_date=start_text, end_date=end_text)
        if str(result.error_code) != "0":
            raise RuntimeError(
                "BaoStock query_trade_dates failed: "
                f"code={result.error_code} msg={result.error_msg}"
            )
        rows: list[list[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()

    if frame.empty or not {"calendar_date", "is_trading_day"}.issubset(frame.columns):
        raise RuntimeError("BaoStock returned no usable trading-calendar rows")
    trading = frame.loc[
        frame["is_trading_day"].astype(str).str.strip().eq("1"), "calendar_date"
    ]
    dates = _dates(trading)
    if dates.empty:
        raise RuntimeError(
            f"BaoStock returned no trading dates in {start_text}..{end_text}"
        )
    return dates


def _latest_phase(nav: pd.DataFrame) -> dict[str, Any]:
    required = {
        "date",
        "day_index",
        "is_rebalance_day",
        "rebalance_every",
        "rebalance_offset",
    }
    missing = required - set(nav.columns)
    if nav.empty or missing:
        raise RuntimeError(f"tracking NAV lacks phase columns: {sorted(missing)}")
    work = nav.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"]).sort_values("date")
    if work.empty:
        raise RuntimeError("tracking NAV has no valid phase date")
    row = work.iloc[-1]
    every = int(row["rebalance_every"])
    offset = int(row["rebalance_offset"])
    day_index = int(row["day_index"])
    if every <= 0 or not 0 <= offset < every:
        raise RuntimeError(
            f"invalid persisted rebalance phase: every={every} offset={offset}"
        )
    expected = (day_index - offset) % every == 0
    actual = _bool_value(row["is_rebalance_day"])
    if expected != actual:
        raise RuntimeError(
            "latest tracking phase is internally inconsistent: "
            f"date={pd.Timestamp(row['date']):%Y-%m-%d} day_index={day_index} "
            f"every={every} offset={offset} stored_rebalance={actual}"
        )
    return {
        "date": pd.Timestamp(row["date"]).normalize(),
        "day_index": day_index,
        "rebalance_every": every,
        "rebalance_offset": offset,
        "is_rebalance_day": actual,
    }


def project_rebalance_dates(
    tracking_nav: pd.DataFrame,
    future_trade_dates: Iterable[Any],
) -> pd.DataFrame:
    """Project one account's frozen phase onto later exchange trading dates."""
    phase = _latest_phase(tracking_nav)
    future = _dates(future_trade_dates)
    future = future[future > phase["date"]]
    if future.empty:
        return pd.DataFrame(
            columns=["date", "day_index", "is_rebalance_day", "rebalance_every", "rebalance_offset"]
        )
    day_indices = phase["day_index"] + np.arange(1, len(future) + 1, dtype=int)
    is_rebalance = (
        (day_indices - int(phase["rebalance_offset"]))
        % int(phase["rebalance_every"])
        == 0
    )
    return pd.DataFrame(
        {
            "date": future,
            "day_index": day_indices,
            "is_rebalance_day": is_rebalance,
            "rebalance_every": int(phase["rebalance_every"]),
            "rebalance_offset": int(phase["rebalance_offset"]),
        }
    )


def _anchor_frame(phase: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": phase["date"],
                "day_index": int(phase["day_index"]),
                "is_rebalance_day": bool(phase["is_rebalance_day"]),
                "rebalance_every": int(phase["rebalance_every"]),
                "rebalance_offset": int(phase["rebalance_offset"]),
                "schedule_source": "tracking_actual",
            }
        ]
    )


def build_rebalance_schedule(
    matrix_root: Path,
    future_trade_dates: Iterable[Any],
    *,
    production_experiment: str = DEFAULT_PRODUCTION_EXPERIMENT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-form schedule plus per-strategy status.

    Each strategy contributes its latest actual tracking day plus projected later
    exchange sessions. A strategy with missing/corrupt tracking phase is retained
    in the status table with ``status=error`` but contributes no guessed rows.
    """
    matrix_root = Path(matrix_root).expanduser().resolve()
    future = _dates(future_trade_dates)
    schedule_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    for name in discover_experiment_names(matrix_root):
        identity = parse_experiment_name(name)
        item = load_experiment(matrix_root, name)
        try:
            phase = _latest_phase(item["tracking_nav"])
            projection = project_rebalance_dates(item["tracking_nav"], future)
        except Exception as exc:
            status_rows.append(
                {
                    "experiment": name,
                    "display_name": identity.display_name,
                    "target": identity.target,
                    "signal": identity.signal,
                    "is_production": name == production_experiment,
                    "status": "error",
                    "latest_tracking_date": None,
                    "rebalance_every": identity.rebalance_every,
                    "rebalance_offset": None,
                    "next_rebalance_date": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        anchor = _anchor_frame(phase)
        if not projection.empty:
            projection = projection.copy()
            projection["schedule_source"] = "phase_projection"
            combined = pd.concat([anchor, projection], ignore_index=True)
            reb = projection.loc[projection["is_rebalance_day"].astype(bool), "date"]
            next_rebalance = pd.Timestamp(reb.iloc[0]).normalize() if len(reb) else None
        else:
            combined = anchor
            next_rebalance = None

        combined["experiment"] = name
        combined["display_name"] = identity.display_name
        combined["target"] = identity.target
        combined["signal"] = identity.signal
        combined["is_production"] = name == production_experiment
        schedule_rows.append(combined)

        status_rows.append(
            {
                "experiment": name,
                "display_name": identity.display_name,
                "target": identity.target,
                "signal": identity.signal,
                "is_production": name == production_experiment,
                "status": "ok",
                "latest_tracking_date": phase["date"],
                "rebalance_every": phase["rebalance_every"],
                "rebalance_offset": phase["rebalance_offset"],
                "next_rebalance_date": next_rebalance,
                "error": None,
            }
        )

    schedule = pd.concat(schedule_rows, ignore_index=True) if schedule_rows else pd.DataFrame()
    if not schedule.empty:
        schedule["date"] = pd.to_datetime(schedule["date"]).dt.normalize()
        schedule = schedule.sort_values(["date", "rebalance_every", "signal"]).reset_index(drop=True)
    status = pd.DataFrame(status_rows)
    return schedule, status


def daily_rebalance_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long schedule into one row per represented trading day."""
    if schedule.empty:
        return pd.DataFrame(
            columns=["date", "rebalance_count", "rebalance_experiments", "production_rebalance"]
        )
    rows: list[dict[str, Any]] = []
    for trade_date, group in schedule.groupby("date", sort=True):
        due = group.loc[group["is_rebalance_day"].astype(bool)].copy()
        rows.append(
            {
                "date": pd.Timestamp(trade_date).normalize(),
                "rebalance_count": int(len(due)),
                "rebalance_experiments": due["display_name"].astype(str).tolist(),
                "rebalance_codes": due["experiment"].astype(str).tolist(),
                "production_rebalance": bool(due["is_production"].astype(bool).any()) if not due.empty else False,
            }
        )
    return pd.DataFrame(rows)
