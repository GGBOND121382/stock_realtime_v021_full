#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared tracking-account helpers for the AS1455 strategy dashboard.

The canonical strict-forward experiments remain immutable research artifacts.
The dashboard tracking account is a separate state machine that starts empty on
``tracking_start_date`` and preserves the frozen historical rebalance phase.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

USER_CONFIG = Path(".dashboard") / "user_config.json"
TRACKING_SEMANTICS_VERSION = 3
TRACKING_MANIFEST = "tracking_forward_manifest.json"
TRACKING_RESULT = "tracking_forward_result.csv"
TRACKING_NAV = "tracking_forward_nav.csv"
TRACKING_ORDERS = "tracking_forward_orders.csv"
TRACKING_REJECTIONS = "tracking_forward_rejections.csv"
TRACKING_POSITIONS = "tracking_forward_positions.csv"
TRACKING_LATEST_STATE = "tracking_forward_latest_state.json"
TRACKING_LATEST_POSITIONS = "tracking_forward_latest_positions.csv"
TRACKING_MATRIX_SUMMARY = "tracking_matrix_summary.csv"
TRACKING_MATRIX_MANIFEST = "tracking_matrix_manifest.json"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def tracking_start_date(matrix_root: Path) -> pd.Timestamp | None:
    payload = read_json(matrix_root / USER_CONFIG, {}) or {}
    value = payload.get("tracking_start_date")
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def contiguous_tracking_dates(
    prediction_dates: pd.DatetimeIndex,
    execution_calendar: pd.DatetimeIndex,
    lower_bound: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return contiguous executable/predicted dates from ``lower_bound``.

    Dates before the first available prediction may be skipped (for example a
    weekend start). Once tracking starts, a missing prediction stops advancement
    so the account never jumps across an unprocessed trading day.
    """
    prediction_dates = pd.DatetimeIndex(prediction_dates).normalize().unique().sort_values()
    execution_calendar = pd.DatetimeIndex(execution_calendar).normalize().unique().sort_values()
    candidates = execution_calendar[execution_calendar >= pd.Timestamp(lower_bound).normalize()]
    if candidates.empty:
        return pd.DatetimeIndex([])
    available = set(prediction_dates)
    started = False
    selected: list[pd.Timestamp] = []
    for value in candidates:
        date = pd.Timestamp(value).normalize()
        if not started:
            if date not in available:
                continue
            started = True
            selected.append(date)
            continue
        if date not in available:
            break
        selected.append(date)
    return pd.DatetimeIndex(selected)


def experiment_tracking_paths(experiment_root: Path) -> dict[str, Path]:
    return {
        "manifest": experiment_root / TRACKING_MANIFEST,
        "result": experiment_root / TRACKING_RESULT,
        "nav": experiment_root / TRACKING_NAV,
        "orders": experiment_root / TRACKING_ORDERS,
        "rejections": experiment_root / TRACKING_REJECTIONS,
        "positions": experiment_root / TRACKING_POSITIONS,
        "latest_state": experiment_root / TRACKING_LATEST_STATE,
        "latest_positions": experiment_root / TRACKING_LATEST_POSITIONS,
    }


def tracking_manifest_matches(experiment_root: Path, start: pd.Timestamp) -> bool:
    payload = read_json(experiment_root / TRACKING_MANIFEST, {}) or {}
    return (
        payload.get("status") == "ok"
        and payload.get("tracking_start_date") == start.strftime("%Y-%m-%d")
        and int(payload.get("tracking_semantics_version", 0) or 0)
        == TRACKING_SEMANTICS_VERSION
    )


def resolve_initial_cash(experiment_root: Path, default: float = 200_000.0) -> float:
    """Read the frozen strict-forward run's configured initial cash."""
    strict_file = (
        experiment_root
        / "strict_oos_forward"
        / "01_close_auction_grid"
        / "strict_oos_manifest.json"
    )
    strict = read_json(strict_file, {}) or {}
    run_name = strict.get("retained_run_name")
    if run_name:
        config = read_json(
            strict_file.parent / "01_runs" / str(run_name) / "config.json", {}
        ) or {}
        value = pd.to_numeric(
            pd.Series([config.get("initial_cash")]), errors="coerce"
        ).iloc[0]
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return float(default)


def load_latest_tracking_state(
    experiment_root: Path,
    start: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = experiment_tracking_paths(experiment_root)
    manifest = read_json(paths["manifest"], {}) or {}
    if manifest.get("tracking_start_date") != start.strftime("%Y-%m-%d"):
        raise RuntimeError(
            f"tracking start mismatch for {experiment_root.name}: "
            f"expected={start:%Y-%m-%d} actual={manifest.get('tracking_start_date')}"
        )
    if int(manifest.get("tracking_semantics_version", 0) or 0) != TRACKING_SEMANTICS_VERSION:
        raise RuntimeError(
            f"tracking semantics are stale for {experiment_root.name}: "
            f"expected={TRACKING_SEMANTICS_VERSION} "
            f"actual={manifest.get('tracking_semantics_version')}"
        )
    state = read_json(paths["latest_state"], {}) or {}
    if not state:
        raise FileNotFoundError(paths["latest_state"])
    if paths["latest_positions"].is_file():
        try:
            positions = pd.read_csv(paths["latest_positions"], encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            positions = pd.DataFrame()
    else:
        positions = pd.DataFrame()
    return state, positions
