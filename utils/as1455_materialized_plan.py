#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted start-date-aware 14:55 plan cache.

The canonical live plan remains immutable evidence of what the live job produced.
When the user changes the tracking start date, the dashboard rebuild workflow
materializes a separate derived plan cache for every existing live date.  The UI
can then read these files without replaying nine portfolios on every view.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MATERIALIZED_DIR = "start_date_plan"
MATERIALIZED_MATRIX_MANIFEST = Path(".dashboard") / "start_date_plan_manifest.json"
PLAN_CACHE_VERSION = 1


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def day_root(live_root: Path, date_token: str) -> Path:
    return Path(live_root) / date_token / "nine_strategy" / MATERIALIZED_DIR


def matrix_manifest_path(matrix_root: Path) -> Path:
    return Path(matrix_root) / MATERIALIZED_MATRIX_MANIFEST


def materialized_ready(matrix_root: Path, start: pd.Timestamp) -> bool:
    manifest = read_json(matrix_manifest_path(matrix_root), {}) or {}
    return (
        manifest.get("status") == "ok"
        and int(manifest.get("plan_cache_version", 0) or 0) == PLAN_CACHE_VERSION
        and manifest.get("tracking_start_date")
        == pd.Timestamp(start).normalize().strftime("%Y-%m-%d")
    )


def write_materialized_day(
    live_root: Path,
    date_token: str,
    start: pd.Timestamp,
    preview: dict[str, Any],
    tracking_semantics_version: int,
) -> None:
    root = day_root(live_root, date_token)
    root.mkdir(parents=True, exist_ok=True)
    summary = preview.get("summary")
    if not isinstance(summary, pd.DataFrame) or len(summary) != 9:
        raise RuntimeError(f"materialized {date_token} requires 9 summary rows")
    details = preview.get("details") or {}
    if len(details) != 9:
        raise RuntimeError(f"materialized {date_token} requires 9 strategy details")

    # Manifest is written last, so readers never accept a partially refreshed day.
    atomic_csv(summary, root / "live_nine_strategy_summary.csv")
    for experiment, detail in details.items():
        strategy_root = root / "strategies" / str(experiment)
        manifest = dict(detail.get("manifest") or {})
        manifest.update(
            {
                "tracking_start_date": pd.Timestamp(start).normalize().strftime("%Y-%m-%d"),
                "tracking_semantics_version": int(tracking_semantics_version),
                "plan_cache_version": PLAN_CACHE_VERSION,
                "plan_source": "materialized_start_date_plan",
            }
        )
        atomic_csv(pd.DataFrame(detail.get("orders", pd.DataFrame())), strategy_root / "16_live_orders.csv")
        atomic_csv(pd.DataFrame(detail.get("rejections", pd.DataFrame())), strategy_root / "16_live_rejections.csv")
        atomic_csv(pd.DataFrame(detail.get("target_positions", pd.DataFrame())), strategy_root / "16_live_target_portfolio.csv")
        atomic_csv(pd.DataFrame(detail.get("current_positions", pd.DataFrame())), strategy_root / "current_positions_before_plan.csv")
        atomic_csv(pd.DataFrame(detail.get("rank", pd.DataFrame())), strategy_root / "live_rank.csv")
        atomic_csv(pd.DataFrame(detail.get("nav", pd.DataFrame())), strategy_root / "16_live_nav.csv")
        atomic_json(strategy_root / "strategy_manifest.json", manifest)

    atomic_json(
        root / "live_nine_strategy_manifest.json",
        {
            "status": "ok",
            "protocol": "as1455_materialized_start_date_plan_v1",
            "plan_cache_version": PLAN_CACHE_VERSION,
            "tracking_start_date": pd.Timestamp(start).normalize().strftime("%Y-%m-%d"),
            "tracking_semantics_version": int(tracking_semantics_version),
            "trade_date": pd.to_datetime(date_token, format="%Y%m%d").strftime("%Y-%m-%d"),
            "experiment_count": 9,
            "model_inference_rerun": bool(preview.get("model_inference_rerun", False)),
            "historical_grid_rerun": bool(preview.get("historical_grid_rerun", False)),
            "execution_source": preview.get("execution_source"),
            "raw_daily_fallback_dates": preview.get("raw_daily_fallback_dates") or [],
        },
    )


def load_materialized_day(
    live_root: Path,
    date_token: str,
    start: pd.Timestamp,
    tracking_semantics_version: int,
) -> dict[str, Any]:
    root = day_root(live_root, date_token)
    manifest = read_json(root / "live_nine_strategy_manifest.json", {}) or {}
    expected_start = pd.Timestamp(start).normalize().strftime("%Y-%m-%d")
    if (
        manifest.get("status") != "ok"
        or int(manifest.get("plan_cache_version", 0) or 0) != PLAN_CACHE_VERSION
        or manifest.get("tracking_start_date") != expected_start
        or int(manifest.get("tracking_semantics_version", 0) or 0)
        != int(tracking_semantics_version)
    ):
        raise FileNotFoundError(
            f"materialized plan is not ready for date={date_token} start={expected_start}"
        )
    return {
        "date_token": date_token,
        "nine_root": root,
        "manifest": manifest,
        "summary": read_csv(root / "live_nine_strategy_summary.csv"),
    }


def load_materialized_strategy(day: dict[str, Any], experiment: str) -> dict[str, Any]:
    root = Path(day["nine_root"]) / "strategies" / experiment
    return {
        "root": root,
        "manifest": read_json(root / "strategy_manifest.json", {}) or {},
        "orders": read_csv(root / "16_live_orders.csv"),
        "rejections": read_csv(root / "16_live_rejections.csv"),
        "target_positions": read_csv(root / "16_live_target_portfolio.csv"),
        "current_positions": read_csv(root / "current_positions_before_plan.csv"),
        "rank": read_csv(root / "live_rank.csv"),
        "nav": read_csv(root / "16_live_nav.csv"),
    }
