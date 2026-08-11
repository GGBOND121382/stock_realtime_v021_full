#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only start-date-aware plan access for the Streamlit dashboard.

Expensive/lightweight replay alike is deliberately excluded from the page read
path.  A tracking-start change rebuilds accounts and materializes all saved live
plans once in the background.  The dashboard then only loads persisted CSV/JSON
artifacts.  A freshly generated canonical live plan may also be read directly
when its tracking-start metadata already matches the current configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.as1455_live_data import load_live_day, load_strategy
from utils.as1455_materialized_plan import (
    load_materialized_day,
    load_materialized_strategy,
    materialized_ready,
)
from utils.as1455_tracking import TRACKING_SEMANTICS_VERSION


def _normalize_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _details_from_materialized(day: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = day["summary"]
    details: dict[str, dict[str, Any]] = {}
    if summary.empty or "experiment" not in summary.columns:
        return details
    for experiment in summary["experiment"].astype(str):
        item = load_materialized_strategy(day, experiment)
        details[experiment] = {
            "manifest": item["manifest"],
            "orders": item["orders"],
            "target_positions": item["target_positions"],
            "current_positions": item["current_positions"],
            "rejections": item["rejections"],
            "rank": item["rank"],
            "nav": item["nav"],
            "phase": {},
        }
    return details


def _details_from_canonical(day: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = day["summary"]
    details: dict[str, dict[str, Any]] = {}
    if summary.empty or "experiment" not in summary.columns:
        return details
    for experiment in summary["experiment"].astype(str):
        item = load_strategy(day, experiment)
        details[experiment] = {
            "manifest": item["manifest"],
            "orders": item["orders"],
            "target_positions": item["target_positions"],
            "current_positions": item["current_positions"],
            "rejections": item["rejections"],
            "rank": item["rank"],
            "nav": item["nav"],
            "phase": {},
        }
    return details


def _canonical_matches_start(
    day: dict[str, Any],
    start: pd.Timestamp,
) -> bool:
    manifest = day.get("manifest") or {}
    return (
        manifest.get("status") == "ok"
        and manifest.get("tracking_start_date") == start.strftime("%Y-%m-%d")
        and int(manifest.get("tracking_semantics_version", 0) or 0)
        == TRACKING_SEMANTICS_VERSION
        and int(manifest.get("experiment_count", 0) or 0) == 9
    )


def preview_nine_strategy_day(
    matrix_root: Path,
    live_root: Path,
    start: pd.Timestamp,
    selected: pd.Timestamp,
    **_: Any,
) -> dict[str, Any]:
    """Load, never compute, one start-date-aware nine-strategy plan."""
    matrix_root = Path(matrix_root).expanduser().resolve()
    live_root = Path(live_root).expanduser().resolve()
    start = _normalize_date(start)
    selected = _normalize_date(selected)
    token = selected.strftime("%Y%m%d")

    if selected < start:
        return {
            "status": "before_start",
            "tracking_start_date": start.strftime("%Y-%m-%d"),
            "selected_date": selected.strftime("%Y-%m-%d"),
            "summary": pd.DataFrame(),
            "details": {},
            "execution_source": "before_tracking_start",
            "raw_daily_fallback_dates": [],
            "model_inference_rerun": False,
            "historical_grid_rerun": False,
            "dashboard_replay_rerun": False,
        }

    if materialized_ready(matrix_root, start):
        try:
            day = load_materialized_day(
                live_root,
                token,
                start,
                TRACKING_SEMANTICS_VERSION,
            )
            details = _details_from_materialized(day)
            if len(day["summary"]) != 9 or len(details) != 9:
                raise RuntimeError(
                    f"materialized plan is incomplete: summary={len(day['summary'])} details={len(details)}"
                )
            manifest = day["manifest"]
            return {
                "status": "ok",
                "tracking_start_date": start.strftime("%Y-%m-%d"),
                "selected_date": selected.strftime("%Y-%m-%d"),
                "summary": day["summary"],
                "details": details,
                "prediction_source": "persisted_saved_predictions",
                "execution_source": manifest.get("execution_source")
                or "materialized_start_date_plan",
                "raw_daily_fallback_dates": manifest.get("raw_daily_fallback_dates") or [],
                "model_inference_rerun": False,
                "historical_grid_rerun": False,
                "dashboard_replay_rerun": False,
                "plan_source": "materialized_start_date_plan",
            }
        except FileNotFoundError:
            pass

    # Today's normal live job already uses the current tracking account.  It can
    # be read immediately without waiting for another materialization pass.
    canonical = load_live_day(live_root, token)
    if _canonical_matches_start(canonical, start):
        details = _details_from_canonical(canonical)
        if len(canonical["summary"]) == 9 and len(details) == 9:
            return {
                "status": "ok",
                "tracking_start_date": start.strftime("%Y-%m-%d"),
                "selected_date": selected.strftime("%Y-%m-%d"),
                "summary": canonical["summary"],
                "details": details,
                "prediction_source": "canonical_live_job",
                "execution_source": "canonical_1455_plan",
                "raw_daily_fallback_dates": [],
                "model_inference_rerun": False,
                "historical_grid_rerun": False,
                "dashboard_replay_rerun": False,
                "plan_source": "canonical_current_start_plan",
            }

    raise RuntimeError(
        "当前起算日的盯盘计划缓存尚未完成。修改起算日后请等待一次后台统一重建；"
        "页面查看本身不会再即时重放9个策略。"
    )
