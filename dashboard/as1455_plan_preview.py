#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only start-date-aware plan access for the Streamlit dashboard.

Expensive/lightweight replay alike is deliberately excluded from the page read
path.  A tracking-start change rebuilds accounts and materializes all saved live
plans once in the background.  The dashboard then only loads persisted CSV/JSON
artifacts.  A freshly generated canonical live plan may also be read directly
when its tracking-start metadata already matches the current configuration.

The dashboard monitor selection is intentionally separate from production.  The
production experiment remains fixed elsewhere; this module only filters the
persisted plans that the page attempts to display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.as1455_live_data import load_live_day, load_strategy
from dashboard.as1455_monitor_config import load_monitor_experiments
from utils.as1455_materialized_plan import (
    load_materialized_day,
    load_materialized_strategy,
    materialized_ready,
)
from utils.as1455_tracking import TRACKING_SEMANTICS_VERSION


def _normalize_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _filter_summary(summary: pd.DataFrame, requested: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if summary.empty or "experiment" not in summary.columns:
        return pd.DataFrame(), list(requested)
    available = set(summary["experiment"].astype(str))
    selected = [name for name in requested if name in available]
    missing = [name for name in requested if name not in available]
    if not selected:
        return pd.DataFrame(), missing
    order = {name: index for index, name in enumerate(requested)}
    out = summary.loc[summary["experiment"].astype(str).isin(selected)].copy()
    out["_monitor_order"] = out["experiment"].astype(str).map(order)
    out = out.sort_values("_monitor_order", kind="stable").drop(columns=["_monitor_order"])
    return out.reset_index(drop=True), missing


def _details_from_materialized(
    day: dict[str, Any],
    requested: list[str],
) -> dict[str, dict[str, Any]]:
    summary, _ = _filter_summary(day["summary"], requested)
    details: dict[str, dict[str, Any]] = {}
    if summary.empty:
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


def _details_from_canonical(
    day: dict[str, Any],
    requested: list[str],
) -> dict[str, dict[str, Any]]:
    summary, _ = _filter_summary(day["summary"], requested)
    details: dict[str, dict[str, Any]] = {}
    if summary.empty:
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
    )


def preview_nine_strategy_day(
    matrix_root: Path,
    live_root: Path,
    start: pd.Timestamp,
    selected: pd.Timestamp,
    **_: Any,
) -> dict[str, Any]:
    """Load, never compute, one start-date-aware configured-monitor plan.

    The legacy function name is retained for callers.  The returned rows follow
    ``user_config.json.monitor_experiments`` and default to the production
    ``r21_best`` strategy.  Missing configured research-monitor plans are
    reported but do not invalidate an otherwise available production plan.
    """
    matrix_root = Path(matrix_root).expanduser().resolve()
    live_root = Path(live_root).expanduser().resolve()
    start = _normalize_date(start)
    selected = _normalize_date(selected)
    token = selected.strftime("%Y%m%d")
    requested = load_monitor_experiments(matrix_root)

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
            "monitor_experiments": requested,
            "missing_monitor_experiments": [],
        }

    if materialized_ready(matrix_root, start):
        try:
            day = load_materialized_day(
                live_root,
                token,
                start,
                TRACKING_SEMANTICS_VERSION,
            )
            filtered, missing = _filter_summary(day["summary"], requested)
            details = _details_from_materialized(day, requested)
            if not filtered.empty and len(details) == len(filtered):
                manifest = day["manifest"]
                return {
                    "status": "ok",
                    "tracking_start_date": start.strftime("%Y-%m-%d"),
                    "selected_date": selected.strftime("%Y-%m-%d"),
                    "summary": filtered,
                    "details": details,
                    "prediction_source": "persisted_saved_predictions",
                    "execution_source": manifest.get("execution_source")
                    or "materialized_start_date_plan",
                    "raw_daily_fallback_dates": manifest.get("raw_daily_fallback_dates") or [],
                    "model_inference_rerun": False,
                    "historical_grid_rerun": False,
                    "dashboard_replay_rerun": False,
                    "plan_source": "materialized_start_date_plan",
                    "monitor_experiments": requested,
                    "missing_monitor_experiments": missing,
                }
        except FileNotFoundError:
            pass

    # Today's production job intentionally persists only the production strategy.
    # A partial canonical day is therefore valid as long as its tracking metadata
    # matches the configured account and at least one requested monitor exists.
    canonical = load_live_day(live_root, token)
    if _canonical_matches_start(canonical, start):
        filtered, missing = _filter_summary(canonical["summary"], requested)
        details = _details_from_canonical(canonical, requested)
        if not filtered.empty and len(details) == len(filtered):
            return {
                "status": "ok",
                "tracking_start_date": start.strftime("%Y-%m-%d"),
                "selected_date": selected.strftime("%Y-%m-%d"),
                "summary": filtered,
                "details": details,
                "prediction_source": "canonical_live_job",
                "execution_source": "canonical_1455_plan",
                "raw_daily_fallback_dates": [],
                "model_inference_rerun": False,
                "historical_grid_rerun": False,
                "dashboard_replay_rerun": False,
                "plan_source": "canonical_current_start_plan",
                "monitor_experiments": requested,
                "missing_monitor_experiments": missing,
            }

    raise RuntimeError(
        "当前起算日的已配置盯盘模型尚无可用计划缓存。页面只读取已经落盘的计划，"
        "不会即时运行模型或重放策略。"
    )
