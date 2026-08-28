#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure dashboard status semantics for start-date-aware tracking accounts.

This module is intentionally presentation-only.  It never starts a refresh,
changes account state, or alters production trading behavior.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

WAITING_FOR_MARKET_DAY = "waiting_for_completed_market_day"


def tracking_status_view(
    summary: pd.DataFrame,
    manifest: dict[str, Any],
    selected_start: pd.Timestamp,
    semantics_version: int,
    refresh_state: str,
    *,
    expected_experiments: int = 9,
) -> dict[str, Any]:
    """Return one unambiguous UI state for tracking readiness.

    ``refresh_state`` describes the background process.  ``summary/manifest``
    describe the account state.  Keeping these concepts separate prevents a
    successfully completed rebuild that is merely waiting for today's market
    data from being displayed as "still rebuilding".
    """
    start = pd.Timestamp(selected_start).normalize()
    expected_start = start.strftime("%Y-%m-%d")
    refresh = str(refresh_state or "idle")

    try:
        total = int(manifest.get("experiment_count", expected_experiments) or expected_experiments)
    except (TypeError, ValueError):
        total = expected_experiments
    total = total if total > 0 else expected_experiments
    try:
        completed = int(manifest.get("completed_experiment_count", 0) or 0)
    except (TypeError, ValueError):
        completed = 0

    statuses: set[str] = set()
    if not summary.empty and "status" in summary.columns:
        statuses = set(summary["status"].dropna().astype(str))

    config_matches = (
        manifest.get("tracking_start_date") == expected_start
        and int(manifest.get("tracking_semantics_version", 0) or 0) == int(semantics_version)
    )
    ready = (
        config_matches
        and manifest.get("status") == "ok"
        and completed == expected_experiments
        and len(summary) == expected_experiments
        and (not statuses or statuses == {"ok"})
    )

    if refresh == "running":
        state = "running"
        headline = "正在重建"
        detail = f"当前起算日账户正在后台刷新（{completed}/{expected_experiments}）。"
    elif ready:
        state = "ready"
        headline = "已就绪"
        detail = f"当前起算日账户已完成（{completed}/{expected_experiments}）。"
    elif config_matches and len(summary) == expected_experiments and statuses == {WAITING_FOR_MARKET_DAY}:
        state = "waiting_market_day"
        headline = "等待收盘数据"
        detail = (
            f"9个账户已按 {expected_start} 起算口径初始化，但尚无从该日期开始的完整市场日；"
            "无需重复重建。若今天为交易日，通常在20:20市场数据刷新后自动继续。"
        )
    elif refresh in {"failed", "stale"}:
        state = "failed"
        headline = "刷新失败"
        detail = "最近一次后台账户刷新未正常完成，请查看任务日志。"
    elif config_matches and manifest.get("status") == "partial":
        state = "partial"
        headline = f"部分完成 {completed}/{expected_experiments}"
        detail = "已有部分策略账户完成，但并非全部处于可用状态，请查看各策略状态或任务日志。"
    else:
        state = "not_ready"
        headline = "未就绪"
        detail = "当前起算日账户尚未形成一致的可用状态。"

    return {
        "state": state,
        "headline": headline,
        "detail": detail,
        "completed": completed,
        "total": total,
        "expected": expected_experiments,
        "config_matches": config_matches,
        "statuses": sorted(statuses),
    }
