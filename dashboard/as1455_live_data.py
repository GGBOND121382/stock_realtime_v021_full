#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only helpers for the AS1455 nine-strategy live-monitor page."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

DATE_RE = re.compile(r"^\d{8}$")


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


def discover_live_dates(live_root: Path) -> list[str]:
    if not live_root.is_dir():
        return []
    dates = []
    for item in live_root.iterdir():
        if not item.is_dir() or not DATE_RE.fullmatch(item.name):
            continue
        nine = item / "nine_strategy"
        canonical = nine / "live_nine_strategy_manifest.json"
        materialized = nine / "start_date_plan" / "live_nine_strategy_manifest.json"
        if canonical.is_file() or materialized.is_file():
            dates.append(item.name)
    return sorted(dates, reverse=True)


def load_live_day(live_root: Path, date_token: str) -> dict[str, Any]:
    day_root = live_root / date_token
    nine_root = day_root / "nine_strategy"
    return {
        "date_token": date_token,
        "day_root": day_root,
        "nine_root": nine_root,
        "manifest": read_json(nine_root / "live_nine_strategy_manifest.json", {}) or {},
        "summary": read_csv(nine_root / "live_nine_strategy_summary.csv"),
        "rebalance": read_csv(nine_root / "live_rebalance_strategies.csv"),
    }


def load_strategy(live_day: dict[str, Any], experiment: str) -> dict[str, Any]:
    root = Path(live_day["nine_root"]) / "strategies" / experiment
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


def load_job_status(live_root: Path, stage: str) -> dict[str, Any]:
    path = live_root / ".dashboard" / f"nine_strategy_{stage}_status.json"
    payload = read_json(path, {}) or {}
    payload["status_file"] = str(path)
    log_file = payload.get("log_file")
    if log_file:
        log = Path(str(log_file)).expanduser()
        if not log.is_absolute():
            log = Path.cwd() / log
        payload["resolved_log_file"] = str(log)
    return payload
