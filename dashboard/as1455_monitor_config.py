#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dashboard-only configuration for which frozen AS1455 strategies are monitored.

This setting never changes the production trading strategy.  The live production
experiment remains fixed elsewhere; this module only controls what the dashboard
tries to display.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DEFAULT_PRODUCTION_EXPERIMENT = "r21_best_reb21_fold0_4_forward"
MONITOR_CONFIG_KEY = "monitor_experiments"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def config_file(matrix_root: Path) -> Path:
    return Path(matrix_root).expanduser().resolve() / ".dashboard" / "user_config.json"


def normalize_monitor_experiments(
    values: Iterable[object] | None,
    available: Iterable[str] | None = None,
) -> list[str]:
    allowed = set(str(x) for x in available) if available is not None else None
    result: list[str] = []
    for value in values or []:
        name = str(value).strip()
        if not name or (allowed is not None and name not in allowed):
            continue
        if name not in result:
            result.append(name)
    if DEFAULT_PRODUCTION_EXPERIMENT not in result and (
        allowed is None or DEFAULT_PRODUCTION_EXPERIMENT in allowed
    ):
        result.insert(0, DEFAULT_PRODUCTION_EXPERIMENT)
    return result or [DEFAULT_PRODUCTION_EXPERIMENT]


def load_monitor_experiments(
    matrix_root: Path,
    available: Iterable[str] | None = None,
) -> list[str]:
    payload = _read_json(config_file(matrix_root))
    raw = payload.get(MONITOR_CONFIG_KEY)
    if not isinstance(raw, list):
        raw = [DEFAULT_PRODUCTION_EXPERIMENT]
    return normalize_monitor_experiments(raw, available)


def save_monitor_experiments(
    matrix_root: Path,
    values: Iterable[object],
    available: Iterable[str] | None = None,
) -> list[str]:
    selected = normalize_monitor_experiments(values, available)
    path = config_file(matrix_root)
    payload = _read_json(path)
    payload[MONITOR_CONFIG_KEY] = selected
    _write_json(path, payload)
    return selected
