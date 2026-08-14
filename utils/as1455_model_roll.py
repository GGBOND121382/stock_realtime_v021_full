#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State transitions for AS1455 rolling production model generations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from utils import as1455_ch17_common as common
from utils.as1455_model_registry import (
    DEFAULT_REGISTRY_ROOT,
    LEGACY_GENERATION,
    atomic_write_json,
    bootstrap_registry,
    generation_id,
    generation_index,
    period_id,
    validate_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION_CACHE_BASE = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_prediction_cache"
)


def _prediction_cache_file(
    feature_preset: str,
    target_col: str,
    cache_base: Path = DEFAULT_PREDICTION_CACHE_BASE,
) -> Path:
    return (
        cache_base
        / f"{feature_preset}_{target_col}_top5"
        / "fold0_forward_latest"
        / "00_predictions"
        / "fold0_forward_preds.h5"
    )


def _prediction_dates(path: Path) -> pd.DatetimeIndex:
    if not path.is_file():
        return pd.DatetimeIndex([])
    frame = pd.read_hdf(path, "predictions")
    if not isinstance(frame.index, pd.MultiIndex) or "date" not in frame.index.names:
        raise RuntimeError(f"unexpected prediction index in {path}: {frame.index.names}")
    dates = (
        pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("date")))
        .normalize()
        .unique()
        .sort_values()
    )
    del frame
    return dates


def legacy_common_forward_dates(
    feature_preset: str,
    cache_base: Path = DEFAULT_PREDICTION_CACHE_BASE,
) -> pd.DatetimeIndex:
    common_dates: set[pd.Timestamp] | None = None
    for target_col in common.TARGET_SPECS:
        dates = set(
            pd.Timestamp(value).normalize()
            for value in _prediction_dates(
                _prediction_cache_file(feature_preset, target_col, cache_base)
            )
        )
        if not dates:
            return pd.DatetimeIndex([])
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    return pd.DatetimeIndex(sorted(common_dates or set())).normalize()


def _normalized_date_strings(values: list[Any]) -> list[str]:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna().normalize()
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates.unique().sort_values()]


def ensure_legacy_period_initialized(
    registry_root: Path | str | None = None,
    *,
    feature_preset: str = "rotation_addon_onehot",
    cache_base: Path = DEFAULT_PREDICTION_CACHE_BASE,
) -> dict[str, Any]:
    """Merge the pre-registry legacy forward history into period000 once.

    This can read the historical fold0-forward HDF caches, so callers on the
    14:55 critical path must not invoke it.  The 21:30 rollover checker calls it
    instead.  Live planning only appends the successfully completed current day.
    """
    root = Path(registry_root or DEFAULT_REGISTRY_ROOT).expanduser().resolve()
    registry = bootstrap_registry(root, feature_preset=feature_preset)
    current = dict(registry.get("current_period") or {})
    if registry.get("active_generation") != LEGACY_GENERATION:
        return registry
    if bool(current.get("legacy_cache_initialized")):
        return registry

    existing = _normalized_date_strings(list(current.get("observed_dates") or []))
    legacy_dates = legacy_common_forward_dates(feature_preset, cache_base)
    if not len(legacy_dates):
        return registry
    merged = _normalized_date_strings(
        existing + [value.strftime("%Y-%m-%d") for value in legacy_dates]
    )
    current["period_id"] = current.get("period_id") or period_id(0)
    current["period_index"] = int(current.get("period_index", 0) or 0)
    current["generation_id"] = LEGACY_GENERATION
    current["observed_dates"] = merged
    current["observed_days"] = len(merged)
    current["required_days"] = int(registry.get("period_length", 63) or 63)
    current["start_date"] = merged[0] if merged else None
    current["last_observed_date"] = merged[-1] if merged else None
    current["legacy_cache_initialized"] = True
    registry = deepcopy(registry)
    registry["current_period"] = current
    atomic_write_json(root / "registry.json", registry)
    return registry


def record_live_generation_use(
    registry_root: Path | str | None,
    *,
    trade_date: str,
    feature_preset: str = "rotation_addon_onehot",
    initialize_legacy: bool = False,
) -> dict[str, Any]:
    """Append one successfully completed live day to the active period.

    ``initialize_legacy`` defaults to False so production post-processing stays
    O(1).  Historical HDF reconciliation is deferred to ``rollover_status``.
    """
    root = Path(registry_root or DEFAULT_REGISTRY_ROOT).expanduser().resolve()
    registry = (
        ensure_legacy_period_initialized(root, feature_preset=feature_preset)
        if initialize_legacy
        else bootstrap_registry(root, feature_preset=feature_preset)
    )
    validate_registry(registry)
    date_text = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    active_generation = str(registry["active_generation"])
    updated = deepcopy(registry)

    for target_col in common.TARGET_SPECS:
        entry = updated["active_models"][target_col]
        if str(entry.get("generation_id")) != active_generation:
            raise RuntimeError(
                "active target generations are inconsistent: "
                f"registry={active_generation} {target_col}={entry.get('generation_id')}"
            )
        if not entry.get("model_updated_date"):
            entry["model_updated_date"] = date_text
            entry["effective_from"] = date_text

    for generation in updated.get("generations", []):
        if str(generation.get("generation_id")) != active_generation:
            continue
        if not generation.get("model_updated_date"):
            generation["model_updated_date"] = date_text
        targets = generation.get("targets")
        if isinstance(targets, dict):
            for target_col in common.TARGET_SPECS:
                target = targets.get(target_col)
                if isinstance(target, dict) and not target.get("model_updated_date"):
                    target["model_updated_date"] = date_text
                    target["effective_from"] = date_text

    current = dict(updated.get("current_period") or {})
    if str(current.get("generation_id")) != active_generation:
        raise RuntimeError(
            "current period generation differs from active generation: "
            f"period={current.get('generation_id')} active={active_generation}"
        )
    dates = _normalized_date_strings(list(current.get("observed_dates") or []) + [date_text])
    current["observed_dates"] = dates
    current["observed_days"] = len(dates)
    current["required_days"] = int(updated.get("period_length", 63) or 63)
    current["start_date"] = dates[0] if dates else None
    current["last_observed_date"] = dates[-1] if dates else None
    updated["current_period"] = current
    atomic_write_json(root / "registry.json", updated)
    return updated


def rollover_status(
    registry_root: Path | str | None = None,
    *,
    feature_preset: str = "rotation_addon_onehot",
) -> dict[str, Any]:
    # This is intentionally the non-live path where the one-time legacy HDF scan
    # is allowed to happen.
    registry = ensure_legacy_period_initialized(
        registry_root, feature_preset=feature_preset
    )
    current = dict(registry.get("current_period") or {})
    dates = _normalized_date_strings(list(current.get("observed_dates") or []))
    required = int(current.get("required_days") or registry.get("period_length", 63) or 63)
    due = len(dates) >= required
    boundary = dates[-1] if due else None
    return {
        "status": "due" if due else "waiting",
        "due": due,
        "active_generation": registry.get("active_generation"),
        "period_id": current.get("period_id"),
        "period_index": int(current.get("period_index", 0) or 0),
        "observed_days": len(dates),
        "required_days": required,
        "remaining_days": max(0, required - len(dates)),
        "period_start": dates[0] if dates else None,
        "period_last_observed": dates[-1] if dates else None,
        "rollover_boundary": boundary,
        "legacy_cache_initialized": bool(current.get("legacy_cache_initialized")),
    }


def next_generation(registry: dict[str, Any]) -> str:
    indices = [
        generation_index(str(item["generation_id"]))
        for item in registry.get("generations", [])
        if isinstance(item, dict) and item.get("generation_id")
    ]
    return generation_id((max(indices) if indices else -1) + 1)


def activate_generation(
    registry_root: Path | str,
    *,
    generation: dict[str, Any],
    period_end: str,
    feature_preset: str = "rotation_addon_onehot",
) -> dict[str, Any]:
    root = Path(registry_root).expanduser().resolve()
    registry = bootstrap_registry(root, feature_preset=feature_preset)
    validate_registry(registry)
    new_id = str(generation.get("generation_id"))
    if not new_id:
        raise RuntimeError("generation record lacks generation_id")
    generation_index(new_id)
    if any(str(item.get("generation_id")) == new_id for item in registry.get("generations", [])):
        raise RuntimeError(f"generation already registered: {new_id}")
    targets = generation.get("targets")
    if not isinstance(targets, dict):
        raise RuntimeError("generation record lacks targets")
    missing = [target for target in common.TARGET_SPECS if target not in targets]
    if missing:
        raise RuntimeError(f"generation misses targets: {missing}")

    updated = deepcopy(registry)
    closed = dict(updated.get("current_period") or {})
    closed["end_date"] = pd.Timestamp(period_end).strftime("%Y-%m-%d")
    updated.setdefault("completed_periods", []).append(closed)
    new_period_index = int(closed.get("period_index", 0) or 0) + 1

    clean_targets: dict[str, dict[str, Any]] = {}
    for target_col in common.TARGET_SPECS:
        entry = dict(targets[target_col])
        entry["generation_id"] = new_id
        entry["model_updated_date"] = None
        entry["effective_from"] = None
        clean_targets[target_col] = entry
    generation = dict(generation)
    generation["targets"] = clean_targets
    generation["model_updated_date"] = None
    generation["effective_after"] = pd.Timestamp(period_end).strftime("%Y-%m-%d")

    updated.setdefault("generations", []).append(generation)
    updated["active_generation"] = new_id
    updated["active_models"] = clean_targets
    updated["current_period"] = {
        "period_id": period_id(new_period_index),
        "period_index": new_period_index,
        "generation_id": new_id,
        "start_date": None,
        "observed_dates": [],
        "observed_days": 0,
        "required_days": int(updated.get("period_length", 63) or 63),
        "last_observed_date": None,
        "legacy_cache_initialized": True,
    }
    atomic_write_json(root / "registry.json", updated)
    return updated
