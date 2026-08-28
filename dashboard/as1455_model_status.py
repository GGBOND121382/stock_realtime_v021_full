#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only model-generation helpers for the AS1455 Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.as1455_model_registry import (
    load_registry,
    load_snapshot,
    model_display_for_experiment,
    snapshot_model,
    target_col_from_experiment,
)


def load_dashboard_registry(
    registry_root: Path,
    *,
    feature_preset: str = "rotation_addon_onehot",
) -> dict[str, Any]:
    return load_registry(
        registry_root,
        feature_preset=feature_preset,
        allow_virtual_legacy=True,
    )


def current_model_info(
    registry: dict[str, Any], experiment: str
) -> dict[str, Any]:
    return model_display_for_experiment(registry, experiment)


def attach_current_model_columns(
    summary: pd.DataFrame,
    registry: dict[str, Any],
) -> pd.DataFrame:
    out = summary.copy()
    if out.empty or "experiment" not in out.columns:
        return out
    for index, experiment in out["experiment"].astype(str).items():
        info = current_model_info(registry, experiment)
        for key, value in info.items():
            out.at[index, key] = value
    return out


def model_info_for_live_date(
    live_root: Path,
    date_token: str,
    experiment: str,
    registry: dict[str, Any],
) -> dict[str, Any]:
    target_col = target_col_from_experiment(experiment)
    if target_col is None:
        return current_model_info(registry, experiment)
    snapshot_path = live_root / str(date_token) / "13_active_model_snapshot.json"
    if snapshot_path.is_file():
        try:
            entry = snapshot_model(load_snapshot(snapshot_path), target_col)
            return {
                "model_generation": entry.get("generation_id"),
                "model_updated_date": entry.get("model_updated_date")
                or entry.get("effective_from"),
                "model_source_type": entry.get("source_type"),
                "model_train_start": entry.get("train_start"),
                "model_train_end": entry.get("train_end"),
                "model_trained_at": entry.get("trained_at"),
                "model_target_col": target_col,
            }
        except Exception:
            pass
    return current_model_info(registry, experiment)


def attach_live_model_columns(
    summary: pd.DataFrame,
    live_root: Path,
    date_token: str,
    registry: dict[str, Any],
) -> pd.DataFrame:
    out = summary.copy()
    if out.empty or "experiment" not in out.columns:
        return out
    for index, experiment in out["experiment"].astype(str).items():
        info = model_info_for_live_date(
            live_root, date_token, experiment, registry
        )
        for key, value in info.items():
            out.at[index, key] = value
    return out
