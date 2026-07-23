#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory feature overlay for AS1455 live inference.

The historical forward helper reads HDF from disk.  Live inference needs to
append only today's 31 base-feature rows without rewriting the whole model-data
file, while preserving the same rotation/addon/sector-encoding definitions.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from utils import as1455_ch17_common as common


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_inference_features_from_frame(
    data: pd.DataFrame,
    target_col: str,
    feature_preset: str,
    sector_encoding: str = "onehot",
    *,
    source_label: str = "historical_model_data+live_overlay",
) -> common.FeatureBuildResult:
    if feature_preset not in common.FEATURE_PRESETS:
        raise RuntimeError(
            f"bad feature_preset={feature_preset!r}; expected {common.FEATURE_PRESETS}"
        )
    spec = common.target_spec(target_col)
    frame = data.copy()
    rows_before = int(len(frame))
    if list(frame.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {frame.index.names}")
    if list(frame.columns) != common.base.EXPECTED_MODEL_COLUMNS:
        raise RuntimeError(f"unexpected model_data columns: {list(frame.columns)}")
    outcomes = frame.filter(like="fwd").columns.tolist()
    if outcomes != common.base.EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")

    all_dates = pd.DatetimeIndex(frame.index.get_level_values("date"))
    target_valid = frame[target_col].notna()
    target_dates = all_dates[target_valid.to_numpy()]
    target_valid_max = pd.Timestamp(target_dates.max()) if len(target_dates) else None
    feature_columns = [
        column for column in frame.columns
        if column not in common.base.EXPECTED_OUTCOMES
    ]
    frame = frame.dropna(subset=feature_columns).sort_index()
    if frame.empty:
        raise RuntimeError("no feature-complete rows available for live inference")
    y = frame[target_col].copy()
    X_base = frame[feature_columns].copy()
    if X_base.shape[1] != 31 or any("fwd" in c for c in X_base.columns):
        raise RuntimeError(f"bad live base X shape/columns: {X_base.shape}")

    X_rot, rotation_cols = common.base.add_sector_rotation_features(X_base)
    addon_cols: list[str] = []
    feature_groups: dict[str, list[str]] = {}
    if feature_preset == "rotation_onehot":
        X_context = X_rot
    else:
        X_context, addon_cols, feature_groups = (
            common.addon.add_compact_addon_features(X_rot)
        )
    X_final, no_scale_cols, sector_onehot_cols = common.base.apply_sector_encoding(
        X_context, sector_encoding
    )
    dates = pd.DatetimeIndex(X_final.index.get_level_values("date"))
    report = {
        "row_mode": "inference_features_only_live_overlay",
        "require_target": False,
        "rows_before_dropna": rows_before,
        "rows_after_feature_dropna": int(len(X_final)),
        "model_data_max_date": _format_date(all_dates.max()),
        "feature_valid_max_date": _format_date(dates.max()),
        "target_valid_max_date": _format_date(target_valid_max),
        "target_col": target_col,
        "target_lookahead": spec.lookahead,
        "feature_preset": feature_preset,
        "source_label": source_label,
        "base_feature_count": int(X_base.shape[1]),
        "rotation_feature_count": int(len(rotation_cols)),
        "addon_feature_count": int(len(addon_cols)),
        "final_feature_count": int(X_final.shape[1]),
        "sector_encoding": sector_encoding,
        "sector_onehot_count": int(len(sector_onehot_cols)),
        "addon_feature_cols": list(addon_cols),
        "addon_feature_groups": feature_groups,
    }
    return common.FeatureBuildResult(
        X=X_final,
        y=y,
        no_scale_cols=list(no_scale_cols),
        rotation_cols=list(rotation_cols),
        addon_cols=list(addon_cols),
        feature_groups=feature_groups,
        sector_onehot_cols=list(sector_onehot_cols),
        report=report,
    )
