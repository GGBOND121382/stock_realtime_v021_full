#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature construction for true forward inference.

Training and historical backtests require a realized target.  Fold0-forward
inference does not: the newest feature rows must remain available even when
r01_fwd/r05_fwd/r21_fwd are not realized yet.  This module keeps the existing
feature definitions unchanged and only changes the row-retention contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils import as1455_ch17_common as common


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_inference_xy(
    path: Path,
    train_end: str | None,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load model features without requiring the future target to be known."""
    spec = common.target_spec(target_col)
    data = pd.read_hdf(path, "model_data")
    rows_before = int(len(data))

    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    if list(data.columns) != common.base.EXPECTED_MODEL_COLUMNS:
        raise RuntimeError(f"unexpected model_data columns: {list(data.columns)}")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != common.base.EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")

    all_dates = pd.DatetimeIndex(data.index.get_level_values("date"))
    model_data_max_date = pd.Timestamp(all_dates.max())
    end_ts = common.base.parse_train_end(train_end)
    if end_ts is not None:
        data = data.loc[all_dates <= end_ts]

    scoped_dates = pd.DatetimeIndex(data.index.get_level_values("date"))
    target_valid = data[target_col].notna()
    target_valid_dates = scoped_dates[target_valid.to_numpy()]
    target_valid_max_date = (
        pd.Timestamp(target_valid_dates.max()) if len(target_valid_dates) else None
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in common.base.EXPECTED_OUTCOMES
    ]
    data = data.dropna(subset=feature_columns).sort_index()
    if data.empty:
        raise RuntimeError("no feature-complete rows available for forward inference")

    y = data[target_col].copy()
    X = data[feature_columns].copy()
    if X.shape[1] != 31 or any("fwd" in column for column in X.columns):
        raise RuntimeError(f"bad forward X shape/columns: {X.shape}")

    feature_dates = pd.DatetimeIndex(X.index.get_level_values("date"))
    unlabeled = y.isna()
    report = {
        "row_mode": "inference_features_only",
        "require_target": False,
        "rows_before_dropna": rows_before,
        "rows_after_feature_dropna": int(len(X)),
        "unlabeled_prediction_rows": int(unlabeled.sum()),
        "unlabeled_prediction_dates": int(
            feature_dates[unlabeled.to_numpy()].nunique()
        ),
        "model_data_max_date": _format_date(model_data_max_date),
        "train_end_effective": _format_date(
            pd.Timestamp(feature_dates.max()) if end_ts is None else end_ts
        ),
        "feature_valid_max_date": _format_date(feature_dates.max()),
        "target_valid_max_date": _format_date(target_valid_max_date),
        "target_col": target_col,
        "target_lookahead": spec.lookahead,
        "dropna_mode": "feature_columns_only",
    }
    return X, y, report


def build_inference_features(
    model_data: Path,
    train_end: str | None,
    target_col: str,
    feature_preset: str,
    sector_encoding: str = "onehot",
) -> common.FeatureBuildResult:
    """Build the same feature matrix as training while retaining latest rows."""
    if feature_preset not in common.FEATURE_PRESETS:
        raise RuntimeError(
            f"bad feature_preset={feature_preset!r}; "
            f"expected {common.FEATURE_PRESETS}"
        )

    X_base, y, meta = load_inference_xy(model_data, train_end, target_col)
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
    report = {
        **meta,
        "feature_preset": feature_preset,
        "model_data": str(model_data.resolve()),
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
