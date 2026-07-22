#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared target-independent fold calendar for Ch17 AS1455.

The historical model-data tail differs by target because r1/r5/r21 labels become
available after different lookaheads. Fold windows must nevertheless denote the
same market dates. This module derives one canonical calendar from rows with
complete model features and cuts it at the latest date on which every configured
target is realized. Target-specific lookahead is used only for the training
embargo; validation/fold dates are identical across targets and feature presets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils import as1455_ch17_common as common

CALENDAR_MODE = "shared_feature_complete_all_targets"


@dataclass(frozen=True)
class AlignedFoldWindow:
    fold_index: int
    target_col: str
    lookahead: int
    train_lower_exclusive: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    fold_lower_exclusive: pd.Timestamp
    fold_start: pd.Timestamp
    fold_end: pd.Timestamp
    common_target_end: pd.Timestamp
    calendar_start: pd.Timestamp
    calendar_end: pd.Timestamp
    calendar_n_dates: int
    target_valid_ends: dict[str, str]

    def report(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, pd.Timestamp):
                payload[key] = value.strftime("%Y-%m-%d")
        payload.update(
            {
                "fold_calendar_mode": CALENDAR_MODE,
                "validation_start": payload["fold_start"],
                "validation_end": payload["fold_end"],
                # Backward-compatible aliases consumed by older forward code.
                "test_start": payload["fold_start"],
                "test_end": payload["fold_end"],
            }
        )
        return payload


def _feature_columns(data: pd.DataFrame) -> list[str]:
    outcomes = list(common.base.EXPECTED_OUTCOMES)
    missing = [column for column in outcomes if column not in data.columns]
    if missing:
        raise RuntimeError(f"model_data missing outcomes: {missing}")
    return [column for column in data.columns if column not in outcomes]


def canonical_calendar_from_frame(
    data: pd.DataFrame,
    train_end: str | None = None,
    targets: Iterable[str] | None = None,
) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    targets = list(targets or common.base.EXPECTED_OUTCOMES)
    unknown = [target for target in targets if target not in data.columns]
    if unknown:
        raise RuntimeError(f"unknown calendar targets: {unknown}")

    end_ts = common.base.parse_train_end(train_end)
    if end_ts is not None:
        dates = pd.DatetimeIndex(data.index.get_level_values("date"))
        data = data.loc[dates <= end_ts]

    feature_complete = data.dropna(subset=_feature_columns(data)).sort_index()
    if feature_complete.empty:
        raise RuntimeError("no feature-complete rows for aligned fold calendar")

    scoped_dates = pd.DatetimeIndex(feature_complete.index.get_level_values("date"))
    target_valid_ends: dict[str, pd.Timestamp] = {}
    for target in targets:
        valid_dates = scoped_dates[feature_complete[target].notna().to_numpy()]
        if len(valid_dates) == 0:
            raise RuntimeError(f"no realized rows for calendar target={target}")
        target_valid_ends[target] = pd.Timestamp(valid_dates.max()).normalize()

    common_target_end = min(target_valid_ends.values())
    canonical_dates = pd.DatetimeIndex(
        sorted(scoped_dates[scoped_dates <= common_target_end].unique())
    ).normalize()
    if canonical_dates.empty:
        raise RuntimeError("aligned fold calendar is empty")

    report = {
        "fold_calendar_mode": CALENDAR_MODE,
        "calendar_targets": targets,
        "target_valid_ends": {
            key: value.strftime("%Y-%m-%d")
            for key, value in target_valid_ends.items()
        },
        "common_target_end": common_target_end.strftime("%Y-%m-%d"),
        "calendar_start": canonical_dates.min().strftime("%Y-%m-%d"),
        "calendar_end": canonical_dates.max().strftime("%Y-%m-%d"),
        "calendar_n_dates": int(len(canonical_dates)),
    }
    return canonical_dates, report


def load_canonical_calendar(
    model_data: Path,
    train_end: str | None = None,
) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    data = pd.read_hdf(model_data, "model_data")
    return canonical_calendar_from_frame(data, train_end=train_end)


def fold_window_from_dates(
    canonical_dates: pd.DatetimeIndex,
    fold_index: int,
    target_col: str,
    calendar_report: dict[str, Any] | None = None,
) -> AlignedFoldWindow:
    if target_col not in common.TARGET_SPECS:
        raise RuntimeError(f"unsupported target_col={target_col}")
    if not 0 <= int(fold_index) < int(common.base.N_SPLITS):
        raise RuntimeError(
            f"fold_index must be 0..{common.base.N_SPLITS - 1}, got {fold_index}"
        )

    dates_desc = pd.DatetimeIndex(sorted(canonical_dates.unique(), reverse=True))
    lookahead = common.target_lookahead(target_col)
    train_length = int(common.base.TRAIN_PERIOD_LENGTH)
    fold_length = int(common.base.TEST_PERIOD_LENGTH)
    required = train_length + lookahead + common.base.N_SPLITS * fold_length
    if len(dates_desc) < required:
        raise RuntimeError(
            "not enough canonical dates for aligned folds: "
            f"need={required} got={len(dates_desc)} target={target_col}"
        )

    fold_end_idx = fold_index * fold_length
    fold_boundary_idx = fold_end_idx + fold_length
    train_end_idx = fold_boundary_idx + lookahead - 1
    train_boundary_idx = train_end_idx + train_length + lookahead - 1

    fold_end = pd.Timestamp(dates_desc[fold_end_idx]).normalize()
    fold_lower_exclusive = pd.Timestamp(dates_desc[fold_boundary_idx]).normalize()
    fold_start = pd.Timestamp(dates_desc[fold_boundary_idx - 1]).normalize()
    train_end = pd.Timestamp(dates_desc[train_end_idx]).normalize()
    train_lower_exclusive = pd.Timestamp(dates_desc[train_boundary_idx]).normalize()
    train_start = pd.Timestamp(dates_desc[train_boundary_idx - 1]).normalize()

    report = calendar_report or {}
    target_ends = dict(report.get("target_valid_ends", {}))
    common_end = pd.Timestamp(
        report.get("common_target_end", canonical_dates.max())
    ).normalize()
    return AlignedFoldWindow(
        fold_index=int(fold_index),
        target_col=target_col,
        lookahead=int(lookahead),
        train_lower_exclusive=train_lower_exclusive,
        train_start=train_start,
        train_end=train_end,
        fold_lower_exclusive=fold_lower_exclusive,
        fold_start=fold_start,
        fold_end=fold_end,
        common_target_end=common_end,
        calendar_start=pd.Timestamp(canonical_dates.min()).normalize(),
        calendar_end=pd.Timestamp(canonical_dates.max()).normalize(),
        calendar_n_dates=int(len(canonical_dates)),
        target_valid_ends=target_ends,
    )


def aligned_fold_split(
    X: pd.DataFrame,
    model_data: Path,
    fold_index: int,
    target_col: str,
    train_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    canonical_dates, calendar_report = load_canonical_calendar(
        model_data, train_end=train_end
    )
    window = fold_window_from_dates(
        canonical_dates, fold_index, target_col, calendar_report
    )
    dates = pd.DatetimeIndex(X.index.get_level_values("date")).normalize()
    train_mask = (dates > window.train_lower_exclusive) & (dates <= window.train_end)
    fold_mask = (dates > window.fold_lower_exclusive) & (dates <= window.fold_end)
    train_idx = np.flatnonzero(train_mask)
    fold_idx = np.flatnonzero(fold_mask)
    if len(train_idx) == 0 or len(fold_idx) == 0:
        raise RuntimeError(
            "aligned fold has no rows after target-specific row filtering: "
            f"target={target_col} fold={fold_index} train={len(train_idx)} "
            f"fold_rows={len(fold_idx)} window={window.report()}"
        )
    report = window.report()
    report.update(
        {
            "n_train_rows": int(len(train_idx)),
            "n_fold_rows": int(len(fold_idx)),
            "n_test_rows": int(len(fold_idx)),
            "fold_semantics": "internal_model_selection_window",
        }
    )
    return train_idx, fold_idx, report
