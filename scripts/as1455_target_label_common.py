#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import run_as1455_sector_rotation_fold0_param_search as base

TARGET_LOOKAHEAD = {"r01_fwd": 1, "r05_fwd": 5, "r21_fwd": 21}


def target_lookahead(target_col: str) -> int:
    if target_col not in TARGET_LOOKAHEAD:
        raise RuntimeError(f"unsupported target_col={target_col!r}; expected {sorted(TARGET_LOOKAHEAD)}")
    return int(TARGET_LOOKAHEAD[target_col])


def load_xy_target(path: Path, train_end: str | None, dropna_mode: str, target_col: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if target_col not in base.EXPECTED_OUTCOMES:
        raise RuntimeError(f"target_col must be one of {base.EXPECTED_OUTCOMES}, got {target_col!r}")
    data = pd.read_hdf(path, "model_data")
    n_before = int(len(data))
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    if list(data.columns) != base.EXPECTED_MODEL_COLUMNS:
        raise RuntimeError(f"unexpected model_data columns: {list(data.columns)}")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != base.EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")
    end_ts = base.parse_train_end(train_end)
    dates = data.index.get_level_values("date")
    effective_end = pd.Timestamp(dates.max()) if end_ts is None else end_ts
    if end_ts is not None:
        data = data.loc[dates <= end_ts]
    if dropna_mode == "strict_original":
        data = data.dropna()
    elif dropna_mode == "target_only":
        required = [c for c in data.columns if c not in base.EXPECTED_OUTCOMES or c == target_col]
        data = data.dropna(subset=required)
    else:
        raise RuntimeError(f"bad dropna_mode: {dropna_mode}")
    data = data.sort_index()
    y = data[target_col].copy()
    X = data.drop(base.EXPECTED_OUTCOMES, axis=1)
    if X.shape[1] != 31 or any("fwd" in c for c in X.columns):
        raise RuntimeError(f"bad X shape/columns: {X.shape}")
    meta = {
        "rows_before_dropna": n_before,
        "rows_after_dropna": int(len(data)),
        "train_end_effective": effective_end.strftime("%Y-%m-%d"),
        "target_col": target_col,
        "target_lookahead": target_lookahead(target_col),
        "dropna_mode": dropna_mode,
    }
    return X, y, meta


def get_fold_target(X: pd.DataFrame, fold_index: int, target_col: str):
    lookahead = target_lookahead(target_col)
    cv = base.MultipleTimeSeriesCV(base.N_SPLITS, base.TRAIN_PERIOD_LENGTH, base.TEST_PERIOD_LENGTH, lookahead)
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        if i == fold_index:
            train_index = X.iloc[train_idx].index
            test_index = X.iloc[test_idx].index
            report = {
                "fold_index": i,
                "target_col": target_col,
                "lookahead": lookahead,
                "train_start": pd.Timestamp(train_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
                "train_end": pd.Timestamp(train_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
                "test_start": pd.Timestamp(test_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(test_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
            }
            return train_idx, test_idx, report
    raise RuntimeError(f"fold_index must be 0..{base.N_SPLITS - 1}, got {fold_index}")
