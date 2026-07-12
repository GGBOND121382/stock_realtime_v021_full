#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-signal ranking cache for the unchanged v7 trading engine.

The v7 engine calls ``day_frame.sort_values('score', ascending=False)`` for each
configuration.  This module pre-sorts each date once with exactly that operation
and returns an explicit pandas ``DataFrame`` subclass whose identical subsequent
sort request is a no-op copy.  No trading function is patched or duplicated.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


class PreSortedPredictionFrame(pd.DataFrame):
    """DataFrame carrying a verified date/score pre-sort contract."""

    _metadata = ["_as1455_presorted_date_score"]

    @property
    def _constructor(self):  # type: ignore[override]
        return PreSortedPredictionFrame

    def sort_values(self, by: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        ascending = kwargs.get("ascending", True)
        axis = kwargs.get("axis", 0)
        inplace = kwargs.get("inplace", False)
        kind = kwargs.get("kind", "quicksort")
        na_position = kwargs.get("na_position", "last")
        ignore_index = kwargs.get("ignore_index", False)
        key = kwargs.get("key", None)

        is_v7_daily_sort = (
            bool(getattr(self, "_as1455_presorted_date_score", False))
            and by == "score"
            and ascending is False
            and axis in (0, "index")
            and not inplace
            and kind == "quicksort"
            and na_position == "last"
            and not ignore_index
            and key is None
            and not args
        )
        if is_v7_daily_sort:
            return self.copy()
        return super().sort_values(by, *args, **kwargs)


def prepare_presorted_predictions(predictions: pd.DataFrame) -> PreSortedPredictionFrame:
    """Sort every date once exactly as v7 would sort it.

    Sorting group-by-group, rather than globally by two columns, preserves the
    original per-date tie behavior of ``DataFrame.sort_values``.
    """
    if predictions.empty:
        raise ValueError("empty predictions")
    required = {"date", "symbol", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame missing columns: {sorted(missing)}")

    groups = [
        group.sort_values("score", ascending=False).copy()
        for _date, group in predictions.groupby("date", sort=True)
    ]
    ordered = pd.concat(groups, axis=0) if groups else predictions.copy()
    cached = PreSortedPredictionFrame(ordered)
    cached._as1455_presorted_date_score = True
    return cached


def validate_presorted_predictions(predictions: pd.DataFrame) -> None:
    """Fail if a date is not monotonically sorted by descending score."""
    for date, group in predictions.groupby("date", sort=True):
        score = pd.to_numeric(group["score"], errors="coerce")
        non_null = score.dropna()
        if not non_null.is_monotonic_decreasing:
            raise RuntimeError(f"prediction rank cache invalid for date={date}")
