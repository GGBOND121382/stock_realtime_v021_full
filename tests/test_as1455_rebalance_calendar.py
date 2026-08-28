from __future__ import annotations

import pandas as pd
import pytest

from dashboard.as1455_rebalance_calendar import (
    _latest_phase,
    daily_rebalance_summary,
    project_rebalance_dates,
)


def _nav(*, day_index: int, every: int, offset: int, is_rebalance: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-08-14",
                "day_index": day_index,
                "is_rebalance_day": is_rebalance,
                "rebalance_every": every,
                "rebalance_offset": offset,
                "nav": 140000.0,
            }
        ]
    )


def test_projection_continues_latest_tracking_phase() -> None:
    nav = _nav(day_index=4, every=5, offset=0, is_rebalance=False)
    future = pd.to_datetime(["2026-08-17", "2026-08-18", "2026-08-19"])
    actual = project_rebalance_dates(nav, future)
    assert actual["day_index"].tolist() == [5, 6, 7]
    assert actual["is_rebalance_day"].tolist() == [True, False, False]


def test_projection_respects_nonzero_offset() -> None:
    nav = _nav(day_index=6, every=5, offset=1, is_rebalance=True)
    future = pd.to_datetime(
        ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"]
    )
    actual = project_rebalance_dates(nav, future)
    assert actual["is_rebalance_day"].tolist() == [False, False, False, False, True]


def test_every_one_strategy_rebalances_every_future_session() -> None:
    nav = _nav(day_index=12, every=1, offset=0, is_rebalance=True)
    future = pd.to_datetime(["2026-08-17", "2026-08-18", "2026-08-19"])
    actual = project_rebalance_dates(nav, future)
    assert actual["is_rebalance_day"].tolist() == [True, True, True]


def test_latest_phase_parses_csv_string_boolean_safely() -> None:
    phase = _latest_phase(_nav(day_index=4, every=5, offset=0, is_rebalance="False"))
    assert phase["is_rebalance_day"] is False


def test_latest_phase_rejects_inconsistent_persisted_state() -> None:
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        _latest_phase(_nav(day_index=4, every=5, offset=0, is_rebalance=True))


def test_daily_summary_identifies_production_due() -> None:
    schedule = pd.DataFrame(
        [
            {
                "date": "2026-08-17",
                "experiment": "r21_best_reb21_fold0_4_forward",
                "display_name": "21日目标 · 最优单模型",
                "is_rebalance_day": True,
                "is_production": True,
            },
            {
                "date": "2026-08-17",
                "experiment": "r05_best_reb5_fold0_4_forward",
                "display_name": "5日目标 · 最优单模型",
                "is_rebalance_day": False,
                "is_production": False,
            },
        ]
    )
    schedule["date"] = pd.to_datetime(schedule["date"])
    actual = daily_rebalance_summary(schedule)
    assert len(actual) == 1
    assert int(actual.iloc[0]["rebalance_count"]) == 1
    assert bool(actual.iloc[0]["production_rebalance"]) is True
    assert actual.iloc[0]["rebalance_codes"] == ["r21_best_reb21_fold0_4_forward"]
