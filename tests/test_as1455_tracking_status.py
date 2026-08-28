from __future__ import annotations

import pandas as pd

from dashboard.as1455_tracking_status import tracking_status_view


def _summary(status: str, count: int = 9) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": [f"s{i}" for i in range(count)],
            "status": [status] * count,
        }
    )


def _manifest(status: str, completed: int) -> dict:
    return {
        "status": status,
        "tracking_start_date": "2026-08-17",
        "tracking_semantics_version": 5,
        "experiment_count": 9,
        "completed_experiment_count": completed,
    }


def test_waiting_market_day_is_not_reported_as_running_or_failed() -> None:
    view = tracking_status_view(
        _summary("waiting_for_completed_market_day"),
        _manifest("partial", 0),
        pd.Timestamp("2026-08-17"),
        5,
        "success",
    )
    assert view["state"] == "waiting_market_day"
    assert view["headline"] == "等待收盘数据"
    assert "无需重复重建" in view["detail"]


def test_background_running_takes_precedence_while_rebuild_is_active() -> None:
    view = tracking_status_view(
        _summary("waiting_for_completed_market_day"),
        _manifest("partial", 0),
        pd.Timestamp("2026-08-17"),
        5,
        "running",
    )
    assert view["state"] == "running"
    assert view["headline"] == "正在重建"


def test_ready_requires_all_nine_accounts() -> None:
    view = tracking_status_view(
        _summary("ok"),
        _manifest("ok", 9),
        pd.Timestamp("2026-08-17"),
        5,
        "success",
    )
    assert view["state"] == "ready"
    assert view["completed"] == 9


def test_partial_is_distinct_from_waiting_market_day() -> None:
    mixed = _summary("ok")
    mixed.loc[0, "status"] = "waiting_for_completed_market_day"
    view = tracking_status_view(
        mixed,
        _manifest("partial", 8),
        pd.Timestamp("2026-08-17"),
        5,
        "success",
    )
    assert view["state"] == "partial"
    assert view["headline"] == "部分完成 8/9"
