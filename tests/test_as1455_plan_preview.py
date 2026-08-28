from __future__ import annotations

import pandas as pd
import pytest

from dashboard import as1455_plan_compute as compute
from dashboard import as1455_plan_preview as preview
from utils.as1455_materialized_plan import (
    PLAN_CACHE_VERSION,
    atomic_json,
    matrix_manifest_path,
    write_materialized_day,
)
from utils.as1455_tracking import TRACKING_SEMANTICS_VERSION


class FakeSelection:
    pass


def _predictions() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [
            ["000001.SZ", "000002.SZ", "000003.SZ"],
            pd.to_datetime(["2026-08-07", "2026-08-10", "2026-08-10"]),
        ],
        names=["symbol", "date"],
    )
    return pd.DataFrame(
        {
            0: [0.10, 0.30, 0.20],
            1: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
        },
        index=index,
    )


def test_rank_table_filters_datetimeindex_with_vector_mask(monkeypatch) -> None:
    monkeypatch.setattr(
        compute.live,
        "score_predictions",
        lambda frame, selection: frame[0].rename("score"),
    )

    ranked = compute._rank_table(
        _predictions(), FakeSelection(), pd.Timestamp("2026-08-10")
    )

    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["symbol"].tolist() == ["000002.SZ", "000003.SZ"]
    assert pd.to_datetime(ranked["date"]).dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-10",
        "2026-08-10",
    ]


def test_rank_table_missing_date_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        compute.live,
        "score_predictions",
        lambda frame, selection: frame[0].rename("score"),
    )

    ranked = compute._rank_table(
        _predictions(), FakeSelection(), pd.Timestamp("2026-08-11")
    )

    assert ranked.empty
    assert list(ranked.columns) == ["rank", "symbol", "date", "score"]


def _fake_preview(start: pd.Timestamp, selected: pd.Timestamp) -> dict:
    rows = []
    details = {}
    for index in range(9):
        experiment = f"strategy_{index}"
        row = {
            "trade_date": selected.strftime("%Y-%m-%d"),
            "experiment": experiment,
            "status": "ok",
            "action": "调仓" if index % 2 == 0 else "非调仓日·继续持有",
            "is_rebalance_day": index % 2 == 0,
            "planned_buys": 1 if index % 2 == 0 else 0,
            "planned_sells": 0,
            "target_positions": 1,
            "tracking_bootstrap": index == 0,
        }
        rows.append(row)
        details[experiment] = {
            "manifest": row,
            "orders": pd.DataFrame(
                [{"symbol": "000001.SZ", "side": "buy", "shares": 100}]
                if index % 2 == 0
                else []
            ),
            "target_positions": pd.DataFrame(
                [{"symbol": "000001.SZ", "shares": 100}]
            ),
            "current_positions": pd.DataFrame(),
            "rejections": pd.DataFrame(),
            "rank": pd.DataFrame(
                [{"rank": 1, "symbol": "000001.SZ", "score": 0.1}]
            ),
            "nav": pd.DataFrame(
                [{"date": selected, "nav": 100000.0, "is_rebalance_day": index % 2 == 0}]
            ),
            "phase": {},
        }
    return {
        "status": "ok",
        "tracking_start_date": start.strftime("%Y-%m-%d"),
        "selected_date": selected.strftime("%Y-%m-%d"),
        "summary": pd.DataFrame(rows),
        "details": details,
        "execution_source": "saved_1455_sidecars",
        "raw_daily_fallback_dates": [],
        "model_inference_rerun": False,
        "historical_grid_rerun": False,
    }


def test_dashboard_preview_reads_materialized_plan_without_compute(tmp_path) -> None:
    matrix_root = tmp_path / "matrix"
    live_root = tmp_path / "live"
    matrix_root.mkdir()
    start = pd.Timestamp("2026-08-07")
    selected = pd.Timestamp("2026-08-10")
    token = selected.strftime("%Y%m%d")
    payload = _fake_preview(start, selected)

    write_materialized_day(
        live_root,
        token,
        start,
        payload,
        TRACKING_SEMANTICS_VERSION,
    )
    atomic_json(
        matrix_manifest_path(matrix_root),
        {
            "status": "ok",
            "plan_cache_version": PLAN_CACHE_VERSION,
            "tracking_start_date": start.strftime("%Y-%m-%d"),
            "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
            "completed_date_count": 1,
            "dates": [token],
        },
    )

    result = preview.preview_nine_strategy_day(
        matrix_root, live_root, start, selected
    )

    assert result["status"] == "ok"
    assert result["plan_source"] == "materialized_start_date_plan"
    assert result["dashboard_replay_rerun"] is False
    assert len(result["summary"]) == 9
    assert len(result["details"]) == 9
    assert not hasattr(preview, "compute_nine_strategy_day")


def test_dashboard_preview_never_falls_back_to_on_demand_compute(tmp_path) -> None:
    matrix_root = tmp_path / "matrix"
    live_root = tmp_path / "live"
    matrix_root.mkdir()
    live_root.mkdir()

    with pytest.raises(RuntimeError, match="页面查看本身不会再即时重放"):
        preview.preview_nine_strategy_day(
            matrix_root,
            live_root,
            pd.Timestamp("2026-08-07"),
            pd.Timestamp("2026-08-10"),
        )
