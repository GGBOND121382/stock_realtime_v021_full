from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.as1455_tracking import (
    contiguous_tracking_dates,
    resolve_initial_cash,
    tracking_start_date,
)


def test_tracking_start_date_reads_user_config(tmp_path: Path) -> None:
    dashboard = tmp_path / ".dashboard"
    dashboard.mkdir()
    (dashboard / "user_config.json").write_text(
        json.dumps({"tracking_start_date": "2026-08-07"}), encoding="utf-8"
    )
    assert tracking_start_date(tmp_path) == pd.Timestamp("2026-08-07")


def test_contiguous_tracking_dates_skips_nontrading_start_then_stops_at_gap() -> None:
    predictions = pd.DatetimeIndex(
        ["2026-08-10", "2026-08-11", "2026-08-13", "2026-08-14"]
    )
    calendar = pd.DatetimeIndex(
        ["2026-08-07", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"]
    )
    selected = contiguous_tracking_dates(
        predictions, calendar, pd.Timestamp("2026-08-08")
    )
    assert selected.tolist() == [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")]


def test_resolve_initial_cash_from_retained_forward_run(tmp_path: Path) -> None:
    experiment = tmp_path / "r01_all5_reb1_fold0_5_forward"
    grid = experiment / "strict_oos_forward" / "01_close_auction_grid"
    run = grid / "01_runs" / "selected"
    run.mkdir(parents=True)
    (grid / "strict_oos_manifest.json").write_text(
        json.dumps({"retained_run_name": "selected"}), encoding="utf-8"
    )
    (run / "config.json").write_text(
        json.dumps({"initial_cash": 345678.0}), encoding="utf-8"
    )
    assert resolve_initial_cash(experiment) == 345678.0
