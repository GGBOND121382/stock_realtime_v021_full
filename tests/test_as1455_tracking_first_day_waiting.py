from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import update_as1455_tracking_accounts as tracker
from utils.as1455_tracking import (
    TRACKING_MATRIX_MANIFEST,
    TRACKING_MATRIX_SUMMARY,
    TRACKING_SEMANTICS_VERSION,
    experiment_tracking_paths,
)


def test_raw_daily_probe_distinguishes_first_day_from_completed_day(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "date": ["2026-08-26", "2026-08-27"],
            "close": [10.0, 10.1],
        }
    ).to_csv(raw / "600000_daily_raw.csv", index=False)

    found, latest = tracker.raw_daily_has_completed_date(raw, pd.Timestamp("2026-08-28"))
    assert not found
    assert latest == pd.Timestamp("2026-08-27")

    pd.DataFrame(
        {
            "trade_date": [20260827, 20260828],
            "close": [8.0, 8.2],
        }
    ).to_csv(raw / "000001_daily_raw.csv", index=False)

    found, latest = tracker.raw_daily_has_completed_date(raw, pd.Timestamp("2026-08-28"))
    assert found
    assert latest == pd.Timestamp("2026-08-28")


def test_waiting_matrix_replaces_stale_start_without_predictions(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    dashboard = matrix / ".dashboard"
    dashboard.mkdir(parents=True)
    (dashboard / "user_config.json").write_text(
        json.dumps(
            {
                "tracking_start_date": "2026-08-28",
                "tracking_initial_cash": 120000.0,
            }
        ),
        encoding="utf-8",
    )

    experiments = []
    for index in range(9):
        name = f"exp{index}"
        experiments.append({"experiment": name})
        root = matrix / name
        root.mkdir()
        paths = experiment_tracking_paths(root)
        paths["manifest"].write_text(
            json.dumps(
                {
                    "status": "ok",
                    "tracking_start_date": "2026-08-19",
                    "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
                    "initial_cash": 120000.0,
                }
            ),
            encoding="utf-8",
        )
        paths["latest_state"].write_text(
            json.dumps({"asof_date": "2026-08-27", "cash": 120000.0}),
            encoding="utf-8",
        )
        pd.DataFrame({"date": ["2026-08-27"], "nav": [120000.0]}).to_csv(
            paths["nav"], index=False
        )

    manifest = tracker.initialize_waiting_matrix(
        experiments,
        matrix,
        pd.Timestamp("2026-08-28"),
        "rebuild",
        FileNotFoundError("no prediction source for r01_fwd"),
        pd.Timestamp("2026-08-27"),
    )

    assert manifest["status"] == "partial"
    assert manifest["completed_experiment_count"] == 0
    assert manifest["waiting_for_completed_market_day"] is True
    assert manifest["raw_daily_latest_date"] == "2026-08-27"

    summary = pd.read_csv(matrix / TRACKING_MATRIX_SUMMARY)
    assert len(summary) == 9
    assert set(summary["status"]) == {tracker.WAITING_FOR_MARKET_DAY}
    assert set(summary["tracking_start_date"]) == {"2026-08-28"}

    matrix_manifest = json.loads((matrix / TRACKING_MATRIX_MANIFEST).read_text(encoding="utf-8"))
    assert matrix_manifest["tracking_start_date"] == "2026-08-28"

    for item in experiments:
        paths = experiment_tracking_paths(matrix / item["experiment"])
        per_experiment = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        assert per_experiment["status"] == tracker.WAITING_FOR_MARKET_DAY
        assert per_experiment["tracking_start_date"] == "2026-08-28"
        assert not paths["latest_state"].exists()
        assert not paths["nav"].exists()
