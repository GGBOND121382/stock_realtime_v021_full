from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import run_as1455_live_production_strategy_planner_entry as production


def test_best_prediction_loader_accepts_top1_only(tmp_path: Path) -> None:
    path = tmp_path / "top5_live_predictions.csv"
    pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "date": "2026-08-17",
                "0": 0.25,
            }
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    actual = production.load_best_prediction_panel(
        path, pd.Timestamp("2026-08-17")
    )
    assert list(actual.columns) == [0]
    assert len(actual) == 1
    assert actual.index.names == ["symbol", "date"]
    assert float(actual.iloc[0, 0]) == pytest.approx(0.25)


def test_best_prediction_loader_requires_model_zero(tmp_path: Path) -> None:
    path = tmp_path / "top5_live_predictions.csv"
    pd.DataFrame(
        [{"symbol": "600000.SH", "date": "2026-08-17", "1": 0.25}]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(RuntimeError, match="missing production prediction columns"):
        production.load_best_prediction_panel(path, pd.Timestamp("2026-08-17"))


def test_default_production_experiment_is_r21_best() -> None:
    assert production.DEFAULT_PRODUCTION_EXPERIMENT == "r21_best_reb21_fold0_4_forward"
