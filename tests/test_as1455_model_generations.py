from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.as1455_model_registry import (
    atomic_write_json,
    bootstrap_registry,
    generation_id,
    model_display_for_experiment,
    period_id,
    target_col_from_experiment,
)
from utils.as1455_model_roll import (
    activate_generation,
    next_generation,
    record_live_generation_use,
    rollover_status,
)


def test_generation_and_period_names_do_not_reuse_fold_namespace() -> None:
    assert generation_id(0) == "gen000"
    assert generation_id(37) == "gen037"
    assert period_id(0) == "period000"
    assert period_id(108) == "period108"
    assert "fold" not in generation_id(37)
    assert "fold" not in period_id(108)


def test_bootstrap_keeps_legacy_fold0_as_gen000_reference(tmp_path: Path) -> None:
    registry = bootstrap_registry(tmp_path)
    assert registry["active_generation"] == "gen000"
    assert registry["current_period"]["period_id"] == "period000"
    for target in ("r01_fwd", "r05_fwd", "r21_fwd"):
        entry = registry["active_models"][target]
        assert entry["generation_id"] == "gen000"
        assert entry["source_type"] == "legacy_cv_fold"
        assert entry["source_fold"] == 0
        assert "fold0" in entry["model_dir"]


def test_only_successfully_recorded_live_days_advance_period(tmp_path: Path) -> None:
    registry = bootstrap_registry(tmp_path)
    dates = pd.bdate_range("2026-01-01", periods=62).strftime("%Y-%m-%d").tolist()
    registry["current_period"].update(
        {
            "observed_dates": dates,
            "observed_days": 62,
            "start_date": dates[0],
            "last_observed_date": dates[-1],
            "required_days": 63,
        }
    )
    atomic_write_json(tmp_path / "registry.json", registry)

    before = rollover_status(tmp_path)
    assert before["due"] is False
    assert before["remaining_days"] == 1

    # The production pipeline calls this function only after the nine-strategy
    # planner has completed successfully.
    final_date = pd.bdate_range(pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), periods=1)[0]
    record_live_generation_use(tmp_path, trade_date=final_date.strftime("%Y-%m-%d"))
    after = rollover_status(tmp_path)
    assert after["due"] is True
    assert after["observed_days"] == 63
    assert after["rollover_boundary"] == final_date.strftime("%Y-%m-%d")

    # Idempotent retry of the same successful date must not create day 64.
    record_live_generation_use(tmp_path, trade_date=final_date.strftime("%Y-%m-%d"))
    repeated = rollover_status(tmp_path)
    assert repeated["observed_days"] == 63


def test_activation_is_atomic_across_all_three_targets(tmp_path: Path) -> None:
    registry = bootstrap_registry(tmp_path)
    dates = pd.bdate_range("2026-01-01", periods=63).strftime("%Y-%m-%d").tolist()
    registry["current_period"].update(
        {
            "observed_dates": dates,
            "observed_days": 63,
            "start_date": dates[0],
            "last_observed_date": dates[-1],
            "required_days": 63,
        }
    )
    atomic_write_json(tmp_path / "registry.json", registry)
    assert next_generation(registry) == "gen001"

    generation = {
        "generation_id": "gen001",
        "generation_index": 1,
        "source_type": "rolling_refit",
        "source_generation": "gen000",
        "source_period": "period000",
        "trained_at": "2026-04-01T21:30:00+08:00",
        "targets": {
            target: {
                "model_dir": str(tmp_path / "generations" / "gen001" / target),
                "source_type": "rolling_refit",
                "source_generation": "gen000",
                "trained_at": "2026-04-01T21:30:00+08:00",
                "train_start": "2022-01-01",
                "train_end": "2026-03-31",
                "label_valid_end": "2026-03-31",
            }
            for target in ("r01_fwd", "r05_fwd", "r21_fwd")
        },
    }
    activated = activate_generation(
        tmp_path,
        generation=generation,
        period_end=dates[-1],
    )
    assert activated["active_generation"] == "gen001"
    assert set(
        entry["generation_id"] for entry in activated["active_models"].values()
    ) == {"gen001"}
    assert activated["completed_periods"][-1]["period_id"] == "period000"
    assert activated["current_period"]["period_id"] == "period001"
    assert activated["current_period"]["observed_days"] == 0
    assert activated["current_period"]["observed_dates"] == []


def test_nine_strategy_names_map_to_target_model_date(tmp_path: Path) -> None:
    registry = bootstrap_registry(tmp_path)
    registry["active_models"]["r05_fwd"]["model_updated_date"] = "2026-08-14"
    atomic_write_json(tmp_path / "registry.json", registry)
    registry = bootstrap_registry(tmp_path)

    assert target_col_from_experiment("r05_all5_reb5_fold0_5_forward") == "r05_fwd"
    assert target_col_from_experiment("r05_first3_reb5_fold0_5_forward") == "r05_fwd"
    assert target_col_from_experiment("r05_best_reb5_fold0_5_forward") == "r05_fwd"
    info = model_display_for_experiment(
        registry, "r05_best_reb5_fold0_5_forward"
    )
    assert info["model_generation"] == "gen000"
    assert info["model_updated_date"] == "2026-08-14"
