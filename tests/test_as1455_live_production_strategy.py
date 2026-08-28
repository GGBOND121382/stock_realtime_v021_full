from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_as1455_live_nine_strategy_planner_entry as entry
from scripts import run_as1455_live_production_strategy_planner_entry as production
from scripts import run_as1455_live_simulation_strategy_planner_entry as simulation


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


def test_default_simulation_experiment_is_r01_best() -> None:
    assert simulation.DEFAULT_SIMULATION_EXPERIMENT == "r01_best_reb1_fold0_5_forward"


def test_production_tracking_start_is_fail_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(entry, "tracking_start_date", lambda matrix_root: None)
    production.install_fail_closed_tracking_start()
    with pytest.raises(RuntimeError, match="legacy strict-forward account fallback is forbidden"):
        entry.tracking_start_date(tmp_path)


def test_execution_batch_is_committed_after_manifest_writes(monkeypatch, tmp_path: Path) -> None:
    experiment = production.DEFAULT_PRODUCTION_EXPERIMENT
    strategy = tmp_path / "strategies" / experiment
    strategy.mkdir(parents=True)
    batch_path = strategy / "execution_batch.json"
    manifest_path = strategy / "strategy_manifest.json"
    events: list[str] = []

    def fake_atomic(path: Path, payload: dict) -> None:
        events.append(Path(path).name)

    def fake_publish() -> None:
        entry._atomic_write_json(batch_path, {"experiment": experiment, "status": "ready"})
        entry._atomic_write_json(manifest_path, {"status": "ok"})

    monkeypatch.setattr(entry, "_atomic_write_json", fake_atomic)
    monkeypatch.setattr(entry, "publish_execution_batches", fake_publish)

    actual = production.publish_execution_batch_as_final_commit(experiment)
    assert actual == batch_path
    assert events == ["strategy_manifest.json", "execution_batch.json"]


def test_execution_batch_not_committed_if_precommit_publication_fails(
    monkeypatch, tmp_path: Path
) -> None:
    experiment = production.DEFAULT_PRODUCTION_EXPERIMENT
    strategy = tmp_path / "strategies" / experiment
    strategy.mkdir(parents=True)
    batch_path = strategy / "execution_batch.json"
    events: list[str] = []

    def fake_atomic(path: Path, payload: dict) -> None:
        events.append(Path(path).name)

    def fake_publish() -> None:
        entry._atomic_write_json(batch_path, {"experiment": experiment, "status": "ready"})
        raise RuntimeError("manifest update failed")

    monkeypatch.setattr(entry, "_atomic_write_json", fake_atomic)
    monkeypatch.setattr(entry, "publish_execution_batches", fake_publish)

    with pytest.raises(RuntimeError, match="manifest update failed"):
        production.publish_execution_batch_as_final_commit(experiment)
    assert events == []


def test_r01_simulation_publish_does_not_mutate_r21_batch(tmp_path: Path) -> None:
    r01 = simulation.DEFAULT_SIMULATION_EXPERIMENT
    r21 = production.DEFAULT_PRODUCTION_EXPERIMENT
    staging = tmp_path / "staging"
    publish = tmp_path / "live"

    source = staging / "strategies" / r01 / "execution_batch.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "status": "ready",
                "protocol": entry.EXECUTION_BATCH_PROTOCOL,
                "trade_date": "2026-08-17",
                "experiment": r01,
                "order_count": 0,
                "orders": [],
            }
        ),
        encoding="utf-8",
    )

    r21_batch = publish / "strategies" / r21 / "execution_batch.json"
    r21_batch.parent.mkdir(parents=True)
    r21_batch.write_text('{"sentinel":"production"}', encoding="utf-8")

    actual = simulation.publish_simulation_batch(
        staging,
        publish,
        r01,
        "20260817",
    )

    assert actual == publish / "strategies" / r01 / "execution_batch.json"
    assert json.loads(actual.read_text(encoding="utf-8"))["experiment"] == r01
    assert json.loads(r21_batch.read_text(encoding="utf-8")) == {"sentinel": "production"}
