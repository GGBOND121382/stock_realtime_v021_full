from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.serve_as1455_execution_api import (
    batch_path,
    load_ready_batch,
    select_request_experiment,
    validate_experiment_name,
)


EXPERIMENT = "r21_best_reb21_fold0_4_forward"
R01_BEST = "r01_best_reb1_fold0_5_forward"


def write_batch(root: Path, trade_date: str, experiment: str = EXPERIMENT, **overrides):
    path = batch_path(root, trade_date, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ready",
        "protocol": "as1455_execution_batch_v1",
        "trade_date": trade_date,
        "experiment": experiment,
        "order_count": 1,
        "orders": [
            {
                "signal_id": "0123456789abcdef",
                "sequence": 1,
                "code": "600000",
                "side": "buy",
                "qty": 100,
                "submit_price": "10.00",
            }
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_ready_batch(tmp_path: Path):
    write_batch(tmp_path, "2026-08-17")
    payload = load_ready_batch(tmp_path, "2026-08-17", EXPERIMENT)
    assert payload is not None
    assert payload["order_count"] == 1


def test_load_ready_batch_for_explicit_test_experiment(tmp_path: Path):
    write_batch(tmp_path, "2026-08-17", experiment=R01_BEST)
    payload = load_ready_batch(tmp_path, "2026-08-17", R01_BEST)
    assert payload is not None
    assert payload["experiment"] == R01_BEST


def test_missing_or_not_ready_returns_none(tmp_path: Path):
    assert load_ready_batch(tmp_path, "2026-08-17", EXPERIMENT) is None
    write_batch(tmp_path, "2026-08-17", status="building")
    assert load_ready_batch(tmp_path, "2026-08-17", EXPERIMENT) is None


def test_protocol_mismatch_fails_closed(tmp_path: Path):
    write_batch(tmp_path, "2026-08-17", protocol="wrong")
    with pytest.raises(RuntimeError, match="unexpected execution protocol"):
        load_ready_batch(tmp_path, "2026-08-17", EXPERIMENT)


def test_count_mismatch_fails_closed(tmp_path: Path):
    write_batch(tmp_path, "2026-08-17", order_count=2)
    with pytest.raises(RuntimeError, match="invalid order payload"):
        load_ready_batch(tmp_path, "2026-08-17", EXPERIMENT)


def test_request_without_override_keeps_production_experiment():
    assert select_request_experiment("", EXPERIMENT) == EXPERIMENT


def test_request_override_selects_r01_best():
    query = "experiment=r01_best_reb1_fold0_5_forward"
    assert select_request_experiment(query, EXPERIMENT) == R01_BEST


@pytest.mark.parametrize(
    "query",
    [
        "experiment=",
        "experiment=../../etc/passwd",
        "experiment=r01_best_reb1_fold0_5_forward&experiment=r21_best_reb21_fold0_4_forward",
    ],
)
def test_invalid_request_override_is_rejected(query: str):
    with pytest.raises(ValueError):
        select_request_experiment(query, EXPERIMENT)


def test_invalid_experiment_name_is_rejected():
    with pytest.raises(ValueError, match="invalid experiment"):
        validate_experiment_name("../r01_best_reb1_fold0_5_forward")
