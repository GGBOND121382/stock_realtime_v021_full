from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.serve_as1455_execution_api import (
    batch_path,
    build_temporary_execution_batch,
    load_ready_batch,
    load_request_batch,
    request_has_experiment_override,
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_temporary_plan(root: Path, trade_date: str, experiment: str = R01_BEST) -> Path:
    token = trade_date.replace("-", "")
    plan_root = (
        root
        / token
        / "nine_strategy"
        / "start_date_plan"
        / "strategies"
        / experiment
    )
    plan_root.mkdir(parents=True, exist_ok=True)
    (plan_root / "strategy_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "trade_date": trade_date,
                "experiment": experiment,
                "plan_source": "materialized_start_date_plan",
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        plan_root / "16_live_orders.csv",
        [
            {
                "symbol": "600000.SH",
                "side": "buy",
                "filled_shares": 100,
                "raw_exec_price": "10.00",
                "rank": 1,
                "reason": "test-buy",
                "position_before": 0,
            },
            {
                "symbol": "000002.SZ",
                "side": "sell",
                "filled_shares": 200,
                "raw_exec_price": "20.00",
                "rank": 2,
                "reason": "test-sell",
                "position_before": 200,
            },
        ],
    )
    write_csv(
        root / token / "08_live_execution_sidecar.csv",
        [
            {
                "symbol": "600000.SH",
                "up_limit": "11.00",
                "down_limit": "9.00",
            },
            {
                "symbol": "000002.SZ",
                "up_limit": "22.00",
                "down_limit": "18.00",
            },
        ],
    )
    return plan_root


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
    assert request_has_experiment_override("") is False


def test_request_override_selects_r01_best():
    query = "experiment=r01_best_reb1_fold0_5_forward"
    assert select_request_experiment(query, EXPERIMENT) == R01_BEST
    assert request_has_experiment_override(query) is True


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


def test_temporary_batch_uses_materialized_plan_and_limit_prices(tmp_path: Path):
    trade_date = "2026-08-18"
    write_temporary_plan(tmp_path, trade_date)

    payload = build_temporary_execution_batch(tmp_path, trade_date, R01_BEST)

    assert payload is not None
    assert payload["status"] == "ready"
    assert payload["protocol"] == "as1455_execution_batch_v1"
    assert payload["experiment"] == R01_BEST
    assert payload["temporary_test_batch"] is True
    assert payload["order_count"] == 2
    assert [order["side"] for order in payload["orders"]] == ["sell", "buy"]
    assert [order["sequence"] for order in payload["orders"]] == [1, 2]
    assert payload["orders"][0]["code"] == "000002"
    assert payload["orders"][0]["submit_price"] == "18.00"
    assert payload["orders"][1]["code"] == "600000"
    assert payload["orders"][1]["submit_price"] == "11.00"
    assert all(len(order["signal_id"]) == 64 for order in payload["orders"])
    assert not batch_path(tmp_path, trade_date, R01_BEST).exists()


def test_default_request_never_uses_temporary_plan(tmp_path: Path):
    trade_date = "2026-08-18"
    write_temporary_plan(tmp_path, trade_date, experiment=EXPERIMENT)

    assert load_request_batch(tmp_path, trade_date, EXPERIMENT, "") is None


def test_explicit_request_uses_temporary_plan_when_committed_batch_missing(tmp_path: Path):
    trade_date = "2026-08-18"
    write_temporary_plan(tmp_path, trade_date)

    payload = load_request_batch(
        tmp_path,
        trade_date,
        EXPERIMENT,
        f"experiment={R01_BEST}",
    )

    assert payload is not None
    assert payload["experiment"] == R01_BEST
    assert payload["temporary_test_batch"] is True


def test_committed_batch_wins_over_temporary_fallback(tmp_path: Path):
    trade_date = "2026-08-18"
    write_temporary_plan(tmp_path, trade_date)
    write_batch(tmp_path, trade_date, experiment=R01_BEST)

    payload = load_request_batch(
        tmp_path,
        trade_date,
        EXPERIMENT,
        f"experiment={R01_BEST}",
    )

    assert payload is not None
    assert payload["experiment"] == R01_BEST
    assert "temporary_test_batch" not in payload
