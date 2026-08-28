from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from utils.as1455_manual_calibration import apply_manual_calibration
from utils.as1455_tracking import TRACKING_SEMANTICS_VERSION


EXPERIMENT = "r21_best_reb21_fold0_4_forward"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_account(tmp_path: Path) -> tuple[Path, Path, Path]:
    matrix = tmp_path / "matrix"
    live = tmp_path / "live"
    root = matrix / EXPERIMENT
    root.mkdir(parents=True)
    (matrix / ".dashboard").mkdir(parents=True)
    _write_json(
        matrix / ".dashboard" / "user_config.json",
        {
            "tracking_start_date": "2026-08-03",
            "tracking_initial_cash": 140000.0,
        },
    )
    _write_json(
        root / "tracking_forward_manifest.json",
        {
            "status": "ok",
            "experiment": EXPERIMENT,
            "tracking_start_date": "2026-08-03",
            "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
            "initial_cash": 140000.0,
        },
    )
    _write_json(
        root / "tracking_forward_latest_state.json",
        {
            "status": "ok",
            "experiment": EXPERIMENT,
            "tracking_start_date": "2026-08-03",
            "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
            "asof_date": "2026-08-14",
            "initial_cash": 140000.0,
            "nav": 150000.0,
            "cash": 10000.0,
            "n_positions": 1,
        },
    )
    positions = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "shares": 1000,
                "buy_date": "2026-08-10",
                "avg_entry_price": 10.0,
                "entry_rank": 1.0,
                "entry_score": 0.3,
                "cost_basis_notional": 10000.0,
                "cost_basis_fee": 5.0,
            }
        ]
    )
    positions.to_csv(root / "tracking_forward_latest_positions.csv", index=False)
    positions.assign(date="2026-08-14").to_csv(
        root / "tracking_forward_positions.csv", index=False
    )
    pd.DataFrame(
        [
            {"date": "2026-08-13", "nav": 140000.0, "cash": 140000.0, "n_positions": 0},
            {"date": "2026-08-14", "nav": 150000.0, "cash": 10000.0, "n_positions": 1},
        ]
    ).to_csv(root / "tracking_forward_nav.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": EXPERIMENT,
                "initial_cash": 140000.0,
                "final_nav": 150000.0,
                "total_return": 150000.0 / 140000.0 - 1.0,
                "annual_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "forward_end": "2026-08-14",
                "forward_n_days": 2,
            }
        ]
    ).to_csv(root / "tracking_forward_result.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": EXPERIMENT,
                "initial_cash": 140000.0,
                "final_nav": 150000.0,
                "total_return": 150000.0 / 140000.0 - 1.0,
                "annual_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "forward_end": "2026-08-14",
                "forward_n_days": 2,
            }
        ]
    ).to_csv(matrix / "tracking_matrix_summary.csv", index=False)
    _write_json(
        matrix / "tracking_matrix_manifest.json",
        {
            "status": "ok",
            "tracking_start_date": "2026-08-03",
            "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
            "completed_experiment_count": 9,
        },
    )

    # The calibration helper invalidates the current Shanghai-calendar-day READY.
    today = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    strategy = live / today / "nine_strategy" / "strategies" / EXPERIMENT
    strategy.mkdir(parents=True)
    _write_json(strategy / "execution_batch.json", {"status": "ready", "experiment": EXPERIMENT})
    _write_json(
        strategy / "strategy_manifest.json",
        {"execution_batch_file": str(strategy / "execution_batch.json")},
    )
    _write_json(
        live / today / "nine_strategy" / "live_nine_strategy_manifest.json",
        {"execution_batch_files": {EXPERIMENT: str(strategy / "execution_batch.json")}},
    )
    return matrix, live, root


def test_manual_calibration_updates_latest_state_and_invalidates_ready(tmp_path: Path) -> None:
    matrix, live, root = _seed_account(tmp_path)
    actual_positions = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "shares": 1031,
                "buy_date": "2026-08-10",
                "avg_entry_price": 10.2,
            },
            {
                "symbol": "000001.SZ",
                "shares": 500,
                "buy_date": "2026-08-14",
                "avg_entry_price": 12.3,
            },
        ]
    )

    audit = apply_manual_calibration(
        matrix,
        experiment=EXPERIMENT,
        asof_date=pd.Timestamp("2026-08-14"),
        strategy_cash=12345.67,
        strategy_nav=152345.67,
        positions=actual_positions,
        note="broker reconciliation",
        live_root=live,
    )

    state = json.loads((root / "tracking_forward_latest_state.json").read_text())
    assert state["cash"] == pytest.approx(12345.67)
    assert state["nav"] == pytest.approx(152345.67)
    assert state["n_positions"] == 2
    assert state["account_state_source"] == "manual_broker_calibration"
    latest = pd.read_csv(root / "tracking_forward_latest_positions.csv")
    assert latest["shares"].tolist() == [1031, 500]

    today = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    ready = live / today / "nine_strategy" / "strategies" / EXPERIMENT / "execution_batch.json"
    assert not ready.exists()
    archived = audit["archived_same_day_execution_batch"]
    assert archived and Path(archived).is_file()
    assert Path(audit["backup_dir"]).is_dir()


def test_manual_calibration_rejects_fractional_shares(tmp_path: Path) -> None:
    matrix, live, _ = _seed_account(tmp_path)
    positions = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "shares": 1031.25,
                "buy_date": "2026-08-10",
                "avg_entry_price": 10.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="正整数"):
        apply_manual_calibration(
            matrix,
            experiment=EXPERIMENT,
            asof_date=pd.Timestamp("2026-08-14"),
            strategy_cash=10000.0,
            strategy_nav=150000.0,
            positions=positions,
            live_root=live,
        )
