from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import run_as1455_live_nine_strategy_planner_entry as entry


def test_empty_tracking_positions_are_valid_when_state_count_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "tracking_forward_latest_positions.csv"
    path.write_bytes(b"")

    positions = entry._load_tracking_positions(path, 0)

    assert positions.empty
    assert list(positions.columns) == [
        "symbol",
        "shares",
        "buy_date",
        "avg_entry_price",
        "entry_rank",
        "entry_score",
        "cost_basis_notional",
        "cost_basis_fee",
    ]


def test_empty_tracking_positions_fail_closed_when_state_expects_holdings(tmp_path: Path) -> None:
    path = tmp_path / "tracking_forward_latest_positions.csv"
    path.write_bytes(b"")

    with pytest.raises(RuntimeError, match="state expects 1 positions"):
        entry._load_tracking_positions(path, 1)


def test_header_only_empty_tracking_positions_are_valid(tmp_path: Path) -> None:
    path = tmp_path / "tracking_forward_latest_positions.csv"
    pd.DataFrame(columns=["symbol", "shares", "buy_date"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )

    positions = entry._load_tracking_positions(path, 0)

    assert positions.empty
    assert "symbol" in positions.columns
    assert "shares" in positions.columns
    assert "buy_date" in positions.columns
