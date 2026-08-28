from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.as1455_cash_metrics import aggregate_cash_requirements, order_amount_metrics


def test_order_amount_metrics_uses_upper_limit_for_conservative_cash() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-08-12", "2026-08-12", "2026-08-12"],
            "symbol": ["600000.SH", "000001.SZ", "600001.SH"],
            "side": ["buy", "sell", "buy"],
            "filled_shares": [1000, 2000, 500],
            "notional": [10_000.0, -30_000.0, 10_000.0],
            "up_limit": [11.0, 16.5, 22.0],
            "total_fee": [5.0, 35.0, 5.0],
        }
    )
    actual = order_amount_metrics(frame)
    assert actual["buy_amount"] == 20_000.0
    assert actual["sell_amount"] == 30_000.0
    assert actual["gross_trade_amount"] == 50_000.0
    assert actual["net_buy_amount"] == -10_000.0
    assert actual["limit_buy_notional"] == 22_000.0
    # Saved 5-yuan buy fees are conservatively scaled by the 1.1 limit-price ratio.
    assert actual["limit_buy_fee_reserve"] == 11.0
    assert actual["conservative_cash_required"] == 22_011.0
    assert actual["cash_requirement_complete"] is True


def test_order_amount_metrics_reads_saved_1455_sidecar_when_order_lacks_limit(tmp_path: Path) -> None:
    day = tmp_path / "20260812"
    day.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ"],
            "up_limit": [11.0, 13.2],
        }
    ).to_csv(day / "08_live_execution_sidecar.csv", index=False, encoding="utf-8-sig")
    frame = pd.DataFrame(
        {
            "date": ["2026-08-12", "2026-08-12"],
            "symbol": ["600000.SH", "000001.SZ"],
            "side": ["buy", "buy"],
            "shares": [1000, 1000],
            "raw_exec_price": [10.0, 12.0],
            "notional": [10_000.0, 12_000.0],
            "total_fee": [5.0, 5.0],
        }
    )
    actual = order_amount_metrics(frame, live_root=tmp_path)
    assert actual["limit_buy_notional"] == 24_200.0
    assert actual["conservative_cash_required"] == 24_211.0
    assert actual["cash_requirement_complete"] is True


def test_order_amount_metrics_does_not_fake_conservative_cash_without_upper_limit(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-08-12"],
            "symbol": ["600000.SH"],
            "side": ["buy"],
            "shares": [1000],
            "raw_exec_price": [10.0],
            "notional": [10_000.0],
            "total_fee": [5.0],
        }
    )
    actual = order_amount_metrics(frame, live_root=tmp_path)
    assert np.isnan(actual["conservative_cash_required"])
    assert actual["cash_requirement_complete"] is False


def test_order_amount_metrics_no_buys_requires_no_extra_cash() -> None:
    frame = pd.DataFrame(
        {
            "side": ["sell"],
            "shares": [200],
            "raw_exec_price": [12.0],
            "notional": [2_400.0],
        }
    )
    actual = order_amount_metrics(frame)
    assert actual["buy_amount"] == 0.0
    assert actual["sell_amount"] == 2_400.0
    assert actual["conservative_cash_required"] == 0.0
    assert actual["cash_requirement_complete"] is True


def test_order_amount_metrics_fills_missing_notional_row_by_row() -> None:
    frame = pd.DataFrame(
        {
            "side": ["sell", "sell"],
            "notional": [5_000.0, None],
            "filled_shares": [None, 200],
            "shares": [100, 250],
            "raw_exec_price": [50.0, 12.0],
        }
    )
    actual = order_amount_metrics(frame)
    assert actual["sell_amount"] == 7_400.0


def test_aggregate_cash_requirements_reports_max_and_sum() -> None:
    summary = pd.DataFrame(
        {
            "buy_amount": [9_000.0, 25_000.0, 0.0],
            "conservative_cash_required": [10_000.0, 30_000.0, 0.0],
            "gross_trade_amount": [12_000.0, 50_000.0, 0.0],
        }
    )
    assert aggregate_cash_requirements(summary) == {
        "max_single_strategy_cash_required": 30_000.0,
        "sum_cash_required": 40_000.0,
        "sum_gross_trade_amount": 62_000.0,
    }


def test_aggregate_cash_requirements_fails_closed_when_any_buy_limit_is_unknown() -> None:
    summary = pd.DataFrame(
        {
            "buy_amount": [10_000.0, 20_000.0],
            "conservative_cash_required": [11_005.0, np.nan],
            "gross_trade_amount": [10_000.0, 20_000.0],
        }
    )
    actual = aggregate_cash_requirements(summary)
    assert np.isnan(actual["max_single_strategy_cash_required"])
    assert np.isnan(actual["sum_cash_required"])
    assert actual["sum_gross_trade_amount"] == 30_000.0
