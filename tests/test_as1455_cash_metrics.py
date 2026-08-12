from __future__ import annotations

import pandas as pd

from dashboard.as1455_cash_metrics import aggregate_cash_requirements, order_amount_metrics


def test_order_amount_metrics_prefers_notional_and_uses_absolute_magnitudes() -> None:
    frame = pd.DataFrame(
        {
            "side": ["buy", "sell", "buy"],
            "notional": [50_000.0, -30_000.0, 25_000.0],
        }
    )
    actual = order_amount_metrics(frame)
    assert actual == {
        "buy_amount": 75_000.0,
        "sell_amount": 30_000.0,
        "gross_trade_amount": 105_000.0,
        "net_buy_amount": 45_000.0,
        "conservative_cash_required": 75_000.0,
    }


def test_order_amount_metrics_falls_back_to_shares_times_price() -> None:
    frame = pd.DataFrame(
        {
            "side": ["buy", "sell"],
            "shares": [100, 200],
            "raw_exec_price": [10.0, 12.0],
        }
    )
    actual = order_amount_metrics(frame)
    assert actual["buy_amount"] == 1_000.0
    assert actual["sell_amount"] == 2_400.0
    assert actual["net_buy_amount"] == -1_400.0
    assert actual["conservative_cash_required"] == 1_000.0


def test_order_amount_metrics_fills_missing_notional_row_by_row() -> None:
    frame = pd.DataFrame(
        {
            "side": ["buy", "sell"],
            "notional": [5_000.0, None],
            "filled_shares": [None, 200],
            "shares": [100, 250],
            "raw_exec_price": [50.0, 12.0],
        }
    )
    actual = order_amount_metrics(frame)
    assert actual["buy_amount"] == 5_000.0
    assert actual["sell_amount"] == 2_400.0


def test_aggregate_cash_requirements_reports_max_and_sum() -> None:
    summary = pd.DataFrame(
        {
            "conservative_cash_required": [10_000.0, 30_000.0, 0.0],
            "gross_trade_amount": [12_000.0, 50_000.0, 0.0],
        }
    )
    assert aggregate_cash_requirements(summary) == {
        "max_single_strategy_cash_required": 30_000.0,
        "sum_cash_required": 40_000.0,
        "sum_gross_trade_amount": 62_000.0,
    }
