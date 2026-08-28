from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard.as1455_cash_history import (
    current_position_cash_estimate,
    daily_cash_requirements,
)


def _write_raw_daily(root: Path, code: str, date: str, preclose: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": date,
                "close": preclose,
                "preclose": preclose,
                "volume": 100000,
                "amount": 1000000,
                "tradestatus": 1,
            }
        ]
    ).to_csv(root / f"{code}_daily_raw.csv", index=False)


def test_historical_daily_cash_uses_raw_preclose_upper_limit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    live = tmp_path / "live"
    _write_raw_daily(raw, "600000", "2026-08-14", 10.0)
    orders = pd.DataFrame(
        [
            {
                "date": "2026-08-14",
                "symbol": "600000.SH",
                "side": "buy",
                "filled_shares": 1000,
                "raw_exec_price": 10.0,
                "notional": 10000.0,
                "total_fee": 5.0,
            }
        ]
    )
    daily = daily_cash_requirements(
        orders,
        live_root=live,
        raw_daily_cache_dir=raw,
    )
    assert len(daily) == 1
    assert bool(daily.iloc[0]["cash_requirement_complete"])
    assert abs(float(daily.iloc[0]["limit_buy_notional"]) - 11000.0) < 1e-9
    assert abs(float(daily.iloc[0]["limit_buy_fee_reserve"]) - 5.5) < 1e-9
    assert abs(float(daily.iloc[0]["conservative_cash_required"]) - 11005.5) < 1e-9


def test_current_position_estimate_reprices_same_shares_at_upper_limit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    live = tmp_path / "live"
    _write_raw_daily(raw, "600000", "2026-08-14", 10.0)
    historical_orders = pd.DataFrame(
        [
            {
                "date": "2026-08-01",
                "symbol": "600001.SH",
                "side": "buy",
                "filled_shares": 1000,
                "notional": 10000.0,
                "total_fee": 5.0,
            }
        ]
    )
    item = {
        "tracking_latest_state": {"asof_date": "2026-08-14"},
        "tracking_latest_positions": pd.DataFrame(
            [{"symbol": "600000.SH", "shares": 1000}]
        ),
        "tracking_orders": pd.DataFrame(),
        "tracking_nav": pd.DataFrame(),
    }
    estimate = current_position_cash_estimate(
        item,
        live_root=live,
        raw_daily_cache_dir=raw,
        historical_orders=historical_orders,
        forward_orders=pd.DataFrame(),
    )
    assert estimate["complete"]
    assert abs(float(estimate["upper_limit_notional"]) - 11000.0) < 1e-9
    assert abs(float(estimate["observed_fee_rate"]) - 0.0005) < 1e-12
    assert abs(float(estimate["estimated_cash_required"]) - 11005.5) < 1e-9
