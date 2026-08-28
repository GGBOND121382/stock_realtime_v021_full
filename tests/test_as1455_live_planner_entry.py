from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from scripts import run_as1455_live_nine_strategy_planner_entry as entry


@dataclass(frozen=True)
class FakeConfig:
    initial_cash: float


class FakeV7:
    @staticmethod
    def summarize_nav(
        nav,
        orders,
        rejects,
        cfg,
        actions=None,
        round_trips=None,
        daily_drawdown=None,
    ):
        total_return = float(nav["nav"].iloc[-1] / cfg.initial_cash - 1.0)
        return {
            "initial_cash": cfg.initial_cash,
            "total_return": total_return,
        }


def test_starting_portfolio_nav_recovers_pre_day_nav() -> None:
    nav = pd.DataFrame([{"nav": 201_000.0, "daily_return": 0.005}])
    assert entry._starting_portfolio_nav(nav) == pytest.approx(200_000.0)


def test_live_summary_adapter_uses_portfolio_nav_only_for_summary(monkeypatch) -> None:
    fake = FakeV7()
    original_cfg = FakeConfig(initial_cash=1_000.0)
    nav = pd.DataFrame([{"nav": 201_000.0, "daily_return": 0.005}])

    monkeypatch.setattr(entry.planner.live, "load_v7_module", lambda: fake)
    entry.install_live_summary_adapter()
    patched = entry.planner.live.load_v7_module()
    summary = patched.summarize_nav(
        nav,
        pd.DataFrame(),
        pd.DataFrame(),
        original_cfg,
    )

    assert summary["initial_cash"] == pytest.approx(200_000.0)
    assert summary["total_return"] == pytest.approx(0.005)
    # The adapter passes a replaced dataclass only to the summary function; it
    # must never mutate the actual residual cash available to the trading loop.
    assert original_cfg.initial_cash == 1_000.0


@pytest.mark.parametrize(
    "historical_mode",
    ["synthetic_share_factor_from_preclose", "synthetic_cash_from_preclose", "none"],
)
def test_live_tracking_corporate_action_policy_keeps_share_quantity_executable(
    historical_mode: str,
) -> None:
    assert entry.planner.synthetic_corporate_action_mode(
        {"corporate_action_mode": historical_mode}
    ) == "synthetic_cash_from_preclose"


def test_execution_batch_accepts_integer_odd_lot_sell_quantity() -> None:
    # A position can legitimately contain an integer odd-lot remainder after a
    # real corporate action. The batch contract requires integer shares, not a
    # 100-share multiple, for sells.
    assert entry._order_qty(pd.Series({"shares": 1031.0})) == 1031


def test_execution_batch_rejects_fractional_share_quantity() -> None:
    with pytest.raises(RuntimeError, match="invalid shares"):
        entry._order_qty(pd.Series({"shares": 1031.042129}))
