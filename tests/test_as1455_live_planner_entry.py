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
