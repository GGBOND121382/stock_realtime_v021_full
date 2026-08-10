from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scripts import update_as1455_tracking_accounts as tracker


@dataclass(frozen=True)
class FakeConfig:
    rebalance_offset: int
    corporate_action_mode: str = "none"


class FakeSelection:
    historical_rebalance_every = 5
    historical_rebalance_offset = 2


class FakeV7:
    def __init__(self) -> None:
        self.captured_config = None

    def backtest(self, predictions, execution, cfg, **kwargs):
        self.captured_config = cfg
        return {
            "nav": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "rejections": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "final_state": {"cash": 100.0, "positions": []},
        }


def test_tracking_chunk_keeps_phase_aligned_offset(monkeypatch) -> None:
    index = pd.MultiIndex.from_arrays(
        [["000001"], pd.to_datetime(["2026-08-07"])],
        names=["symbol", "date"],
    )
    predictions = pd.DataFrame(
        {0: [1.0], 1: [1.0], 2: [1.0], 3: [1.0], 4: [1.0]}, index=index
    )
    execution = pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-07"), "symbol": "000001"}]
    )

    monkeypatch.setattr(
        tracker.live,
        "score_predictions",
        lambda frame, selection: pd.Series(
            [1.0] * len(frame), index=frame.index, name="score"
        ),
    )
    monkeypatch.setattr(
        tracker,
        "align_forward_rebalance_phase",
        lambda **kwargs: {"effective_forward_offset": 3},
    )
    monkeypatch.setattr(
        tracker.live,
        "build_trade_config",
        lambda *args, **kwargs: FakeConfig(rebalance_offset=3),
    )
    monkeypatch.setattr(
        tracker.planner,
        "synthetic_corporate_action_mode",
        lambda historical_config: "none",
    )

    v7 = FakeV7()
    tracker.run_chunk(
        v7,
        FakeSelection(),
        {},
        {
            "historical_n_days": 100,
            "historical_first_date": "2025-01-01",
            "historical_last_date": "2025-12-31",
        },
        predictions,
        execution,
        pd.DatetimeIndex(["2026-08-07"]),
        pd.DatetimeIndex(["2026-08-07"]),
        100.0,
        pd.DataFrame(),
        "none",
        0.05,
    )

    assert v7.captured_config is not None
    assert v7.captured_config.rebalance_offset == 3


def test_first_entry_marker_uses_first_buy_not_account_start() -> None:
    frames = {
        "nav": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-07", "2026-08-10"]),
                "nav": [200000.0, 200000.0],
            }
        ),
        "orders": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-10"]),
                "side": ["buy"],
                "symbol": ["000001"],
            }
        ),
        "positions": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-10"]),
                "symbol": ["000001"],
            }
        ),
        "rejections": pd.DataFrame(),
    }

    first_entry = tracker.annotate_first_entry(frames)

    assert first_entry == pd.Timestamp("2026-08-10")
    assert frames["nav"]["tracking_bootstrap"].tolist() == [False, True]
