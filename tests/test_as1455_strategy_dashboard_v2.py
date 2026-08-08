from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.as1455_live_data import discover_live_dates, load_live_day
from scripts.export_as1455_global_forward_latest_states import export_experiment


def test_live_day_discovery(tmp_path: Path) -> None:
    live_root = tmp_path / "live_as1455"
    nine = live_root / "20260807" / "nine_strategy"
    nine.mkdir(parents=True)
    (nine / "live_nine_strategy_manifest.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    pd.DataFrame(
        [{"experiment": "r01_all5_reb1_fold0_5_forward", "is_rebalance_day": True}]
    ).to_csv(nine / "live_nine_strategy_summary.csv", index=False)
    assert discover_live_dates(live_root) == ["20260807"]
    item = load_live_day(live_root, "20260807")
    assert item["manifest"]["status"] == "ok"
    assert len(item["summary"]) == 1


def test_latest_forward_state_export(tmp_path: Path) -> None:
    experiment = tmp_path / "r05_best_reb5_fold0_5_forward"
    grid = experiment / "strict_oos_forward" / "01_close_auction_grid"
    run = grid / "01_runs" / "model_0"
    run.mkdir(parents=True)
    (grid / "strict_oos_manifest.json").write_text(
        json.dumps({"retained_run_name": "model_0"}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "date": "2026-08-06",
                "nav": 200000.0,
                "cash": 20000.0,
                "n_positions": 1,
                "day_index": 10,
                "is_rebalance_day": False,
                "rebalance_every": 5,
                "rebalance_offset": 2,
            },
            {
                "date": "2026-08-07",
                "nav": 201000.0,
                "cash": 21000.0,
                "n_positions": 1,
                "day_index": 11,
                "is_rebalance_day": True,
                "rebalance_every": 5,
                "rebalance_offset": 2,
            },
        ]
    ).to_csv(run / "close_auction_nav.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-08-06",
                "symbol": "600000.SH",
                "shares": 1000,
                "buy_date": "2026-08-01",
                "avg_entry_price": 10.0,
            },
            {
                "date": "2026-08-07",
                "symbol": "600001.SH",
                "shares": 900,
                "buy_date": "2026-08-07",
                "avg_entry_price": 20.0,
            },
        ]
    ).to_csv(run / "close_auction_positions.csv", index=False)

    state = export_experiment(experiment)
    assert state["asof_date"] == "2026-08-07"
    assert state["cash"] == 21000.0
    assert state["n_positions"] == 1
    positions = pd.read_csv(experiment / "strict_forward_latest_positions.csv")
    assert positions["symbol"].tolist() == ["600001.SH"]
