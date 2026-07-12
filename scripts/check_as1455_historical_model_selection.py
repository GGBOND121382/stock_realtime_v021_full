#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic checks for shared AS1455 historical model selection."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_model_selection import (  # noqa: E402
    select_historical_signal,
    signal_spec_from_row,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rotation_onehot_r05_fwd_reb5_20260712"
        summary_dir = root / "01_close_auction_grid" / "02_summary"
        summary_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run_name": "model_2_max10_sell100_reb5_off0",
                    "status": "ok",
                    "signal_name": "model_2",
                    "signal_cols": "2",
                    "signal_mode": "single",
                    "max_positions": 10,
                    "sell_rank": 100,
                    "rebalance_every": 5,
                    "rebalance_offset": 0,
                    "sharpe": 1.2,
                },
                {
                    "run_name": "ensemble_first3_mean_max20_sell150_reb5_off3",
                    "status": "ok",
                    "signal_name": "ensemble_first3_mean",
                    "signal_cols": "0,1,2",
                    "signal_mode": "mean",
                    "max_positions": 20,
                    "sell_rank": 150,
                    "rebalance_every": 5,
                    "rebalance_offset": 3,
                    "sharpe": 1.5,
                },
                {
                    "run_name": "failed_model_4",
                    "status": "failed",
                    "signal_name": "model_4",
                    "signal_cols": "4",
                    "signal_mode": "single",
                    "sharpe": 9.9,
                },
            ]
        ).to_csv(summary_dir / "grid_summary_compact.csv", index=False)

        selected = select_historical_signal(
            backtest_root=root,
            rank_metric="sharpe",
        )
        assert selected.signal_spec == "ensemble_first3_mean:0,1,2:mean"
        assert selected.required_top_n == 3
        assert selected.historical_max_positions == 20
        assert selected.historical_sell_rank == 150
        assert selected.historical_rebalance_offset == 3

    model4 = pd.Series(
        {
            "signal_name": "model_4",
            "signal_cols": 4.0,
            "signal_mode": "single",
        }
    )
    spec, required_top_n = signal_spec_from_row(model4)
    assert spec == "model_4:4:single"
    assert required_top_n == 5
    print("[PASS] historical model signal selection")


if __name__ == "__main__":
    main()
