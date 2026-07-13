#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic regression checks for AS1455 storage and strict-OOS fixes."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_forward_features import load_inference_xy  # noqa: E402
from utils.as1455_model_selection import HistoricalSignalSelection  # noqa: E402
from utils.as1455_rebalance_phase import align_forward_rebalance_phase  # noqa: E402
from utils.as1455_strict_oos import (  # noqa: E402
    apply_strict_oos_args,
    finalize_strict_oos_grid,
)


def check_forward_dates(tmp: Path) -> None:
    dates = pd.bdate_range("2026-06-22", periods=15)
    symbols = ["000001.SZ", "600000.SH"]
    index = pd.MultiIndex.from_product(
        [symbols, dates], names=["symbol", "date"]
    )
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(len(index), len(common.base.EXPECTED_MODEL_COLUMNS))),
        index=index,
        columns=common.base.EXPECTED_MODEL_COLUMNS,
    )
    frame["sector"] = 1
    frame["year"] = frame.index.get_level_values("date").year
    frame["month"] = frame.index.get_level_values("date").month
    frame["weekday"] = frame.index.get_level_values("date").weekday
    for symbol in symbols:
        symbol_index = frame.xs(symbol, level="symbol").index
        frame.loc[(symbol, symbol_index[-1]), "r01_fwd"] = np.nan
        frame.loc[(symbol, symbol_index[-5:]), "r05_fwd"] = np.nan
        frame.loc[(symbol, symbol_index), "r21_fwd"] = np.nan

    path = tmp / "model_data.h5"
    frame.to_hdf(path, "model_data", mode="w", format="table")
    X, y, report = load_inference_xy(path, None, "r05_fwd")
    x_dates = pd.DatetimeIndex(X.index.get_level_values("date"))
    assert x_dates.max() == dates.max()
    assert y.loc[y.index.get_level_values("date") == dates.max()].isna().all()
    assert report["feature_valid_max_date"] == dates.max().strftime("%Y-%m-%d")
    assert report["target_valid_max_date"] == dates[-6].strftime("%Y-%m-%d")
    assert report["unlabeled_prediction_dates"] == 5


def make_selection(root: Path) -> HistoricalSignalSelection:
    return HistoricalSignalSelection(
        backtest_root=str(root),
        summary_file=str(root / "summary.csv"),
        rank_metric="sharpe",
        rank_metric_value=1.5,
        run_name="ensemble_first3_mean_max20_sell150_reb5_off3",
        signal_name="ensemble_first3_mean",
        signal_cols="0,1,2",
        signal_mode="mean",
        signal_spec="ensemble_first3_mean:0,1,2:mean",
        required_top_n=3,
        historical_max_positions=20,
        historical_sell_rank=150,
        historical_rebalance_every=5,
        historical_rebalance_offset=3,
        historical_date_min="2025-01-02",
        historical_date_max="2026-06-30",
        historical_n_days=378,
    )


def check_strict_oos(tmp: Path) -> None:
    selection = make_selection(tmp / "historical")
    args = SimpleNamespace(
        rebalance_every=5,
        max_positions_list="5,10,15,20,25",
        sell_rank_list="75,100,150,200,250,300",
        offset_mode="full",
        rebalance_offset_list=None,
    )
    config = apply_strict_oos_args(args, selection)
    assert config == {
        "max_positions": 20,
        "sell_rank": 150,
        "rebalance_every": 5,
        "rebalance_offset": 3,
    }
    assert args.max_positions_list == "20"
    assert args.sell_rank_list == "150"
    assert args.offset_mode == "full"
    assert args.rebalance_offset_list is None
    assert args.rebalance_phase_history_offset == 3
    assert args.rebalance_phase_history_n_days == 378

    calendar = pd.bdate_range("2026-07-01", periods=5)
    alignment = align_forward_rebalance_phase(
        rebalance_every=5,
        historical_offset=3,
        historical_n_days=378,
        historical_first_date="2025-01-02",
        historical_last_date="2026-06-30",
        forward_prediction_dates=calendar,
        execution_calendar_dates=calendar,
    )
    assert alignment["forward_global_index"] == 378
    assert alignment["effective_forward_offset"] == 0
    assert alignment["historical_offset_numeric_reused"] is False

    out_root = tmp / "forward"
    grid = out_root / "01_close_auction_grid"
    summary_dir = grid / "02_summary"
    runs_dir = grid / "01_runs"
    logs_dir = grid / "04_logs"
    summary_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    run_name = "ensemble_first3_mean_max20_sell150_reb5_off0"
    pd.DataFrame(
        [
            {
                "run_name": run_name,
                "status": "ok",
                "signal_name": "ensemble_first3_mean",
                "signal_cols": "0,1,2",
                "signal_mode": "mean",
                "max_positions": 20,
                "sell_rank": 150,
                "rebalance_every": 5,
                "rebalance_offset": 0,
                "sharpe": 1.0,
                "total_return": 0.1,
            }
        ]
    ).to_csv(summary_dir / "grid_summary.csv", index=False)
    run_dir = runs_dir / run_name
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (logs_dir / f"{run_name}.log").write_text("ok", encoding="utf-8")
    (grid / "grid_engine_manifest.json").write_text(
        json.dumps(
            {
                "engine": "synthetic",
                "rebalance_phase_alignment": alignment,
            }
        ),
        encoding="utf-8",
    )

    manifest = finalize_strict_oos_grid(out_root, selection)
    assert manifest["retained_run_name"].endswith("off0")
    assert manifest["retained_config_count"] == 1
    assert manifest["historical_config"]["rebalance_offset"] == 3
    assert manifest["retained_config"]["rebalance_offset"] == 0
    assert manifest["historical_rebalance_phase_reused"] is True
    assert manifest["historical_offset_numeric_reused"] is False
    retained = pd.read_csv(summary_dir / "grid_summary_compact.csv")
    assert len(retained) == 1
    assert int(retained.iloc[0]["rebalance_offset"]) == 0
    assert sorted(path.name for path in runs_dir.iterdir()) == [run_name]
    engine = json.loads((grid / "grid_engine_manifest.json").read_text(encoding="utf-8"))
    assert engine["historical_trading_parameters_reused"] is True
    assert engine["historical_rebalance_phase_reused"] is True


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        check_forward_dates(tmp)
        check_strict_oos(tmp)
    print("[PASS] AS1455 forward dates, phase alignment and strict OOS")


if __name__ == "__main__":
    main()
