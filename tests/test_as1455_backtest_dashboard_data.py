from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.as1455_backtest_data import (
    build_forward_comparison,
    discover_experiment_names,
    load_experiment,
    load_matrix_summary,
    parse_experiment_name,
)


def build_experiment(root: Path, name: str, nav_values: list[float]) -> None:
    experiment = root / name
    run = experiment / "strict_oos_forward" / "01_close_auction_grid" / "01_runs" / "winner"
    run.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=len(nav_values), freq="D"),
            "nav": nav_values,
        }
    ).to_csv(run / "close_auction_nav.csv", index=False)
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "config.json").write_text("{}", encoding="utf-8")
    strict = experiment / "strict_oos_forward" / "01_close_auction_grid" / "strict_oos_manifest.json"
    strict.write_text(json.dumps({"retained_run_name": "winner"}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "total_return": nav_values[-1] / nav_values[0] - 1,
                "annual_return": 0.2,
                "sharpe": 1.2,
                "max_drawdown": -0.1,
                "forward_start": "2026-07-01",
                "forward_end": "2026-07-03",
            }
        ]
    ).to_csv(experiment / "strict_forward_result.csv", index=False)
    (experiment / "global_fold0_to_fold5_forward_manifest.json").write_text(
        json.dumps({"fixed_signal_kind": parse_experiment_name(name).signal}),
        encoding="utf-8",
    )


def test_parse_experiment_name() -> None:
    identity = parse_experiment_name("r05_first3_reb5_fold0_5_forward")
    assert identity.target_col == "r05_fwd"
    assert identity.rebalance_every == 5
    assert identity.signal == "first3"


def test_load_summary_and_forward_comparison(tmp_path: Path) -> None:
    names = [
        "r01_all5_reb1_fold0_5_forward",
        "r05_best_reb5_fold0_5_forward",
    ]
    build_experiment(tmp_path, names[0], [100.0, 101.0, 103.0])
    build_experiment(tmp_path, names[1], [100.0, 99.0, 102.0])
    (tmp_path / "expected_experiments.txt").write_text("\n".join(names), encoding="utf-8")

    assert discover_experiment_names(tmp_path) == names
    summary = load_matrix_summary(tmp_path)
    assert len(summary) == 2
    assert set(summary["signal"]) == {"all5", "best"}

    comparison = build_forward_comparison(tmp_path, names)
    assert comparison.shape == (3, 2)
    assert round(float(comparison.iloc[-1, 0]), 6) == 3.0
    assert round(float(comparison.iloc[-1, 1]), 6) == 2.0

    item = load_experiment(tmp_path, names[0])
    assert item["forward_run"].name == "winner"
    assert len(item["forward_nav"]) == 3
