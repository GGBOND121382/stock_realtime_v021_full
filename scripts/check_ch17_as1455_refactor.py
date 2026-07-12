#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static and synthetic checks for the Ch17 AS1455 refactor.

The checks avoid market data and model files. They verify that the repository
does not silently reintroduce duplicated protocol, trading, path, CLI, ranking,
or plotting implementations.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_rank_cache import (  # noqa: E402
    PreSortedPredictionFrame,
    prepare_presorted_predictions,
    validate_presorted_predictions,
)
from utils.as1455_signal_specs import signal_specs_for_top_n  # noqa: E402


def read(relative: str) -> str:
    path = PROJECT_DIR / relative
    if not path.exists():
        raise AssertionError(f"missing file: {relative}")
    return path.read_text(encoding="utf-8")


def function_names(relative: str) -> set[str]:
    tree = ast.parse(read(relative), filename=relative)
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def assert_target_specs() -> None:
    actual = {
        key: (value.lookahead, value.rebalance_every, value.offset_mode)
        for key, value in common.TARGET_SPECS.items()
    }
    expected = {
        "r01_fwd": (1, 1, "zero"),
        "r05_fwd": (5, 5, "full"),
        "r21_fwd": (21, 21, "full"),
    }
    assert actual == expected, (actual, expected)


def assert_signal_specs() -> None:
    assert signal_specs_for_top_n(1) == ["model_0:0:single"]
    assert signal_specs_for_top_n(2) == [
        "model_0:0:single",
        "model_1:1:single",
        "ensemble_all2_mean:0,1:mean",
    ]
    assert signal_specs_for_top_n(3) == [
        "model_0:0:single",
        "model_1:1:single",
        "model_2:2:single",
        "ensemble_first3_mean:0,1,2:mean",
    ]
    top5 = signal_specs_for_top_n(5)
    assert len(top5) == 7
    assert top5[-2:] == [
        "ensemble_first3_mean:0,1,2:mean",
        "ensemble_all5_mean:0,1,2,3,4:mean",
    ]


def assert_path_contracts() -> None:
    assert as1455_paths.DEFAULT_MODEL_DATA == (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch12_as1455"
        / "model_data_as1455.h5"
    )
    for preset in common.FEATURE_PRESETS:
        r1_template = common.default_fold_dir_template(preset, "r01_fwd")
        r1_fold0 = common.fold_dir_from_template(r1_template, 0)
        assert "fold0_search" in r1_fold0.name
        assert "r01_fwd" not in r1_fold0.parts

        r5_template = common.default_fold_dir_template(preset, "r05_fwd")
        r5_fold0 = common.fold_dir_from_template(r5_template, 0)
        assert "r05_fwd" in r5_fold0.parts
        assert r5_fold0.name == "fold0_search"


def assert_rank_cache_equivalence() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-02", "2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05"]
            ),
            "symbol": ["A", "B", "C", "D", "E"],
            "score": [0.2, 0.5, 0.5, -0.1, 0.3],
        },
        index=[10, 11, 12, 20, 21],
    )
    expected = {
        date: group.copy().sort_values("score", ascending=False).copy()
        for date, group in predictions.groupby("date", sort=True)
    }
    cached = prepare_presorted_predictions(predictions)
    assert isinstance(cached, PreSortedPredictionFrame)
    validate_presorted_predictions(cached)
    actual = {
        date: group.copy().sort_values("score", ascending=False).copy()
        for date, group in cached.groupby("date", sort=True)
    }
    assert set(expected) == set(actual)
    for date in expected:
        pd.testing.assert_frame_equal(expected[date], actual[date])


def assert_single_trade_engine() -> None:
    entry = read("code/backtest/run_as1455_close_auction_grid_inprocess.py")
    assert "utils.as1455_grid_runner" in entry
    assert "def backtest(" not in entry

    runner = read("utils/as1455_grid_runner.py")
    assert "bt.backtest(" in runner
    assert "prepare_presorted_predictions" in runner
    for forbidden in (
        "def backtest_prepared(",
        "def backtest(",
        "inspect.getsource",
        "exec(",
        "base.plot_frequency =",
    ):
        assert forbidden not in runner, f"forbidden duplicate mechanism: {forbidden}"

    v7_functions = function_names(
        "code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py"
    )
    assert "backtest" in v7_functions
    runner_functions = function_names("utils/as1455_grid_runner.py")
    assert "backtest" not in runner_functions
    assert "backtest_prepared" not in runner_functions


def assert_thin_compatibility_wrappers() -> None:
    wrappers = [
        "scripts/run_as1455_rotation_one_lag_daily_backtest.py",
        "scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py",
        "scripts/run_as1455_r05_target_search_all.sh",
        "scripts/run_as1455_r21_target_search_all.sh",
        "scripts/run_as1455_r05_natural_backtest.sh",
        "scripts/run_as1455_r21_natural_backtest.sh",
    ]
    forbidden_python_functions = {
        "read_top_checkpoints",
        "transform_for_source_model",
        "build_feature_matrix",
        "make_one_lag_predictions",
        "backtest",
    }
    for relative in wrappers:
        source = read(relative)
        if relative.endswith(".py"):
            names = function_names(relative)
            duplicates = names & forbidden_python_functions
            assert not duplicates, f"{relative} reimplements {sorted(duplicates)}"
        assert "inspect.getsource" not in source
        assert "exec(" not in source


def assert_shared_prediction_and_cli_layers() -> None:
    common_functions = function_names("utils/as1455_ch17_common.py")
    required_common = {
        "build_target_features",
        "get_fold_target",
        "read_top_checkpoints",
        "load_preprocess",
        "transform_for_source_model",
        "predict_checkpoint_set",
        "write_prediction_artifacts",
        "build_grid_command",
    }
    missing = required_common - common_functions
    assert not missing, f"missing shared functions: {sorted(missing)}"

    cli_functions = function_names("utils/as1455_cli.py")
    required_cli = {
        "add_prediction_grid_arguments",
        "normalize_common_prediction_args",
        "run_prediction_grid",
        "resolve_existing_prediction",
    }
    missing_cli = required_cli - cli_functions
    assert not missing_cli, f"missing shared CLI functions: {sorted(missing_cli)}"

    for relative in (
        "scripts/run_as1455_target_one_lag_backtest.py",
        "scripts/run_as1455_fold0_forward_backtest.py",
    ):
        source = read(relative)
        names = function_names(relative)
        duplicated = names & {
            "read_top_checkpoints",
            "load_preprocess",
            "transform_for_source_model",
            "predict_checkpoint_set",
            "write_prediction_artifacts",
            "build_grid_command",
            "add_prediction_grid_arguments",
        }
        assert not duplicated, f"{relative} duplicates {sorted(duplicated)}"
        assert "as1455_cli.add_prediction_grid_arguments" in source
        assert "as1455_cli.run_prediction_grid" in source


def assert_unified_plotter() -> None:
    assert not (
        PROJECT_DIR
        / "scripts"
        / "plot_as1455_backtest_return_curves_accessible.py"
    ).exists()
    shell = read("scripts/plot_as1455_default_ab_nav_curves.sh")
    assert "scripts/plot_as1455_backtest_return_curves.py" in shell
    plotter = read("scripts/plot_as1455_backtest_return_curves.py")
    assert "utils.as1455_plotting" in plotter


def main() -> None:
    checks = [
        assert_target_specs,
        assert_signal_specs,
        assert_path_contracts,
        assert_rank_cache_equivalence,
        assert_single_trade_engine,
        assert_thin_compatibility_wrappers,
        assert_shared_prediction_and_cli_layers,
        assert_unified_plotter,
    ]
    for check in checks:
        check()
        print(f"[OK] {check.__name__}")
    print("[PASS] Ch17 AS1455 structural checks passed")


if __name__ == "__main__":
    main()
