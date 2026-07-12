#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static structural checks for the Ch17 AS1455 refactor.

The checks deliberately avoid market data and model files. They verify that the
repository structure does not silently reintroduce duplicated protocol, trading,
or plotting implementations.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_ch17_common import TARGET_SPECS  # noqa: E402
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
        for key, value in TARGET_SPECS.items()
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
    top5 = signal_specs_for_top_n(5)
    assert len(top5) == 7
    assert top5[-2:] == [
        "ensemble_first3_mean:0,1,2:mean",
        "ensemble_all5_mean:0,1,2,3,4:mean",
    ]


def assert_single_trade_engine() -> None:
    grid = read("code/backtest/run_as1455_close_auction_grid_inprocess.py")
    assert "bt.backtest(" in grid
    for forbidden in (
        "def backtest_prepared(",
        "inspect.getsource",
        "exec(",
        "base.plot_frequency =",
    ):
        assert forbidden not in grid, f"forbidden duplicate mechanism: {forbidden}"

    v7_functions = function_names(
        "code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py"
    )
    assert "backtest" in v7_functions
    grid_functions = function_names(
        "code/backtest/run_as1455_close_auction_grid_inprocess.py"
    )
    assert "backtest" not in grid_functions
    assert "backtest_prepared" not in grid_functions


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


def assert_shared_prediction_layer() -> None:
    common = function_names("utils/as1455_ch17_common.py")
    required = {
        "build_target_features",
        "get_fold_target",
        "read_top_checkpoints",
        "load_preprocess",
        "transform_for_source_model",
        "predict_checkpoint_set",
        "write_prediction_artifacts",
        "build_grid_command",
    }
    missing = required - common
    assert not missing, f"missing shared functions: {sorted(missing)}"

    for relative in (
        "scripts/run_as1455_target_one_lag_backtest.py",
        "scripts/run_as1455_fold0_forward_backtest.py",
    ):
        names = function_names(relative)
        duplicated = names & {
            "read_top_checkpoints",
            "load_preprocess",
            "transform_for_source_model",
            "predict_checkpoint_set",
            "write_prediction_artifacts",
            "build_grid_command",
        }
        assert not duplicated, f"{relative} duplicates {sorted(duplicated)}"


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
        assert_single_trade_engine,
        assert_thin_compatibility_wrappers,
        assert_shared_prediction_layer,
        assert_unified_plotter,
    ]
    for check in checks:
        check()
        print(f"[OK] {check.__name__}")
    print("[PASS] Ch17 AS1455 structural checks passed")


if __name__ == "__main__":
    main()
