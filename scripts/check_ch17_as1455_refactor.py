#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural checks for the AS1455 Ch17 refactor."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_rank_cache as rank_cache  # noqa: E402


def read(relative: str) -> str:
    return (PROJECT_DIR / relative).read_text(encoding="utf-8")


def function_names(relative: str) -> set[str]:
    tree = ast.parse(read(relative), filename=relative)
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def assert_target_specs() -> None:
    expected = {
        "r01_fwd": (1, 1, "zero"),
        "r05_fwd": (5, 5, "full"),
        "r21_fwd": (21, 21, "full"),
    }
    actual = {
        name: (spec.lookahead, spec.rebalance_every, spec.offset_mode)
        for name, spec in common.TARGET_SPECS.items()
    }
    assert actual == expected, (actual, expected)


def assert_signal_specs() -> None:
    for value in common.default_signal_specs(5):
        parts = value.split(":")
        assert len(parts) == 3, value
        name, columns, mode = parts
        assert name
        assert mode in {"single", "mean"}
        assert all(token.isdigit() for token in columns.split(","))


def assert_path_contracts() -> None:
    from utils import as1455_paths

    assert as1455_paths.DEFAULT_MODEL_DATA.name == "model_data_as1455.h5"
    assert as1455_paths.TARGET_BACKTEST_ROOT.name == "ch17_as1455_target_backtest"
    assert (
        as1455_paths.FOLD0_FORWARD_BACKTEST_ROOT.name
        == "ch17_as1455_fold0_forward_backtest"
    )


def _make_prediction_frame() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2026-01-01", periods=4, freq="B"):
        for symbol, score in (
            ("000001.SZ", 0.2),
            ("000002.SZ", 0.5),
            ("600000.SH", -0.1),
        ):
            rows.append({"date": date, "symbol": symbol, "score": score})
    return pd.DataFrame(rows)


def assert_rank_cache_equivalence() -> None:
    source = _make_prediction_frame()
    expected = {
        date: group.sort_values("score", ascending=False).reset_index(drop=True)
        for date, group in source.groupby("date", sort=True)
    }
    prepared = rank_cache.prepare_presorted_predictions(source)
    rank_cache.validate_presorted_predictions(prepared)
    actual = {
        date: group.sort_values("score", ascending=False).reset_index(drop=True)
        for date, group in prepared.groupby("date", sort=True)
    }
    assert expected.keys() == actual.keys()
    for date in expected:
        pd.testing.assert_frame_equal(
            pd.DataFrame(expected[date]),
            pd.DataFrame(actual[date]),
        )


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


def assert_storage_maintenance_entrypoint() -> None:
    shell_path = PROJECT_DIR / "scripts" / "run_as1455_storage_maintenance.sh"
    exporter_path = PROJECT_DIR / "scripts" / "export_as1455_storage_diagnostics.py"
    guide_path = PROJECT_DIR / "AS1455_STORAGE_MAINTENANCE.md"
    assert shell_path.exists()
    assert exporter_path.exists()
    assert guide_path.exists()

    shell = shell_path.read_text(encoding="utf-8")
    assert 'APPLY="${APPLY:-0}"' in shell
    assert 'SHARE_FILE="$OUT_DIR/share_me.txt"' in shell
    assert "cleanup_dry_run.json" in shell
    assert "cleanup_apply.json" in shell
    assert "scripts/export_as1455_storage_diagnostics.py" in shell
    assert "scripts/cleanup_as1455_storage.py" in shell

    exporter_functions = function_names("scripts/export_as1455_storage_diagnostics.py")
    for required in {
        "active_as1455_processes",
        "scan_files",
        "important_paths",
        "main",
    }:
        assert required in exporter_functions


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
        assert_storage_maintenance_entrypoint,
    ]
    for check in checks:
        check()
        print(f"[OK] {check.__name__}")
    print("[PASS] Ch17 AS1455 structural checks passed")


if __name__ == "__main__":
    main()
