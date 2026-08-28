#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic-only cross-fold evaluation for the r01 fold0 first3 ensemble.

This experiment deliberately violates temporal OOS ordering: the three highest
ranked checkpoints from *fold0* are held fixed and evaluated on every historical
fold0..fold6 test window.  The result is useful only for inspecting model
stability across market regimes; it must not be reported as a strict-OOS result.

Safety / isolation contract:
- read existing r01 fold0 checkpoints, historical model_data and raw-daily cache;
- read the currently frozen r01-first3 trading configuration from the nine-
  strategy matrix;
- never train a model and never run a trading-parameter grid;
- never write tracking, live, model-registry, matrix or checkpoint artifacts;
- write only under a fresh ch17_as1455_diagnostics output directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_model_selection import (  # noqa: E402
    find_summary_file,
    select_historical_signal,
)

FEATURE_PRESET = "rotation_addon_onehot"
TARGET_COL = "r01_fwd"
FIXED_SIGNAL_SPEC = "ensemble_first3_mean:0,1,2:mean"
DEFAULT_MODEL_DATA = (
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "model_data_as1455.h5"
)
DEFAULT_RAW_DAILY = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch12_as1455"
    / "baostock_raw_daily_cache"
)
DEFAULT_MATRIX_ROOT = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)
DEFAULT_UNIVERSE = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_static_universe"
    / "07_universe_allA_top1000_static.csv"
)
DEFAULT_OUT_BASE = (
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_diagnostics"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fold_helpers = load_module(
    "as1455_crossfold_helpers",
    PROJECT_DIR / "scripts" / "run_as1455_independent_fold_backtests.py",
)
bt = fold_helpers.bt


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )


def resolved_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def parse_folds(value: str) -> list[int]:
    folds = common.parse_int_list(value)
    if len(folds) != len(set(folds)):
        raise argparse.ArgumentTypeError(f"duplicate folds: {folds}")
    invalid = [fold for fold in folds if fold < 0 or fold > 6]
    if invalid:
        raise argparse.ArgumentTypeError(f"folds must be in 0..6, got {invalid}")
    return folds


def resolve_formal_r01_first3(
    matrix_root: Path,
) -> tuple[Path, Path, Any, dict[str, Any], Path]:
    """Resolve the frozen trading config used by the production r01-first3 row."""
    expected_file = matrix_root / "expected_experiments.txt"
    names: list[str] = []
    if expected_file.is_file():
        names = [
            line.strip()
            for line in expected_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    candidates = [
        name
        for name in names
        if name.startswith("r01_first3_reb1_") and name.endswith("_forward")
    ]
    if not candidates:
        candidates = sorted(
            path.name
            for path in matrix_root.glob("r01_first3_reb1_*_forward")
            if path.is_dir()
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "cannot uniquely resolve the production r01-first3 experiment: "
            f"matrix_root={matrix_root} candidates={candidates}"
        )

    experiment_root = matrix_root / candidates[0]
    manifest_file = experiment_root / "global_fold0_to_fold5_forward_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest = read_json(manifest_file)
    if str(manifest.get("target_col")) != TARGET_COL:
        raise RuntimeError(f"unexpected target_col in {manifest_file}: {manifest.get('target_col')}")
    if str(manifest.get("fixed_signal_kind")) != "first3":
        raise RuntimeError(
            f"unexpected fixed_signal_kind in {manifest_file}: {manifest.get('fixed_signal_kind')}"
        )
    manifest_spec = str(manifest.get("fixed_signal_spec") or "")
    if manifest_spec and manifest_spec != FIXED_SIGNAL_SPEC:
        raise RuntimeError(
            f"production first3 signal mismatch: {manifest_spec} != {FIXED_SIGNAL_SPEC}"
        )

    raw_history_root = manifest.get("historical_result_root") or manifest.get(
        "historical_result_root"
    )
    if not raw_history_root:
        fallback = experiment_root / "historical_fold_selection"
        if not fallback.is_dir():
            raise RuntimeError(
                f"manifest lacks historical_result_root and fallback is absent: {manifest_file}"
            )
        history_root = fallback.resolve()
    else:
        history_root = resolved_project_path(str(raw_history_root))
        if not history_root.is_dir():
            fallback = experiment_root / "historical_fold_selection"
            if fallback.is_dir():
                history_root = fallback.resolve()
            else:
                raise FileNotFoundError(history_root)

    selection = select_historical_signal(
        backtest_root=history_root,
        rank_metric="sharpe",
    )
    if selection.signal_spec != FIXED_SIGNAL_SPEC:
        raise RuntimeError(
            "resolved production trading selection is not first3: "
            f"{selection.signal_spec}"
        )
    if int(selection.historical_rebalance_every or -1) != 1:
        raise RuntimeError(
            "r01 diagnostic requires the frozen daily rebalance rule; "
            f"got {selection.historical_rebalance_every}"
        )

    _summary_file, grid_dir = find_summary_file(history_root)
    config_file = grid_dir / "01_runs" / selection.run_name / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config = read_json(config_file)
    if str(config.get("signal_name")) != selection.signal_name:
        raise RuntimeError("stored config signal_name differs from selected summary")
    if str(config.get("signal_mode")) != selection.signal_mode:
        raise RuntimeError("stored config signal_mode differs from selected summary")
    config_cols = ",".join(
        str(int(float(token.strip())))
        for token in str(config.get("signal_cols", "")).split(",")
        if token.strip()
    )
    if config_cols != "0,1,2":
        raise RuntimeError(f"stored config is not first3: signal_cols={config_cols}")
    if str(config.get("capacity_mode", "none")) != "none":
        raise RuntimeError(
            "diagnostic intentionally refuses capacity-dependent configs; "
            f"production capacity_mode={config.get('capacity_mode')}"
        )
    return experiment_root, history_root, selection, config, config_file


def build_fixed_fold0_predictions(
    *,
    features: common.FeatureBuildResult,
    fold0_dir: Path,
    folds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]]]:
    fold_reports: dict[int, dict[str, Any]] = {}
    indices: list[np.ndarray] = []
    for fold in folds:
        _train_idx, test_idx, report = common.get_fold_target(
            features.X, fold, TARGET_COL
        )
        if not len(test_idx):
            raise RuntimeError(f"fold{fold} has no test rows")
        fold_reports[fold] = report
        indices.append(np.asarray(test_idx, dtype=int))

    union_idx = np.unique(np.concatenate(indices))
    predictions, checkpoint_rows, source_manifest = common.predict_checkpoint_set(
        features.X,
        union_idx,
        fold0_dir,
        3,
        metadata={
            "source_fold": 0,
            "diagnostic": "fold0_first3_fixed_across_all_folds",
        },
    )
    expected_columns = [0, 1, 2]
    normalized_columns = [int(column) for column in predictions.columns]
    if normalized_columns != expected_columns:
        raise RuntimeError(
            f"unexpected prediction columns: {normalized_columns} != {expected_columns}"
        )

    score = predictions[expected_columns].mean(axis=1).rename("score").reset_index()
    score["date"] = pd.to_datetime(score["date"], errors="coerce").dt.normalize()
    score["symbol"] = score["symbol"].map(bt.normalize_symbol)
    if score[["date", "symbol", "score"]].isna().any().any():
        raise RuntimeError("fold0 first3 score panel contains missing values")
    if score.duplicated(["date", "symbol"]).any():
        raise RuntimeError("fold0 first3 score panel contains duplicate date/symbol rows")
    score = score.sort_values(["date", "symbol"]).reset_index(drop=True)

    checkpoint_table = pd.DataFrame(checkpoint_rows)
    checkpoint_table["diagnostic_source_fold"] = 0
    checkpoint_table["fixed_signal_spec"] = FIXED_SIGNAL_SPEC
    for report in fold_reports.values():
        report["source_model_fold"] = 0
        report["source_model_train_start"] = source_manifest.get("train_start")
        report["source_model_train_end"] = source_manifest.get("train_end")
        report["source_model_test_start"] = source_manifest.get("test_start")
        report["source_model_test_end"] = source_manifest.get("test_end")
    return score, checkpoint_table, fold_reports


def build_execution(
    *,
    score: pd.DataFrame,
    raw_daily_cache_dir: Path,
    universe_file: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = bt.read_universe(universe_file if universe_file and universe_file.exists() else None)
    st_symbols = bt.load_st_symbols(None)
    st_status = bt.load_st_status(None)
    symbols = sorted(score["symbol"].astype(str).unique())
    execution, report = bt.build_execution_panel(
        symbols,
        raw_daily_cache_dir,
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=pd.DataFrame(),
        raw_5m_cache_dir=None,
    )
    if execution.empty:
        raise RuntimeError("execution panel is empty")
    execution = execution.copy()
    execution["date"] = pd.to_datetime(execution["date"], errors="coerce").dt.normalize()
    return execution, report


def write_result(root: Path, result: dict[str, Any], cfg: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame_names = {
        "nav.csv": "nav",
        "orders.csv": "orders",
        "rejections.csv": "rejections",
        "positions.csv": "positions",
        "round_trips.csv": "round_trips",
    }
    for filename, key in frame_names.items():
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(root / filename, index=False, encoding="utf-8-sig")
    write_json(root / "summary.json", result["summary"])
    write_json(root / "config.json", dataclasses.asdict(cfg))


def normalized_curve(result: dict[str, Any], initial_cash: float) -> pd.DataFrame:
    nav = result["nav"].copy()
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    nav["trading_day"] = np.arange(1, len(nav) + 1)
    nav["return_pct"] = (nav["nav"] / float(initial_cash) - 1.0) * 100.0
    return nav[["date", "trading_day", "nav", "return_pct"]]


def plot_fold_comparison(curves: dict[int, pd.DataFrame], out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for fold in sorted(curves):
        curve = curves[fold]
        ax.plot(curve["trading_day"], curve["return_pct"], label=f"fold{fold}")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Trading day within fold")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("R01 fold0 first3 fixed ensemble: independent cross-fold diagnostic")
    ax.legend(ncol=4)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def plot_continuous(curve: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve["date"], curve["return_pct"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("R01 fold0 first3 fixed ensemble: fold6→fold0 continuous diagnostic")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic: hold r01 fold0 top-3 checkpoints fixed and evaluate them on fold0..fold6."
    )
    parser.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    parser.add_argument(
        "--fold0-dir",
        default=str(common.default_fold0_dir(FEATURE_PRESET, TARGET_COL)),
    )
    parser.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    parser.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY))
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--folds", default="0,1,2,3,4,5,6")
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Resolve all existing artifacts/configs but do not load TensorFlow or run the experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    model_data = resolved_project_path(args.model_data)
    fold0_dir = resolved_project_path(args.fold0_dir)
    matrix_root = resolved_project_path(args.matrix_root)
    raw_daily_cache_dir = resolved_project_path(args.raw_daily_cache_dir)
    universe_file = resolved_project_path(args.universe) if args.universe else None

    for label, path, kind in (
        ("model_data", model_data, "file"),
        ("fold0_dir", fold0_dir, "dir"),
        ("matrix_root", matrix_root, "dir"),
        ("raw_daily_cache_dir", raw_daily_cache_dir, "dir"),
    ):
        ok = path.is_file() if kind == "file" else path.is_dir()
        if not ok:
            raise FileNotFoundError(f"missing {label}: {path}")

    experiment_root, history_root, selection, config, config_file = resolve_formal_r01_first3(
        matrix_root
    )
    initial_cash = float(config.get("initial_cash", 200000.0))
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise RuntimeError(f"invalid production initial_cash={initial_cash}")
    fixed_offset = int(selection.historical_rebalance_offset or 0)
    cfg = fold_helpers.config_to_trade_config(
        config,
        initial_cash=initial_cash,
        effective_offset=fixed_offset,
    )

    print("[DIAGNOSTIC] strict_oos=false; temporal look-ahead is intentional")
    print(f"[SOURCE MODEL] fold0_dir={fold0_dir}")
    print(f"[SOURCE SIGNAL] {FIXED_SIGNAL_SPEC}")
    print(f"[FORMAL EXPERIMENT] {experiment_root}")
    print(f"[FORMAL HISTORY] {history_root}")
    print(f"[FORMAL CONFIG] {config_file}")
    print(
        "[FROZEN TRADE] "
        f"max_positions={selection.historical_max_positions} "
        f"sell_rank={selection.historical_sell_rank} "
        f"rebalance_every={selection.historical_rebalance_every} "
        f"offset={fixed_offset} initial_cash={initial_cash:.2f}"
    )
    print(f"[TARGET FOLDS] {folds}")
    if args.check_only:
        print("[PASS] diagnostic inputs resolved; no prediction/backtest executed")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        resolved_project_path(args.out_root)
        if args.out_root
        else (DEFAULT_OUT_BASE / f"r01_fold0_first3_crossfold_{stamp}").resolve()
    )
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"diagnostic output root already exists and is non-empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    features = common.build_target_features(
        model_data,
        args.train_end,
        "target_only",
        TARGET_COL,
        FEATURE_PRESET,
        "onehot",
    )
    score, checkpoints, fold_reports = build_fixed_fold0_predictions(
        features=features,
        fold0_dir=fold0_dir,
        folds=folds,
    )
    score.to_csv(out_root / "fold0_first3_scores_all_folds.csv", index=False, encoding="utf-8-sig")
    checkpoints.to_csv(
        out_root / "fold0_first3_selected_checkpoints.csv", index=False, encoding="utf-8-sig"
    )

    execution, execution_report = build_execution(
        score=score,
        raw_daily_cache_dir=raw_daily_cache_dir,
        universe_file=universe_file,
    )
    execution_report.to_csv(
        out_root / "execution_data_report.csv", index=False, encoding="utf-8-sig"
    )

    rows: list[dict[str, Any]] = []
    curves: dict[int, pd.DataFrame] = {}
    for fold in sorted(folds):
        report = fold_reports[fold]
        start = pd.Timestamp(report["test_start"]).normalize()
        end = pd.Timestamp(report["test_end"]).normalize()
        pred = score.loc[(score["date"] >= start) & (score["date"] <= end)].copy()
        pred_dates = pd.DatetimeIndex(pred["date"].unique()).sort_values()
        if len(pred_dates) < 2:
            raise RuntimeError(f"fold{fold} has fewer than two prediction dates")
        exec_part = execution.loc[execution["date"].isin(pred_dates)].copy()
        result = bt.backtest(
            pred[["date", "symbol", "score"]],
            exec_part,
            cfg,
            corporate_actions=None,
        )
        fold_root = out_root / "independent_folds" / f"fold{fold}"
        write_result(fold_root, result, cfg)
        curve = normalized_curve(result, initial_cash)
        curve.to_csv(fold_root / "normalized_curve.csv", index=False, encoding="utf-8-sig")
        curves[fold] = curve
        summary = dict(result["summary"])
        rows.append(
            {
                "fold": fold,
                "test_start": start.strftime("%Y-%m-%d"),
                "test_end": end.strftime("%Y-%m-%d"),
                "n_prediction_dates": int(len(pred_dates)),
                "source_model_fold": 0,
                "signal_spec": FIXED_SIGNAL_SPEC,
                "max_positions": int(selection.historical_max_positions),
                "sell_rank": int(selection.historical_sell_rank),
                "rebalance_every": 1,
                "rebalance_offset": fixed_offset,
                **summary,
            }
        )
        print(
            "[FOLD] "
            f"fold={fold} dates={start:%Y-%m-%d}..{end:%Y-%m-%d} "
            f"return={float(summary.get('total_return', np.nan)):.6f} "
            f"sharpe={float(summary.get('sharpe', np.nan)):.6f}"
        )

    summary_table = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    summary_table.to_csv(out_root / "crossfold_summary.csv", index=False, encoding="utf-8-sig")
    plot_fold_comparison(curves, out_root / "crossfold_return_comparison.png")

    # Optional continuous diagnostic: same fold0 model and same frozen trading
    # config, with one account carried chronologically from the oldest requested
    # fold to the newest.  This is still intentionally non-OOS.
    all_dates = pd.DatetimeIndex(score["date"].unique()).sort_values()
    exec_all = execution.loc[execution["date"].isin(all_dates)].copy()
    continuous = bt.backtest(
        score[["date", "symbol", "score"]].sort_values(["date", "symbol"]),
        exec_all,
        cfg,
        corporate_actions=None,
    )
    continuous_root = out_root / "continuous_all_requested_folds"
    write_result(continuous_root, continuous, cfg)
    continuous_curve = normalized_curve(continuous, initial_cash)
    continuous_curve.to_csv(
        continuous_root / "normalized_curve.csv", index=False, encoding="utf-8-sig"
    )
    plot_continuous(continuous_curve, out_root / "continuous_return_curve.png")

    manifest = {
        "status": "ok",
        "protocol": "diagnostic_fold0_first3_fixed_crossfold_v1",
        "strict_oos": False,
        "reportable_as_oos": False,
        "warning": (
            "fold0 checkpoints are evaluated backward on older folds; temporal look-ahead is intentional "
            "for diagnostics and these results must not be used as OOS performance evidence"
        ),
        "feature_preset": FEATURE_PRESET,
        "target_col": TARGET_COL,
        "source_model_fold": 0,
        "fixed_signal_spec": FIXED_SIGNAL_SPEC,
        "requested_folds": sorted(folds),
        "fold0_dir": str(fold0_dir),
        "model_data": str(model_data),
        "raw_daily_cache_dir": str(raw_daily_cache_dir),
        "formal_experiment_root": str(experiment_root),
        "formal_historical_root": str(history_root),
        "formal_config_file": str(config_file),
        "formal_selection": selection.to_dict(),
        "frozen_trade_config": dataclasses.asdict(cfg),
        "outputs": {
            "crossfold_summary": str(out_root / "crossfold_summary.csv"),
            "crossfold_plot": str(out_root / "crossfold_return_comparison.png"),
            "continuous_plot": str(out_root / "continuous_return_curve.png"),
            "continuous_summary": str(continuous_root / "summary.json"),
        },
    }
    write_json(out_root / "manifest.json", manifest)

    show_cols = [
        column
        for column in (
            "fold",
            "test_start",
            "test_end",
            "total_return",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "n_orders",
        )
        if column in summary_table.columns
    ]
    print(summary_table[show_cols].to_string(index=False))
    print("[PASS] diagnostic-only r01 fold0 first3 cross-fold experiment finished")
    print(f"[PASS] output={out_root}")


if __name__ == "__main__":
    main()
