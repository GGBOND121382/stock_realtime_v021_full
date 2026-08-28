#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic-only fold0-model evaluation across all historical AS1455 folds.

For one target (r01/r05/r21), load the already-trained fold0 Top-5 checkpoints
once, predict requested historical fold test windows, and evaluate any subset of
three fixed signals:

- best:   model_0:0:single
- first3: ensemble_first3_mean:0,1,2:mean
- all5:   ensemble_all5_mean:0,1,2,3,4:mean

Each signal reuses its own frozen trading configuration from the production
nine-strategy historical matrix.  No model training or trading-parameter Grid
is run.  Results are diagnostic only: older folds overlap the fold0 model's
training window and therefore are not strict-OOS evidence.

Isolation contract:
- read existing checkpoints, model_data, raw-daily cache and matrix results;
- never write tracking/live/model-registry/matrix/checkpoint artifacts;
- write only under ch17_as1455_diagnostics (or an explicit --out-root).
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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_model_selection import find_summary_file, select_historical_signal  # noqa: E402

FEATURE_PRESET = "rotation_addon_onehot"
SIGNALS: dict[str, dict[str, Any]] = {
    "best": {
        "spec": "model_0:0:single",
        "cols": [0],
        "mode": "single",
    },
    "first3": {
        "spec": "ensemble_first3_mean:0,1,2:mean",
        "cols": [0, 1, 2],
        "mode": "mean",
    },
    "all5": {
        "spec": "ensemble_all5_mean:0,1,2,3,4:mean",
        "cols": [0, 1, 2, 3, 4],
        "mode": "mean",
    },
}
TARGET_SHORT = {"r01_fwd": "r01", "r05_fwd": "r05", "r21_fwd": "r21"}
DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "model_data_as1455.h5"
DEFAULT_RAW_DAILY = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "baostock_raw_daily_cache"
DEFAULT_MATRIX_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_global_fixed_signal_matrix" / "refresh_all_v1"
DEFAULT_OUT_BASE = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_diagnostics"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fold_helpers = load_module(
    "as1455_generic_crossfold_helpers",
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


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def parse_ints(value: str) -> list[int]:
    folds = common.parse_int_list(value)
    if len(folds) != len(set(folds)):
        raise argparse.ArgumentTypeError(f"duplicate folds: {folds}")
    bad = [fold for fold in folds if fold < 0 or fold > 6]
    if bad:
        raise argparse.ArgumentTypeError(f"folds must be in 0..6, got {bad}")
    return folds


def parse_signals(value: str) -> list[str]:
    items = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--signals is empty")
    aliases = {"first5": "all5"}
    items = [aliases.get(item, item) for item in items]
    bad = [item for item in items if item not in SIGNALS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unsupported signals={bad}; expected best,first3,all5"
        )
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError(f"duplicate signals: {items}")
    return items


def normalized_dates(values: Iterable[Any]) -> pd.DatetimeIndex:
    return (
        pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )


def resolve_formal_strategy(
    *, matrix_root: Path, target_col: str, signal_kind: str
) -> tuple[Path, Path, Any, dict[str, Any], Path]:
    spec = common.target_spec(target_col)
    short = TARGET_SHORT[target_col]
    expected_file = matrix_root / "expected_experiments.txt"
    names: list[str] = []
    if expected_file.is_file():
        names = [
            line.strip()
            for line in expected_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    prefix = f"{short}_{signal_kind}_reb{spec.rebalance_every}_"
    candidates = [
        name for name in names if name.startswith(prefix) and name.endswith("_forward")
    ]
    if not candidates:
        candidates = sorted(
            path.name
            for path in matrix_root.glob(f"{prefix}*_forward")
            if path.is_dir()
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "cannot uniquely resolve formal strategy: "
            f"target={target_col} signal={signal_kind} candidates={candidates}"
        )

    experiment_root = (matrix_root / candidates[0]).resolve()
    manifest_file = experiment_root / "global_fold0_to_fold5_forward_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest = read_json(manifest_file)
    if str(manifest.get("target_col")) != target_col:
        raise RuntimeError(
            f"target mismatch in {manifest_file}: {manifest.get('target_col')}"
        )
    if str(manifest.get("fixed_signal_kind")) != signal_kind:
        raise RuntimeError(
            f"signal mismatch in {manifest_file}: {manifest.get('fixed_signal_kind')}"
        )
    expected_spec = str(SIGNALS[signal_kind]["spec"])
    manifest_spec = str(manifest.get("fixed_signal_spec") or "")
    if manifest_spec and manifest_spec != expected_spec:
        raise RuntimeError(
            f"fixed signal spec mismatch: {manifest_spec} != {expected_spec}"
        )

    raw_history = manifest.get("historical_result_root")
    fallback = experiment_root / "historical_fold_selection"
    if raw_history:
        history_root = project_path(str(raw_history))
        if not history_root.is_dir() and fallback.is_dir():
            history_root = fallback.resolve()
    else:
        history_root = fallback.resolve()
    if not history_root.is_dir():
        raise FileNotFoundError(history_root)

    selection = select_historical_signal(
        backtest_root=history_root,
        rank_metric="sharpe",
    )
    if selection.signal_spec != expected_spec:
        raise RuntimeError(
            f"selected historical signal mismatch: {selection.signal_spec} != {expected_spec}"
        )
    if int(selection.historical_rebalance_every or -1) != spec.rebalance_every:
        raise RuntimeError(
            "selected rebalance interval mismatch: "
            f"{selection.historical_rebalance_every} != {spec.rebalance_every}"
        )
    required = {
        "max_positions": selection.historical_max_positions,
        "sell_rank": selection.historical_sell_rank,
        "rebalance_offset": selection.historical_rebalance_offset,
        "history_first_date": selection.historical_date_min,
        "history_last_date": selection.historical_date_max,
        "history_n_days": selection.historical_n_days,
    }
    missing = [key for key, val in required.items() if val is None]
    if missing:
        raise RuntimeError(
            f"formal frozen strategy metadata is incomplete: {missing}"
        )

    _summary_file, grid_dir = find_summary_file(history_root)
    config_file = grid_dir / "01_runs" / selection.run_name / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config = read_json(config_file)
    if str(config.get("capacity_mode", "none")) != "none":
        raise RuntimeError(
            "diagnostic supports the current production capacity_mode=none only; "
            f"got {config.get('capacity_mode')}"
        )
    return experiment_root, history_root, selection, config, config_file


def all_fold_indices_and_reports(
    features: common.FeatureBuildResult, target_col: str
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    indices: dict[int, np.ndarray] = {}
    reports: dict[int, dict[str, Any]] = {}
    for fold in range(7):
        _train_idx, test_idx, report = common.get_fold_target(
            features.X, fold, target_col
        )
        indices[fold] = np.asarray(test_idx, dtype=int)
        reports[fold] = report
    return indices, reports


def predict_fold0_topn(
    *,
    features: common.FeatureBuildResult,
    fold0_dir: Path,
    requested_folds: list[int],
    fold_indices: dict[int, np.ndarray],
    top_n: int,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    union_idx = np.unique(
        np.concatenate([fold_indices[fold] for fold in requested_folds])
    )
    predictions, checkpoint_rows, source_manifest = common.predict_checkpoint_set(
        features.X,
        union_idx,
        fold0_dir,
        top_n,
        metadata={
            "source_fold": 0,
            "target_col": target_col,
            "diagnostic": "fold0_fixed_models_across_historical_folds",
        },
    )
    expected = list(range(top_n))
    actual = [int(column) for column in predictions.columns]
    if actual != expected:
        raise RuntimeError(f"unexpected model columns: {actual} != {expected}")
    return predictions.sort_index(), pd.DataFrame(checkpoint_rows), source_manifest


def signal_score(predictions: pd.DataFrame, signal_kind: str) -> pd.DataFrame:
    signal = SIGNALS[signal_kind]
    cols = list(signal["cols"])
    if signal["mode"] == "single":
        score = predictions[cols[0]]
    else:
        score = predictions[cols].mean(axis=1)
    out = score.rename("score").reset_index()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].map(bt.normalize_symbol)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["date", "symbol", "score"])
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_execution(
    score: pd.DataFrame, raw_daily_cache_dir: Path, universe_file: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = bt.read_universe(
        universe_file if universe_file is not None and universe_file.exists() else None
    )
    execution, report = bt.build_execution_panel(
        sorted(score["symbol"].astype(str).unique()),
        raw_daily_cache_dir,
        universe,
        bt.load_st_symbols(None),
        st_status=bt.load_st_status(None),
        last5_panel=pd.DataFrame(),
        raw_5m_cache_dir=None,
    )
    if execution.empty:
        raise RuntimeError("execution panel is empty")
    execution = execution.copy()
    execution["date"] = pd.to_datetime(
        execution["date"], errors="coerce"
    ).dt.normalize()
    return execution, report


def effective_fold_offset(
    *,
    selection: Any,
    formal_config: dict[str, Any],
    full_fold_calendar: pd.DatetimeIndex,
    fold_calendar: pd.DatetimeIndex,
) -> int:
    every = int(selection.historical_rebalance_every)
    if every == 1:
        return 0
    original = int(selection.historical_rebalance_offset)
    reference = pd.Timestamp(selection.historical_date_min).normalize()
    ref_loc = full_fold_calendar.get_indexer([reference])
    fold_loc = full_fold_calendar.get_indexer([fold_calendar[0]])
    if int(ref_loc[0]) < 0:
        raise RuntimeError(
            f"formal history first date is absent from fold calendar: {reference:%Y-%m-%d}"
        )
    if int(fold_loc[0]) < 0:
        raise RuntimeError(
            f"fold first date is absent from fold calendar: {fold_calendar[0]:%Y-%m-%d}"
        )
    delta = int(fold_loc[0]) - int(ref_loc[0])
    effective = int((original - delta) % every)
    stored_every = int(formal_config.get("rebalance_every", every))
    if stored_every != every:
        raise RuntimeError(
            f"stored rebalance_every mismatch: {stored_every} != {every}"
        )
    return effective


def write_result(root: Path, result: dict[str, Any], cfg: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, key in (
        ("nav.csv", "nav"),
        ("orders.csv", "orders"),
        ("rejections.csv", "rejections"),
        ("positions.csv", "positions"),
    ):
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


def plot_curves(
    curves: dict[int, pd.DataFrame], *, target_col: str, signal_kind: str, out_file: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for fold in sorted(curves):
        curve = curves[fold]
        ax.plot(curve["trading_day"], curve["return_pct"], label=f"fold{fold}")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Trading day within fold")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        f"{target_col} fold0 {signal_kind}: independent cross-fold diagnostic"
    )
    ax.legend(ncol=4)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic only: hold one target's fold0 models fixed and evaluate "
            "best/first3/all5 on historical fold test windows."
        )
    )
    parser.add_argument(
        "--target",
        choices=list(TARGET_SHORT),
        required=True,
    )
    parser.add_argument(
        "--signals",
        default="best,first3,all5",
        help="Comma list: best,first3,all5. first5 is accepted as an alias of all5.",
    )
    parser.add_argument("--folds", default="0,1,2,3,4,5,6")
    parser.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    parser.add_argument("--fold0-dir", default=None)
    parser.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    parser.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY))
    parser.add_argument("--universe", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Resolve model/config inputs only; do not load TensorFlow or backtest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_col = args.target
    requested_signals = parse_signals(args.signals)
    requested_folds = parse_ints(args.folds)
    model_data = project_path(args.model_data)
    matrix_root = project_path(args.matrix_root)
    raw_daily_cache = project_path(args.raw_daily_cache_dir)
    fold0_dir = (
        project_path(args.fold0_dir)
        if args.fold0_dir
        else common.default_fold0_dir(FEATURE_PRESET, target_col).resolve()
    )
    universe_file = project_path(args.universe) if args.universe else None

    for label, path, kind in (
        ("model_data", model_data, "file"),
        ("fold0_dir", fold0_dir, "dir"),
        ("matrix_root", matrix_root, "dir"),
        ("raw_daily_cache", raw_daily_cache, "dir"),
    ):
        ok = path.is_file() if kind == "file" else path.is_dir()
        if not ok:
            raise FileNotFoundError(f"missing {label}: {path}")

    formal: dict[str, dict[str, Any]] = {}
    for signal_kind in requested_signals:
        experiment_root, history_root, selection, config, config_file = (
            resolve_formal_strategy(
                matrix_root=matrix_root,
                target_col=target_col,
                signal_kind=signal_kind,
            )
        )
        initial_cash = float(config.get("initial_cash", 200000.0))
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise RuntimeError(
                f"invalid initial_cash for {signal_kind}: {initial_cash}"
            )
        formal[signal_kind] = {
            "experiment_root": experiment_root,
            "history_root": history_root,
            "selection": selection,
            "config": config,
            "config_file": config_file,
            "initial_cash": initial_cash,
        }
        print(
            "[FORMAL] "
            f"target={target_col} signal={signal_kind} "
            f"max_positions={selection.historical_max_positions} "
            f"sell_rank={selection.historical_sell_rank} "
            f"rebalance_every={selection.historical_rebalance_every} "
            f"history_offset={selection.historical_rebalance_offset} "
            f"history={selection.historical_date_min}..{selection.historical_date_max}"
        )

    print("[DIAGNOSTIC] strict_oos=false; older folds overlap fold0 training data")
    print(f"[SOURCE MODEL] target={target_col} fold0_dir={fold0_dir}")
    print(f"[SIGNALS] {requested_signals}")
    print(f"[FOLDS] {requested_folds}")
    if args.check_only:
        print("[PASS] diagnostic inputs resolved; no prediction/backtest executed")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        project_path(args.out_root)
        if args.out_root
        else (
            DEFAULT_OUT_BASE
            / f"{target_col}_fold0_crossfold_{stamp}"
        ).resolve()
    )
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"output root already exists and is non-empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    features = common.build_target_features(
        model_data,
        args.train_end,
        "target_only",
        target_col,
        FEATURE_PRESET,
        "onehot",
    )
    fold_indices, fold_reports = all_fold_indices_and_reports(features, target_col)
    top_n = max(max(SIGNALS[kind]["cols"]) + 1 for kind in requested_signals)
    predictions, checkpoints, source_manifest = predict_fold0_topn(
        features=features,
        fold0_dir=fold0_dir,
        requested_folds=requested_folds,
        fold_indices=fold_indices,
        top_n=top_n,
        target_col=target_col,
    )
    predictions.to_hdf(
        out_root / "fold0_model_predictions.h5", key="predictions", mode="w"
    )
    checkpoints.to_csv(
        out_root / "selected_fold0_checkpoints.csv",
        index=False,
        encoding="utf-8-sig",
    )

    execution_seed = signal_score(predictions, requested_signals[-1])
    execution, execution_report = build_execution(
        execution_seed, raw_daily_cache, universe_file
    )
    execution_report.to_csv(
        out_root / "execution_data_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exec_dates = normalized_dates(execution["date"])

    # Reconstruct the complete fold0..fold6 execution calendar.  This lets r05
    # and r21 preserve the frozen production rebalance phase even when an
    # individual fold is restarted from empty positions for comparison.
    all_test_dates: list[pd.Timestamp] = []
    for fold in range(7):
        idx = fold_indices[fold]
        all_test_dates.extend(
            pd.DatetimeIndex(features.X.iloc[idx].index.get_level_values("date"))
            .normalize()
            .unique()
            .tolist()
        )
    full_fold_calendar = normalized_dates(all_test_dates).intersection(exec_dates)
    if len(full_fold_calendar) < 2:
        raise RuntimeError("full historical fold execution calendar is empty")

    summary_rows: list[dict[str, Any]] = []
    for signal_kind in requested_signals:
        score = signal_score(predictions, signal_kind)
        score.to_csv(
            out_root / f"fold0_{signal_kind}_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
        info = formal[signal_kind]
        selection = info["selection"]
        config = info["config"]
        initial_cash = float(info["initial_cash"])
        curves: dict[int, pd.DataFrame] = {}

        for fold in sorted(requested_folds):
            report = fold_reports[fold]
            start = pd.Timestamp(report["test_start"]).normalize()
            end = pd.Timestamp(report["test_end"]).normalize()
            pred = score.loc[
                (score["date"] >= start) & (score["date"] <= end)
            ].copy()
            pred_dates = normalized_dates(pred["date"])
            fold_calendar = pred_dates.intersection(exec_dates)
            if len(fold_calendar) < 2:
                raise RuntimeError(
                    f"target={target_col} signal={signal_kind} fold{fold} "
                    "has fewer than two prediction/execution dates"
                )
            effective_offset = effective_fold_offset(
                selection=selection,
                formal_config=config,
                full_fold_calendar=full_fold_calendar,
                fold_calendar=fold_calendar,
            )
            cfg = fold_helpers.config_to_trade_config(
                config,
                initial_cash=initial_cash,
                effective_offset=effective_offset,
            )
            exec_part = execution.loc[execution["date"].isin(fold_calendar)].copy()
            result = bt.backtest(
                pred[["date", "symbol", "score"]],
                exec_part,
                cfg,
                corporate_actions=None,
            )
            fold_root = out_root / "signals" / signal_kind / "independent_folds" / f"fold{fold}"
            write_result(fold_root, result, cfg)
            curve = normalized_curve(result, initial_cash)
            curve.to_csv(
                fold_root / "normalized_curve.csv",
                index=False,
                encoding="utf-8-sig",
            )
            curves[fold] = curve
            summary = dict(result["summary"])
            row = {
                "target_col": target_col,
                "signal_kind": signal_kind,
                "signal_spec": SIGNALS[signal_kind]["spec"],
                "fold": fold,
                "test_start": start.strftime("%Y-%m-%d"),
                "test_end": end.strftime("%Y-%m-%d"),
                "n_prediction_dates": int(len(pred_dates)),
                "n_execution_dates": int(len(fold_calendar)),
                "source_model_fold": 0,
                "max_positions": int(selection.historical_max_positions),
                "sell_rank": int(selection.historical_sell_rank),
                "rebalance_every": int(selection.historical_rebalance_every),
                "formal_history_offset": int(selection.historical_rebalance_offset),
                "effective_fold_offset": int(effective_offset),
                **summary,
            }
            summary_rows.append(row)
            print(
                "[FOLD] "
                f"target={target_col} signal={signal_kind} fold={fold} "
                f"dates={start:%Y-%m-%d}..{end:%Y-%m-%d} "
                f"return={float(summary.get('total_return', np.nan)):.6f} "
                f"sharpe={float(summary.get('sharpe', np.nan)):.6f}"
            )

        plot_curves(
            curves,
            target_col=target_col,
            signal_kind=signal_kind,
            out_file=out_root / f"crossfold_return_{signal_kind}.png",
        )

    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["signal_kind", "fold"]
    ).reset_index(drop=True)
    summary_table.to_csv(
        out_root / "crossfold_summary.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "status": "ok",
        "protocol": "diagnostic_fold0_fixed_models_crossfold_v1",
        "strict_oos": False,
        "reportable_as_oos": False,
        "warning": (
            "fold0 checkpoints are evaluated on older historical folds that overlap "
            "their training window; results are diagnostic/in-sample and must not be "
            "used as OOS performance evidence"
        ),
        "feature_preset": FEATURE_PRESET,
        "target_col": target_col,
        "signals": requested_signals,
        "signal_specs": {kind: SIGNALS[kind]["spec"] for kind in requested_signals},
        "requested_folds": sorted(requested_folds),
        "source_model_fold": 0,
        "fold0_dir": str(fold0_dir),
        "model_data": str(model_data),
        "raw_daily_cache_dir": str(raw_daily_cache),
        "source_model_manifest": source_manifest,
        "formal_strategies": {
            kind: {
                "experiment_root": str(formal[kind]["experiment_root"]),
                "history_root": str(formal[kind]["history_root"]),
                "config_file": str(formal[kind]["config_file"]),
                "selection": formal[kind]["selection"].to_dict(),
            }
            for kind in requested_signals
        },
        "summary_file": str(out_root / "crossfold_summary.csv"),
    }
    write_json(out_root / "manifest.json", manifest)

    show_cols = [
        column
        for column in (
            "signal_kind",
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
    print("[PASS] generic fold0 cross-fold diagnostic finished")
    print(f"[PASS] output={out_root}")


if __name__ == "__main__":
    main()
