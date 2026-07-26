#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nested walk-forward selection for AS1455 Chapter-17 strategies.

Protocol for each source fold k:

1. Use the already-trained fold-k checkpoint candidates.
2. Predict fold k's own held-out validation window.
3. Run the full signal/trading grid only on that validation window.
4. Freeze the complete winning configuration C_k.
5. Apply C_k exactly once to target fold k-1 (or fold0-forward for k=0).

No target-fold or forward result participates in selection.  This script is the
correct replacement for the former global grid over concatenated target folds.
It reuses existing search checkpoints and model_data; it does not retrain models
or refresh data.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_forward_features import build_inference_features  # noqa: E402
from utils.as1455_model_selection import (  # noqa: E402
    HistoricalSignalSelection,
    select_historical_signal,
)
from utils.as1455_signal_specs import append_signal_specs  # noqa: E402
from utils.as1455_strict_oos import finalize_strict_oos_grid  # noqa: E402
from utils.as1455_backtest_io import output_frames  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fold_helpers = load_module(
    "as1455_nested_fold_helpers",
    PROJECT_DIR / "scripts" / "run_as1455_independent_fold_backtests.py",
)
bt = fold_helpers.bt
comparison_v2 = load_module(
    "as1455_nested_bridge_helpers",
    PROJECT_DIR / "scripts" / "run_as1455_r05_addon_fold_comparison_v2.py",
)


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


def normalized_dates(values: Iterable[Any]) -> pd.DatetimeIndex:
    return (
        pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )


def source_to_target(source_fold: int) -> int | None:
    if source_fold < 0 or source_fold > 6:
        raise ValueError(f"source_fold must be in 0..6, got {source_fold}")
    return source_fold - 1 if source_fold > 0 else None


def validation_window(
    features: common.FeatureBuildResult,
    source_fold: int,
    target_col: str,
) -> tuple[np.ndarray, dict[str, Any], pd.DatetimeIndex]:
    _train_idx, validation_idx, report = common.get_fold_target(
        features.X, source_fold, target_col
    )
    dates = normalized_dates(
        features.X.iloc[validation_idx].index.get_level_values("date")
    )
    if not dates.size:
        raise RuntimeError(f"source fold{source_fold} validation window is empty")
    return validation_idx, report, dates


def with_validation_phase(
    selection: HistoricalSignalSelection,
    report: dict[str, Any],
    dates: pd.DatetimeIndex,
) -> HistoricalSignalSelection:
    del report
    return dataclasses.replace(
        selection,
        historical_date_min=pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
        historical_date_max=pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        historical_n_days=int(len(dates)),
    )


def prediction_artifact(
    *,
    out_root: Path,
    features: common.FeatureBuildResult,
    row_idx: np.ndarray,
    source_dir: Path,
    source_fold: int,
    target_col: str,
    top_n: int,
    protocol: str,
    target_fold: int | None,
    filename: str,
    force: bool,
) -> Path:
    path = out_root / "00_predictions" / filename
    manifest_path = out_root / "00_predictions" / f"{Path(filename).stem}_manifest.json"
    if path.exists() and manifest_path.exists() and not force:
        print(f"[RESUME] predictions={path}")
        return path

    predictions, checkpoint_rows, source_manifest = common.predict_checkpoint_set(
        features.X,
        row_idx,
        source_dir,
        top_n,
        metadata={
            "source_fold": int(source_fold),
            "target_fold": target_fold,
            "selection_role": protocol,
        },
    )
    return common.write_prediction_artifacts(
        out_root=out_root,
        predictions=predictions,
        y=features.y,
        target_col=target_col,
        prediction_filename=filename,
        manifest_filename=manifest_path.name,
        checkpoint_filename=f"{Path(filename).stem}_checkpoints.csv",
        manifest={
            "protocol": protocol,
            "source_fold": int(source_fold),
            "target_fold": target_fold,
            "source_dir": str(source_dir),
            "source_manifest": source_manifest,
            "feature_meta": features.report,
            "top_n": int(top_n),
            "selection_data_only": protocol == "source_fold_validation_grid",
        },
        checkpoint_rows=checkpoint_rows,
    )


def append_optional_path(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, str(Path(value).expanduser().resolve())])


def base_grid_command(
    args: argparse.Namespace,
    *,
    grid_out: Path,
    prediction_file: Path,
    output_mode: str,
    max_positions_list: str,
    sell_rank_list: str,
    offset_mode: str,
    force: bool,
) -> list[str]:
    command = common.build_grid_command(
        python_bin=args.python_bin,
        grid_script=Path(args.grid_script).expanduser().resolve(),
        grid_out=grid_out,
        prediction_file=prediction_file,
        raw_daily_cache_dir=Path(args.raw_daily_cache_dir).expanduser().resolve(),
        profile=args.profile,
        capacity_mode=args.capacity_mode,
        output_mode=output_mode,
        offset_mode=offset_mode,
        rebalance_every=args.rebalance_every,
        max_positions_list=max_positions_list,
        sell_rank_list=sell_rank_list,
        model_family=f"AS1455 nested {args.feature_preset} {args.target_col}",
        model_run="per-source-fold validation selection",
        force_grid=force,
    )
    command.extend(["--initial-cash", str(args.initial_cash)])
    if args.skip_parity_check:
        command.append("--skip-parity-check")
    append_optional_path(command, "--raw-5m-cache-dir", args.raw_5m_cache_dir)
    append_optional_path(command, "--last5-panel", args.last5_panel)
    append_optional_path(command, "--universe", args.universe)
    append_optional_path(command, "--st-symbols", args.st_symbols)
    append_optional_path(command, "--st-status", args.st_status)
    append_optional_path(command, "--corporate-actions", args.corporate_actions)
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    print("[CMD] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def run_validation_grid(
    args: argparse.Namespace,
    *,
    root: Path,
    prediction_file: Path,
) -> HistoricalSignalSelection:
    grid_root = root / "01_close_auction_grid"
    command = base_grid_command(
        args,
        grid_out=grid_root,
        prediction_file=prediction_file,
        output_mode=args.validation_output_mode,
        max_positions_list=args.max_positions_list,
        sell_rank_list=args.sell_rank_list,
        offset_mode="full",
        force=args.force,
    )
    command = append_signal_specs(command, args.top_n)
    run_command(command, args.dry_run)
    if args.dry_run:
        raise RuntimeError("dry-run stops before validation selection")
    return select_historical_signal(
        backtest_root=root,
        rank_metric=args.selection_rank_metric,
    )


def run_frozen_target(
    args: argparse.Namespace,
    *,
    root: Path,
    prediction_file: Path,
    selection: HistoricalSignalSelection,
) -> dict[str, Any]:
    if selection.historical_rebalance_every != args.rebalance_every:
        raise RuntimeError(
            "selected validation rebalance period differs from target protocol: "
            f"selection={selection.historical_rebalance_every} "
            f"target={args.rebalance_every}"
        )
    required = {
        "max_positions": selection.historical_max_positions,
        "sell_rank": selection.historical_sell_rank,
        "rebalance_offset": selection.historical_rebalance_offset,
        "date_min": selection.historical_date_min,
        "date_max": selection.historical_date_max,
        "n_days": selection.historical_n_days,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"frozen target selection is incomplete: {missing}")

    command = base_grid_command(
        args,
        grid_out=root / "01_close_auction_grid",
        prediction_file=prediction_file,
        output_mode=args.target_output_mode,
        max_positions_list=str(selection.historical_max_positions),
        sell_rank_list=str(selection.historical_sell_rank),
        offset_mode="full",
        force=args.force,
    )
    command.extend(["--signal-spec", selection.signal_spec])
    command.extend(
        [
            "--rebalance-phase-history-offset",
            str(selection.historical_rebalance_offset),
            "--rebalance-phase-history-first-date",
            str(selection.historical_date_min),
            "--rebalance-phase-history-last-date",
            str(selection.historical_date_max),
            "--rebalance-phase-history-n-days",
            str(selection.historical_n_days),
        ]
    )
    run_command(command, args.dry_run)
    if args.dry_run:
        raise RuntimeError("dry-run stops before strict target finalization")
    return finalize_strict_oos_grid(root, selection)


def target_prediction_indices(
    features: common.FeatureBuildResult,
    target_fold: int,
    target_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    _train_idx, target_idx, report = common.get_fold_target(
        features.X, target_fold, target_col
    )
    return target_idx, report


def forward_prediction_indices(
    features: common.FeatureBuildResult,
    fold0_test_end: pd.Timestamp,
    start_date: str | None,
    end_date: str | None,
) -> np.ndarray:
    dates = pd.DatetimeIndex(features.X.index.get_level_values("date"))
    mask = dates > fold0_test_end
    if start_date:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date:
        mask &= dates <= pd.Timestamp(end_date)
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise RuntimeError(
            "no forward feature rows after fold0 validation; "
            f"fold0_test_end={fold0_test_end:%Y-%m-%d} "
            f"available_max={pd.Timestamp(dates.max()):%Y-%m-%d}"
        )
    return idx


def retained_run(root: Path, strict_manifest: dict[str, Any]) -> tuple[Path, Path]:
    run_name = str(strict_manifest["retained_run_name"])
    run_dir = root / "01_close_auction_grid" / "01_runs" / run_name
    config_file = run_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(config_file)
    return run_dir, config_file


def run_continuous_account(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    out_root: Path,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: pd.Timestamp(row["target_start"]))
    loaded: list[dict[str, Any]] = []
    symbols: set[str] = set()
    capacity_modes: set[str] = set()
    for record in ordered:
        selection = record["selection"]
        pred = fold_helpers.load_selected_predictions(
            Path(record["prediction_file"]), selection
        )
        config = read_json(Path(record["config_file"]))
        symbols.update(pred["symbol"].astype(str).unique())
        capacity_modes.add(str(config["capacity_mode"]))
        loaded.append({**record, "pred": pred, "config": config})

    universe = bt.read_universe(
        Path(args.universe).expanduser().resolve() if args.universe else None
    )
    st_symbols = bt.load_st_symbols(
        Path(args.st_symbols).expanduser().resolve() if args.st_symbols else None
    )
    st_status = bt.load_st_status(
        Path(args.st_status).expanduser().resolve() if args.st_status else None
    )
    last5 = bt.load_last5_panel(
        Path(args.last5_panel).expanduser().resolve() if args.last5_panel else None
    )
    actions = bt.load_corporate_actions(
        Path(args.corporate_actions).expanduser().resolve()
        if args.corporate_actions
        else None
    )
    raw_5m = (
        Path(args.raw_5m_cache_dir).expanduser().resolve()
        if args.raw_5m_cache_dir
        else None
    )
    if all(mode == "none" for mode in capacity_modes):
        last5 = pd.DataFrame()
        raw_5m = None
    execution, execution_report = bt.build_execution_panel(
        sorted(symbols),
        Path(args.raw_daily_cache_dir).expanduser().resolve(),
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=last5,
        raw_5m_cache_dir=raw_5m,
    )
    continuous_root = out_root / "continuous_target_folds_plus_forward"
    continuous_root.mkdir(parents=True, exist_ok=True)
    execution_report.to_csv(
        continuous_root / "execution_data_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    execution_dates = normalized_dates(execution["date"])

    state: dict[str, Any] = {"cash": float(args.initial_cash), "positions": []}
    previous_nav = float(args.initial_cash)
    previous_end: pd.Timestamp | None = None
    global_index = 0
    nav_parts: list[pd.DataFrame] = []
    order_parts: list[pd.DataFrame] = []
    reject_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    trip_parts: list[pd.DataFrame] = []
    segment_rows: list[dict[str, Any]] = []
    trip_offset = 0
    first_cfg: Any | None = None

    for record in loaded:
        pred = record["pred"]
        pred_dates = normalized_dates(pred["date"])
        if previous_end is not None:
            bridge_dates = execution_dates[
                (execution_dates > previous_end) & (execution_dates < pred_dates[0])
            ]
        else:
            bridge_dates = pd.DatetimeIndex([])

        config = record["config"]
        cfg = fold_helpers.config_to_trade_config(
            config,
            initial_cash=float(state["cash"]),
            effective_offset=int(config["rebalance_offset"]),
        )
        if first_cfg is None:
            first_cfg = dataclasses.replace(cfg, initial_cash=float(args.initial_cash))

        if len(bridge_dates):
            state, bnav, bpos, bactions = comparison_v2.bridge_state(
                state,
                bridge_dates,
                execution,
                actions,
                cfg,
                previous_nav,
                global_index,
            )
            nav_parts.append(bnav)
            position_parts.append(bpos)
            action_parts.append(bactions)
            global_index += len(bnav)
            previous_nav = float(bnav.iloc[-1]["nav"])

        exec_part = execution[execution["date"].isin(pred_dates)]
        result = bt.backtest(
            pred,
            exec_part,
            cfg,
            corporate_actions=actions,
            initial_positions=state["positions"],
            day_index_start=0,
        )
        label = str(record["segment"])
        nav = result["nav"].copy()
        nav["protocol_segment"] = label
        nav["segment_day_index"] = pd.to_numeric(
            nav["day_index"], errors="coerce"
        )
        nav["global_day_index"] = nav["segment_day_index"] + global_index
        nav_parts.append(nav)

        for key, parts in (
            ("orders", order_parts),
            ("rejections", reject_parts),
            ("positions", position_parts),
            ("corporate_actions", action_parts),
        ):
            frame = result[key].copy()
            if not frame.empty:
                frame["protocol_segment"] = label
                if "day_index" in frame.columns:
                    frame["global_day_index"] = (
                        pd.to_numeric(frame["day_index"], errors="coerce")
                        + global_index
                    )
                parts.append(frame)

        trips = result["round_trips"].copy()
        if not trips.empty:
            trips["round_trip_id"] = (
                pd.to_numeric(trips["round_trip_id"], errors="coerce")
                + trip_offset
            )
            trips["protocol_segment"] = label
            trip_offset += len(trips)
            trip_parts.append(trips)

        segment_rows.append(
            {
                "segment": label,
                "source_fold": record["source_fold"],
                "target_fold": record["target_fold"],
                "start": pred_dates[0],
                "end": pred_dates[-1],
                "n_days": len(pred_dates),
                "validation_selected_signal": record["selection"]["signal_spec"],
                "validation_selected_max_positions": record["selection"][
                    "historical_max_positions"
                ],
                "validation_selected_sell_rank": record["selection"][
                    "historical_sell_rank"
                ],
                "validation_selected_offset": record["selection"][
                    "historical_rebalance_offset"
                ],
                "effective_target_offset": config["rebalance_offset"],
                "start_nav": float(result["nav"].iloc[0]["nav"]),
                "end_nav": float(result["nav"].iloc[-1]["nav"]),
            }
        )
        state = result["final_state"]
        previous_nav = float(result["nav"].iloc[-1]["nav"])
        previous_end = pd.Timestamp(pred_dates[-1])
        global_index += len(result["nav"])

    nav = (
        pd.concat(nav_parts, ignore_index=True, sort=False)
        .sort_values("date")
        .reset_index(drop=True)
    )
    nav["daily_return"] = pd.to_numeric(nav["nav"], errors="coerce").pct_change()
    nav.loc[0, "daily_return"] = (
        float(nav.loc[0, "nav"]) / float(args.initial_cash) - 1.0
    )
    orders = (
        pd.concat(order_parts, ignore_index=True, sort=False)
        if order_parts
        else pd.DataFrame()
    )
    rejections = (
        pd.concat(reject_parts, ignore_index=True, sort=False)
        if reject_parts
        else pd.DataFrame()
    )
    positions = (
        pd.concat(position_parts, ignore_index=True, sort=False)
        if position_parts
        else pd.DataFrame()
    )
    corporate_actions = (
        pd.concat(action_parts, ignore_index=True, sort=False)
        if action_parts
        else pd.DataFrame()
    )
    round_trips = (
        pd.concat(trip_parts, ignore_index=True, sort=False)
        if trip_parts
        else pd.DataFrame()
    )
    assert first_cfg is not None
    drawdown = bt.build_daily_drawdown(nav)
    result = {
        "nav": nav,
        "orders": orders,
        "trades": orders.copy(),
        "rejections": rejections,
        "positions": positions,
        "corporate_actions": corporate_actions,
        "round_trips": round_trips,
        "daily_drawdown": drawdown,
        "monthly_summary": bt.build_period_summary(nav, "M"),
        "yearly_summary": bt.build_period_summary(nav, "Y"),
        "fee_summary": bt.build_fee_summary(orders, first_cfg),
        "turnover_summary": bt.build_turnover_summary(nav, orders),
    }
    result["summary"] = bt.summarize_nav(
        nav,
        orders,
        rejections,
        first_cfg,
        corporate_actions,
        round_trips,
        drawdown,
    )
    selected, _full = output_frames(result, args.target_output_mode)
    for filename, frame in selected.items():
        frame.to_csv(continuous_root / filename, index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(
        continuous_root / "protocol_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(continuous_root / "summary.json", result["summary"])
    write_json(
        continuous_root / "config.json",
        {
            "protocol": "nested_per_source_fold_grid_then_next_window",
            "initial_cash": args.initial_cash,
            "account_state_continuous": True,
            "segments": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"pred", "config"}
                }
                for record in loaded
            ],
        },
    )
    write_json(
        continuous_root / "close_auction_summary.json",
        {
            "protocol": "nested_per_source_fold_grid_then_next_window",
            "selection_data_never_includes_target": True,
            "target_data_never_used_for_grid": True,
            "final_state": state,
            "n_segments": len(loaded),
        },
    )
    return {
        "root": str(continuous_root),
        "summary": result["summary"],
        "final_state": state,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-source-fold nested model/trading selection followed by one frozen "
            "next-window evaluation"
        )
    )
    parser.add_argument("--model-data", default=str(as1455_paths.DEFAULT_MODEL_DATA))
    parser.add_argument(
        "--feature-preset",
        choices=list(common.FEATURE_PRESETS),
        default="rotation_addon_onehot",
    )
    parser.add_argument(
        "--target-col", choices=list(common.TARGET_SPECS), default="r05_fwd"
    )
    parser.add_argument("--fold-dir-template", default=None)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--raw-daily-cache-dir",
        default=str(as1455_paths.DEFAULT_RAW_DAILY_CACHE_DIR),
    )
    parser.add_argument("--raw-5m-cache-dir", default=None)
    parser.add_argument("--last5-panel", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--st-symbols", default=None)
    parser.add_argument("--st-status", default=None)
    parser.add_argument("--corporate-actions", default=None)
    parser.add_argument("--grid-script", default=str(as1455_paths.DEFAULT_GRID_SCRIPT))
    parser.add_argument("--python-bin", default=sys.executable or "python3")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--selection-rank-metric", default="sharpe")
    parser.add_argument("--max-positions-list", default="5,10,15,20,25")
    parser.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    parser.add_argument("--rebalance-every", type=int, default=None)
    parser.add_argument("--profile", default="close_auction_skip_limit")
    parser.add_argument("--capacity-mode", default="none")
    parser.add_argument("--initial-cash", type=float, default=200000.0)
    parser.add_argument(
        "--validation-output-mode",
        choices=["summary", "compact", "full"],
        default="summary",
    )
    parser.add_argument(
        "--target-output-mode", choices=["compact", "full"], default="compact"
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-parity-check", action="store_true")
    parser.add_argument("--skip-continuous", action="store_true")
    args = parser.parse_args()
    spec = common.target_spec(args.target_col)
    if args.rebalance_every is None:
        args.rebalance_every = spec.rebalance_every
    if args.fold_dir_template is None:
        args.fold_dir_template = common.default_fold_dir_template(
            args.feature_preset, args.target_col
        )
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    if args.initial_cash <= 0:
        raise SystemExit("--initial-cash must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model_data = Path(args.model_data).expanduser().resolve()
    if not model_data.exists():
        raise FileNotFoundError(model_data)
    if not Path(args.raw_daily_cache_dir).expanduser().resolve().is_dir():
        raise FileNotFoundError(args.raw_daily_cache_dir)

    historical_features = common.build_target_features(
        model_data,
        None,
        "target_only",
        args.target_col,
        args.feature_preset,
        "onehot",
    )
    forward_features: common.FeatureBuildResult | None = None
    records: list[dict[str, Any]] = []

    for source_fold in range(6, -1, -1):
        target_fold = source_to_target(source_fold)
        source_dir = common.fold_dir_from_template(
            args.fold_dir_template, source_fold
        )
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        source_root = out_root / f"source_fold{source_fold}"
        validation_root = source_root / "validation_selection"
        validation_idx, source_report, validation_dates = validation_window(
            historical_features, source_fold, args.target_col
        )
        validation_pred = prediction_artifact(
            out_root=validation_root,
            features=historical_features,
            row_idx=validation_idx,
            source_dir=source_dir,
            source_fold=source_fold,
            target_col=args.target_col,
            top_n=args.top_n,
            protocol="source_fold_validation_grid",
            target_fold=source_fold,
            filename="validation_preds.h5",
            force=args.force,
        )
        selection = run_validation_grid(
            args, root=validation_root, prediction_file=validation_pred
        )
        selection = with_validation_phase(
            selection, source_report, validation_dates
        )
        selection_payload = selection.to_dict()
        write_json(
            source_root / "selected_for_next_window.json",
            {
                "protocol": "source_validation_select_then_freeze",
                "source_fold": source_fold,
                "target_fold": target_fold,
                "validation_fold_report": source_report,
                "validation_dates": {
                    "start": validation_dates[0],
                    "end": validation_dates[-1],
                    "n_days": len(validation_dates),
                },
                "selection": selection_payload,
                "target_results_used_for_selection": False,
            },
        )

        if target_fold is not None:
            target_root = source_root / f"target_fold{target_fold}"
            target_idx, target_report = target_prediction_indices(
                historical_features, target_fold, args.target_col
            )
            target_pred = prediction_artifact(
                out_root=target_root,
                features=historical_features,
                row_idx=target_idx,
                source_dir=source_dir,
                source_fold=source_fold,
                target_col=args.target_col,
                top_n=args.top_n,
                protocol="frozen_source_fold_to_next_target_fold",
                target_fold=target_fold,
                filename="target_preds.h5",
                force=args.force,
            )
            segment = f"target_fold{target_fold}"
        else:
            target_root = source_root / "forward"
            if forward_features is None:
                forward_features = build_inference_features(
                    model_data,
                    None,
                    args.target_col,
                    args.feature_preset,
                    "onehot",
                )
            fold0_end = common.resolve_fold_test_end(source_dir)
            target_idx = forward_prediction_indices(
                forward_features,
                fold0_end,
                args.start_date,
                args.end_date,
            )
            target_report = {
                "target_fold": None,
                "test_start": pd.Timestamp(
                    forward_features.X.iloc[target_idx]
                    .index.get_level_values("date")
                    .min()
                ).strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(
                    forward_features.X.iloc[target_idx]
                    .index.get_level_values("date")
                    .max()
                ).strftime("%Y-%m-%d"),
            }
            target_pred = prediction_artifact(
                out_root=target_root,
                features=forward_features,
                row_idx=target_idx,
                source_dir=source_dir,
                source_fold=source_fold,
                target_col=args.target_col,
                top_n=args.top_n,
                protocol="frozen_fold0_to_strict_forward",
                target_fold=None,
                filename="forward_preds.h5",
                force=args.force,
            )
            segment = "strict_oos_forward"

        strict_manifest = run_frozen_target(
            args,
            root=target_root,
            prediction_file=target_pred,
            selection=selection,
        )
        run_dir, config_file = retained_run(target_root, strict_manifest)
        summary = read_json(run_dir / "summary.json")
        target_dates = normalized_dates(
            pd.read_hdf(target_pred, "predictions").index.get_level_values("date")
        )
        record = {
            "segment": segment,
            "source_fold": source_fold,
            "target_fold": target_fold,
            "source_dir": str(source_dir),
            "validation_root": str(validation_root),
            "selection": selection_payload,
            "prediction_file": str(target_pred),
            "target_root": str(target_root),
            "run_dir": str(run_dir),
            "config_file": str(config_file),
            "summary_file": str(run_dir / "summary.json"),
            "strict_manifest": str(
                target_root
                / "01_close_auction_grid"
                / "strict_oos_manifest.json"
            ),
            "target_start": target_dates[0],
            "target_end": target_dates[-1],
            "target_n_days": len(target_dates),
            "target_fold_report": target_report,
            "summary": summary,
        }
        records.append(record)
        print(
            f"[OK] source_fold{source_fold} selected on "
            f"{validation_dates[0]:%Y-%m-%d}..{validation_dates[-1]:%Y-%m-%d} "
            f"and frozen for {segment}"
        )

    rows = []
    for record in records:
        summary = record["summary"]
        rows.append(
            {
                "segment": record["segment"],
                "source_fold": record["source_fold"],
                "target_fold": record["target_fold"],
                "selection_signal": record["selection"]["signal_spec"],
                "selection_max_positions": record["selection"][
                    "historical_max_positions"
                ],
                "selection_sell_rank": record["selection"][
                    "historical_sell_rank"
                ],
                "selection_offset": record["selection"][
                    "historical_rebalance_offset"
                ],
                "target_start": record["target_start"],
                "target_end": record["target_end"],
                "total_return": summary.get("total_return"),
                "annual_return": summary.get("annual_return"),
                "sharpe": summary.get("sharpe"),
                "max_drawdown": summary.get("max_drawdown"),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_root / "nested_fold_target_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    continuous = None
    if not args.skip_continuous:
        continuous = run_continuous_account(args, records, out_root)

    manifest = {
        "protocol": "nested_per_source_fold_validation_grid",
        "feature_preset": args.feature_preset,
        "target_col": args.target_col,
        "model_training": False,
        "data_refresh": False,
        "model_data_rebuild": False,
        "validation_grid_count": 7,
        "target_grid_count": 0,
        "target_fixed_backtest_count": 7,
        "global_concatenated_target_grid": False,
        "target_results_used_for_selection": False,
        "forward_results_used_for_selection": False,
        "records": records,
        "continuous": continuous,
        "duration_seconds": int(round(time.time() - started)),
    }
    write_json(out_root / "nested_fold_protocol_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "out_root": str(out_root),
                "records": len(records),
                "duration_seconds": manifest["duration_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
