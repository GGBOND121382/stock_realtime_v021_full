#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from functools import reduce
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_backtest_io import output_frames
from utils.as1455_plotting import plot_frequency


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bt = load_module(
    "as1455_independent_fold_bt_v7",
    PROJECT_DIR / "code" / "backtest" / "run_as1455_close_auction_backtest_v7_maxpos_grid.py",
)

RULES = {"daily": None, "weekly": "W-FRI", "monthly": "M"}
TRADE_FIELDS = tuple(field.name for field in dataclasses.fields(bt.TradeConfig))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def normalize_dates(values: Any) -> pd.DatetimeIndex:
    return (
        pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )


def sample_curve(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "daily":
        return frame.copy()
    rule = RULES[frequency]
    return (
        frame.set_index("date")[["nav", "return_pct"]]
        .resample(rule)
        .last()
        .dropna()
        .reset_index()
    )


def config_to_trade_config(
    config: dict[str, Any],
    *,
    initial_cash: float,
    effective_offset: int,
) -> Any:
    missing = [name for name in TRADE_FIELDS if name not in config]
    if missing:
        raise RuntimeError(f"stored config is missing TradeConfig fields: {missing}")
    values = {name: config[name] for name in TRADE_FIELDS}
    values["initial_cash"] = float(initial_cash)
    values["rebalance_offset"] = int(effective_offset)
    return bt.TradeConfig(**values)


def load_selected_predictions(
    path: Path,
    selection: dict[str, Any],
) -> pd.DataFrame:
    signal_cols = [
        token.strip()
        for token in str(selection["signal_cols"]).split(",")
        if token.strip()
    ]
    predictions, _metadata = bt.load_predictions(
        path,
        "predictions",
        None,
        signal_cols=signal_cols,
        signal_mode=str(selection["signal_mode"]),
        signal_name=str(selection["signal_name"]),
        prediction_file_sha256="existing-artifact-not-rehashed",
        model_params_file=None,
    )
    return predictions


def mapping_by_fold(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        fold = int(row["source_fold"])
        start = pd.Timestamp(row["start"]).normalize()
        end = pd.Timestamp(row["end"]).normalize()
        if start > end:
            raise RuntimeError(f"fold{fold} mapping reversed: {start} > {end}")
        result[fold] = {**row, "start_ts": start, "end_ts": end}
    return result


def available_dates(
    prediction_dates: pd.DatetimeIndex,
    execution_dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    dates = prediction_dates.intersection(execution_dates).sort_values()
    if len(dates) < 2:
        raise RuntimeError(
            f"not enough prediction/execution overlap: prediction={len(prediction_dates)} "
            f"execution={len(execution_dates)} overlap={len(dates)}"
        )
    return dates


def common_fold_calendar(
    candidates: list[tuple[str, pd.DatetimeIndex]],
    fold: int,
) -> pd.DatetimeIndex:
    if not candidates:
        raise RuntimeError(f"fold{fold} has no available strategies")
    for label, dates in candidates:
        if len(dates) < 2:
            raise RuntimeError(f"fold{fold} has fewer than two dates for {label}")
    common = reduce(
        lambda left, right: left.intersection(right),
        (dates for _, dates in candidates),
    ).sort_values()
    if len(common) < 2:
        raise RuntimeError(f"fold{fold} has fewer than two common trading dates")

    first = common[0]
    last = common[-1]
    for label, dates in candidates:
        local = dates[(dates >= first) & (dates <= last)]
        if not local.equals(common):
            missing = common.difference(local)
            extra = local.difference(common)
            raise RuntimeError(
                f"fold{fold} calendar mismatch for {label}; "
                f"missing={list(map(str, missing[:10]))} "
                f"extra={list(map(str, extra[:10]))}"
            )
    return common


def effective_offset_for_crop(
    *,
    full_dates: pd.DatetimeIndex,
    crop_dates: pd.DatetimeIndex,
    original_offset: int,
    rebalance_every: int,
) -> tuple[int, int]:
    first = crop_dates[0]
    locations = full_dates.get_indexer([first])
    if int(locations[0]) < 0:
        raise RuntimeError(f"crop start is not in full backtest calendar: {first}")
    skipped = int(locations[0])
    every = int(rebalance_every)
    if every <= 0:
        raise RuntimeError(f"invalid rebalance_every={every}")
    original = int(original_offset)
    if not 0 <= original < every:
        raise RuntimeError(
            f"invalid original rebalance_offset={original} for rebalance_every={every}"
        )
    return int((original - skipped) % every), skipped


def capacity_precheck(
    execution_panel: pd.DataFrame,
    predictions: pd.DataFrame,
    capacity_mode: str,
) -> dict[str, Any]:
    report = bt.build_capacity_precheck(execution_panel, predictions, capacity_mode)
    if capacity_mode != "none" and not bool(report.get("passed")):
        raise RuntimeError(f"capacity precheck failed: {report}")
    return report


def write_independent_run(
    *,
    run_dir: Path,
    result: dict[str, Any],
    cfg: Any,
    output_mode: str,
    manifest: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    selected, _full = output_frames(result, output_mode)
    for filename, frame in selected.items():
        frame.to_csv(run_dir / filename, index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(
        json.dumps(
            result["summary"],
            ensure_ascii=False,
            indent=2,
            default=bt.json_default,
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            dataclasses.asdict(cfg),
            ensure_ascii=False,
            indent=2,
            default=bt.json_default,
        ),
        encoding="utf-8",
    )
    (run_dir / "independent_fold_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )


def curve_from_result(result: dict[str, Any], initial_cash: float) -> pd.DataFrame:
    nav = result["nav"].copy()
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["date", "nav"]).sort_values("date")
    nav["return_pct"] = (nav["nav"] / float(initial_cash) - 1.0) * 100.0
    return nav[["date", "nav", "return_pct"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run each AS1455 fold independently from empty positions using one "
            "previously frozen signal/trading configuration per strategy."
        )
    )
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--raw-daily-cache-dir", required=True)
    parser.add_argument("--raw-5m-cache-dir", default=None)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--initial-cash", type=float, default=200000.0)
    parser.add_argument("--output-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--frequencies", default="daily,weekly,monthly")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.initial_cash <= 0:
        raise SystemExit("--initial-cash must be positive")

    pair_path = Path(args.pair_manifest).expanduser().resolve()
    pair_payload = read_json(pair_path)
    pairs = pair_payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 6:
        raise RuntimeError(f"expected six paired strategies in {pair_path}")

    out_root = Path(args.out_root).expanduser().resolve()
    runs_root = out_root / "runs"
    plots_root = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)

    states: list[dict[str, Any]] = []
    all_symbols: set[str] = set()
    capacity_modes: set[str] = set()

    for pair in pairs:
        label = str(pair["label"])
        historical = pair["historical"]
        forward = pair["forward"]
        selection = historical["selection"]

        historical_predictions = load_selected_predictions(
            Path(historical["prediction_file"]), selection
        )
        forward_predictions = load_selected_predictions(
            Path(forward["prediction_file"]), selection
        )
        all_symbols.update(historical_predictions["symbol"].astype(str).unique())
        all_symbols.update(forward_predictions["symbol"].astype(str).unique())

        historical_config = historical["config"]
        forward_config = forward["config"]
        capacity_modes.add(str(historical_config["capacity_mode"]))
        capacity_modes.add(str(forward_config["capacity_mode"]))

        states.append(
            {
                "label": label,
                "target_col": str(pair["target_col"]),
                "feature_preset": str(pair["feature_preset"]),
                "historical": historical,
                "forward": forward,
                "selection": selection,
                "historical_predictions": historical_predictions,
                "forward_predictions": forward_predictions,
                "historical_config": historical_config,
                "forward_config": forward_config,
                "mapping": mapping_by_fold(historical["fold_mapping"]),
            }
        )

    raw_daily_cache = Path(args.raw_daily_cache_dir).expanduser().resolve()
    if not raw_daily_cache.is_dir():
        raise FileNotFoundError(raw_daily_cache)
    raw_5m_cache = (
        Path(args.raw_5m_cache_dir).expanduser().resolve()
        if args.raw_5m_cache_dir
        else None
    )
    if any(mode != "none" for mode in capacity_modes):
        if raw_5m_cache is None or not raw_5m_cache.is_dir():
            raise RuntimeError(
                "at least one frozen config uses capacity data, but --raw-5m-cache-dir "
                "is missing or invalid"
            )
    else:
        raw_5m_cache = None

    empty_universe = pd.DataFrame(columns=["symbol", "board", "is_mainboard"])
    execution_panel, execution_report = bt.build_execution_panel(
        all_symbols,
        raw_daily_cache,
        empty_universe,
        set(),
        st_status=pd.DataFrame(),
        last5_panel=pd.DataFrame(),
        raw_5m_cache_dir=raw_5m_cache,
    )
    if execution_panel.empty:
        raise RuntimeError("execution panel is empty")
    execution_report.to_csv(
        out_root / "execution_data_report.csv", index=False, encoding="utf-8-sig"
    )
    execution_dates = normalize_dates(execution_panel["date"])

    for state in states:
        state["historical_dates"] = available_dates(
            normalize_dates(state["historical_predictions"]["date"]),
            execution_dates,
        )
        state["forward_dates"] = available_dates(
            normalize_dates(state["forward_predictions"]["date"]),
            execution_dates,
        )

    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    invalid_frequencies = sorted(set(frequencies) - set(RULES))
    if invalid_frequencies:
        raise RuntimeError(f"unsupported frequencies: {invalid_frequencies}")

    global_manifest: dict[str, Any] = {
        "mode": "independent_fold_frozen_config",
        "initial_state": "empty_positions_and_initial_cash",
        "initial_cash": float(args.initial_cash),
        "prediction_generation": False,
        "parameter_grid": False,
        "training": False,
        "data_refresh": False,
        "pair_manifest": str(pair_path),
        "execution_symbols": int(execution_panel["symbol"].nunique()),
        "execution_dates": int(execution_panel["date"].nunique()),
        "folds": {},
        "runs": [],
    }
    backtest_count = 0

    for fold in range(6, -1, -1):
        available_states: list[dict[str, Any]] = []
        candidate_calendars: list[tuple[str, pd.DatetimeIndex]] = []

        for state in states:
            if fold == 0:
                candidate = state["forward_dates"]
            else:
                mapping = state["mapping"].get(fold)
                if mapping is None:
                    continue
                candidate = state["historical_dates"]
                candidate = candidate[
                    (candidate >= mapping["start_ts"])
                    & (candidate <= mapping["end_ts"])
                ]
            available_states.append(state)
            candidate_calendars.append((state["label"], candidate))

        common_dates = common_fold_calendar(candidate_calendars, fold)
        common_start = pd.Timestamp(common_dates[0]).strftime("%Y-%m-%d")
        common_end = pd.Timestamp(common_dates[-1]).strftime("%Y-%m-%d")
        curves: list[dict[str, Any]] = []
        fold_runs: list[dict[str, Any]] = []

        for state in available_states:
            if fold == 0:
                predictions = state["forward_predictions"]
                full_dates = state["forward_dates"]
                stored_config = state["forward_config"]
                source_prediction = state["forward"]["prediction_file"]
                source_config = state["forward"]["config_file"]
                original_offset = int(stored_config["rebalance_offset"])
                source_kind = "strict_oos_forward"
            else:
                predictions = state["historical_predictions"]
                full_dates = state["historical_dates"]
                stored_config = state["historical_config"]
                source_prediction = state["historical"]["prediction_file"]
                source_config = state["historical"]["config_file"]
                original_offset = int(stored_config["rebalance_offset"])
                source_kind = "historical_one_lag"

            effective_offset, skipped_dates = effective_offset_for_crop(
                full_dates=full_dates,
                crop_dates=common_dates,
                original_offset=original_offset,
                rebalance_every=int(stored_config["rebalance_every"]),
            )
            config = config_to_trade_config(
                stored_config,
                initial_cash=args.initial_cash,
                effective_offset=effective_offset,
            )

            prediction_slice = predictions[predictions["date"].isin(common_dates)].copy()
            execution_slice = execution_panel[
                execution_panel["date"].isin(common_dates)
            ].copy()
            prediction_dates = normalize_dates(prediction_slice["date"])
            if not prediction_dates.equals(common_dates):
                raise RuntimeError(
                    f"fold{fold} prediction coverage mismatch for {state['label']}"
                )

            precheck = capacity_precheck(
                execution_slice,
                prediction_slice,
                str(config.capacity_mode),
            )
            result = bt.backtest(
                prediction_slice,
                execution_slice,
                config,
                corporate_actions=pd.DataFrame(),
            )
            run_name = f"{state['label']}_fold{fold}"
            run_dir = runs_root / f"fold{fold}" / state["label"]
            run_manifest = {
                "label": state["label"],
                "fold": fold,
                "source_kind": source_kind,
                "source_prediction_file": source_prediction,
                "source_config_file": source_config,
                "source_historical_root": state["historical"]["root"],
                "source_forward_root": state["forward"]["root"],
                "signal": state["selection"],
                "initial_state": "empty_positions_and_initial_cash",
                "initial_cash": float(args.initial_cash),
                "common_start": common_start,
                "common_end": common_end,
                "common_trading_days": int(len(common_dates)),
                "original_rebalance_offset": original_offset,
                "skipped_overlap_dates_before_common_start": skipped_dates,
                "effective_local_rebalance_offset": effective_offset,
                "frozen_config": dataclasses.asdict(config),
                "capacity_precheck": precheck,
            }
            write_independent_run(
                run_dir=run_dir,
                result=result,
                cfg=config,
                output_mode=args.output_mode,
                manifest=run_manifest,
            )
            curve = curve_from_result(result, args.initial_cash)
            curves.append(
                {"label": state["label"], "run_name": run_name, "curve": curve}
            )
            run_record = {
                **run_manifest,
                "run_dir": str(run_dir),
                "summary": result["summary"],
            }
            fold_runs.append(run_record)
            global_manifest["runs"].append(run_record)
            backtest_count += 1
            print(
                f"[OK] fold{fold} {state['label']} "
                f"{common_start}..{common_end} days={len(common_dates)} "
                f"offset={original_offset}->{effective_offset}"
            )

        fold_dir = plots_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for frequency in frequencies:
            frame = plot_frequency(
                curves=curves,
                frequency=frequency,
                out_file=fold_dir / f"return_curve_{frequency}.png",
                title=f"AS1455 independent fold{fold} return ({frequency})",
                sample_curve=sample_curve,
                plt=plt,
            )
            frame.to_csv(
                fold_dir / f"return_curve_{frequency}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        fold_payload = {
            "fold": fold,
            "common_start": common_start,
            "common_end": common_end,
            "common_trading_days": int(len(common_dates)),
            "strategies": [state["label"] for state in available_states],
            "backtest_count": len(fold_runs),
            "runs": fold_runs,
        }
        global_manifest["folds"][f"fold{fold}"] = fold_payload
        (fold_dir / "fold_manifest.json").write_text(
            json.dumps(
                fold_payload,
                ensure_ascii=False,
                indent=2,
                default=bt.json_default,
            ),
            encoding="utf-8",
        )

    expected_backtests = 40
    expected_plots = 7 * len(frequencies)
    actual_plots = len(list(plots_root.glob("fold*/return_curve_*.png")))
    global_manifest.update(
        {
            "expected_backtests": expected_backtests,
            "backtest_count": backtest_count,
            "expected_plots": expected_plots,
            "plot_count": actual_plots,
            "duration_seconds": int(round(time.time() - started)),
        }
    )
    global_manifest["all_ok"] = (
        backtest_count == expected_backtests and actual_plots == expected_plots
    )
    manifest_path = out_root / "independent_fold_manifest.json"
    manifest_path.write_text(
        json.dumps(
            global_manifest,
            ensure_ascii=False,
            indent=2,
            default=bt.json_default,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": global_manifest["mode"],
                "backtest_count": backtest_count,
                "expected_backtests": expected_backtests,
                "plot_count": actual_plots,
                "expected_plots": expected_plots,
                "duration_seconds": global_manifest["duration_seconds"],
                "all_ok": global_manifest["all_ok"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not global_manifest["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
