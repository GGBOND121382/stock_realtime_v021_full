#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared AS1455 in-process grid runner.

This module owns grid orchestration and reusable input preparation.  Portfolio
simulation remains exclusively in the v7 ``backtest`` function.  Predictions
are sorted once per date and signal through ``as1455_rank_cache``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utils.as1455_backtest_io import build_trade_config, write_run
from utils.as1455_rank_cache import (
    prepare_presorted_predictions,
    validate_presorted_predictions,
)
from utils.as1455_rebalance_phase import align_forward_rebalance_phase

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKTEST_DIR = PROJECT_DIR / "code" / "backtest"
ENGINE_NAME = "inprocess_shared_rank_v5_phase_aligned"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module(
    "as1455_grid_legacy",
    BACKTEST_DIR / "run_as1455_close_auction_grid_v1.py",
)
bt = load_module(
    "as1455_bt_v7",
    BACKTEST_DIR / "run_as1455_close_auction_backtest_v7_maxpos_grid.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AS1455 in-process grid using one v7 trade engine and "
            "one daily score sort per signal"
        )
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-key", default=None)
    parser.add_argument("--raw-daily-cache-dir", required=True)
    parser.add_argument("--raw-5m-cache-dir", default=None)
    parser.add_argument("--last5-panel", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--st-symbols", default=None)
    parser.add_argument("--st-status", default=None)
    parser.add_argument("--corporate-actions", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--profile",
        default="close_auction_skip_limit",
        choices=["close_auction_simple", "close_auction_skip_limit"],
    )
    parser.add_argument(
        "--capacity-mode",
        default="none",
        choices=["none", "last5_amount", "last5_volume", "last5_both"],
    )
    parser.add_argument(
        "--capacity-missing-policy",
        default="fail",
        choices=["fail", "reject", "disable"],
    )
    parser.add_argument("--min-last5-coverage", type=float, default=0.95)
    parser.add_argument("--participation-rate", type=float, default=0.05)
    parser.add_argument("--initial-cash", type=float, default=200000)
    parser.add_argument("--commission-rate", type=float, default=0.000085)
    parser.add_argument("--min-commission", type=float, default=5)
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--slippage-bps", type=float, default=0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--allow-non-mainboard", action="store_true")
    parser.add_argument("--allow-st", action="store_true")
    parser.add_argument(
        "--corporate-action-mode",
        default="synthetic_share_factor_from_preclose",
        choices=[
            "none",
            "synthetic_share_factor_from_preclose",
            "synthetic_cash_from_preclose",
        ],
    )
    parser.add_argument("--corporate-action-threshold", type=float, default=1e-3)
    parser.add_argument("--min-price", type=float, default=0)
    parser.add_argument("--limit-eps", type=float, default=1e-6)
    parser.add_argument(
        "--max-positions-list",
        type=legacy.parse_int_list,
        default=legacy.DEFAULT_MAX_POSITIONS,
    )
    parser.add_argument(
        "--sell-rank-list",
        type=legacy.parse_int_list,
        default=legacy.DEFAULT_SELL_RANKS,
    )
    parser.add_argument(
        "--rebalance-every-list",
        type=legacy.parse_int_list,
        default=legacy.DEFAULT_REBALANCE_EVERY,
    )
    parser.add_argument(
        "--rebalance-offset-list",
        type=legacy.parse_int_list,
        default=None,
        help=(
            "Optional exact local offset subset. Configs are first constructed "
            "using --offset-mode and then restricted to these offsets."
        ),
    )
    parser.add_argument("--rebalance-phase-history-offset", type=int, default=None)
    parser.add_argument("--rebalance-phase-history-first-date", default=None)
    parser.add_argument("--rebalance-phase-history-last-date", default=None)
    parser.add_argument("--rebalance-phase-history-n-days", type=int, default=None)
    parser.add_argument(
        "--signal-spec",
        dest="signal_specs",
        action="append",
        type=legacy.parse_signal_spec,
        default=None,
    )
    parser.add_argument("--offset-mode", choices=["zero", "full"], default="zero")
    parser.add_argument(
        "--run-output-mode",
        choices=["summary", "compact", "full"],
        default="compact",
        help=(
            "File-retention level only; summary=JSON only, compact=core NAV "
            "and conclusion CSVs, full=all audit CSVs."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-parity-check", action="store_true")
    parser.add_argument(
        "--parity-check-only",
        action="store_true",
        help="Build shared inputs, execute one v7 smoke run, then exit.",
    )
    parser.add_argument("--model-family", default="ML4T Ch17 NN")
    parser.add_argument("--model-run", default=None)
    parser.add_argument("--model-params-file", default=None)
    parser.add_argument("--prediction-file-sha256", default=None)
    args = parser.parse_args()
    if args.signal_specs is None:
        args.signal_specs = [
            legacy.parse_signal_spec(value)
            for value in legacy.DEFAULT_SIGNAL_SPECS
        ]
    if args.parity_check_only and args.skip_parity_check:
        raise SystemExit(
            "--parity-check-only cannot be combined with --skip-parity-check"
        )
    _validate_phase_args(args)
    return args


def _phase_values(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "historical_offset": args.rebalance_phase_history_offset,
        "historical_first_date": args.rebalance_phase_history_first_date,
        "historical_last_date": args.rebalance_phase_history_last_date,
        "historical_n_days": args.rebalance_phase_history_n_days,
    }


def _validate_phase_args(args: argparse.Namespace) -> None:
    values = _phase_values(args)
    supplied = [name for name, value in values.items() if value is not None]
    if supplied and len(supplied) != len(values):
        missing = [name for name, value in values.items() if value is None]
        raise SystemExit(
            "rebalance phase alignment requires all historical fields; "
            f"supplied={supplied} missing={missing}"
        )
    if supplied and args.rebalance_offset_list is not None:
        raise SystemExit(
            "--rebalance-offset-list cannot be combined with historical phase alignment"
        )


def phase_alignment_requested(args: argparse.Namespace) -> bool:
    return args.rebalance_phase_history_offset is not None


def build_configs(args: argparse.Namespace) -> list[Any]:
    configs = legacy.build_configs(args)
    if args.rebalance_offset_list is None:
        return configs
    requested = set(int(value) for value in args.rebalance_offset_list)
    invalid = sorted(
        {
            offset
            for _spec, _max_pos, _sell_rank, rebalance_every, offset in configs
            if offset in requested and not 0 <= offset < rebalance_every
        }
    )
    if invalid:
        raise SystemExit(f"invalid rebalance offsets for requested periods: {invalid}")
    filtered = [config for config in configs if int(config[4]) in requested]
    if not filtered:
        raise SystemExit(
            "--rebalance-offset-list removed every grid config; "
            f"requested={sorted(requested)}"
        )
    return filtered


def load_shared_inputs(args: argparse.Namespace) -> tuple[
    dict[str, dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DatetimeIndex,
]:
    prediction_path = Path(args.predictions)
    prediction_sha = (
        args.prediction_file_sha256 or legacy.sha256_file(prediction_path)
    )
    model_params_file = (
        args.model_params_file
        or legacy.infer_model_params_file(args.predictions)
    )

    signals: dict[str, dict[str, Any]] = {}
    all_symbols: set[str] = set()
    for spec in args.signal_specs:
        predictions, metadata = bt.load_predictions(
            prediction_path,
            args.prediction_key,
            None,
            signal_cols=bt.parse_csv_tokens(spec["signal_cols"]),
            signal_mode=spec["signal_mode"],
            signal_name=spec["signal_name"],
            prediction_file_sha256=prediction_sha,
            model_params_file=(
                Path(model_params_file) if model_params_file else None
            ),
        )
        predictions = bt.apply_date_filters(
            predictions, args.start_date, args.end_date
        )
        if predictions.empty:
            raise SystemExit(f"empty predictions for {spec['signal_name']}")

        ranked_predictions = prepare_presorted_predictions(predictions)
        validate_presorted_predictions(ranked_predictions)
        signals[spec["signal_name"]] = {
            "preds": ranked_predictions,
            "meta": metadata,
        }
        all_symbols.update(ranked_predictions["symbol"].unique())

    universe = bt.read_universe(Path(args.universe) if args.universe else None)
    st_symbols = bt.load_st_symbols(
        Path(args.st_symbols) if args.st_symbols else None
    )
    st_status = bt.load_st_status(
        Path(args.st_status) if args.st_status else None
    )
    last5_panel = bt.load_last5_panel(
        Path(args.last5_panel) if args.last5_panel else None
    )
    corporate_actions = bt.load_corporate_actions(
        Path(args.corporate_actions) if args.corporate_actions else None
    )
    full_execution_panel, execution_report = bt.build_execution_panel(
        all_symbols,
        Path(args.raw_daily_cache_dir),
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=last5_panel,
        raw_5m_cache_dir=(
            Path(args.raw_5m_cache_dir) if args.raw_5m_cache_dir else None
        ),
    )
    if full_execution_panel.empty:
        raise SystemExit("empty execution panel")
    execution_calendar = pd.DatetimeIndex(
        pd.to_datetime(full_execution_panel["date"], errors="coerce").dropna().unique()
    ).normalize().sort_values()
    execution_panel = bt.apply_date_filters(
        full_execution_panel, args.start_date, args.end_date
    )
    if execution_panel.empty:
        raise SystemExit("empty execution panel after date filters")

    for name, payload in signals.items():
        effective = args.capacity_mode
        precheck = bt.build_capacity_precheck(
            execution_panel, payload["preds"], effective
        )
        if effective != "none":
            coverage = float(precheck.get("coverage_rate", 0))
            positive = float(precheck.get("positive_rate", 0))
            if coverage < args.min_last5_coverage or positive <= 0:
                if args.capacity_missing_policy == "fail":
                    raise SystemExit(
                        f"capacity data insufficient for {name}: "
                        f"coverage={coverage:.6f} positive={positive:.6f}"
                    )
                if args.capacity_missing_policy == "disable":
                    effective = "none"
                    precheck["policy_action"] = "disabled_capacity_mode"
                else:
                    precheck["policy_action"] = "reject_on_missing_capacity"
        payload["capacity_mode"] = effective
        payload["capacity_precheck"] = precheck

    return (
        signals,
        execution_panel,
        execution_report,
        corporate_actions,
        execution_calendar,
    )


def resolve_rebalance_phase(
    args: argparse.Namespace,
    signals: dict[str, dict[str, Any]],
    execution_calendar: pd.DatetimeIndex,
) -> dict[str, Any] | None:
    if not phase_alignment_requested(args):
        return None
    if len(args.rebalance_every_list) != 1:
        raise SystemExit(
            "historical phase alignment requires exactly one rebalance period"
        )
    if len(signals) != 1:
        raise SystemExit(
            "historical phase alignment requires exactly one frozen signal"
        )
    payload = next(iter(signals.values()))
    prediction_dates = pd.DatetimeIndex(
        pd.to_datetime(payload["preds"]["date"], errors="coerce")
        .dropna()
        .unique()
    ).normalize().sort_values()
    alignment = align_forward_rebalance_phase(
        rebalance_every=int(args.rebalance_every_list[0]),
        historical_offset=int(args.rebalance_phase_history_offset),
        historical_n_days=int(args.rebalance_phase_history_n_days),
        historical_first_date=str(args.rebalance_phase_history_first_date),
        historical_last_date=str(args.rebalance_phase_history_last_date),
        forward_prediction_dates=prediction_dates,
        execution_calendar_dates=execution_calendar,
    )
    args.rebalance_offset_list = [int(alignment["effective_forward_offset"])]
    print(
        "[PHASE ALIGN] "
        f"historical_offset={alignment['historical_offset']} "
        f"history_days={alignment['historical_n_days']} "
        f"bridge_days={alignment['bridge_execution_days']} "
        f"forward_global_index={alignment['forward_global_index']} "
        f"effective_forward_offset={alignment['effective_forward_offset']}"
    )
    return alignment


def engine_manifest(
    *,
    args: argparse.Namespace,
    configs: list[Any],
    prediction_sha: str,
    smoke_check: dict[str, Any] | None,
    parity_check_only: bool,
    phase_alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "engine": ENGINE_NAME,
        "configs": len(configs),
        "signals": [spec["signal_name"] for spec in args.signal_specs],
        "single_trade_engine": True,
        "trade_engine_source": (
            "code/backtest/"
            "run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest"
        ),
        "execution_panel_built_once": True,
        "predictions_loaded_once_per_signal": True,
        "daily_rankings_built_once_per_signal": True,
        "rank_cache_source": "utils/as1455_rank_cache.py",
        "rank_cache_strategy": (
            "pre-sort each date once with the same pandas call used by v7; "
            "subsequent identical sort requests return a copy without sorting"
        ),
        "dynamic_source_rewrite": False,
        "prediction_file_sha256": prediction_sha,
        "output_mode": args.run_output_mode,
        "parity_check_only": parity_check_only,
        "shared_engine_smoke_check": smoke_check,
        "rebalance_phase_alignment": phase_alignment,
    }


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    if not prediction_path.exists():
        raise SystemExit(f"prediction file not found: {prediction_path}")

    out_root = Path(args.out_root)
    runs_root = out_root / "01_runs"
    logs_root = out_root / "04_logs"
    summary_root = out_root / "02_summary"
    for path in (runs_root, logs_root, summary_root):
        path.mkdir(parents=True, exist_ok=True)

    if args.dry_run and not phase_alignment_requested(args):
        configs = build_configs(args)
        legacy.write_grid_config(out_root / "00_grid_config.csv", configs)
        print(f"[DRY RUN] configs={len(configs)} engine={ENGINE_NAME}")
        return

    prediction_sha = (
        args.prediction_file_sha256 or legacy.sha256_file(prediction_path)
    )
    model_run = args.model_run or legacy.infer_model_run(args.predictions)
    (
        signals,
        execution_panel,
        execution_report,
        corporate_actions,
        execution_calendar,
    ) = load_shared_inputs(args)
    phase_alignment = resolve_rebalance_phase(
        args, signals, execution_calendar
    )
    configs = build_configs(args)
    legacy.write_grid_config(out_root / "00_grid_config.csv", configs)
    if args.dry_run:
        print(
            f"[DRY RUN] configs={len(configs)} engine={ENGINE_NAME} "
            f"phase_alignment={phase_alignment}"
        )
        return

    first_result: dict[str, Any] | None = None
    first_run_name: str | None = None
    smoke_check: dict[str, Any] | None = None
    if not args.skip_parity_check and configs:
        spec, max_positions, sell_rank, rebalance_every, offset = configs[0]
        payload = signals[spec["signal_name"]]
        cfg = build_trade_config(
            bt,
            args,
            payload,
            max_positions=max_positions,
            sell_rank=sell_rank,
            rebalance_every=rebalance_every,
            rebalance_offset=offset,
        )
        first_run_name = legacy.run_name(
            spec["signal_name"],
            max_positions,
            sell_rank,
            rebalance_every,
            offset,
        )
        print(
            "[PARITY] single v7 trade engine smoke run "
            f"for {first_run_name}"
        )
        first_result = bt.backtest(
            payload["preds"],
            execution_panel,
            cfg,
            corporate_actions=corporate_actions,
        )
        smoke_check = {
            "passed": True,
            "run_name": first_run_name,
            "final_nav": first_result["summary"].get("final_nav"),
            "n_orders": first_result["summary"].get("n_orders"),
            "n_rejections": first_result["summary"].get("n_rejections"),
        }
        print("[PARITY] PASS")

    manifest_path = out_root / "grid_engine_manifest.json"
    if args.parity_check_only:
        manifest_path.write_text(
            json.dumps(
                engine_manifest(
                    args=args,
                    configs=configs,
                    prediction_sha=prediction_sha,
                    smoke_check=smoke_check,
                    parity_check_only=True,
                    phase_alignment=phase_alignment,
                ),
                ensure_ascii=False,
                indent=2,
                default=bt.json_default,
            ),
            encoding="utf-8",
        )
        print("[PARITY] check-only completed; grid was not executed")
        return

    rows: list[dict[str, Any]] = []
    for index, config_tuple in enumerate(configs, 1):
        spec, max_positions, sell_rank, rebalance_every, offset = config_tuple
        run_name = legacy.run_name(
            spec["signal_name"],
            max_positions,
            sell_rank,
            rebalance_every,
            offset,
        )
        run_dir = runs_root / run_name
        log_path = logs_root / f"{run_name}.log"

        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[{index}/{len(configs)}] SKIP existing {run_name}")
            rows.append(
                legacy.flatten_summary(
                    run_dir, config_tuple, "ok", returncode=0
                )
            )
            continue

        payload = signals[spec["signal_name"]]
        cfg = build_trade_config(
            bt,
            args,
            payload,
            max_positions=max_positions,
            sell_rank=sell_rank,
            rebalance_every=rebalance_every,
            rebalance_offset=offset,
        )
        print(f"[{index}/{len(configs)}] RUN {run_name}")
        try:
            if first_result is not None and run_name == first_run_name:
                result = first_result
                first_result = None
            else:
                result = bt.backtest(
                    payload["preds"],
                    execution_panel,
                    cfg,
                    corporate_actions=corporate_actions,
                )
            write_run(
                run_dir=run_dir,
                result=result,
                cfg=cfg,
                signal_meta=payload["meta"],
                args=args,
                prediction_sha=prediction_sha,
                exec_panel=execution_panel,
                capacity_precheck=payload["capacity_precheck"],
                model_run=model_run,
                json_default=bt.json_default,
                engine_name=ENGINE_NAME,
            )
            log_path.write_text(
                f"[OK] {ENGINE_NAME}\n", encoding="utf-8"
            )
            rows.append(
                legacy.flatten_summary(
                    run_dir, config_tuple, "ok", returncode=0
                )
            )
        except Exception as exc:
            log_path.write_text(
                f"[FAILED] {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            rows.append(
                legacy.flatten_summary(
                    run_dir, config_tuple, "failed", returncode=1
                )
            )
            print(f"    FAILED {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "grid_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(
        out_root / "grid_summary.csv", index=False, encoding="utf-8-sig"
    )
    legacy.write_leaderboards(summary_csv, out_root)

    manifest_path.write_text(
        json.dumps(
            engine_manifest(
                args=args,
                configs=configs,
                prediction_sha=prediction_sha,
                smoke_check=smoke_check,
                parity_check_only=False,
                phase_alignment=phase_alignment,
            ),
            ensure_ascii=False,
            indent=2,
            default=bt.json_default,
        ),
        encoding="utf-8",
    )
    if args.run_output_mode == "full":
        execution_report.to_csv(
            out_root / "execution_panel_build_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"[OK] configs={len(configs)} summary={summary_csv}")


if __name__ == "__main__":
    main()
