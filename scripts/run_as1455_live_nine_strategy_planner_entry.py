#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tracking-aware entry point for the nine-strategy 14:55 planner.

Two adapters are installed before invoking the canonical planner:
1. continuation summaries use portfolio NAV rather than residual cash;
2. account state comes from the user-selected tracking account.  On the first
   effective tracking day the account is empty and a one-off bootstrap
   rebalance is forced, so the day can contain buys but cannot contain sells.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner  # noqa: E402
from utils.as1455_tracking import (  # noqa: E402
    experiment_tracking_paths,
    read_json,
    resolve_initial_cash,
    tracking_start_date,
)

_BOOTSTRAP_NEXT_CONFIG = False
_STATE_SOURCE_NEXT = "canonical_strict_forward"
_EXECUTION_CALENDAR = pd.DatetimeIndex([])


def _arg_value(flag: str) -> str | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _load_execution_calendar_from_cli() -> pd.DatetimeIndex:
    value = _arg_value("--execution-calendar")
    if not value:
        return pd.DatetimeIndex([])
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return pd.DatetimeIndex([])
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in frame.columns:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(
        pd.to_datetime(frame["date"], errors="coerce").dropna()
    ).normalize().unique().sort_values()


def _starting_portfolio_nav(nav: pd.DataFrame) -> float | None:
    if nav.empty or "nav" not in nav.columns or "daily_return" not in nav.columns:
        return None
    first_nav = pd.to_numeric(pd.Series([nav.iloc[0]["nav"]]), errors="coerce").iloc[0]
    first_ret = pd.to_numeric(pd.Series([nav.iloc[0]["daily_return"]]), errors="coerce").iloc[0]
    if not np.isfinite(first_nav) or not np.isfinite(first_ret):
        return None
    gross = 1.0 + float(first_ret)
    if gross <= 0.0:
        return None
    starting_nav = float(first_nav) / gross
    return starting_nav if np.isfinite(starting_nav) and starting_nav > 0 else None


def install_live_summary_adapter() -> None:
    original_loader = planner.live.load_v7_module

    def load_v7_with_live_summary():
        v7 = original_loader()
        original_summarize = v7.summarize_nav

        def summarize_nav_live(
            nav: pd.DataFrame,
            orders: pd.DataFrame,
            rejects: pd.DataFrame,
            cfg: Any,
            actions: pd.DataFrame | None = None,
            round_trips: pd.DataFrame | None = None,
            daily_drawdown: pd.DataFrame | None = None,
        ) -> dict:
            starting_nav = _starting_portfolio_nav(nav)
            summary_cfg = replace(cfg, initial_cash=starting_nav) if starting_nav else cfg
            return original_summarize(
                nav, orders, rejects, summary_cfg, actions, round_trips, daily_drawdown
            )

        v7.summarize_nav = summarize_nav_live
        return v7

    planner.live.load_v7_module = load_v7_with_live_summary


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol", "shares", "buy_date", "avg_entry_price", "entry_rank",
            "entry_score", "cost_basis_notional", "cost_basis_fee",
        ]
    )


def install_tracking_state_adapter() -> None:
    original_load_state = planner.load_state
    original_build_trade_config = planner.live.build_trade_config

    def tracking_load_state(experiment_root: Path, trade_date: pd.Timestamp):
        global _BOOTSTRAP_NEXT_CONFIG, _STATE_SOURCE_NEXT
        matrix_root = experiment_root.parent
        start = tracking_start_date(matrix_root)
        if start is None:
            _BOOTSTRAP_NEXT_CONFIG = False
            _STATE_SOURCE_NEXT = "canonical_strict_forward"
            return original_load_state(experiment_root, trade_date)

        trade_date = pd.Timestamp(trade_date).normalize()
        if trade_date < start:
            raise RuntimeError(
                f"trade_date {trade_date:%Y-%m-%d} is before tracking_start_date {start:%Y-%m-%d}"
            )

        prior_days = _EXECUTION_CALENDAR[
            (_EXECUTION_CALENDAR >= start) & (_EXECUTION_CALENDAR < trade_date)
        ]
        paths = experiment_tracking_paths(experiment_root)
        manifest = read_json(paths["manifest"], {}) or {}
        state = read_json(paths["latest_state"], {}) or {}

        bootstrap = trade_date == start or len(prior_days) == 0
        if bootstrap:
            cash = resolve_initial_cash(experiment_root)
            _BOOTSTRAP_NEXT_CONFIG = True
            _STATE_SOURCE_NEXT = "empty_tracking_start"
            return (
                {
                    "status": "empty_tracking_start",
                    "asof_date": None,
                    "tracking_start_date": start.strftime("%Y-%m-%d"),
                    "cash": cash,
                    "nav": cash,
                    "n_positions": 0,
                    "tracking_state_source": _STATE_SOURCE_NEXT,
                },
                _empty_positions(),
            )

        if manifest.get("tracking_start_date") != start.strftime("%Y-%m-%d"):
            raise RuntimeError(
                f"tracking account is not rebuilt for start={start:%Y-%m-%d}: "
                f"experiment={experiment_root.name}"
            )
        if not state:
            raise FileNotFoundError(
                f"tracking state missing for {experiment_root.name}; run the nightly/incremental refresh first"
            )
        asof = pd.Timestamp(state["asof_date"]).normalize()
        expected_asof = pd.Timestamp(prior_days[-1]).normalize()
        if asof != expected_asof:
            raise RuntimeError(
                f"stale tracking state for {experiment_root.name}: "
                f"asof={asof:%Y-%m-%d} expected={expected_asof:%Y-%m-%d}"
            )
        positions = planner.live.load_positions(
            paths["latest_positions"], allow_missing_buy_date=False
        )
        if len(positions) != int(state.get("n_positions", 0)):
            raise RuntimeError(
                f"tracking position count mismatch for {experiment_root.name}"
            )
        state = dict(state)
        state["tracking_state_source"] = "latest_tracking_account"
        _BOOTSTRAP_NEXT_CONFIG = False
        _STATE_SOURCE_NEXT = "latest_tracking_account"
        return state, positions

    def tracking_build_trade_config(*args, **kwargs):
        global _BOOTSTRAP_NEXT_CONFIG
        cfg = original_build_trade_config(*args, **kwargs)
        if _BOOTSTRAP_NEXT_CONFIG:
            cfg = replace(cfg, rebalance_offset=0)
            _BOOTSTRAP_NEXT_CONFIG = False
        return cfg

    planner.load_state = tracking_load_state
    planner.live.build_trade_config = tracking_build_trade_config


def postprocess_manifests() -> None:
    out_value = _arg_value("--out-root")
    matrix_value = _arg_value("--matrix-root")
    if not out_value or not matrix_value:
        return
    out_root = Path(out_value).expanduser().resolve()
    matrix_root = Path(matrix_value).expanduser().resolve()
    start = tracking_start_date(matrix_root)
    if start is None:
        return
    strategies_root = out_root / "strategies"
    if strategies_root.is_dir():
        for path in strategies_root.glob("*/strategy_manifest.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            initial = payload.get("initial_state") or {}
            payload["account_state_source"] = initial.get(
                "tracking_state_source", "latest_tracking_account"
            )
            payload["tracking_start_date"] = start.strftime("%Y-%m-%d")
            payload["tracking_bootstrap"] = payload["account_state_source"] == "empty_tracking_start"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest_file = out_root / "live_nine_strategy_manifest.json"
    if manifest_file.is_file():
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        payload["tracking_start_date"] = start.strftime("%Y-%m-%d")
        payload["account_state_semantics"] = "empty_on_tracking_start_then_latest_completed_tracking_account"
        manifest_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    global _EXECUTION_CALENDAR
    _EXECUTION_CALENDAR = _load_execution_calendar_from_cli()
    install_live_summary_adapter()
    install_tracking_state_adapter()
    planner.main()
    postprocess_manifests()


if __name__ == "__main__":
    main()
