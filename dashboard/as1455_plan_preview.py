#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast start-date-aware 14:55 plan preview for the Streamlit dashboard.

This module never runs model inference. It reuses saved Top-5 predictions/ranks,
the frozen historical strategy parameters and saved 14:55 execution sidecars.
Changing the tracking start date therefore only replays lightweight portfolio
state and recomputes the selected day's orders.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner  # noqa: E402
from scripts import run_as1455_live_strict_oos_monitor as live  # noqa: E402
from scripts import update_as1455_tracking_accounts as tracker  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_model_selection import select_corresponding_historical_signal  # noqa: E402
from utils.as1455_strict_oos import historical_phase_window, historical_trading_config  # noqa: E402
from utils.as1455_tracking import resolve_initial_cash  # noqa: E402

DATE_RE = re.compile(r"^\d{8}$")
DEFAULT_CACHE_BASE = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_prediction_cache"
)
DEFAULT_RAW_DAILY = (
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch12_as1455"
    / "baostock_raw_daily_cache"
)


def _date_token(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _normalize_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _read_live_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column) for column in frame.columns]
    required = {"symbol", "date", "0", "1", "2", "3", "4"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing prediction columns: {sorted(missing)}")
    return tracker.normalize_predictions(frame)


def _load_live_predictions(
    live_root: Path,
    target: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not live_root.is_dir():
        return pd.DataFrame()
    for day in sorted(live_root.iterdir()):
        if not day.is_dir() or not DATE_RE.fullmatch(day.name):
            continue
        date = pd.to_datetime(day.name, format="%Y%m%d").normalize()
        if date < start or date > end:
            continue
        path = day / "nine_strategy" / "shared_predictions" / target / "top5_live_predictions.csv"
        if path.is_file():
            frames.append(_read_live_prediction_file(path))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def _prediction_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if frame.empty:
        return pd.DatetimeIndex([])
    return (
        pd.DatetimeIndex(frame.index.get_level_values("date"))
        .normalize()
        .unique()
        .sort_values()
    )


def _calendar_from_live_day(live_root: Path, selected: pd.Timestamp) -> pd.DatetimeIndex:
    path = live_root / _date_token(selected) / "05_execution_calendar.csv"
    return live.load_execution_calendar(path, selected)


def _load_sidecar_range(
    live_root: Path,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for date in dates:
        date = _normalize_date(date)
        path = live_root / _date_token(date) / "08_live_execution_sidecar.csv"
        if not path.is_file():
            missing.append(date.strftime("%Y-%m-%d"))
            continue
        frames.append(live.load_execution_panel(path, date))
    execution = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not execution.empty:
        execution["date"] = pd.to_datetime(execution["date"], errors="raise").dt.normalize()
        execution = execution.drop_duplicates(["date", "symbol"], keep="last")
    return execution, missing


def _overlay_execution(base: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return base
    if base.empty:
        return overlay.copy()
    work = base.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    keys = pd.MultiIndex.from_frame(overlay[["date", "symbol"]])
    base_keys = pd.MultiIndex.from_frame(work[["date", "symbol"]])
    work = work.loc[~base_keys.isin(keys)]
    return pd.concat([work, overlay], ignore_index=True, sort=False)


def _build_execution_context(
    v7: Any,
    target_predictions: dict[str, pd.DataFrame],
    live_root: Path,
    raw_daily: Path,
    start: pd.Timestamp,
    selected: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    # The selected live day's calendar is the same raw-daily market calendar used
    # by the canonical planner and already contains the bridge needed for exact
    # historical-offset alignment.
    calendar = _calendar_from_live_day(live_root, selected)
    required = calendar[(calendar >= start) & (calendar <= selected)]
    sidecars, missing = _load_sidecar_range(live_root, required)
    if not missing and len(sidecars):
        return sidecars, calendar, {
            "execution_source": "saved_1455_sidecars",
            "raw_daily_fallback_dates": [],
        }

    # Fallback only when some old live day lacks its sidecar. This is slower but
    # still does not run model inference or historical Grid/Fold work.
    union = pd.concat(list(target_predictions.values()))
    union = union[~union.index.duplicated(keep="last")].sort_index()
    base, raw_calendar = tracker.execution_inputs(v7, union, raw_daily)
    execution = _overlay_execution(base, sidecars)
    merged_calendar = raw_calendar.union(calendar).unique().sort_values()
    return execution, merged_calendar, {
        "execution_source": "raw_daily_with_1455_sidecar_overlay",
        "raw_daily_fallback_dates": missing,
    }


def _selection_context(
    item: dict[str, Any],
    matrix_root: Path,
    feature_preset: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    root = matrix_root / item["experiment"]
    manifest = tracker.read_json(
        root / "global_fold0_to_fold5_forward_manifest.json", {}
    ) or {}
    history_root = planner.resolve_history_root(root, manifest)
    selection = select_corresponding_historical_signal(
        base_root=Path(as1455_paths.TARGET_BACKTEST_ROOT),
        feature_preset=feature_preset,
        target_col=item["target_col"],
        rebalance_every=item["rebalance_every"],
        rank_metric="sharpe",
        explicit_backtest_root=history_root,
    )
    expected = planner.EXPECTED_SIGNAL_SPEC[item["signal"]]
    actual = planner.selection_spec(selection)
    if actual != expected:
        raise RuntimeError(
            f"fixed signal mismatch for {item['experiment']}: {actual} != {expected}"
        )
    historical_trading_config(selection, item["rebalance_every"])
    history_window = historical_phase_window(selection)
    historical_config, _ = live.load_historical_run_config(selection)
    return selection, historical_config, history_window, actual


def _rank_table(predictions: pd.DataFrame, selection: Any, date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.DatetimeIndex(predictions.index.get_level_values("date")).normalize()
    day = predictions.loc[dates.eq(date)]
    score = live.score_predictions(day, selection)
    ranked = score.reset_index()
    if "score" not in ranked.columns:
        ranked = ranked.rename(columns={score.name: "score", 0: "score"})
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked = ranked.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "shares",
            "buy_date",
            "avg_entry_price",
            "entry_rank",
            "entry_score",
            "cost_basis_notional",
            "cost_basis_fee",
        ]
    )


def _action_text(
    is_rebalance: bool,
    orders: pd.DataFrame,
    current_positions: pd.DataFrame,
    buy_count: int,
    rejection_count: int,
) -> tuple[str, bool]:
    first_entry = current_positions.empty and buy_count > 0
    if first_entry:
        return "首次建仓", True
    if is_rebalance and len(orders):
        return "调仓", False
    if is_rebalance and rejection_count:
        return "调仓日·未成交", False
    if is_rebalance:
        return "调仓日·无需成交", False
    if current_positions.empty:
        return "非调仓日·保持空仓", False
    return "非调仓日·继续持有", False


def preview_nine_strategy_day(
    matrix_root: Path,
    live_root: Path,
    start: pd.Timestamp,
    selected: pd.Timestamp,
    *,
    feature_preset: str = "rotation_addon_onehot",
    cache_base: Path = DEFAULT_CACHE_BASE,
    raw_daily: Path = DEFAULT_RAW_DAILY,
    capacity_mode: str = "none",
    participation_rate: float = 0.05,
) -> dict[str, Any]:
    """Recompute one day's nine-strategy plan for an arbitrary tracking start.

    The expensive model inference is never called. For recent dates the fast
    path uses only saved prediction CSVs, one saved calendar CSV and the saved
    per-day 14:55 execution sidecars.
    """
    matrix_root = matrix_root.expanduser().resolve()
    live_root = live_root.expanduser().resolve()
    start = _normalize_date(start)
    selected = _normalize_date(selected)
    if selected < start:
        return {
            "status": "before_start",
            "tracking_start_date": start.strftime("%Y-%m-%d"),
            "selected_date": selected.strftime("%Y-%m-%d"),
            "summary": pd.DataFrame(),
            "details": {},
        }
    if capacity_mode != "none":
        raise RuntimeError("dashboard preview currently requires capacity_mode=none")

    experiments = planner.parse_experiments(matrix_root)
    targets = sorted({item["target"] for item in experiments})
    target_predictions: dict[str, pd.DataFrame] = {}
    prediction_meta: dict[str, Any] = {}
    for target in targets:
        live_predictions = _load_live_predictions(live_root, target, start, selected)
        # If recent live files are incomplete, reuse the already-generated
        # fold0 forward prediction cache. This is disk I/O only, not inference.
        calendar = _calendar_from_live_day(live_root, selected)
        required = calendar[(calendar >= start) & (calendar <= selected)]
        live_dates = set(_prediction_dates(live_predictions))
        missing = [date for date in required if date not in live_dates]
        if missing:
            cached, meta = tracker.load_predictions(
                cache_base, live_root, feature_preset, target, start
            )
            dates = pd.DatetimeIndex(cached.index.get_level_values("date")).normalize()
            cached = cached.loc[(dates >= start) & (dates <= selected)]
            target_predictions[target] = cached
            prediction_meta[target] = {
                **meta,
                "source": "live_plus_existing_prediction_cache",
            }
        else:
            target_predictions[target] = live_predictions
            prediction_meta[target] = {
                "source": "saved_live_top5_predictions",
                "date_min": _prediction_dates(live_predictions).min().strftime("%Y-%m-%d"),
                "date_max": _prediction_dates(live_predictions).max().strftime("%Y-%m-%d"),
            }

    v7 = live.load_v7_module()
    tracker.patch_summary(v7)
    execution, calendar, execution_meta = _build_execution_context(
        v7,
        target_predictions,
        live_root,
        raw_daily,
        start,
        selected,
    )

    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for item in experiments:
        experiment = item["experiment"]
        root = matrix_root / experiment
        predictions = target_predictions[item["target"]]
        selection, historical_config, history_window, signal_spec = _selection_context(
            item, matrix_root, feature_preset
        )
        pred_dates = _prediction_dates(predictions)
        dates = tracker.contiguous_tracking_dates(pred_dates, calendar, start)
        dates = dates[dates <= selected]
        if selected not in set(dates):
            rows.append(
                {
                    "experiment": experiment,
                    "status": "unavailable",
                    "action": "缺少该日可重放数据",
                    "is_rebalance_day": False,
                    "planned_orders": 0,
                    "planned_buys": 0,
                    "planned_sells": 0,
                    "rejections": 0,
                    "current_positions": 0,
                    "target_positions": 0,
                    "fixed_signal_spec": signal_spec,
                }
            )
            continue

        initial_cash = resolve_initial_cash(root)
        current_positions = _empty_positions()
        cash_before = initial_cash
        prior_dates = dates[dates < selected]
        if len(prior_dates):
            prior_result, _ = tracker.run_chunk(
                v7,
                selection,
                historical_config,
                history_window,
                predictions,
                execution,
                calendar,
                prior_dates,
                initial_cash,
                _empty_positions(),
                capacity_mode,
                participation_rate,
            )
            prior_state = prior_result["final_state"]
            cash_before = float(prior_state["cash"])
            current_positions = pd.DataFrame(prior_state.get("positions", []))

        day_result, day_meta = tracker.run_chunk(
            v7,
            selection,
            historical_config,
            history_window,
            predictions,
            execution,
            calendar,
            pd.DatetimeIndex([selected]),
            cash_before,
            current_positions,
            capacity_mode,
            participation_rate,
        )
        nav = day_result["nav"].copy()
        orders = day_result["orders"].copy()
        rejections = day_result["rejections"].copy()
        target_positions = day_result["positions"].copy()
        final_state = day_result["final_state"]
        is_rebalance = bool(nav["is_rebalance_day"].iloc[-1])
        buy_count = (
            int(orders["side"].astype(str).str.lower().eq("buy").sum())
            if not orders.empty and "side" in orders.columns
            else 0
        )
        sell_count = (
            int(orders["side"].astype(str).str.lower().eq("sell").sum())
            if not orders.empty and "side" in orders.columns
            else 0
        )
        rejection_count = int(len(rejections))
        action, first_entry = _action_text(
            is_rebalance,
            orders,
            current_positions,
            buy_count,
            rejection_count,
        )
        phase = day_meta["phase"]
        rank = _rank_table(predictions, selection, selected)
        row = {
            "trade_date": selected.strftime("%Y-%m-%d"),
            "experiment": experiment,
            "target_col": item["target_col"],
            "signal_kind": item["signal"],
            "status": "ok",
            "action": action,
            "is_rebalance_day": is_rebalance,
            "planned_orders": int(len(orders)),
            "planned_buys": buy_count,
            "planned_sells": sell_count,
            "rejections": rejection_count,
            "current_positions": int(len(current_positions)),
            "target_positions": int(len(final_state.get("positions", []))),
            "cash_before": cash_before,
            "planned_cash_after": float(final_state.get("cash", np.nan)),
            "max_positions": int(selection.historical_max_positions),
            "sell_rank": int(selection.historical_sell_rank),
            "rebalance_every": int(selection.historical_rebalance_every),
            "historical_offset": int(selection.historical_rebalance_offset),
            "effective_preview_offset": int(phase["effective_forward_offset"]),
            "tracking_bootstrap": first_entry,
            "fixed_signal_spec": signal_spec,
            "source": "实时重算·复用已有预测/rank",
        }
        rows.append(row)
        details[experiment] = {
            "manifest": row,
            "orders": orders,
            "target_positions": target_positions,
            "current_positions": current_positions,
            "rejections": rejections,
            "rank": rank,
            "nav": nav,
            "phase": phase,
        }

    summary = pd.DataFrame(rows)
    return {
        "status": "ok" if len(details) == 9 else "partial",
        "tracking_start_date": start.strftime("%Y-%m-%d"),
        "selected_date": selected.strftime("%Y-%m-%d"),
        "summary": summary,
        "details": details,
        "prediction_source": prediction_meta,
        **execution_meta,
        "model_inference_rerun": False,
        "historical_grid_rerun": False,
    }
