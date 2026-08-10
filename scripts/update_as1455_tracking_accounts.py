#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maintain start-date-aware AS1455 tracking accounts without rerunning Fold/Grid.

A tracking account is separate from the frozen research strict-forward artifact:
- before the configured start date it is empty;
- the first executable date is a forced bootstrap rebalance from empty;
- later dates keep the historically validated rebalance phase;
- incremental refresh processes only dates after the latest account state.

Predictions prefer saved 14:55 live files and fall back to the existing fold0
forward prediction cache. Execution uses completed BaoStock raw-daily rows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
from scripts import run_as1455_live_strict_oos_monitor as live  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_model_selection import select_corresponding_historical_signal  # noqa: E402
from utils.as1455_rebalance_phase import align_forward_rebalance_phase  # noqa: E402
from utils.as1455_strict_oos import historical_phase_window, historical_trading_config  # noqa: E402
from utils.as1455_tracking import (  # noqa: E402
    TRACKING_MATRIX_MANIFEST,
    TRACKING_MATRIX_SUMMARY,
    contiguous_tracking_dates,
    experiment_tracking_paths,
    read_json,
    resolve_initial_cash,
    tracking_start_date,
    write_json,
)

DATE_RE = re.compile(r"^\d{8}$")
DEFAULT_CACHE_BASE = PROJECT_DIR / "saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_prediction_cache"
DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data/ashare_ml4t/live_as1455"
DEFAULT_RAW_DAILY = PROJECT_DIR / "saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if not isinstance(work.index, pd.MultiIndex):
        if not {"symbol", "date"}.issubset(work.columns):
            raise RuntimeError("prediction frame requires symbol/date")
        work["symbol"] = work["symbol"].map(live.exchange_symbol)
        work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
        work = work.set_index(["symbol", "date"])
    symbols = work.index.get_level_values(0).map(live.exchange_symbol)
    dates = pd.to_datetime(work.index.get_level_values(1), errors="raise").normalize()
    work.index = pd.MultiIndex.from_arrays([symbols, dates], names=["symbol", "date"])
    work.columns = [int(str(c)) if str(c).isdigit() else c for c in work.columns]
    if set(range(5)) - set(work.columns):
        raise RuntimeError("prediction frame lacks Top-5 columns 0..4")
    for column in range(5):
        work[column] = pd.to_numeric(work[column], errors="raise")
    return work[list(range(5))].sort_index()


def prediction_cache_file(cache_base: Path, preset: str, target_col: str) -> Path:
    return cache_base / f"{preset}_{target_col}_top5/fold0_forward_latest/00_predictions/fold0_forward_preds.h5"


def load_predictions(
    cache_base: Path,
    live_root: Path,
    preset: str,
    target: str,
    start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_col = f"{target}_fwd"
    cache_file = prediction_cache_file(cache_base, preset, target_col)
    frames: list[pd.DataFrame] = []
    cache_rows = 0
    live_rows = 0
    if cache_file.is_file():
        base = normalize_predictions(pd.read_hdf(cache_file, "predictions"))
        base = base.loc[base.index.get_level_values("date") >= start]
        cache_rows = len(base)
        if not base.empty:
            frames.append(base)
    if live_root.is_dir():
        live_frames: list[pd.DataFrame] = []
        for day in sorted(live_root.iterdir()):
            if not day.is_dir() or not DATE_RE.fullmatch(day.name):
                continue
            if pd.to_datetime(day.name, format="%Y%m%d").normalize() < start:
                continue
            path = day / "nine_strategy/shared_predictions" / target / "top5_live_predictions.csv"
            if path.is_file():
                candidate = pd.read_csv(path, encoding="utf-8-sig")
                candidate.columns = [str(c) for c in candidate.columns]
                if {"symbol", "date", "0", "1", "2", "3", "4"}.issubset(candidate.columns):
                    live_frames.append(normalize_predictions(candidate))
        if live_frames:
            overlay = pd.concat(live_frames).sort_index()
            live_rows = len(overlay)
            frames.append(overlay)
    if not frames:
        raise FileNotFoundError(f"no prediction source for {target_col}: {cache_file}")
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    dates = pd.DatetimeIndex(combined.index.get_level_values("date"))
    return combined, {
        "cache_file": str(cache_file),
        "cache_rows": cache_rows,
        "live_overlay_rows": live_rows,
        "date_min": dates.min().strftime("%Y-%m-%d"),
        "date_max": dates.max().strftime("%Y-%m-%d"),
    }


def patch_summary(v7: Any) -> None:
    """Use portfolio NAV, not residual cash, for continuation-chunk summaries."""
    original = v7.summarize_nav

    def wrapped(nav, orders, rejects, cfg, actions=None, round_trips=None, daily_drawdown=None):
        summary_cfg = cfg
        if not nav.empty and {"nav", "daily_return"}.issubset(nav.columns):
            first_nav = pd.to_numeric(pd.Series([nav.iloc[0]["nav"]]), errors="coerce").iloc[0]
            first_ret = pd.to_numeric(pd.Series([nav.iloc[0]["daily_return"]]), errors="coerce").iloc[0]
            if pd.notna(first_nav) and pd.notna(first_ret) and 1 + float(first_ret) > 0:
                start_nav = float(first_nav) / (1 + float(first_ret))
                if math.isfinite(start_nav) and start_nav > 0:
                    summary_cfg = replace(cfg, initial_cash=start_nav)
        return original(nav, orders, rejects, summary_cfg, actions, round_trips, daily_drawdown)

    v7.summarize_nav = wrapped


def execution_inputs(v7: Any, predictions: pd.DataFrame, raw_daily: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    symbols = sorted(set(predictions.index.get_level_values("symbol").astype(str)))
    panel, _ = v7.build_execution_panel(
        symbols, raw_daily, pd.DataFrame(), set(),
        st_status=pd.DataFrame(), last5_panel=pd.DataFrame(), raw_5m_cache_dir=None,
    )
    if panel.empty:
        raise RuntimeError("execution panel is empty")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    calendar = pd.DatetimeIndex(panel["date"].dropna().unique()).normalize().sort_values()
    return panel, calendar


def run_chunk(
    v7: Any,
    selection: Any,
    historical_config: dict[str, Any],
    history_window: dict[str, Any],
    predictions: pd.DataFrame,
    execution: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    dates: pd.DatetimeIndex,
    cash: float,
    positions: pd.DataFrame,
    bootstrap: bool,
    capacity_mode: str,
    participation_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    date_set = set(pd.DatetimeIndex(dates).normalize())
    pred_dates = pd.DatetimeIndex(predictions.index.get_level_values("date")).normalize()
    selected = predictions.loc[pred_dates.isin(date_set)]
    score = live.score_predictions(selected, selection)
    preds = score.reset_index()
    if "score" not in preds.columns:
        preds = preds.rename(columns={score.name: "score", 0: "score"})
    phase = align_forward_rebalance_phase(
        rebalance_every=int(selection.historical_rebalance_every),
        historical_offset=int(selection.historical_rebalance_offset),
        historical_n_days=int(history_window["historical_n_days"]),
        historical_first_date=history_window["historical_first_date"],
        historical_last_date=history_window["historical_last_date"],
        forward_prediction_dates=dates,
        execution_calendar_dates=calendar,
    )
    cfg = live.build_trade_config(
        v7, selection, historical_config, phase, float(cash), capacity_mode, participation_rate
    )
    cfg = replace(cfg, corporate_action_mode=planner.synthetic_corporate_action_mode(historical_config))
    if bootstrap:
        cfg = replace(cfg, rebalance_offset=0)
    result = v7.backtest(
        preds[["symbol", "date", "score"]],
        execution.loc[execution["date"].isin(date_set)].copy(),
        cfg,
        corporate_actions=None,
        initial_positions=positions,
        day_index_start=0,
        allow_single_date=True,
    )
    return result, {"phase": phase, "trade_config": cfg.__dict__}


def frames(result: dict[str, Any], bootstrap_date: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    out = {key: result[src].copy() for key, src in {
        "nav": "nav", "orders": "orders", "rejections": "rejections", "positions": "positions"
    }.items()}
    for frame in out.values():
        if not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frame["tracking_bootstrap"] = frame["date"].eq(bootstrap_date) if bootstrap_date is not None else False
    return out


def merge_frames(paths: dict[str, Path], fresh: list[dict[str, pd.DataFrame]], rebuild: bool) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for key in ("nav", "orders", "rejections", "positions"):
        pieces: list[pd.DataFrame] = []
        if not rebuild:
            old = read_csv(paths[key])
            if not old.empty:
                pieces.append(old)
        pieces.extend(part[key] for part in fresh if not part[key].empty)
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        if not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            if key == "nav":
                frame = frame.sort_values("date").drop_duplicates("date", keep="last")
            elif key == "positions" and "symbol" in frame.columns:
                frame = frame.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
        merged[key] = frame.reset_index(drop=True)
    return merged


def recompute_nav(nav: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if nav.empty:
        return nav
    out = nav.sort_values("date").drop_duplicates("date", keep="last").copy()
    values = pd.to_numeric(out["nav"], errors="coerce")
    prior = values.shift(1)
    prior.iloc[0] = float(initial_cash)
    out["daily_return"] = values / prior - 1
    out["cumulative_return"] = values / float(initial_cash) - 1
    out["cumulative_return_pct"] = out["cumulative_return"] * 100
    out["drawdown"] = values / values.cummax() - 1
    return out


def account_summary(nav: pd.DataFrame, orders: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    nav = recompute_nav(nav, initial_cash)
    rets = pd.to_numeric(nav["daily_return"], errors="coerce").dropna()
    final_nav = float(nav["nav"].iloc[-1])
    total_return = final_nav / initial_cash - 1
    exponent = math.log1p(total_return) * 252 / max(len(nav), 1) if total_return > -1 else np.nan
    annual_return = math.expm1(exponent) if pd.notna(exponent) and exponent < 700 else np.inf if pd.notna(exponent) else np.nan
    vol = float(rets.std(ddof=0)) if len(rets) else np.nan
    return {
        "forward_start": pd.Timestamp(nav["date"].iloc[0]).strftime("%Y-%m-%d"),
        "forward_end": pd.Timestamp(nav["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "forward_n_days": int(len(nav)),
        "initial_cash": float(initial_cash),
        "final_nav": final_nav,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(rets.mean() / vol * math.sqrt(252)) if len(rets) and vol > 0 else np.nan,
        "max_drawdown": float(pd.to_numeric(nav["drawdown"], errors="coerce").min()),
        "n_orders": int(len(orders)),
    }


def clear_old(paths: dict[str, Path]) -> None:
    for key in ("result", "nav", "orders", "rejections", "positions", "latest_state", "latest_positions"):
        paths[key].unlink(missing_ok=True)


def process_experiment(
    item: dict[str, Any],
    matrix_root: Path,
    start: pd.Timestamp,
    predictions: pd.DataFrame,
    prediction_meta: dict[str, Any],
    execution: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    v7: Any,
    mode: str,
    preset: str,
    capacity_mode: str,
    participation_rate: float,
) -> dict[str, Any]:
    experiment = item["experiment"]
    root = matrix_root / experiment
    manifest = read_json(root / "global_fold0_to_fold5_forward_manifest.json", {}) or {}
    history_root = planner.resolve_history_root(root, manifest)
    selection = select_corresponding_historical_signal(
        base_root=Path(as1455_paths.TARGET_BACKTEST_ROOT),
        feature_preset=preset,
        target_col=item["target_col"],
        rebalance_every=item["rebalance_every"],
        rank_metric="sharpe",
        explicit_backtest_root=history_root,
    )
    expected = planner.EXPECTED_SIGNAL_SPEC[item["signal"]]
    actual = planner.selection_spec(selection)
    if actual != expected:
        raise RuntimeError(f"fixed signal mismatch for {experiment}: {actual} != {expected}")
    historical_trading_config(selection, item["rebalance_every"])
    history_window = historical_phase_window(selection)
    historical_config, historical_config_path = live.load_historical_run_config(selection)
    initial_cash = resolve_initial_cash(root)
    paths = experiment_tracking_paths(root)

    old_manifest = read_json(paths["manifest"], {}) or {}
    old_state = read_json(paths["latest_state"], {}) or {}
    rebuild = mode == "rebuild" or old_manifest.get("tracking_start_date") != start.strftime("%Y-%m-%d") or not old_state
    old_positions = read_csv(paths["latest_positions"])
    if rebuild:
        lower = start
        cash = initial_cash
        positions = pd.DataFrame()
        previous_asof = None
    else:
        previous_asof = pd.Timestamp(old_state["asof_date"]).normalize()
        lower = previous_asof + pd.Timedelta(days=1)
        cash = float(old_state["cash"])
        positions = old_positions

    pred_dates = pd.DatetimeIndex(predictions.index.get_level_values("date")).normalize().unique().sort_values()
    dates = contiguous_tracking_dates(pred_dates, calendar, lower)
    if previous_asof is not None:
        dates = dates[dates > previous_asof]

    fresh: list[dict[str, pd.DataFrame]] = []
    phase_updates: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    if rebuild and len(dates):
        first_date = pd.Timestamp(dates[0]).normalize()
        first_result, meta = run_chunk(
            v7, selection, historical_config, history_window, predictions, execution, calendar,
            pd.DatetimeIndex([first_date]), initial_cash, pd.DataFrame(), True,
            capacity_mode, participation_rate,
        )
        fresh.append(frames(first_result, first_date))
        final_state = first_result["final_state"]
        phase_updates.append({"bootstrap": True, "dates": [first_date.strftime("%Y-%m-%d")], **meta})
        remaining = dates[dates > first_date]
        if len(remaining):
            result, meta = run_chunk(
                v7, selection, historical_config, history_window, predictions, execution, calendar,
                remaining, float(final_state["cash"]), pd.DataFrame(final_state.get("positions", [])), False,
                capacity_mode, participation_rate,
            )
            fresh.append(frames(result))
            final_state = result["final_state"]
            phase_updates.append({"bootstrap": False, "dates": [remaining[0].strftime("%Y-%m-%d"), remaining[-1].strftime("%Y-%m-%d")], **meta})
    elif not rebuild and len(dates):
        result, meta = run_chunk(
            v7, selection, historical_config, history_window, predictions, execution, calendar,
            dates, cash, positions, False, capacity_mode, participation_rate,
        )
        fresh.append(frames(result))
        final_state = result["final_state"]
        phase_updates.append({"bootstrap": False, "dates": [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")], **meta})

    combined = merge_frames(paths, fresh, rebuild)
    combined["nav"] = recompute_nav(combined["nav"], initial_cash)
    if combined["nav"].empty:
        if rebuild:
            clear_old(paths)
        waiting = {
            "status": "waiting_for_completed_market_day",
            "experiment": experiment,
            "tracking_start_date": start.strftime("%Y-%m-%d"),
            "initial_cash": initial_cash,
            "prediction_source": prediction_meta,
        }
        write_json(paths["manifest"], waiting)
        return {"status": waiting["status"], "experiment": experiment, "tracking_start_date": start.strftime("%Y-%m-%d")}

    for key in ("nav", "orders", "rejections", "positions"):
        atomic_csv(combined[key], paths[key])
    last_date = pd.Timestamp(combined["nav"]["date"].iloc[-1]).normalize()
    if final_state is not None:
        latest_positions = pd.DataFrame(final_state.get("positions", []))
        atomic_csv(latest_positions, paths["latest_positions"])
        state = {
            "status": "ok",
            "experiment": experiment,
            "tracking_start_date": start.strftime("%Y-%m-%d"),
            "effective_start_date": pd.Timestamp(combined["nav"]["date"].iloc[0]).strftime("%Y-%m-%d"),
            "asof_date": last_date.strftime("%Y-%m-%d"),
            "nav": float(combined["nav"]["nav"].iloc[-1]),
            "cash": float(final_state["cash"]),
            "n_positions": int(len(latest_positions)),
            "initial_cash": initial_cash,
            "semantics": "empty_on_tracking_start_then_incremental_close_auction_tracking",
        }
        write_json(paths["latest_state"], state)

    summary = account_summary(combined["nav"], combined["orders"], initial_cash)
    row = {
        **summary,
        "status": "ok",
        "experiment": experiment,
        "tracking_start_date": start.strftime("%Y-%m-%d"),
        "effective_start_date": pd.Timestamp(combined["nav"]["date"].iloc[0]).strftime("%Y-%m-%d"),
        "historical_result_reused": True,
        "fixed_signal_spec": actual,
        "max_positions": int(selection.historical_max_positions),
        "sell_rank": int(selection.historical_sell_rank),
        "rebalance_every": int(selection.historical_rebalance_every),
    }
    atomic_csv(pd.DataFrame([row]), paths["result"])
    write_json(paths["manifest"], {
        "status": "ok",
        "protocol": "as1455_tracking_account_v1",
        "experiment": experiment,
        "tracking_start_date": start.strftime("%Y-%m-%d"),
        "effective_start_date": row["effective_start_date"],
        "asof_date": last_date.strftime("%Y-%m-%d"),
        "update_mode": "rebuild" if rebuild else "incremental",
        "new_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "initial_cash": initial_cash,
        "prediction_source": prediction_meta,
        "historical_selection": selection.to_dict(),
        "historical_trade_config_file": str(historical_config_path),
        "phase_updates": phase_updates,
        "bootstrap_semantics": "first effective day is a forced rebalance from empty; later dates resume the frozen historical phase",
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-root", default="saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1")
    ap.add_argument("--live-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY))
    ap.add_argument("--cache-base", default=str(DEFAULT_CACHE_BASE))
    ap.add_argument("--feature-preset", default="rotation_addon_onehot")
    ap.add_argument("--tracking-start-date", default=None)
    ap.add_argument("--mode", choices=["incremental", "rebuild"], default="incremental")
    ap.add_argument("--capacity-mode", default="none")
    ap.add_argument("--participation-rate", type=float, default=0.05)
    args = ap.parse_args()
    if args.capacity_mode != "none":
        raise SystemExit("tracking accounts require --capacity-mode none")

    matrix_root = Path(args.matrix_root).expanduser().resolve()
    live_root = Path(args.live_root).expanduser().resolve()
    raw_daily = Path(args.raw_daily_cache_dir).expanduser().resolve()
    cache_base = Path(args.cache_base).expanduser().resolve()
    start = pd.Timestamp(args.tracking_start_date).normalize() if args.tracking_start_date else tracking_start_date(matrix_root)
    if start is None:
        raise RuntimeError(f"tracking_start_date is missing under {matrix_root / '.dashboard'}")

    experiments = planner.parse_experiments(matrix_root)
    v7 = live.load_v7_module()
    patch_summary(v7)
    target_predictions: dict[str, pd.DataFrame] = {}
    target_meta: dict[str, dict[str, Any]] = {}
    for target in sorted({item["target"] for item in experiments}):
        target_predictions[target], target_meta[target] = load_predictions(
            cache_base, live_root, args.feature_preset, target, start
        )

    # Execution data are target-independent; build once instead of reopening the
    # same ~1000 raw-daily files separately for r01/r05/r21.
    union = pd.concat(list(target_predictions.values()))
    union = union[~union.index.duplicated(keep="last")].sort_index()
    execution, calendar = execution_inputs(v7, union, raw_daily)
    print(f"[EXECUTION] built once symbols={union.index.get_level_values('symbol').nunique()} end={calendar[-1]:%Y-%m-%d}")

    rows = []
    for item in experiments:
        row = process_experiment(
            item, matrix_root, start, target_predictions[item["target"]], target_meta[item["target"]],
            execution, calendar, v7, args.mode, args.feature_preset, args.capacity_mode, args.participation_rate,
        )
        rows.append(row)
        print(f"[TRACK] {item['experiment']} status={row.get('status')} end={row.get('forward_end')}")

    summary = pd.DataFrame(rows)
    summary_file = matrix_root / TRACKING_MATRIX_SUMMARY
    atomic_csv(summary, summary_file)
    ok = summary.loc[summary["status"].astype(str).eq("ok")] if "status" in summary.columns else summary
    manifest = {
        "status": "ok" if len(ok) == 9 else "partial",
        "protocol": "as1455_tracking_matrix_v1",
        "tracking_start_date": start.strftime("%Y-%m-%d"),
        "experiment_count": int(len(summary)),
        "completed_experiment_count": int(len(ok)),
        "asof_dates": sorted(set(ok["forward_end"].dropna().astype(str))) if "forward_end" in ok else [],
        "mode": args.mode,
        "summary_file": str(summary_file),
        "historical_fold_grid_recomputed": False,
        "canonical_strict_forward_recomputed": False,
        "daily_update_semantics": "append only new completed market dates; rebuild only when tracking start changes",
    }
    write_json(matrix_root / TRACKING_MATRIX_MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
