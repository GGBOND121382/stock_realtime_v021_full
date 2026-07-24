#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module(
    "as1455_r05_addon_fold_comparison_legacy",
    PROJECT_DIR / "scripts" / "run_as1455_r05_addon_fold_comparison.py",
)
helpers = legacy.helpers
bt = legacy.bt
TARGET = legacy.TARGET
PRESET = legacy.PRESET
FOLDS = legacy.EXPECTED_TARGET_FOLDS


def load_run(payload: dict[str, Any]) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    nav_path = Path(payload["nav_file"]).resolve()
    return nav_path.parent, legacy.read_json(Path(payload["config_file"]).resolve()), load_nav(nav_path)


def load_nav(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"date", "nav"}.issubset(frame.columns):
        raise RuntimeError(f"NAV lacks date/nav columns: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame.dropna(subset=["date", "nav"]).sort_values("date").drop_duplicates("date", keep="last")
    if len(frame) < 2:
        raise RuntimeError(f"NAV has fewer than two rows: {path}")
    return frame.reset_index(drop=True)


def assert_configs_compatible(history: dict[str, Any], forward: dict[str, Any]) -> None:
    ignored = {"initial_cash", "rebalance_offset", "output_mode", "grid_engine"}
    mismatch = {}
    for field in (item.name for item in dataclasses.fields(bt.TradeConfig)):
        if field in ignored:
            continue
        if str(history.get(field)) != str(forward.get(field)):
            mismatch[field] = (history.get(field), forward.get(field))
    if mismatch:
        raise RuntimeError(f"history/forward frozen configs differ: {mismatch}")


def assert_nav_parity(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    actual_dates = legacy.normalize_dates(actual["date"])
    expected_dates = legacy.normalize_dates(expected["date"])
    if not actual_dates.equals(expected_dates):
        raise RuntimeError(
            "historical rerun/materialized date mismatch: "
            f"missing={list(map(str, expected_dates.difference(actual_dates)[:10]))} "
            f"extra={list(map(str, actual_dates.difference(expected_dates)[:10]))}"
        )
    merged = expected[["date", "nav"]].merge(actual[["date", "nav"]], on="date", suffixes=("_expected", "_actual"))
    diff = (merged["nav_actual"] - merged["nav_expected"]).abs()
    max_abs = float(diff.max()) if len(diff) else 0.0
    max_rel = float((diff / merged["nav_expected"].abs().clip(lower=1.0)).max()) if len(diff) else 0.0
    if max_abs > 1e-5 and max_rel > 1e-10:
        raise RuntimeError(f"historical materialized NAV parity failed: max_abs={max_abs} max_rel={max_rel}")
    return {"passed": True, "rows": int(len(merged)), "max_abs_nav_difference": max_abs, "max_relative_nav_difference": max_rel}


def with_phase(frame: pd.DataFrame, segment: str, global_start: int) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["phase_segment"] = segment
    if "day_index" in out.columns:
        out["engine_day_index"] = pd.to_numeric(out["day_index"], errors="coerce")
        out["global_day_index"] = out["engine_day_index"] + int(global_start)
    return out


def concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(valid, ignore_index=True, sort=False) if valid else pd.DataFrame()


def bridge_state(
    initial_state: dict[str, Any],
    bridge_dates: pd.DatetimeIndex,
    execution: pd.DataFrame,
    actions: pd.DataFrame,
    cfg: Any,
    previous_nav: float,
    global_start: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cash = float(initial_state["cash"])
    positions = bt.normalize_initial_positions(initial_state.get("positions", []))
    exec_by_date = {date: group.set_index("symbol", drop=False) for date, group in execution.groupby("date", sort=True)}
    nav_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    last_nav = float(previous_nav)

    for local_index, date in enumerate(bridge_dates):
        exec_t = exec_by_date.get(date)
        if exec_t is None:
            raise RuntimeError(f"bridge date missing execution panel: {date:%Y-%m-%d}")
        cash_delta, rows = bt.apply_corporate_actions_for_date(date, positions, exec_t, actions, cfg)
        cash += float(cash_delta)
        action_rows.extend(rows)
        missing = [symbol for symbol in positions if symbol not in exec_t.index]
        if missing:
            raise RuntimeError(f"bridge date {date:%Y-%m-%d} missing held-symbol marks: {missing[:20]}")
        values = {}
        for symbol, position in positions.items():
            price = float(exec_t.loc[symbol, "raw_close_1500"])
            if not np.isfinite(price) or price <= 0:
                raise RuntimeError(f"invalid bridge mark {date:%Y-%m-%d} {symbol}: {price}")
            values[symbol] = float(position["shares"]) * price
        holding_value = float(sum(values.values()))
        nav = cash + holding_value
        global_index = int(global_start) + local_index
        nav_rows.append({
            "date": date, "day_index": global_index, "engine_day_index": np.nan,
            "global_day_index": global_index, "phase_segment": "bridge_no_prediction",
            "signal_available": False, "trade_skipped_reason": "no_prediction_bridge_day",
            "is_rebalance_day": bool(bt.is_rebalance_day_index(global_index, cfg)),
            "nav": nav, "daily_return": nav / last_nav - 1.0 if last_nav > 0 else np.nan,
            "cash": cash, "cash_ratio": cash / nav if nav > 0 else np.nan,
            "holding_value": holding_value, "gross_exposure": holding_value / nav if nav > 0 else np.nan,
            "n_positions": len(positions), "turnover": 0.0, "gross_trade_amount": 0.0,
            "total_fee": 0.0, "nav_before_trade": nav, "corporate_action_cash_delta": cash_delta,
            "orders": 0, "buy_orders": 0, "sell_orders": 0, "partial_fill_orders": 0,
            "rejections": 0, "missing_marks": "", "max_positions": cfg.max_positions,
            "buy_candidate_rank": cfg.buy_candidate_rank, "sell_rank": cfg.sell_rank,
            "rebalance_every": cfg.rebalance_every, "rebalance_offset": cfg.rebalance_offset,
        })
        for symbol, value in sorted(values.items()):
            position = positions[symbol]
            position_rows.append({
                "date": date, "symbol": symbol, "rank": np.nan, "score": np.nan,
                "shares": float(position.get("shares", 0.0)),
                "raw_close_1500": float(exec_t.loc[symbol, "raw_close_1500"]),
                "value": value, "weight": value / nav if nav > 0 else np.nan,
                "buy_date": position.get("buy_date"),
                "holding_days": int((pd.Timestamp(date) - pd.Timestamp(position.get("buy_date"))).days),
                "entry_rank": position.get("entry_rank", np.nan), "entry_score": position.get("entry_score", np.nan),
                "avg_entry_price": position.get("avg_entry_price", np.nan),
                "cost_basis_notional": position.get("cost_basis_notional", np.nan),
                "cost_basis_fee": position.get("cost_basis_fee", np.nan),
                "max_positions": cfg.max_positions, "buy_candidate_rank": cfg.buy_candidate_rank,
                "sell_rank": cfg.sell_rank, "rebalance_every": cfg.rebalance_every,
                "rebalance_offset": cfg.rebalance_offset, "global_day_index": global_index,
                "phase_segment": "bridge_no_prediction",
            })
        last_nav = nav

    return (
        {"cash": float(cash), "positions": bt.serialize_position_state(positions), "n_bridge_dates": int(len(bridge_dates))},
        pd.DataFrame(nav_rows), pd.DataFrame(position_rows), pd.DataFrame(action_rows),
    )


def combine_results(
    history: dict[str, Any],
    forward: dict[str, Any],
    bridge_nav: pd.DataFrame,
    bridge_positions: pd.DataFrame,
    bridge_actions: pd.DataFrame,
    history_cfg: Any,
) -> dict[str, Any]:
    history_days = len(history["nav"])
    forward_start = history_days + len(bridge_nav)
    hnav = with_phase(history["nav"], "historical_one_lag", 0)
    hnav["signal_available"], hnav["trade_skipped_reason"] = True, ""
    fnav = with_phase(forward["nav"], "strict_oos_forward", forward_start)
    fnav["signal_available"], fnav["trade_skipped_reason"] = True, ""
    nav = concat_frames(hnav, bridge_nav, fnav).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    nav["daily_return"] = nav["nav"].pct_change()
    nav.loc[0, "daily_return"] = float(nav.loc[0, "nav"]) / float(history_cfg.initial_cash) - 1.0

    orders = concat_frames(with_phase(history["orders"], "historical_one_lag", 0), with_phase(forward["orders"], "strict_oos_forward", forward_start))
    rejections = concat_frames(with_phase(history["rejections"], "historical_one_lag", 0), with_phase(forward["rejections"], "strict_oos_forward", forward_start))
    positions = concat_frames(with_phase(history["positions"], "historical_one_lag", 0), bridge_positions, with_phase(forward["positions"], "strict_oos_forward", forward_start))

    hactions = history["corporate_actions"].copy()
    factions = forward["corporate_actions"].copy()
    if not hactions.empty: hactions["phase_segment"] = "historical_one_lag"
    if not bridge_actions.empty: bridge_actions["phase_segment"] = "bridge_no_prediction"
    if not factions.empty: factions["phase_segment"] = "strict_oos_forward"
    actions = concat_frames(hactions, bridge_actions, factions)

    htrips = history["round_trips"].copy()
    ftrips = forward["round_trips"].copy()
    if not htrips.empty: htrips["phase_segment"] = "historical_one_lag"
    if not ftrips.empty:
        ftrips["round_trip_id"] = pd.to_numeric(ftrips["round_trip_id"], errors="coerce") + len(htrips)
        ftrips["phase_segment"] = "strict_oos_forward"
    trips = concat_frames(htrips, ftrips)

    drawdown = bt.build_daily_drawdown(nav)
    result = {
        "nav": nav, "orders": orders, "trades": orders.copy(), "rejections": rejections,
        "positions": positions, "corporate_actions": actions, "round_trips": trips,
        "daily_drawdown": drawdown, "monthly_summary": bt.build_period_summary(nav, "M"),
        "yearly_summary": bt.build_period_summary(nav, "Y"),
        "fee_summary": bt.build_fee_summary(orders, history_cfg),
        "turnover_summary": bt.build_turnover_summary(nav, orders),
    }
    result["summary"] = bt.summarize_nav(nav, orders, rejections, history_cfg, actions, trips, drawdown)
    result["final_state"] = {
        **forward["final_state"], "global_day_index_end": len(nav) - 1,
        "historical_days": history_days, "bridge_days": len(bridge_nav), "forward_days": len(forward["nav"]),
    }
    return result


def write_result(run_dir: Path, result: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any], mode: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    selected, _ = helpers.output_frames(result, mode)
    for name, frame in selected.items():
        frame.to_csv(run_dir / name, index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=bt.json_default), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=bt.json_default), encoding="utf-8")
    (run_dir / "close_auction_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=bt.json_default), encoding="utf-8")


def segment_table(nav: pd.DataFrame, items: list[tuple[str, pd.DatetimeIndex, int | None, int | None]]) -> pd.DataFrame:
    rows = []
    ordered = nav.sort_values("date").reset_index(drop=True)
    for label, dates, target_fold, source_fold in items:
        part = ordered[ordered["date"].isin(dates)]
        if len(part) != len(dates):
            raise RuntimeError(f"segment coverage mismatch {label}: expected={len(dates)} actual={len(part)}")
        first = int(part.index.min())
        before = float(ordered.loc[first - 1, "nav"]) if first else float(part.iloc[0]["nav"] / (1 + part.iloc[0]["daily_return"]))
        rows.append({
            "segment": label, "target_fold": target_fold, "source_model_fold": source_fold,
            "start_date": part.iloc[0]["date"], "end_date": part.iloc[-1]["date"], "n_days": len(part),
            "start_nav_before_segment": before, "end_nav": float(part.iloc[-1]["nav"]),
            "segment_return": float(part.iloc[-1]["nav"] / before - 1.0),
            "end_positions": int(part.iloc[-1].get("n_positions", 0)), "end_cash": float(part.iloc[-1].get("cash", np.nan)),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r05 addon historical folds plus strict-OOS forward comparison")
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--initial-cash", type=float, default=200000.0)
    parser.add_argument("--output-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--raw-daily-cache-dir", default=None)
    parser.add_argument("--raw-5m-cache-dir", default=None)
    parser.add_argument("--last5-panel", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--st-symbols", default=None)
    parser.add_argument("--st-status", default=None)
    parser.add_argument("--corporate-actions", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    pair_path = Path(args.pair_manifest).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pair = legacy.load_single_pair(pair_path)
    historical, forward = pair["historical"], pair["forward"]
    mapping = legacy.mapping_by_target_fold(historical["fold_mapping"])
    hrun, hconfig, hnav_saved = load_run(historical)
    frun, fconfig, fnav_saved = load_run(forward)
    assert_configs_compatible(hconfig, fconfig)

    meta_path = hrun / "close_auction_summary.json"
    meta = legacy.read_json(meta_path) if meta_path.is_file() else {}
    raw_daily = legacy.choose_path(args.raw_daily_cache_dir, meta.get("raw_daily_cache_dir"), required=True, label="raw daily cache")
    raw_5m = legacy.choose_path(args.raw_5m_cache_dir, meta.get("raw_5m_cache_dir"), required=False, label="raw 5m cache")
    last5_path = legacy.choose_path(args.last5_panel, meta.get("last5_panel"), required=False, label="last5 panel")
    universe_path = legacy.choose_path(args.universe, meta.get("universe"), required=False, label="universe")
    st_symbols_path = legacy.choose_path(args.st_symbols, meta.get("st_symbols"), required=False, label="static ST symbols")
    st_status_path = legacy.choose_path(args.st_status, meta.get("st_status"), required=False, label="historical ST status")
    actions_path = legacy.choose_path(args.corporate_actions, meta.get("corporate_actions"), required=False, label="corporate actions")

    selection = historical["selection"]
    hp = helpers.load_selected_predictions(Path(historical["prediction_file"]).resolve(), selection)
    fp = helpers.load_selected_predictions(Path(forward["prediction_file"]).resolve(), selection)
    hdates, fdates = legacy.normalize_dates(hp["date"]), legacy.normalize_dates(fp["date"])
    if hdates[-1] >= fdates[0]:
        raise RuntimeError(f"historical/forward windows overlap: {hdates[-1]} >= {fdates[0]}")
    symbols = sorted(set(hp["symbol"].astype(str)).union(fp["symbol"].astype(str)))

    universe = bt.read_universe(universe_path)
    st_symbols = bt.load_st_symbols(st_symbols_path)
    st_status = bt.load_st_status(st_status_path)
    last5 = bt.load_last5_panel(last5_path)
    actions = bt.load_corporate_actions(actions_path)
    capacity_modes = {str(hconfig["capacity_mode"]), str(fconfig["capacity_mode"])}
    if all(mode == "none" for mode in capacity_modes):
        raw_5m, last5 = None, pd.DataFrame()
    elif raw_5m is None and last5.empty:
        raise RuntimeError(f"capacity modes require raw 5m or last5 panel: {capacity_modes}")

    execution, execution_report = bt.build_execution_panel(
        symbols, raw_daily, universe, st_symbols, st_status=st_status,
        last5_panel=last5, raw_5m_cache_dir=raw_5m,
    )
    execution_report.to_csv(out_root / "execution_data_report.csv", index=False, encoding="utf-8-sig")
    edates = legacy.normalize_dates(execution["date"])
    audit, fold_dates = legacy.build_boundary_audit(mapping, hdates, edates)
    expected_hdates = legacy.normalize_dates([date for fold in FOLDS for date in fold_dates[fold]])
    if not expected_hdates.equals(legacy.normalize_dates(hnav_saved["date"])):
        raise RuntimeError("historical materialized NAV does not cover exactly target_fold5..target_fold0")
    if not fdates.equals(fdates.intersection(edates)):
        raise RuntimeError(f"forward predictions missing execution dates: {list(fdates.difference(edates)[:20])}")
    if not fdates.equals(legacy.normalize_dates(fnav_saved["date"])):
        raise RuntimeError("retained forward NAV dates do not match forward prediction dates")

    bridge_dates = edates[(edates > hdates[-1]) & (edates < fdates[0])]
    forward_row = {
        "segment": "strict_oos_forward", "target_fold": None, "source_model_fold": 0,
        "manifest_start": fdates[0], "manifest_end": fdates[-1], "actual_start": fdates[0], "actual_end": fdates[-1],
        "prediction_days": len(fdates), "previous_end": hdates[-1],
        "calendar_gap_days": max(0, int((fdates[0] - hdates[-1]).days - 1)), "trading_gap_days": len(bridge_dates),
    }
    pd.concat([audit, pd.DataFrame([forward_row])], ignore_index=True).to_csv(out_root / "fold_boundary_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": bridge_dates}).to_csv(out_root / "forward_bridge_execution_dates.csv", index=False, encoding="utf-8-sig")

    comparison, records, curves = [], [], []
    original_offset = int(hconfig["rebalance_offset"])
    for fold in FOLDS:
        dates = fold_dates[fold]
        offset, skipped = helpers.effective_offset_for_crop(full_dates=hdates, crop_dates=dates, original_offset=original_offset, rebalance_every=int(hconfig["rebalance_every"]))
        cfg = helpers.config_to_trade_config(hconfig, initial_cash=args.initial_cash, effective_offset=offset)
        pred = hp[hp["date"].isin(dates)]
        exec_part = execution[execution["date"].isin(dates)]
        precheck = helpers.capacity_precheck(exec_part, pred, str(cfg.capacity_mode))
        result = bt.backtest(pred, exec_part, cfg, corporate_actions=actions)
        run_dir = out_root / "per_fold" / f"target_fold{fold}"
        manifest = {
            "result_type": "independent_fold", "target_col": TARGET, "feature_preset": PRESET,
            "target_fold": fold, "source_model_fold": int(mapping[fold]["source_fold"]),
            "initial_cash": args.initial_cash, "start_date": dates[0], "end_date": dates[-1],
            "original_rebalance_offset": original_offset, "effective_local_rebalance_offset": offset,
            "skipped_dates_before_fold": skipped, "capacity_precheck": precheck,
            "source_prediction_file": historical["prediction_file"], "source_config_file": historical["config_file"],
        }
        helpers.write_independent_run(run_dir=run_dir, result=result, cfg=cfg, output_mode=args.output_mode, manifest=manifest)
        curves.append({"label": f"target_fold{fold} (source_model_fold{mapping[fold]['source_fold']})", "run_name": f"target_fold{fold}", "curve": helpers.curve_from_result(result, args.initial_cash)})
        comparison.append({"result_type": "independent_fold", "target_fold": fold, "source_model_fold": int(mapping[fold]["source_fold"]), **legacy.scalar_summary(result["summary"])})
        records.append({**manifest, "run_dir": str(run_dir), "summary": result["summary"]})
        print(f"[OK] target_fold{fold} {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d} offset={original_offset}->{offset}")
    legacy.plot_curves(curves, out_root / "per_fold" / "plots", "r05 addon independent historical folds")

    hist_root = out_root / "cross_fold_historical"
    copied_h = legacy.copy_materialized_run(hrun, hist_root / "materialized_run")
    hnav_saved.to_csv(hist_root / "continuous_nav.csv", index=False, encoding="utf-8-sig")
    legacy.plot_curves([{"label": "historical continuous account", "run_name": historical["run_name"], "curve": legacy.curve_from_nav(hnav_saved, float(hconfig.get("initial_cash", args.initial_cash)))}], hist_root / "plots", "r05 addon historical continuous")
    comparison.append({"result_type": "continuous_cross_fold_historical", "target_fold": None, "source_model_fold": None, **legacy.scalar_summary(legacy.read_json(hrun / "summary.json"))})

    forward_root = out_root / "forward_strict_oos"
    copied_f = legacy.copy_materialized_run(frun, forward_root / "retained_run")
    fnav_saved.to_csv(forward_root / "forward_nav.csv", index=False, encoding="utf-8-sig")
    legacy.plot_curves([{"label": "fold0 strict-OOS forward (independent account)", "run_name": forward["run_name"], "curve": legacy.curve_from_nav(fnav_saved, float(fconfig.get("initial_cash", args.initial_cash)))}], forward_root / "plots", "r05 addon fold0 strict-OOS forward")
    comparison.append({"result_type": "independent_strict_oos_forward", "target_fold": "after_fold0", "source_model_fold": 0, **legacy.scalar_summary(legacy.read_json(frun / "summary.json"))})

    hcfg = helpers.config_to_trade_config(hconfig, initial_cash=args.initial_cash, effective_offset=int(hconfig["rebalance_offset"]))
    hpre = helpers.capacity_precheck(execution[execution["date"].isin(hdates)], hp, str(hcfg.capacity_mode))
    hresult = bt.backtest(hp, execution[execution["date"].isin(hdates)], hcfg, corporate_actions=actions)
    parity = assert_nav_parity(hresult["nav"], hnav_saved)
    state, bnav, bpos, bactions = bridge_state(hresult["final_state"], bridge_dates, execution, actions, hcfg, float(hresult["nav"].iloc[-1]["nav"]), len(hresult["nav"]))

    fcfg = helpers.config_to_trade_config(fconfig, initial_cash=float(state["cash"]), effective_offset=int(fconfig["rebalance_offset"]))
    fpre = helpers.capacity_precheck(execution[execution["date"].isin(fdates)], fp, str(fcfg.capacity_mode))
    fresult = bt.backtest(fp, execution[execution["date"].isin(fdates)], fcfg, corporate_actions=actions, initial_positions=state["positions"], day_index_start=0)
    combined = combine_results(hresult, fresult, bnav, bpos, bactions, hcfg)
    combined_root = out_root / "cross_fold_historical_plus_forward"
    combined_manifest = {
        "result_type": "continuous_historical_plus_strict_oos_forward", "target_col": TARGET, "feature_preset": PRESET,
        "historical_prediction_file": historical["prediction_file"], "forward_prediction_file": forward["prediction_file"],
        "historical_materialized_parity": parity, "historical_capacity_precheck": hpre, "forward_capacity_precheck": fpre,
        "historical_start": hdates[0], "historical_end": hdates[-1], "bridge_dates": bridge_dates.tolist(),
        "forward_start": fdates[0], "forward_end": fdates[-1], "bridge_policy": "mark_and_apply_corporate_actions_no_prediction_no_trade",
        "prediction_generation": False, "parameter_grid": False, "training": False, "data_refresh": False,
    }
    combined_config = {
        "target_col": TARGET, "feature_preset": PRESET, "signal": selection,
        "historical_trade_config": hconfig, "forward_trade_config": fconfig,
        "initial_cash": args.initial_cash, "account_state_continuous": True,
    }
    write_result(combined_root, combined, combined_config, combined_manifest, args.output_mode)
    combined["nav"].to_csv(combined_root / "continuous_nav.csv", index=False, encoding="utf-8-sig")
    legacy.plot_curves([{"label": "historical folds + strict-OOS forward", "run_name": "continuous_historical_plus_forward", "curve": legacy.curve_from_nav(combined["nav"], args.initial_cash)}], combined_root / "plots", "r05 addon continuous historical plus forward")
    segment_items = [(f"target_fold{fold}", fold_dates[fold], fold, int(mapping[fold]["source_fold"])) for fold in FOLDS]
    if len(bridge_dates): segment_items.append(("bridge_no_prediction", bridge_dates, None, None))
    segment_items.append(("strict_oos_forward", fdates, None, 0))
    segment_table(combined["nav"], segment_items).to_csv(combined_root / "continuous_segments.csv", index=False, encoding="utf-8-sig")
    comparison.append({"result_type": "continuous_historical_plus_forward", "target_fold": None, "source_model_fold": None, **legacy.scalar_summary(combined["summary"])})

    pd.DataFrame(comparison).to_csv(out_root / "r05_addon_backtest_comparison.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "mode": "r05_addon_historical_forward_complete", "target_col": TARGET, "feature_preset": PRESET,
        "prediction_generation": False, "parameter_grid": False, "training": False, "data_refresh": False,
        "execution_panel_build_count": 1, "continuous_engine_calls": 2,
        "independent_historical_backtest_count": len(records), "bridge_execution_days": len(bridge_dates),
        "historical_materialized_parity": parity, "copied_historical_files": copied_h, "copied_forward_files": copied_f,
        "historical_end": hdates[-1], "forward_start": fdates[0], "forward_end": fdates[-1],
        "per_fold_runs": records, "duration_seconds": int(round(time.time() - started)),
    }
    manifest["all_ok"] = bool(len(records) == 6 and parity["passed"] and len(copied_h) >= 3 and len(copied_f) >= 3 and pd.Timestamp(combined["nav"].iloc[-1]["date"]) == fdates[-1])
    manifest_path = out_root / "r05_addon_fold_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=bt.json_default), encoding="utf-8")
    report = {
        "mode": manifest["mode"], "historical_end": hdates[-1].strftime("%Y-%m-%d"),
        "forward_start": fdates[0].strftime("%Y-%m-%d"), "forward_end": fdates[-1].strftime("%Y-%m-%d"),
        "bridge_execution_days": len(bridge_dates), "duration_seconds": manifest["duration_seconds"],
        "all_ok": manifest["all_ok"], "output_root": str(out_root), "manifest": str(manifest_path),
    }
    (out_root / "r05_addon_fold_comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not manifest["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
