#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the clean AS1455 strict-OOS strategy for one live/replay date.

The restored master fast path supplies today's 31 base features and execution
snapshot. This layer intentionally does *not* restore the legacy 31-feature,
seven-fold-average deploy bundle or its second trading loop. It reuses the clean
rotation/addon feature builders, fold0 search checkpoints, historical best full
run, continuous rebalance phase, and the canonical v7 trading engine.

Outputs are reviewable planned orders only. No broker API is called and no
planned order is persisted as account truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_live_inference import build_inference_features_from_frame  # noqa: E402
from utils.as1455_model_selection import HistoricalSignalSelection, select_corresponding_historical_signal  # noqa: E402
from utils.as1455_rebalance_phase import align_forward_rebalance_phase  # noqa: E402
from utils.as1455_strict_oos import historical_phase_window, historical_trading_config  # noqa: E402

BASE_FEATURE_COLUMNS = [
    column for column in common.base.EXPECTED_MODEL_COLUMNS
    if column not in common.base.EXPECTED_OUTCOMES
]
V7_PATH = PROJECT_DIR / "code" / "backtest" / "run_as1455_close_auction_backtest_v7_maxpos_grid.py"


def load_v7_module():
    spec = importlib.util.spec_from_file_location("as1455_v7_live", V7_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v7 engine: {V7_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def code6(value: object) -> str:
    match = re.search(r"(\d{6})", str(value))
    if not match:
        raise ValueError(f"cannot normalize stock symbol: {value!r}")
    return match.group(1)


def exchange_symbol(value: object) -> str:
    code = code6(value)
    return f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"


def normalize_like_history(value: object, historical_symbols: pd.Index) -> str:
    code = code6(value)
    sample = next((str(x) for x in historical_symbols if str(x)), code)
    return exchange_symbol(code) if (".SH" in sample or ".SZ" in sample) else code


def parse_trade_date(value: str) -> pd.Timestamp:
    if value.lower() == "today":
        return pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        raise ValueError(f"bad trade date: {value!r}")
    return pd.to_datetime(digits, format="%Y%m%d").normalize()


def load_live_base_features(feature_file: Path, historical_symbols: pd.Index, trade_date: pd.Timestamp) -> pd.DataFrame:
    if not feature_file.exists():
        raise FileNotFoundError(feature_file)
    live = pd.read_csv(feature_file, dtype={"symbol": str}, encoding="utf-8-sig")
    missing = [column for column in ["symbol", *BASE_FEATURE_COLUMNS] if column not in live.columns]
    if missing:
        raise RuntimeError(f"live feature file missing columns: {missing}")
    live = live.copy()
    live["symbol"] = live["symbol"].map(lambda value: normalize_like_history(value, historical_symbols))
    live["date"] = trade_date
    for column in BASE_FEATURE_COLUMNS:
        live[column] = pd.to_numeric(live[column], errors="coerce")
    live.replace([np.inf, -np.inf], np.nan, inplace=True)
    live = live.dropna(subset=BASE_FEATURE_COLUMNS)
    if live.empty:
        raise RuntimeError("no complete live base-feature rows")
    if live["symbol"].duplicated().any():
        raise RuntimeError(f"duplicate live symbols: {live.loc[live['symbol'].duplicated(), 'symbol'].head().tolist()}")
    frame = live.set_index(["symbol", "date"])[BASE_FEATURE_COLUMNS]
    for outcome in common.base.EXPECTED_OUTCOMES:
        frame[outcome] = np.nan
    return frame[common.base.EXPECTED_MODEL_COLUMNS].sort_index()


def assemble_inference_frame(model_data_path: Path, live_base: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_hdf(model_data_path, "model_data")
    if list(historical.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected model_data index: {historical.index.names}")
    if list(historical.columns) != common.base.EXPECTED_MODEL_COLUMNS:
        raise RuntimeError("model_data column contract mismatch")
    historical = historical.copy()
    historical.index = pd.MultiIndex.from_arrays(
        [historical.index.get_level_values("symbol").astype(str), pd.to_datetime(historical.index.get_level_values("date")).normalize()],
        names=["symbol", "date"],
    )
    combined = pd.concat([historical, live_base], axis=0)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def score_predictions(predictions: pd.DataFrame, selection: HistoricalSignalSelection) -> pd.Series:
    columns = [int(token) for token in selection.signal_cols.split(",")]
    missing = [column for column in columns if column not in predictions.columns]
    if missing:
        raise RuntimeError(f"selected signal requires unavailable columns: {missing}")
    if selection.signal_mode == "single":
        score = predictions[columns[0]]
    elif selection.signal_mode == "mean":
        score = predictions[columns].mean(axis=1)
    else:
        raise RuntimeError(f"unsupported historical signal mode: {selection.signal_mode}")
    if not np.isfinite(score.to_numpy(dtype=float)).all() or score.nunique(dropna=True) <= 1:
        raise RuntimeError("selected live score is non-finite or constant")
    return score.rename("score")


def validate_live_capacity_mode(capacity_mode: str) -> None:
    if capacity_mode != "none":
        raise RuntimeError(
            "live 14:55 snapshot has no complete 14:55-15:00 capacity data; "
            "use --capacity-mode none"
        )


def load_cash(args: argparse.Namespace) -> float:
    if args.cash is not None:
        cash = float(args.cash)
    elif args.cash_file:
        cash = float(Path(args.cash_file).read_text(encoding="utf-8").strip())
    else:
        raise RuntimeError("live planning requires --cash or --cash-file")
    if not math.isfinite(cash) or cash < 0:
        raise RuntimeError(f"invalid cash balance: {cash}")
    return cash


def load_positions(path: Path, allow_missing_buy_date: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
    if "symbol" not in frame.columns:
        if "code" in frame.columns:
            frame["symbol"] = frame["code"]
        else:
            raise RuntimeError("positions file requires symbol or code")
    if "shares" not in frame.columns:
        raise RuntimeError("positions file requires shares")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(exchange_symbol)
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame = frame[frame["shares"].fillna(0).gt(0)].drop_duplicates("symbol", keep="last")
    buy_col = next((c for c in ["buy_date", "entry_date", "date_bought", "open_date"] if c in frame.columns), None)
    if buy_col is None:
        if not allow_missing_buy_date and not frame.empty:
            raise RuntimeError("positions file lacks buy_date; T+1 cannot be enforced")
        frame["buy_date"] = "1900-01-01"
    else:
        frame["buy_date"] = pd.to_datetime(frame[buy_col], errors="coerce").dt.normalize()
        if frame["buy_date"].isna().any():
            if not allow_missing_buy_date:
                bad = frame.loc[frame["buy_date"].isna(), "symbol"].head().tolist()
                raise RuntimeError(f"positions lack valid buy_date; T+1 unsafe: {bad}")
            frame["buy_date"] = frame["buy_date"].fillna(pd.Timestamp("1900-01-01"))
    if "avg_entry_price" not in frame.columns:
        source = next((c for c in ["cost_price", "entry_price", "price"] if c in frame.columns), None)
        frame["avg_entry_price"] = pd.to_numeric(frame[source], errors="coerce") if source else np.nan
    return frame


def load_execution_panel(sidecar_path: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    if not sidecar_path.exists():
        raise FileNotFoundError(sidecar_path)
    frame = pd.read_csv(sidecar_path, dtype={"symbol": str}, encoding="utf-8-sig")
    if "symbol" not in frame.columns:
        raise RuntimeError("execution sidecar requires symbol")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(exchange_symbol)
    frame["date"] = trade_date
    required_defaults = {
        "raw_close_1500": np.nan,
        "qfq_close_1500": np.nan,
        "raw_preclose": np.nan,
        "prev_raw_close_1500": np.nan,
        "event_ratio": 1.0,
        "tradable": True,
        "is_st": False,
        "is_mainboard": False,
        "up_limit": np.nan,
        "down_limit": np.nan,
        "last5_volume": np.nan,
        "last5_amount": np.nan,
    }
    for column, default in required_defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["raw_close_1500"] = pd.to_numeric(frame["raw_close_1500"], errors="coerce")
    frame["qfq_close_1500"] = pd.to_numeric(frame["qfq_close_1500"], errors="coerce").fillna(frame["raw_close_1500"])
    for column in ["raw_preclose", "prev_raw_close_1500", "event_ratio", "up_limit", "down_limit", "last5_volume", "last5_amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["tradable", "is_st", "is_mainboard"]:
        frame[column] = frame[column].astype(str).str.lower().isin(["1", "true", "yes", "y", "t", "ok"])
    frame = frame.drop_duplicates("symbol", keep="last")
    if frame["raw_close_1500"].isna().any():
        bad = frame.loc[frame["raw_close_1500"].isna(), "symbol"].head().tolist()
        raise RuntimeError(f"execution sidecar has missing prices: {bad}")
    return frame[["date", "symbol", *required_defaults.keys()]]



def load_execution_calendar(path: Path, trade_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Load the raw-daily execution calendar prepared before the live window.

    The calendar is the union of dates observed in the raw-daily cache, matching
    the strict-OOS grid's execution-calendar semantics. The current live date is
    appended because T-day raw daily has not closed when the 14:55 plan is made.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"execution calendar is missing: {path}; rerun the live pre stage"
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in frame.columns:
        raise RuntimeError(f"execution calendar requires a date column: {path}")
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame["date"], errors="coerce").dropna()
    ).normalize().unique().sort_values()
    if dates.empty:
        raise RuntimeError(f"execution calendar is empty: {path}")
    if trade_date not in dates:
        dates = dates.append(pd.DatetimeIndex([trade_date])).unique().sort_values()
    return dates


def load_historical_run_config(selection: HistoricalSignalSelection) -> tuple[dict[str, Any], Path]:
    summary_file = Path(selection.summary_file)
    grid_root = summary_file.parents[1]
    path = grid_root / "01_runs" / selection.run_name / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"selected historical run config is missing: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "max_positions": selection.historical_max_positions,
        "sell_rank": selection.historical_sell_rank,
        "rebalance_every": selection.historical_rebalance_every,
        "rebalance_offset": selection.historical_rebalance_offset,
        "signal_name": selection.signal_name,
        "signal_cols": selection.signal_cols,
        "signal_mode": selection.signal_mode,
    }
    for key, value in expected.items():
        if value is None or key not in config:
            continue
        actual = str(config[key]).replace(".0", "")
        wanted = str(value).replace(".0", "")
        if actual != wanted:
            raise RuntimeError(f"historical config mismatch for {key}: config={actual} selection={wanted}")
    return config, path


def current_marked_nav(cash: float, positions: pd.DataFrame, execution: pd.DataFrame) -> float:
    prices = execution.set_index("symbol")["raw_close_1500"].to_dict()
    missing = sorted(set(positions["symbol"]).difference(prices))
    if missing:
        raise RuntimeError(f"current positions missing live execution marks: {missing[:20]}")
    return cash + sum(float(row["shares"]) * float(prices[row["symbol"]]) for _, row in positions.iterrows())


def config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return default


def build_trade_config(v7, selection: HistoricalSignalSelection, historical_config: dict[str, Any], phase: dict[str, Any], cash: float, capacity_mode: str, participation_rate: float):
    # max/sell/every are authoritative in the selected summary; all other
    # execution semantics come from the selected run config.
    buy_candidate_rank = int(historical_config.get("buy_candidate_rank", selection.historical_sell_rank))
    return v7.TradeConfig(
        max_positions=int(selection.historical_max_positions),
        buy_candidate_rank=buy_candidate_rank,
        sell_rank=int(selection.historical_sell_rank),
        rebalance_every=int(selection.historical_rebalance_every),
        rebalance_offset=int(phase["effective_forward_offset"]),
        initial_cash=float(cash),
        commission_rate=float(historical_config.get("commission_rate", v7.DEFAULT_COMMISSION_RATE)),
        stamp_tax_rate=float(historical_config.get("stamp_tax_rate", v7.DEFAULT_STAMP_TAX_RATE)),
        transfer_fee_rate=float(historical_config.get("transfer_fee_rate", v7.DEFAULT_TRANSFER_FEE_RATE)),
        slippage_bps=float(historical_config.get("slippage_bps", v7.DEFAULT_SLIPPAGE_BPS)),
        profile=str(historical_config.get("profile", "close_auction_skip_limit")),
        mainboard_only=config_bool(historical_config.get("mainboard_only"), True),
        min_price=float(historical_config.get("min_price", 0.0)),
        limit_eps=float(historical_config.get("limit_eps", 1e-6)),
        lot_size=int(historical_config.get("lot_size", v7.DEFAULT_LOT_SIZE)),
        min_commission=float(historical_config.get("min_commission", v7.DEFAULT_MIN_COMMISSION)),
        exclude_st=config_bool(historical_config.get("exclude_st"), True),
        capacity_mode=capacity_mode,
        participation_rate=float(participation_rate),
        # Broker positions already reflect prior company actions. Applying the
        # backtest synthetic adjustment again on the live date would double count.
        corporate_action_mode="none",
        corporate_action_threshold=float(historical_config.get("corporate_action_threshold", 1e-3)),
    )


def write_result_tables(live_dir: Path, result: dict[str, Any], execution: pd.DataFrame) -> dict[str, str]:
    outputs: dict[str, str] = {}
    mapping = {
        "16_live_nav.csv": result["nav"],
        "16_live_orders.csv": result["orders"],
        "16_live_rejections.csv": result["rejections"],
        "16_live_positions_after_plan.csv": result["positions"],
    }
    orders = mapping["16_live_orders.csv"].copy()
    if not orders.empty:
        orders["simulated_fill_status"] = orders.get("order_status", "filled")
        orders["order_status"] = "planned_not_submitted"
    mapping["16_live_orders.csv"] = orders
    for name, frame in mapping.items():
        path = live_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = str(path)
    state = result["final_state"]
    state_positions = pd.DataFrame(state.get("positions", []))
    if not state_positions.empty:
        prices = execution.set_index("symbol")["raw_close_1500"].to_dict()
        state_positions["mark_price"] = state_positions["symbol"].map(prices)
        state_positions["market_value"] = state_positions["shares"] * state_positions["mark_price"]
    state_positions.to_csv(live_dir / "16_live_target_portfolio.csv", index=False, encoding="utf-8-sig")
    outputs["16_live_target_portfolio.csv"] = str(live_dir / "16_live_target_portfolio.csv")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AS1455 clean strict-OOS live monitor")
    parser.add_argument("--trade-date", default="today")
    parser.add_argument("--live-dir", default=None)
    parser.add_argument("--model-data", default=str(as1455_paths.DEFAULT_MODEL_DATA))
    parser.add_argument("--target-col", choices=list(common.TARGET_SPECS), required=True)
    parser.add_argument("--feature-preset", choices=list(common.FEATURE_PRESETS), required=True)
    parser.add_argument("--fold0-dir", default=None)
    parser.add_argument("--selection-backtest-base", default=str(as1455_paths.TARGET_BACKTEST_ROOT))
    parser.add_argument("--selection-backtest-root", default=None)
    parser.add_argument("--selection-rank-metric", default="sharpe")
    parser.add_argument("--feature-file", default=None)
    parser.add_argument("--execution-sidecar", default=None)
    parser.add_argument("--execution-calendar", default=None)
    parser.add_argument("--positions-file", required=True)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--cash-file", default=None)
    parser.add_argument("--allow-missing-buy-date", action="store_true")
    parser.add_argument("--capacity-mode", choices=["none", "last5_amount", "last5_volume", "last5_both"], default="none")
    parser.add_argument("--participation-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_live_capacity_mode(args.capacity_mode)
    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455" / trade_date.strftime("%Y%m%d")
    live_dir.mkdir(parents=True, exist_ok=True)
    feature_file = Path(args.feature_file) if args.feature_file else live_dir / "11_live_model_features_for_prediction.csv"
    sidecar_file = Path(args.execution_sidecar) if args.execution_sidecar else live_dir / "08_live_execution_sidecar.csv"
    calendar_file = Path(args.execution_calendar) if args.execution_calendar else live_dir / "05_execution_calendar.csv"
    model_data_path = Path(args.model_data)

    historical = pd.read_hdf(model_data_path, "model_data")
    live_base = load_live_base_features(feature_file, historical.index.get_level_values("symbol").astype(str), trade_date)
    combined = assemble_inference_frame(model_data_path, live_base)
    feature_result = build_inference_features_from_frame(combined, args.target_col, args.feature_preset, "onehot", source_label=f"{model_data_path}+{feature_file}")

    spec = common.target_spec(args.target_col)
    selection = select_corresponding_historical_signal(
        base_root=Path(args.selection_backtest_base),
        feature_preset=args.feature_preset,
        target_col=args.target_col,
        rebalance_every=spec.rebalance_every,
        rank_metric=args.selection_rank_metric,
        explicit_backtest_root=Path(args.selection_backtest_root) if args.selection_backtest_root else None,
    )
    historical_trading_config(selection, spec.rebalance_every)
    history_window = historical_phase_window(selection)
    historical_config, historical_config_path = load_historical_run_config(selection)

    fold0_dir = Path(args.fold0_dir) if args.fold0_dir else common.default_fold0_dir(args.feature_preset, args.target_col)
    feature_dates = pd.DatetimeIndex(feature_result.X.index.get_level_values("date")).normalize()
    row_indices = np.flatnonzero(feature_dates == trade_date)
    if not len(row_indices):
        raise RuntimeError(f"no current-date inference rows after feature construction: {trade_date:%Y-%m-%d}")
    predictions, checkpoints, source_manifest = common.predict_checkpoint_set(
        feature_result.X, row_indices, fold0_dir, selection.required_top_n,
        metadata={"source_model_fold": 0, "live_trade_date": trade_date.strftime("%Y-%m-%d")},
    )
    score = score_predictions(predictions, selection)
    pred_out = predictions.copy()
    pred_out["pred_score"] = score
    pred_csv = pred_out.reset_index()
    pred_csv["symbol"] = pred_csv["symbol"].map(exchange_symbol)
    pred_csv.to_csv(live_dir / "14_live_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checkpoints).to_csv(live_dir / "14_live_checkpoints.csv", index=False, encoding="utf-8-sig")
    rank = pred_csv[["date", "symbol", "pred_score"]].copy().sort_values("pred_score", ascending=False)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank.to_csv(live_dir / "15_live_rank.csv", index=False, encoding="utf-8-sig")

    execution = load_execution_panel(sidecar_file, trade_date)
    execution_calendar = load_execution_calendar(calendar_file, trade_date)
    phase = align_forward_rebalance_phase(
        rebalance_every=int(selection.historical_rebalance_every),
        historical_offset=int(selection.historical_rebalance_offset),
        historical_n_days=int(history_window["historical_n_days"]),
        historical_first_date=history_window["historical_first_date"],
        historical_last_date=history_window["historical_last_date"],
        forward_prediction_dates=[trade_date],
        execution_calendar_dates=execution_calendar,
    )

    cash = load_cash(args)
    positions = load_positions(Path(args.positions_file), args.allow_missing_buy_date)
    marked_nav_before = current_marked_nav(cash, positions, execution)
    v7 = load_v7_module()
    cfg = build_trade_config(v7, selection, historical_config, phase, cash, args.capacity_mode, args.participation_rate)
    preds_v7 = rank[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "score"})
    result = v7.backtest(
        preds_v7, execution, cfg, corporate_actions=None,
        initial_positions=positions,
        day_index_start=0,
        allow_single_date=True,
    )
    outputs = write_result_tables(live_dir, result, execution)
    orders = result["orders"]
    final_state = result["final_state"]
    manifest = {
        "passed": True,
        "protocol": "as1455_clean_live_strict_oos_v2_v7_single_engine",
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "target_col": args.target_col,
        "feature_preset": args.feature_preset,
        "model_data": str(model_data_path),
        "feature_file": str(feature_file),
        "execution_sidecar": str(sidecar_file),
        "execution_calendar": str(calendar_file),
        "execution_calendar_days": int(len(execution_calendar)),
        "fold0_dir": str(fold0_dir),
        "historical_selection": selection.to_dict(),
        "historical_trade_config_file": str(historical_config_path),
        "historical_trade_config": historical_config,
        "rebalance_phase": phase,
        "source_model_manifest": source_manifest,
        "feature_report": feature_result.report,
        "trade_engine": str(V7_PATH),
        "trade_config": cfg.__dict__,
        "current_account_state": {
            "positions_file": args.positions_file,
            "cash": cash,
            "n_positions": int(len(positions)),
            "marked_nav_before_plan": marked_nav_before,
        },
        "planned_state": final_state,
        "outputs": outputs,
        "planned_order_count": int(len(orders)),
        "broker_orders_submitted": False,
        "planned_orders_persisted_as_account_truth": False,
        "corporate_action_live_policy": "none_broker_positions_already_adjusted",
        "capacity_mode_effective": args.capacity_mode,
        "price_semantics": "14:55 snapshot used as estimated close-auction order price; actual fill must be reconciled from broker",
    }
    write_json(live_dir / "17_live_strict_oos_manifest.json", manifest)
    print(json.dumps({
        "passed": True,
        "live_dir": str(live_dir),
        "rank_rows": int(len(rank)),
        "planned_orders": int(len(orders)),
        "is_rebalance_day": bool(result["nav"]["is_rebalance_day"].iloc[-1]),
        "target_positions": int(len(final_state.get("positions", []))),
        "manifest": str(live_dir / "17_live_strict_oos_manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
