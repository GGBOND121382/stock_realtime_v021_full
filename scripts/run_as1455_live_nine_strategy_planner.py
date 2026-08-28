#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate reviewable 14:55 plans for all nine global fixed-signal strategies.

Inputs:
- one shared Top-5 live prediction panel per target (r01/r05/r21);
- the same live execution sidecar/calendar used by the canonical strict-OOS monitor;
- each strategy's validated historical winner and latest completed strict-forward
  account state from T-1.

Outputs are simulated/planned orders only. No broker API is called.
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

from scripts import run_as1455_live_strict_oos_monitor as live  # noqa: E402
from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_model_selection import select_corresponding_historical_signal  # noqa: E402
from utils.as1455_rebalance_phase import align_forward_rebalance_phase  # noqa: E402
from utils.as1455_strict_oos import historical_phase_window, historical_trading_config  # noqa: E402

EXPERIMENT_RE = re.compile(
    r"^(?P<target>r\d{2})_(?P<signal>all5|first3|best)_reb(?P<reb>\d+)_"
    r"(?P<fold>fold0_[45])_forward$"
)
EXPECTED_SIGNAL_SPEC = {
    "all5": "ensemble_all5_mean:0,1,2,3,4:mean",
    "first3": "ensemble_first3_mean:0,1,2:mean",
    "best": "model_0:0:single",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=live.json_default),
        encoding="utf-8",
    )


def resolve_history_root(experiment_root: Path, manifest: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    for key in ("historical_result_root", "historical_root"):
        value = manifest.get(key)
        if value:
            path = Path(str(value)).expanduser()
            candidates.append(path if path.is_absolute() else PROJECT_DIR / path)
    candidates.extend(
        [
            experiment_root / "historical_fold_selection",
            experiment_root / "historical_fold0_to_fold5_selection",
        ]
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(
        f"cannot resolve historical selection root for {experiment_root.name}: {candidates}"
    )


def load_prediction_panel(path: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"symbol", "date", "0", "1", "2", "3", "4"}
    # Pandas may preserve integer-looking column names as strings in CSV.
    frame.columns = [str(column) for column in frame.columns]
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.loc[frame["date"].eq(trade_date)]
    if frame.empty:
        raise RuntimeError(f"no predictions for {trade_date:%Y-%m-%d}: {path}")
    frame["symbol"] = frame["symbol"].map(live.exchange_symbol)
    for column in ["0", "1", "2", "3", "4"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.rename(columns={str(i): i for i in range(5)})
    frame = frame.set_index(["symbol", "date"])[list(range(5))].sort_index()
    return frame


def selection_spec(selection: Any) -> str:
    return f"{selection.signal_name}:{selection.signal_cols}:{selection.signal_mode}"


def parse_experiments(matrix_root: Path) -> list[dict[str, Any]]:
    expected = matrix_root / "expected_experiments.txt"
    if not expected.is_file():
        raise FileNotFoundError(expected)
    rows: list[dict[str, Any]] = []
    for name in expected.read_text(encoding="utf-8").splitlines():
        name = name.strip()
        if not name:
            continue
        match = EXPERIMENT_RE.fullmatch(name)
        if match is None:
            raise RuntimeError(f"unsupported experiment name: {name}")
        rows.append(
            {
                "experiment": name,
                "target": match.group("target"),
                "target_col": f"{match.group('target')}_fwd",
                "signal": match.group("signal"),
                "rebalance_every": int(match.group("reb")),
                "fold_label": match.group("fold"),
            }
        )
    if len(rows) != 9:
        raise RuntimeError(f"expected nine experiments, got {len(rows)}")
    return rows


def load_state(experiment_root: Path, trade_date: pd.Timestamp) -> tuple[dict[str, Any], pd.DataFrame]:
    state_file = experiment_root / "strict_forward_latest_state.json"
    positions_file = experiment_root / "strict_forward_latest_positions.csv"
    if not state_file.is_file() or not positions_file.is_file():
        raise FileNotFoundError(
            "latest forward state missing; run the 20:00 nine-backtest refresh first: "
            f"{state_file} / {positions_file}"
        )
    state = read_json(state_file)
    asof = pd.Timestamp(state["asof_date"]).normalize()
    if asof >= trade_date:
        raise RuntimeError(
            f"{experiment_root.name} state must be T-1 or earlier for live planning: "
            f"asof={asof:%Y-%m-%d} trade_date={trade_date:%Y-%m-%d}"
        )
    positions = live.load_positions(positions_file, allow_missing_buy_date=False)
    expected = int(state.get("n_positions", 0))
    if len(positions) != expected:
        raise RuntimeError(
            f"state position count mismatch for {experiment_root.name}: "
            f"state={expected} file={len(positions)}"
        )
    return state, positions


def synthetic_corporate_action_mode(historical_config: dict[str, Any]) -> str:
    """Return the executable paper/live corporate-action approximation.

    Historical research may use ``synthetic_share_factor_from_preclose`` to keep
    total-return continuity, but that mode can multiply an integer A-share
    position into fractional shares.  Fractional shares cannot be submitted to
    the broker, so tracking/live accounts must not reuse that representation.

    Until an exact corporate-action feed is supplied, keep the existing share
    quantity unchanged and realize the preclose adjustment as synthetic cash.
    This is exact for a pure cash dividend and remains a value-continuity
    approximation for bonus/split/rights events.  Frozen Fold/Grid artifacts are
    not changed because this function is used only by tracking/live planning.
    """
    historical_mode = str(
        historical_config.get(
            "corporate_action_mode", "synthetic_share_factor_from_preclose"
        )
    )
    if historical_mode not in {
        "synthetic_share_factor_from_preclose",
        "synthetic_cash_from_preclose",
        "none",
    }:
        raise RuntimeError(
            "unsupported corporate_action_mode for live paper state: "
            f"{historical_mode}"
        )
    return "synthetic_cash_from_preclose"


def output_strategy_tables(
    strategy_dir: Path,
    result: dict[str, Any],
    execution: pd.DataFrame,
    current_positions: pd.DataFrame,
) -> dict[str, str]:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    outputs = live.write_result_tables(strategy_dir, result, execution)
    current_file = strategy_dir / "current_positions_before_plan.csv"
    current_positions.to_csv(current_file, index=False, encoding="utf-8-sig")
    outputs[current_file.name] = str(current_file)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", required=True)
    parser.add_argument(
        "--matrix-root",
        default=(
            "saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/"
            "refresh_all_v1"
        ),
    )
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--execution-sidecar", required=True)
    parser.add_argument("--execution-calendar", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument("--selection-rank-metric", default="sharpe")
    parser.add_argument("--capacity-mode", default="none")
    parser.add_argument("--participation-rate", type=float, default=0.05)
    args = parser.parse_args()

    live.validate_live_capacity_mode(args.capacity_mode)
    trade_date = live.parse_trade_date(args.trade_date)
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    execution = live.load_execution_panel(
        Path(args.execution_sidecar).expanduser().resolve(), trade_date
    )
    execution_calendar = live.load_execution_calendar(
        Path(args.execution_calendar).expanduser().resolve(), trade_date
    )
    v7 = live.load_v7_module()
    experiments = parse_experiments(matrix_root)
    prediction_cache: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    strategy_manifests: dict[str, str] = {}

    for item in experiments:
        target = item["target"]
        target_col = item["target_col"]
        signal_kind = item["signal"]
        experiment = item["experiment"]
        experiment_root = matrix_root / experiment
        manifest_file = experiment_root / "global_fold0_to_fold5_forward_manifest.json"
        experiment_manifest = read_json(manifest_file)
        history_root = resolve_history_root(experiment_root, experiment_manifest)

        selection = select_corresponding_historical_signal(
            base_root=Path(as1455_paths.TARGET_BACKTEST_ROOT),
            feature_preset=args.feature_preset,
            target_col=target_col,
            rebalance_every=item["rebalance_every"],
            rank_metric=args.selection_rank_metric,
            explicit_backtest_root=history_root,
        )
        actual_spec = selection_spec(selection)
        expected_spec = EXPECTED_SIGNAL_SPEC[signal_kind]
        if actual_spec != expected_spec:
            raise RuntimeError(
                f"fixed signal mismatch for {experiment}: expected={expected_spec} actual={actual_spec}"
            )
        historical_trading_config(selection, item["rebalance_every"])
        history_window = historical_phase_window(selection)
        historical_config, historical_config_path = live.load_historical_run_config(selection)

        if target not in prediction_cache:
            prediction_file = prediction_root / target / "top5_live_predictions.csv"
            prediction_cache[target] = load_prediction_panel(prediction_file, trade_date)
        predictions = prediction_cache[target]
        score = live.score_predictions(predictions, selection)
        ranked = score.reset_index().rename(columns={"score": "pred_score"})
        ranked["symbol"] = ranked["symbol"].map(live.exchange_symbol)
        ranked = ranked.sort_values("pred_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)

        phase = align_forward_rebalance_phase(
            rebalance_every=int(selection.historical_rebalance_every),
            historical_offset=int(selection.historical_rebalance_offset),
            historical_n_days=int(history_window["historical_n_days"]),
            historical_first_date=history_window["historical_first_date"],
            historical_last_date=history_window["historical_last_date"],
            forward_prediction_dates=[trade_date],
            execution_calendar_dates=execution_calendar,
        )
        state, current_positions = load_state(experiment_root, trade_date)
        cash = float(state["cash"])
        if not math.isfinite(cash) or cash < 0:
            raise RuntimeError(f"invalid T-1 cash for {experiment}: {cash}")

        cfg = live.build_trade_config(
            v7,
            selection,
            historical_config,
            phase,
            cash,
            args.capacity_mode,
            args.participation_rate,
        )
        cfg = replace(
            cfg,
            corporate_action_mode=synthetic_corporate_action_mode(historical_config),
        )
        preds_v7 = ranked[["date", "symbol", "pred_score"]].rename(
            columns={"pred_score": "score"}
        )
        result = v7.backtest(
            preds_v7,
            execution,
            cfg,
            corporate_actions=None,
            initial_positions=current_positions,
            day_index_start=0,
            allow_single_date=True,
        )
        strategy_dir = out_root / "strategies" / experiment
        outputs = output_strategy_tables(
            strategy_dir, result, execution, current_positions
        )
        ranked.to_csv(
            strategy_dir / "live_rank.csv", index=False, encoding="utf-8-sig"
        )

        nav = result["nav"]
        orders = result["orders"].copy()
        rejections = result["rejections"].copy()
        is_rebalance = bool(nav["is_rebalance_day"].iloc[-1])
        buy_count = int((orders["side"].astype(str).str.lower() == "buy").sum()) if not orders.empty and "side" in orders.columns else 0
        sell_count = int((orders["side"].astype(str).str.lower() == "sell").sum()) if not orders.empty and "side" in orders.columns else 0
        final_state = result["final_state"]
        target_positions = int(len(final_state.get("positions", [])))
        action = (
            "调仓"
            if is_rebalance and len(orders) > 0
            else "调仓日·无需成交"
            if is_rebalance
            else "非调仓日·继续持有"
        )
        strategy_manifest = {
            "status": "ok",
            "protocol": "as1455_nine_strategy_live_plan_v1",
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "experiment": experiment,
            "target_col": target_col,
            "signal_kind": signal_kind,
            "fixed_signal_spec": actual_spec,
            "is_rebalance_day": is_rebalance,
            "action": action,
            "historical_selection": selection.to_dict(),
            "historical_trade_config_file": str(historical_config_path),
            "rebalance_phase": phase,
            "trade_config": cfg.__dict__,
            "initial_state": state,
            "current_position_count": int(len(current_positions)),
            "planned_order_count": int(len(orders)),
            "planned_buy_count": buy_count,
            "planned_sell_count": sell_count,
            "rejection_count": int(len(rejections)),
            "target_position_count": target_positions,
            "planned_cash_after": float(final_state.get("cash", np.nan)),
            "outputs": outputs,
            "broker_orders_submitted": False,
            "planned_orders_persisted_as_account_truth": False,
            "account_state_source": "latest_completed_strict_forward_backtest",
            "corporate_action_live_policy": cfg.corporate_action_mode,
        }
        strategy_manifest_file = strategy_dir / "strategy_manifest.json"
        write_json(strategy_manifest_file, strategy_manifest)
        strategy_manifests[experiment] = str(strategy_manifest_file)
        summary_rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "experiment": experiment,
                "target_col": target_col,
                "signal_kind": signal_kind,
                "rebalance_every": int(selection.historical_rebalance_every),
                "is_rebalance_day": is_rebalance,
                "action": action,
                "planned_orders": int(len(orders)),
                "planned_buys": buy_count,
                "planned_sells": sell_count,
                "rejections": int(len(rejections)),
                "current_positions": int(len(current_positions)),
                "target_positions": target_positions,
                "state_asof_date": state["asof_date"],
                "cash_before": cash,
                "planned_cash_after": float(final_state.get("cash", np.nan)),
                "max_positions": int(selection.historical_max_positions),
                "sell_rank": int(selection.historical_sell_rank),
                "historical_offset": int(selection.historical_rebalance_offset),
                "effective_live_offset": int(phase["effective_forward_offset"]),
                "fixed_signal_spec": actual_spec,
                "strategy_dir": str(strategy_dir),
            }
        )
        print(
            "[PLAN] "
            f"experiment={experiment} rebalance={is_rebalance} action={action} "
            f"buys={buy_count} sells={sell_count} target_positions={target_positions}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_file = out_root / "live_nine_strategy_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    rebalance_only = summary.loc[summary["is_rebalance_day"].astype(bool)].copy()
    rebalance_only.to_csv(
        out_root / "live_rebalance_strategies.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "status": "ok",
        "protocol": "as1455_nine_strategy_live_plan_v1",
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "experiment_count": int(len(summary)),
        "rebalance_strategy_count": int(len(rebalance_only)),
        "planned_order_count": int(summary["planned_orders"].sum()),
        "summary_file": str(summary_file),
        "rebalance_summary_file": str(out_root / "live_rebalance_strategies.csv"),
        "strategy_manifests": strategy_manifests,
        "shared_predictions": {
            target: str(prediction_root / target / "top5_live_predictions.csv")
            for target in sorted(prediction_cache)
        },
        "broker_orders_submitted": False,
    }
    manifest_file = out_root / "live_nine_strategy_manifest.json"
    write_json(manifest_file, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
