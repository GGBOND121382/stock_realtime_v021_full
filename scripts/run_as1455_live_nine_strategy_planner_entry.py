#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tracking-aware entry point for the nine-strategy 14:55 planner.

Two adapters are installed before invoking the canonical planner:
1. continuation summaries use portfolio NAV rather than residual cash;
2. account state comes from the user-selected tracking account.

After planning, this entry point also publishes one compact, atomically-written
``execution_batch.json`` per strategy.  That batch contains final limit-order
prices and is the only artifact the Windows executor needs to read.

The tracking start date never resets the frozen rebalance phase. If the current
trade date is the first tracking day (or no earlier trading day has occurred),
the account starts empty, but the canonical planner still decides whether today
is a rebalance day from the historical phase/offset. Therefore a non-rebalance
start date remains cash; the first later scheduled rebalance can buy but cannot
sell because the account is still empty.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner  # noqa: E402
from utils.as1455_tracking import (  # noqa: E402
    TRACKING_SEMANTICS_VERSION,
    experiment_tracking_paths,
    read_json,
    resolve_initial_cash,
    tracking_start_date,
)

_EXECUTION_CALENDAR = pd.DatetimeIndex([])
EXECUTION_BATCH_PROTOCOL = "as1455_execution_batch_v1"


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
    first_nav = pd.to_numeric(
        pd.Series([nav.iloc[0]["nav"]]), errors="coerce"
    ).iloc[0]
    first_ret = pd.to_numeric(
        pd.Series([nav.iloc[0]["daily_return"]]), errors="coerce"
    ).iloc[0]
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
        if getattr(original_summarize, "_as1455_live_summary_adapter", False):
            return v7

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
                nav,
                orders,
                rejects,
                summary_cfg,
                actions,
                round_trips,
                daily_drawdown,
            )

        summarize_nav_live._as1455_live_summary_adapter = True  # type: ignore[attr-defined]
        v7.summarize_nav = summarize_nav_live
        return v7

    planner.live.load_v7_module = load_v7_with_live_summary


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


def _load_tracking_positions(path: Path, expected_count: int) -> pd.DataFrame:
    """Load a persisted tracking position snapshot without masking corruption.

    Older tracking writers may persist an empty account as a zero-byte/headerless
    CSV.  That representation is safe only when the authoritative JSON state says
    ``n_positions == 0``.  A missing/empty file with a positive expected count is
    treated as corruption and must fail closed.
    """
    expected = int(expected_count)
    if expected < 0:
        raise RuntimeError(f"invalid expected tracking position count: {expected}")
    if not path.is_file():
        if expected == 0:
            return _empty_positions()
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        if expected == 0:
            return _empty_positions()
        raise RuntimeError(
            f"tracking positions file is empty but state expects {expected} positions: {path}"
        )
    try:
        positions = planner.live.load_positions(
            path, allow_missing_buy_date=False
        )
    except pd.errors.EmptyDataError:
        if expected == 0:
            return _empty_positions()
        raise RuntimeError(
            f"tracking positions file has no columns but state expects {expected} positions: {path}"
        )
    if len(positions) != expected:
        raise RuntimeError(
            f"tracking position count mismatch: file={len(positions)} expected={expected} path={path}"
        )
    if positions.empty:
        return _empty_positions()
    return positions


def install_tracking_state_adapter() -> None:
    original_load_state = planner.load_state

    def tracking_load_state(experiment_root: Path, trade_date: pd.Timestamp):
        matrix_root = experiment_root.parent
        start = tracking_start_date(matrix_root)
        if start is None:
            return original_load_state(experiment_root, trade_date)

        trade_date = pd.Timestamp(trade_date).normalize()
        if trade_date < start:
            raise RuntimeError(
                f"trade_date {trade_date:%Y-%m-%d} is before "
                f"tracking_start_date {start:%Y-%m-%d}"
            )

        prior_days = _EXECUTION_CALENDAR[
            (_EXECUTION_CALENDAR >= start) & (_EXECUTION_CALENDAR < trade_date)
        ]
        paths = experiment_tracking_paths(experiment_root)
        manifest = read_json(paths["manifest"], {}) or {}
        state = read_json(paths["latest_state"], {}) or {}

        # No earlier tracking-market day: account is empty. Do NOT override the
        # trade config or rebalance offset; the canonical phase logic below will
        # decide whether this date is actually a scheduled rebalance date.
        if len(prior_days) == 0:
            cash = resolve_initial_cash(experiment_root)
            return (
                {
                    "status": "empty_tracking_start",
                    "asof_date": None,
                    "tracking_start_date": start.strftime("%Y-%m-%d"),
                    "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
                    "cash": cash,
                    "nav": cash,
                    "n_positions": 0,
                    "tracking_state_source": "empty_tracking_start",
                },
                _empty_positions(),
            )

        if manifest.get("tracking_start_date") != start.strftime("%Y-%m-%d"):
            raise RuntimeError(
                f"tracking account is not rebuilt for start={start:%Y-%m-%d}: "
                f"experiment={experiment_root.name}"
            )
        if int(manifest.get("tracking_semantics_version", 0) or 0) != TRACKING_SEMANTICS_VERSION:
            raise RuntimeError(
                f"tracking account semantics are stale for {experiment_root.name}; "
                "rebuild the tracking account"
            )
        if not state:
            raise FileNotFoundError(
                f"tracking state missing for {experiment_root.name}; "
                "run the nightly/incremental refresh first"
            )
        asof = pd.Timestamp(state["asof_date"]).normalize()
        expected_asof = pd.Timestamp(prior_days[-1]).normalize()
        if asof != expected_asof:
            raise RuntimeError(
                f"stale tracking state for {experiment_root.name}: "
                f"asof={asof:%Y-%m-%d} expected={expected_asof:%Y-%m-%d}"
            )
        expected_positions = int(state.get("n_positions", 0) or 0)
        positions = _load_tracking_positions(
            paths["latest_positions"], expected_positions
        )
        state = dict(state)
        state["tracking_state_source"] = "latest_tracking_account"
        return state, positions

    planner.load_state = tracking_load_state


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def _positive_price(value: object, field: str) -> Decimal:
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"invalid {field}={value!r}") from exc
    if not price.is_finite() or price <= 0:
        raise RuntimeError(f"invalid {field}={value!r}")
    return price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _order_qty(row: pd.Series) -> int:
    for name in ("filled_shares", "shares", "intended_shares"):
        if name not in row.index or pd.isna(row[name]):
            continue
        value = float(row[name])
        if not math.isfinite(value) or value <= 0 or not value.is_integer():
            raise RuntimeError(f"invalid {name}={row[name]!r}")
        return int(value)
    raise RuntimeError("planned order lacks a positive integer share quantity")


def _reference_price(row: pd.Series) -> Decimal:
    for name in ("raw_exec_price", "raw_close_1500"):
        if name in row.index and not pd.isna(row[name]):
            return _positive_price(row[name], name)
    raise RuntimeError("planned order lacks raw_exec_price/raw_close_1500")


def _server_signal_id(
    trade_date: str,
    experiment: str,
    code: str,
    side: str,
    qty: int,
    submit_price: Decimal,
    rank: object,
    reason: object,
) -> str:
    stable = "|".join(
        [
            trade_date,
            experiment,
            code,
            side,
            str(qty),
            format(submit_price, "f"),
            "" if pd.isna(rank) else str(rank),
            "" if pd.isna(reason) else str(reason),
        ]
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def publish_execution_batches() -> None:
    """Publish the minimal immutable server -> Windows execution contract.

    `execution_batch.json` is written only after every order has a validated final
    submit price.  The final rename is atomic, so its presence itself is the READY
    marker.  Windows no longer needs the strategy manifest, planned-order CSV,
    positions file, quote lookup, or any account read before submitting.
    """
    out_value = _arg_value("--out-root")
    sidecar_value = _arg_value("--execution-sidecar")
    if not out_value or not sidecar_value:
        raise RuntimeError("execution batch publication requires --out-root and --execution-sidecar")

    out_root = Path(out_value).expanduser().resolve()
    sidecar_path = Path(sidecar_value).expanduser().resolve()
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = pd.read_csv(
        sidecar_path,
        dtype={"symbol": str, "code": str},
        encoding="utf-8-sig",
    )
    if "symbol" not in sidecar.columns:
        if "code" in sidecar.columns:
            sidecar["symbol"] = sidecar["code"]
        else:
            raise RuntimeError("execution sidecar requires symbol/code")
    required = {"up_limit", "down_limit"}
    missing = required - set(sidecar.columns)
    if missing:
        raise RuntimeError(f"execution sidecar missing final-price fields: {sorted(missing)}")
    sidecar = sidecar.copy()
    sidecar["symbol"] = sidecar["symbol"].map(planner.live.exchange_symbol)
    if sidecar["symbol"].duplicated().any():
        dup = sidecar.loc[sidecar["symbol"].duplicated(), "symbol"].head().tolist()
        raise RuntimeError(f"duplicate symbols in execution sidecar: {dup}")
    sidecar = sidecar.set_index("symbol", drop=False)

    strategies_root = out_root / "strategies"
    if not strategies_root.is_dir():
        raise FileNotFoundError(strategies_root)
    batch_files: dict[str, str] = {}

    for strategy_dir in sorted(path for path in strategies_root.iterdir() if path.is_dir()):
        manifest_file = strategy_dir / "strategy_manifest.json"
        orders_file = strategy_dir / "16_live_orders.csv"
        if not manifest_file.is_file() or not orders_file.is_file():
            raise FileNotFoundError(
                f"strategy output incomplete before execution publication: {strategy_dir}"
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        trade_date = str(manifest["trade_date"])
        experiment = str(manifest["experiment"])
        try:
            orders = pd.read_csv(orders_file, dtype={"symbol": str}, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            orders = pd.DataFrame()
        expected_count = int(manifest.get("planned_order_count", len(orders)) or 0)
        if expected_count != len(orders):
            raise RuntimeError(
                f"planned order count mismatch for {experiment}: manifest={expected_count} csv={len(orders)}"
            )

        prepared: list[dict[str, Any]] = []
        for source_index, (_, row) in enumerate(orders.iterrows()):
            side = str(row.get("side", "")).strip().lower()
            if side not in {"buy", "sell"}:
                raise RuntimeError(f"unsupported planned order side for {experiment}: {side!r}")
            symbol = planner.live.exchange_symbol(row.get("symbol", ""))
            code = planner.live.code6(symbol)
            if symbol not in sidecar.index:
                raise RuntimeError(f"order symbol missing from execution sidecar: {symbol}")
            execution_row = sidecar.loc[symbol]
            if isinstance(execution_row, pd.DataFrame):
                raise RuntimeError(f"ambiguous execution sidecar symbol: {symbol}")
            upper = _positive_price(execution_row["up_limit"], f"up_limit[{symbol}]")
            lower = _positive_price(execution_row["down_limit"], f"down_limit[{symbol}]")
            if lower >= upper:
                raise RuntimeError(f"invalid daily limit range for {symbol}: lower={lower} upper={upper}")
            reference = _reference_price(row)
            if not (lower <= reference <= upper):
                raise RuntimeError(
                    f"reference price outside server daily limits for {symbol}: "
                    f"reference={reference} lower={lower} upper={upper}"
                )
            qty = _order_qty(row)
            submit_price = upper if side == "buy" else lower
            rank = row.get("rank", np.nan)
            reason = row.get("reason", "")
            prepared.append(
                {
                    "_source_index": source_index,
                    "signal_id": _server_signal_id(
                        trade_date,
                        experiment,
                        code,
                        side,
                        qty,
                        submit_price,
                        rank,
                        reason,
                    ),
                    "code": code,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "submit_price": format(submit_price, ".2f"),
                    "reference_price": format(reference, ".2f"),
                    "upper_limit": format(upper, ".2f"),
                    "lower_limit": format(lower, ".2f"),
                    "rank": None if pd.isna(rank) else float(rank),
                    "position_before": (
                        None
                        if "position_before" not in row.index or pd.isna(row.get("position_before"))
                        else float(row.get("position_before"))
                    ),
                    "price_source": "server_execution_sidecar_limits",
                }
            )

        prepared.sort(key=lambda item: (0 if item["side"] == "sell" else 1, item["_source_index"]))
        final_orders: list[dict[str, Any]] = []
        for sequence, item in enumerate(prepared, start=1):
            item = dict(item)
            item.pop("_source_index", None)
            item["sequence"] = sequence
            final_orders.append(item)

        batch = {
            "status": "ready",
            "protocol": EXECUTION_BATCH_PROTOCOL,
            "trade_date": trade_date,
            "experiment": experiment,
            "order_count": len(final_orders),
            "submit_sequence": "sells_then_buys",
            "price_semantics": (
                "server_final_limit_price: buy=upper_limit, sell=lower_limit; "
                "reference_price is audit/research only"
            ),
            "price_source": "server_execution_sidecar_limits",
            "broker_orders_submitted": False,
            "windows_required_files": ["execution_batch.json"],
            "windows_quote_lookup_required": False,
            "windows_account_reads_required": False,
            "orders": final_orders,
        }
        batch_file = strategy_dir / "execution_batch.json"
        _atomic_write_json(batch_file, batch)
        batch_files[experiment] = str(batch_file)

        manifest["execution_batch_file"] = str(batch_file)
        manifest["execution_batch_protocol"] = EXECUTION_BATCH_PROTOCOL
        manifest["execution_price_source"] = "server_execution_sidecar_limits"
        manifest["windows_required_files"] = ["execution_batch.json"]
        manifest["windows_quote_lookup_required"] = False
        manifest["windows_account_reads_required"] = False
        _atomic_write_json(manifest_file, manifest)

    root_manifest_file = out_root / "live_nine_strategy_manifest.json"
    if root_manifest_file.is_file():
        root_manifest = json.loads(root_manifest_file.read_text(encoding="utf-8"))
        root_manifest["execution_batch_protocol"] = EXECUTION_BATCH_PROTOCOL
        root_manifest["execution_batch_files"] = batch_files
        root_manifest["windows_quote_lookup_required"] = False
        root_manifest["windows_account_reads_required"] = False
        _atomic_write_json(root_manifest_file, root_manifest)


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
            payload["tracking_semantics_version"] = TRACKING_SEMANTICS_VERSION
            # A bootstrap is an actual first entry, not merely the selected
            # account-start date. This is true only when no position existed
            # before the plan and the plan actually contains buys.
            payload["tracking_bootstrap"] = (
                int(payload.get("current_position_count", 0) or 0) == 0
                and int(payload.get("planned_buy_count", 0) or 0) > 0
            )
            _atomic_write_json(path, payload)
    manifest_file = out_root / "live_nine_strategy_manifest.json"
    if manifest_file.is_file():
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        payload["tracking_start_date"] = start.strftime("%Y-%m-%d")
        payload["tracking_semantics_version"] = TRACKING_SEMANTICS_VERSION
        payload["account_state_semantics"] = (
            "empty_from_tracking_start_preserve_historical_rebalance_phase"
        )
        _atomic_write_json(manifest_file, payload)


def main() -> None:
    global _EXECUTION_CALENDAR
    _EXECUTION_CALENDAR = _load_execution_calendar_from_cli()
    install_live_summary_adapter()
    install_tracking_state_adapter()
    planner.main()
    publish_execution_batches()
    postprocess_manifests()


if __name__ == "__main__":
    main()
