#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan one r01-best paper strategy without mutating the r21 production plan.

The canonical nine-strategy planner is executed in an isolated staging root with
its experiment list filtered to exactly one Top-1 ``best`` strategy.  Only the
validated ``execution_batch.json`` is atomically published into the normal live
strategy tree, so the execution API can expose it by explicit experiment query.
The r21 production READY batch and root manifests are never rewritten here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner
from scripts import run_as1455_live_nine_strategy_planner_entry as entry
from scripts import run_as1455_live_production_strategy_planner_entry as production

DEFAULT_SIMULATION_EXPERIMENT = "r01_best_reb1_fold0_5_forward"


def _pop_custom_arg(flag: str, default: str = "") -> str:
    value = default
    cleaned: list[str] = []
    index = 0
    while index < len(sys.argv):
        token = sys.argv[index]
        if token == flag:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{flag} requires a value")
            value = sys.argv[index + 1]
            index += 2
            continue
        prefix = flag + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            index += 1
            continue
        cleaned.append(token)
        index += 1
    sys.argv[:] = cleaned
    return value.strip()


def publish_simulation_batch(
    staging_root: Path,
    publish_root: Path,
    experiment: str,
    trade_date: str,
) -> Path:
    source = staging_root / "strategies" / experiment / "execution_batch.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "ready":
        raise RuntimeError(f"simulation batch is not READY: {source}")
    if payload.get("protocol") != entry.EXECUTION_BATCH_PROTOCOL:
        raise RuntimeError(f"unexpected simulation batch protocol: {payload.get('protocol')}")
    if str(payload.get("experiment")) != experiment:
        raise RuntimeError(
            f"simulation experiment mismatch: actual={payload.get('experiment')} expected={experiment}"
        )
    expected_date = planner.live.parse_trade_date(trade_date).strftime("%Y-%m-%d")
    if str(payload.get("trade_date")) != expected_date:
        raise RuntimeError(
            f"simulation trade_date mismatch: actual={payload.get('trade_date')} expected={expected_date}"
        )
    orders = payload.get("orders")
    if not isinstance(orders, list) or int(payload.get("order_count", -1)) != len(orders):
        raise RuntimeError("invalid simulation execution batch order_count/orders")

    destination = publish_root / "strategies" / experiment / "execution_batch.json"
    entry._atomic_write_json(destination, payload)
    return destination


def main() -> None:
    global_publish_root = _pop_custom_arg("--publish-root")
    experiment = _pop_custom_arg(
        "--simulation-experiment", DEFAULT_SIMULATION_EXPERIMENT
    )
    if experiment != DEFAULT_SIMULATION_EXPERIMENT:
        raise RuntimeError(
            "simulation planner is intentionally restricted to "
            f"{DEFAULT_SIMULATION_EXPERIMENT}; got {experiment}"
        )
    if not global_publish_root:
        raise RuntimeError("simulation planner requires --publish-root")

    trade_value = entry._arg_value("--trade-date")
    staging_value = entry._arg_value("--out-root")
    if not trade_value or not staging_value:
        raise RuntimeError("simulation planner requires --trade-date and --out-root")

    staging_root = Path(staging_value).expanduser().resolve()
    publish_root = Path(global_publish_root).expanduser().resolve()

    # Reuse the proven production Top-1 loader and strict tracking-state contract,
    # but deliberately do not call production invalidation/archive/model-use code.
    production.install_production_filter(experiment)
    production.install_fail_closed_tracking_start()
    production.validate_production_tracking_contract(experiment)

    entry._EXECUTION_CALENDAR = entry._load_execution_calendar_from_cli()
    entry.install_live_summary_adapter()
    entry.install_tracking_state_adapter()

    planner.main()
    entry.postprocess_manifests()
    entry.publish_execution_batches()

    batch_path = publish_simulation_batch(
        staging_root,
        publish_root,
        experiment,
        trade_value,
    )
    print(f"[SIMULATION_READY] {batch_path}")


if __name__ == "__main__":
    main()
