#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed tracking-aware planner for one production best-model strategy.

The nine-strategy research matrix remains intact. Production narrows only the
latency-critical planner to one selected ``best`` experiment and Top-1 model.

Production safety contract:
- a valid configured tracking account is mandatory; legacy-state fallback is
  forbidden;
- an existing same-day execution batch is invalidated before a new run starts;
- planner/post-processing/model-generation bookkeeping must all succeed before
  the READY execution batch is atomically published;
- the execution batch write is the final fallible production commit.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner  # noqa: E402
from scripts import run_as1455_live_nine_strategy_planner_entry as entry  # noqa: E402
from utils.as1455_model_roll import record_live_generation_use  # noqa: E402

DEFAULT_PRODUCTION_EXPERIMENT = "r21_best_reb21_fold0_4_forward"
DEFAULT_MODEL_REGISTRY_ROOT = "saved_data/ashare_ml4t/ch17_as1455_model_registry"


def _pop_custom_arg(flag: str, default: str) -> str:
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


def load_best_prediction_panel(path: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    """Load only model column 0; Top-3/Top-5 columns are intentionally unnecessary."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column) for column in frame.columns]
    required = {"symbol", "date", "0"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing production prediction columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.loc[frame["date"].eq(pd.Timestamp(trade_date).normalize())]
    if frame.empty:
        raise RuntimeError(f"no predictions for {trade_date:%Y-%m-%d}: {path}")
    frame["symbol"] = frame["symbol"].map(planner.live.exchange_symbol)
    frame["0"] = pd.to_numeric(frame["0"], errors="raise")
    frame = frame.rename(columns={"0": 0})
    return frame.set_index(["symbol", "date"])[[0]].sort_index()


def install_production_filter(experiment: str) -> None:
    original_parse = planner.parse_experiments

    def parse_selected(matrix_root: Path):
        all_items = original_parse(matrix_root)
        selected = [item for item in all_items if item["experiment"] == experiment]
        if len(selected) != 1:
            raise RuntimeError(
                f"production experiment must resolve exactly once: {experiment}; "
                f"matches={len(selected)}"
            )
        item = selected[0]
        if item["signal"] != "best":
            raise RuntimeError(
                "latency-critical Top-1 production mode only supports signal=best; "
                f"experiment={experiment} signal={item['signal']}"
            )
        return selected

    planner.parse_experiments = parse_selected
    planner.load_prediction_panel = load_best_prediction_panel


def install_fail_closed_tracking_start() -> None:
    """Production must never fall back to the canonical strict-forward account."""
    original = entry.tracking_start_date

    def strict(matrix_root: Path):
        start = original(matrix_root)
        if start is None:
            raise RuntimeError(
                "production tracking_start_date is missing/invalid; "
                "legacy strict-forward account fallback is forbidden"
            )
        return start

    entry.tracking_start_date = strict


def _production_paths(experiment: str) -> tuple[Path, Path, pd.Timestamp, pd.DatetimeIndex]:
    matrix_value = entry._arg_value("--matrix-root")
    trade_value = entry._arg_value("--trade-date")
    if not matrix_value or not trade_value:
        raise RuntimeError("production validation requires --matrix-root and --trade-date")
    matrix_root = Path(matrix_value).expanduser().resolve()
    experiment_root = matrix_root / experiment
    trade_date = planner.live.parse_trade_date(trade_value)
    calendar = entry._load_execution_calendar_from_cli()
    if calendar.empty:
        raise RuntimeError("production execution calendar is missing/empty")
    return matrix_root, experiment_root, trade_date, calendar


def validate_production_tracking_contract(experiment: str) -> None:
    """Validate the exact account state that the production planner is allowed to use."""
    matrix_root, experiment_root, trade_date, calendar = _production_paths(experiment)
    start = entry.tracking_start_date(matrix_root)
    if trade_date < start:
        raise RuntimeError(
            f"production trade_date {trade_date:%Y-%m-%d} is before tracking start {start:%Y-%m-%d}"
        )
    expected_cash = entry.resolve_initial_cash(experiment_root)
    prior_days = calendar[(calendar >= start) & (calendar < trade_date)]
    if len(prior_days) == 0:
        return

    paths = entry.experiment_tracking_paths(experiment_root)
    manifest = entry.read_json(paths["manifest"], {}) or {}
    state = entry.read_json(paths["latest_state"], {}) or {}
    if manifest.get("status") != "ok" or not state or state.get("status") != "ok":
        raise RuntimeError(f"production tracking state is not ready: {experiment}")
    expected_start = start.strftime("%Y-%m-%d")
    if manifest.get("tracking_start_date") != expected_start:
        raise RuntimeError(
            f"production tracking start mismatch: expected={expected_start} "
            f"actual={manifest.get('tracking_start_date')}"
        )
    if int(manifest.get("tracking_semantics_version", 0) or 0) != entry.TRACKING_SEMANTICS_VERSION:
        raise RuntimeError("production tracking semantics are stale; rebuild before trading")
    for label, payload in (("manifest", manifest), ("state", state)):
        raw = payload.get("initial_cash")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"production {label} lacks valid initial_cash") from exc
        if not math.isfinite(value) or abs(value - expected_cash) > 1e-6:
            raise RuntimeError(
                f"production {label} initial_cash mismatch: expected={expected_cash:.2f} actual={raw}"
            )
    asof = pd.to_datetime(state.get("asof_date"), errors="coerce")
    expected_asof = pd.Timestamp(prior_days[-1]).normalize()
    if pd.isna(asof) or pd.Timestamp(asof).normalize() != expected_asof:
        raise RuntimeError(
            f"production tracking state stale: expected T-1={expected_asof:%Y-%m-%d} "
            f"actual={state.get('asof_date')}"
        )
    expected_positions = int(state.get("n_positions", 0) or 0)
    entry._load_tracking_positions(paths["latest_positions"], expected_positions)


def invalidate_existing_ready_artifacts(experiment: str) -> list[str]:
    """Remove any same-day READY marker before starting a new production run."""
    out_value = entry._arg_value("--out-root")
    if not out_value:
        raise RuntimeError("production invalidation requires --out-root")
    out_root = Path(out_value).expanduser().resolve()
    strategies_root = out_root / "strategies"
    archived: list[str] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_root = out_root / "_superseded_execution_batches" / stamp

    if strategies_root.is_dir():
        for strategy_dir in sorted(p for p in strategies_root.iterdir() if p.is_dir()):
            batch = strategy_dir / "execution_batch.json"
            if batch.is_file():
                archive_root.mkdir(parents=True, exist_ok=True)
                target = archive_root / f"{strategy_dir.name}.json"
                shutil.move(str(batch), str(target))
                archived.append(str(target))
            manifest_file = strategy_dir / "strategy_manifest.json"
            if manifest_file.is_file():
                payload = json.loads(manifest_file.read_text(encoding="utf-8"))
                for key in (
                    "execution_batch_file",
                    "execution_batch_protocol",
                    "execution_price_source",
                ):
                    payload.pop(key, None)
                payload["execution_batch_state"] = "building"
                entry._atomic_write_json(manifest_file, payload)

    root_manifest_file = out_root / "live_nine_strategy_manifest.json"
    if root_manifest_file.is_file():
        payload = json.loads(root_manifest_file.read_text(encoding="utf-8"))
        payload.pop("execution_batch_protocol", None)
        payload["execution_batch_files"] = {}
        payload["execution_batch_state"] = "building"
        entry._atomic_write_json(root_manifest_file, payload)
    return archived


def archive_nonproduction_strategy_dirs(experiment: str) -> list[str]:
    out_value = entry._arg_value("--out-root")
    if not out_value:
        raise RuntimeError("production archive requires --out-root")
    out_root = Path(out_value).expanduser().resolve()
    strategies_root = out_root / "strategies"
    if not strategies_root.is_dir():
        return []
    extras = [p for p in strategies_root.iterdir() if p.is_dir() and p.name != experiment]
    if not extras:
        return []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = out_root / "_superseded_strategies" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for path in extras:
        target = archive / path.name
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return moved


def publish_execution_batch_as_final_commit(experiment: str) -> Path:
    """Stage READY writes while manifests are updated, then commit the one batch last."""
    original_atomic = entry._atomic_write_json
    staged: list[tuple[Path, dict[str, Any]]] = []

    def stage_batch(path: Path, payload: dict[str, Any]) -> None:
        if Path(path).name == "execution_batch.json":
            staged.append((Path(path), dict(payload)))
            return
        original_atomic(Path(path), payload)

    entry._atomic_write_json = stage_batch
    try:
        entry.publish_execution_batches()
    finally:
        entry._atomic_write_json = original_atomic

    if len(staged) != 1:
        raise RuntimeError(f"production must stage exactly one execution batch; got {len(staged)}")
    batch_path, batch = staged[0]
    if batch_path.parent.name != experiment or str(batch.get("experiment")) != experiment:
        raise RuntimeError(
            f"staged execution batch is not the production experiment: path={batch_path} "
            f"payload={batch.get('experiment')} expected={experiment}"
        )

    # This is intentionally the final fallible production write.  The presence
    # of execution_batch.json is the READY marker consumed by Windows.
    original_atomic(batch_path, batch)
    return batch_path


def main() -> None:
    global_registry_root = _pop_custom_arg(
        "--model-registry-root", DEFAULT_MODEL_REGISTRY_ROOT
    )
    experiment = _pop_custom_arg(
        "--production-experiment", DEFAULT_PRODUCTION_EXPERIMENT
    )
    feature_preset = entry._arg_value("--feature-preset") or "rotation_addon_onehot"
    trade_value = entry._arg_value("--trade-date")
    if not trade_value:
        raise RuntimeError("production planner requires --trade-date")

    install_production_filter(experiment)
    install_fail_closed_tracking_start()
    archived_batches = invalidate_existing_ready_artifacts(experiment)
    archived_strategies = archive_nonproduction_strategy_dirs(experiment)
    validate_production_tracking_contract(experiment)

    entry._EXECUTION_CALENDAR = entry._load_execution_calendar_from_cli()
    entry.install_live_summary_adapter()
    entry.install_tracking_state_adapter()

    print(
        f"[PRODUCTION] experiment={experiment} prediction_models=1 "
        "tracking_contract=fail_closed ready_state=invalidated"
    )
    if archived_batches:
        print(f"[PRODUCTION] archived_ready_batches={len(archived_batches)}")
    if archived_strategies:
        print(f"[PRODUCTION] archived_nonproduction_strategy_dirs={len(archived_strategies)}")

    # Everything below this point must complete before READY is published.
    planner.main()
    entry.postprocess_manifests()
    record_live_generation_use(
        Path(global_registry_root),
        trade_date=planner.live.parse_trade_date(trade_value).strftime("%Y-%m-%d"),
        feature_preset=feature_preset,
    )

    # Final commit. Do not add any required/fallible state transition after this.
    batch_path = publish_execution_batch_as_final_commit(experiment)
    print(f"[READY] {batch_path}")


if __name__ == "__main__":
    main()
