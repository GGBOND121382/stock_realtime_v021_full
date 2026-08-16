#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the tracking-aware live planner for one production best-model strategy.

The frozen nine-strategy research matrix is left untouched.  This adapter only
narrows the latency-critical 14:55 planner to one selected ``best`` experiment
and allows its live prediction file to contain only model column 0.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner  # noqa: E402
from scripts import run_as1455_live_nine_strategy_planner_entry as entry  # noqa: E402

DEFAULT_PRODUCTION_EXPERIMENT = "r21_best_reb21_fold0_4_forward"


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


def install_selected_batch_publication(experiment: str) -> None:
    """Archive stale same-day strategy dirs before publishing the one live batch."""
    original_publish = entry.publish_execution_batches

    def publish_selected() -> None:
        out_value = entry._arg_value("--out-root")
        if not out_value:
            raise RuntimeError("production publication requires --out-root")
        out_root = Path(out_value).expanduser().resolve()
        strategies_root = out_root / "strategies"
        extras = (
            [
                path
                for path in strategies_root.iterdir()
                if path.is_dir() and path.name != experiment
            ]
            if strategies_root.is_dir()
            else []
        )
        if extras:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive = out_root / "_superseded_strategies" / stamp
            archive.mkdir(parents=True, exist_ok=False)
            for path in extras:
                shutil.move(str(path), str(archive / path.name))
            print(
                f"[PRODUCTION] archived {len(extras)} stale non-production "
                f"strategy directories under {archive}"
            )
        original_publish()

    entry.publish_execution_batches = publish_selected


def main() -> None:
    experiment = _pop_custom_arg(
        "--production-experiment", DEFAULT_PRODUCTION_EXPERIMENT
    )
    install_production_filter(experiment)
    install_selected_batch_publication(experiment)
    print(
        f"[PRODUCTION] experiment={experiment} prediction_models=1 "
        "research_matrix_unchanged=yes"
    )
    entry.main()


if __name__ == "__main__":
    main()
