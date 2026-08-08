#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export compact latest-account state before full strict-forward audit files are pruned.

The nine-strategy live monitor runs at 14:55 on trade date T.  Its simulated
portfolio state must therefore come from the completed strict-forward backtest
through T-1.  This exporter captures the latest NAV row and latest daily position
rows from every global fixed-signal experiment, then exposes stable files at the
experiment root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT_POSITIONS_FILE = "strict_forward_latest_positions.csv"
EXPERIMENT_STATE_FILE = "strict_forward_latest_state.json"
RUN_POSITIONS_FILE = "close_auction_latest_positions.csv"
RUN_STATE_FILE = "close_auction_latest_state.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_csv(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise


def replace_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def retained_run(experiment_root: Path) -> tuple[Path, dict[str, Any]]:
    strict_file = (
        experiment_root
        / "strict_oos_forward"
        / "01_close_auction_grid"
        / "strict_oos_manifest.json"
    )
    strict = read_json(strict_file)
    run_name = strict.get("retained_run_name")
    if not run_name:
        raise RuntimeError(f"retained_run_name missing: {strict_file}")
    run_dir = strict_file.parent / "01_runs" / str(run_name)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir, strict


def export_experiment(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.expanduser().resolve()
    run_dir, strict = retained_run(experiment_root)
    nav_file = run_dir / "close_auction_nav.csv"
    positions_file = run_dir / "close_auction_positions.csv"
    nav = read_csv(nav_file)
    required_nav = {"date", "nav", "cash", "n_positions"}
    missing_nav = required_nav - set(nav.columns)
    if missing_nav:
        raise RuntimeError(f"{nav_file} missing columns: {sorted(missing_nav)}")
    nav = nav.copy()
    nav["date"] = pd.to_datetime(nav["date"], errors="raise").dt.normalize()
    nav = nav.sort_values("date").drop_duplicates("date", keep="last")
    if nav.empty:
        raise RuntimeError(f"empty NAV: {nav_file}")
    last_nav = nav.iloc[-1].copy()
    latest_date = pd.Timestamp(last_nav["date"]).normalize()

    positions = read_csv(positions_file, allow_empty=True)
    if positions.empty:
        latest_positions = pd.DataFrame(
            columns=[
                "date", "symbol", "shares", "buy_date", "avg_entry_price",
                "entry_rank", "entry_score", "cost_basis_notional",
                "cost_basis_fee",
            ]
        )
    else:
        if not {"date", "symbol", "shares"}.issubset(positions.columns):
            raise RuntimeError(
                f"{positions_file} requires date/symbol/shares; columns={list(positions.columns)}"
            )
        positions = positions.copy()
        positions["date"] = pd.to_datetime(
            positions["date"], errors="raise"
        ).dt.normalize()
        latest_positions = positions.loc[positions["date"].eq(latest_date)].copy()
        latest_positions = latest_positions.sort_values(
            [column for column in ["weight", "symbol"] if column in latest_positions.columns],
            ascending=[False, True] if "weight" in latest_positions.columns else [True],
        )

    expected = int(last_nav["n_positions"])
    actual = int(latest_positions["symbol"].nunique()) if not latest_positions.empty else 0
    if expected != actual:
        raise RuntimeError(
            f"latest position count mismatch for {experiment_root.name}: "
            f"date={latest_date:%Y-%m-%d} nav={expected} positions={actual}"
        )

    run_positions = run_dir / RUN_POSITIONS_FILE
    latest_positions.to_csv(run_positions, index=False, encoding="utf-8-sig")
    experiment_positions = experiment_root / EXPERIMENT_POSITIONS_FILE
    replace_link_or_copy(run_positions, experiment_positions)

    state = {
        "status": "ok",
        "experiment": experiment_root.name,
        "asof_date": latest_date.strftime("%Y-%m-%d"),
        "nav": float(last_nav["nav"]),
        "cash": float(last_nav["cash"]),
        "n_positions": expected,
        "is_rebalance_day": bool(last_nav.get("is_rebalance_day", False)),
        "day_index": int(last_nav["day_index"]) if "day_index" in last_nav and pd.notna(last_nav["day_index"]) else None,
        "rebalance_every": int(last_nav["rebalance_every"]) if "rebalance_every" in last_nav and pd.notna(last_nav["rebalance_every"]) else None,
        "rebalance_offset": int(last_nav["rebalance_offset"]) if "rebalance_offset" in last_nav and pd.notna(last_nav["rebalance_offset"]) else None,
        "retained_run_name": strict.get("retained_run_name"),
        "retained_run_dir": str(run_dir),
        "positions_file": str(experiment_positions),
        "source_nav_file": str(nav_file),
        "source_positions_file": str(positions_file),
        "semantics": "post_close_1500_simulated_strict_forward_state",
    }
    run_state = run_dir / RUN_STATE_FILE
    write_json(run_state, state)
    experiment_state = experiment_root / EXPERIMENT_STATE_FILE
    replace_link_or_copy(run_state, experiment_state)
    print(
        "[OK] latest forward state: "
        f"experiment={experiment_root.name} date={state['asof_date']} "
        f"cash={state['cash']:.2f} positions={expected}"
    )
    return state


def export_matrix(matrix_root: Path) -> dict[str, Any]:
    matrix_root = matrix_root.expanduser().resolve()
    expected_file = matrix_root / "expected_experiments.txt"
    if not expected_file.is_file():
        raise FileNotFoundError(expected_file)
    names = [
        line.strip()
        for line in expected_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != 9:
        raise RuntimeError(f"expected nine experiments, got {len(names)}")
    states = [export_experiment(matrix_root / name) for name in names]
    summary = pd.DataFrame(states)
    summary_file = matrix_root / "strict_forward_latest_states.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "ok",
        "experiment_count": len(states),
        "asof_dates": sorted({item["asof_date"] for item in states}),
        "summary_file": str(summary_file),
        "state_files": {
            item["experiment"]: str(matrix_root / item["experiment"] / EXPERIMENT_STATE_FILE)
            for item in states
        },
        "semantics": (
            "latest completed strict-forward state, intended as T-1 simulated "
            "account input for the next 14:55 nine-strategy monitor"
        ),
    }
    write_json(matrix_root / "strict_forward_latest_states_manifest.json", manifest)
    print(f"[PASS] exported latest account states for {len(states)} experiments")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-root",
        default=(
            "saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/"
            "refresh_all_v1"
        ),
    )
    args = parser.parse_args()
    export_matrix(Path(args.matrix_root))


if __name__ == "__main__":
    main()
