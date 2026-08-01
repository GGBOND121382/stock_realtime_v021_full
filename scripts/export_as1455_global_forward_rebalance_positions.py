#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export post-rebalance holdings for fixed-signal strict-forward backtests.

The grid temporarily retains full audit output for the single frozen forward run.
This script extracts a compact, stable snapshot for every scheduled rebalance day,
including days with no trades and days on which the portfolio is empty.  It can
then remove full all-date audit CSVs so the permanent footprint remains close to
compact mode.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

FULL_ONLY_FILES = (
    "close_auction_orders.csv",
    "close_auction_trades.csv",
    "close_auction_rejections.csv",
    "close_auction_positions.csv",
    "close_auction_corporate_actions.csv",
    "round_trips.csv",
)
RUN_DATES_FILE = "close_auction_rebalance_dates.csv"
RUN_POSITIONS_FILE = "close_auction_rebalance_positions.csv"
EXPERIMENT_DATES_FILE = "strict_forward_rebalance_dates.csv"
EXPERIMENT_POSITIONS_FILE = "strict_forward_rebalance_positions.csv"
EXPERIMENT_MANIFEST_FILE = "strict_forward_rebalance_positions_manifest.json"


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
    if path.stat().st_size == 0 and allow_empty:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def replace_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def retained_run(forward_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    grid_root = forward_root / "01_close_auction_grid"
    strict_file = grid_root / "strict_oos_manifest.json"
    strict = read_json(strict_file)
    run_name = strict.get("retained_run_name")
    if not run_name:
        raise RuntimeError(f"retained_run_name missing: {strict_file}")
    run_dir = grid_root / "01_runs" / str(run_name)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir, strict_file, strict


def prepare_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    nav_file = run_dir / "close_auction_nav.csv"
    positions_file = run_dir / "close_auction_positions.csv"
    nav = read_csv(nav_file)
    required_nav = {"date", "day_index", "is_rebalance_day", "nav", "cash", "n_positions"}
    missing_nav = required_nav - set(nav.columns)
    if missing_nav:
        raise RuntimeError(f"{nav_file} missing columns: {sorted(missing_nav)}")

    nav = nav.copy()
    nav["date"] = pd.to_datetime(nav["date"], errors="raise").dt.normalize()
    nav["is_rebalance_day"] = as_bool(nav["is_rebalance_day"])
    rebalance = (
        nav.loc[nav["is_rebalance_day"]]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if rebalance.empty:
        raise RuntimeError(f"no scheduled rebalance dates in {nav_file}")
    rebalance.insert(0, "rebalance_sequence", range(1, len(rebalance) + 1))
    rebalance["snapshot_phase"] = "post_rebalance_close_1500"

    positions = read_csv(positions_file, allow_empty=True)
    if not positions.empty:
        required_positions = {"date", "symbol", "shares", "value", "weight"}
        missing_positions = required_positions - set(positions.columns)
        if missing_positions:
            raise RuntimeError(
                f"{positions_file} missing columns: {sorted(missing_positions)}"
            )
        positions = positions.copy()
        positions["date"] = pd.to_datetime(
            positions["date"], errors="raise"
        ).dt.normalize()
        positions = positions[positions["date"].isin(set(rebalance["date"]))]
    else:
        positions = pd.DataFrame(
            columns=["date", "symbol", "shares", "value", "weight"]
        )

    portfolio_columns = [
        "rebalance_sequence",
        "date",
        "day_index",
        "nav",
        "cash",
        "cash_ratio",
        "holding_value",
        "gross_exposure",
        "n_positions",
        "turnover",
        "gross_trade_amount",
        "total_fee",
        "orders",
        "buy_orders",
        "sell_orders",
        "partial_fill_orders",
        "rejections",
        "max_positions",
        "buy_candidate_rank",
        "sell_rank",
        "rebalance_every",
        "rebalance_offset",
        "snapshot_phase",
    ]
    portfolio_columns = [column for column in portfolio_columns if column in rebalance.columns]
    dates_table = rebalance[portfolio_columns].copy()
    dates_table = dates_table.rename(
        columns={
            column: f"portfolio_{column}"
            for column in dates_table.columns
            if column not in {"rebalance_sequence", "date", "snapshot_phase"}
        }
    )

    merged = dates_table.merge(positions, on="date", how="left", validate="one_to_many")
    merged["portfolio_empty"] = merged["symbol"].isna()
    merged["scheduled_rebalance_day"] = True
    merged["held_after_rebalance"] = ~merged["portfolio_empty"]
    merged = merged.sort_values(
        ["date", "portfolio_empty", "weight", "symbol"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    merged["position_ordinal"] = (
        merged.loc[~merged["portfolio_empty"]]
        .groupby("date")
        .cumcount()
        .add(1)
        .reindex(merged.index)
        .fillna(0)
        .astype(int)
    )

    expected_counts = (
        dates_table.set_index("date")["portfolio_n_positions"].astype(int)
    )
    actual_counts = (
        merged.loc[~merged["portfolio_empty"]]
        .groupby("date")["symbol"]
        .nunique()
        .reindex(expected_counts.index, fill_value=0)
        .astype(int)
    )
    mismatch = expected_counts[expected_counts != actual_counts]
    if not mismatch.empty:
        details = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "nav_n_positions": int(expected_counts.loc[date]),
                "exported_symbols": int(actual_counts.loc[date]),
            }
            for date in mismatch.index
        ]
        raise RuntimeError(f"rebalance position-count mismatch: {details}")

    if set(merged["date"]) != set(dates_table["date"]):
        raise RuntimeError("not every scheduled rebalance date is represented")

    report = {
        "rebalance_date_count": int(len(dates_table)),
        "position_row_count": int((~merged["portfolio_empty"]).sum()),
        "empty_portfolio_rebalance_date_count": int(
            merged.loc[merged["portfolio_empty"], "date"].nunique()
        ),
        "first_rebalance_date": dates_table["date"].min().strftime("%Y-%m-%d"),
        "last_rebalance_date": dates_table["date"].max().strftime("%Y-%m-%d"),
    }
    return dates_table, merged, report


def update_run_metadata(run_dir: Path, removed: list[str], generated: list[str]) -> None:
    config_file = run_dir / "config.json"
    if config_file.exists():
        config = read_json(config_file)
        config.update(
            {
                "output_mode": "compact_with_rebalance_positions",
                "rebalance_position_snapshot_phase": "post_rebalance_close_1500",
                "full_daily_audit_pruned": bool(removed),
            }
        )
        write_json(config_file, config)

    summary_file = run_dir / "close_auction_summary.json"
    if summary_file.exists():
        summary = read_json(summary_file)
        output_files = sorted(path.name for path in run_dir.glob("*.csv"))
        suppressed = set(summary.get("suppressed_output_files") or [])
        suppressed.update(removed)
        summary.update(
            {
                "output_mode": "compact_with_rebalance_positions",
                "output_files": output_files,
                "suppressed_output_files": sorted(suppressed),
                "rebalance_position_files": generated,
                "rebalance_position_snapshot_phase": "post_rebalance_close_1500",
            }
        )
        write_json(summary_file, summary)


def export_experiment(experiment_root: Path, prune_full_audit: bool) -> dict[str, Any]:
    experiment_root = experiment_root.expanduser().resolve()
    forward_root = experiment_root / "strict_oos_forward"
    run_dir, strict_file, strict = retained_run(forward_root)
    dates_table, positions_table, report = prepare_tables(run_dir)

    run_dates = run_dir / RUN_DATES_FILE
    run_positions = run_dir / RUN_POSITIONS_FILE
    dates_table.to_csv(run_dates, index=False, encoding="utf-8-sig")
    positions_table.to_csv(run_positions, index=False, encoding="utf-8-sig")

    forward_dates = forward_root / RUN_DATES_FILE
    forward_positions = forward_root / RUN_POSITIONS_FILE
    experiment_dates = experiment_root / EXPERIMENT_DATES_FILE
    experiment_positions = experiment_root / EXPERIMENT_POSITIONS_FILE
    for source, target in (
        (run_dates, forward_dates),
        (run_positions, forward_positions),
        (run_dates, experiment_dates),
        (run_positions, experiment_positions),
    ):
        replace_link_or_copy(source, target)

    removed: list[str] = []
    if prune_full_audit:
        for filename in FULL_ONLY_FILES:
            path = run_dir / filename
            if path.exists():
                path.unlink()
                removed.append(filename)

    generated = [RUN_DATES_FILE, RUN_POSITIONS_FILE]
    update_run_metadata(run_dir, removed, generated)

    manifest = {
        "status": "ok",
        "protocol": "strict_forward_all_scheduled_rebalance_days_post_trade_holdings",
        "snapshot_phase": "post_rebalance_close_1500",
        "experiment_root": str(experiment_root),
        "forward_root": str(forward_root),
        "strict_oos_manifest": str(strict_file),
        "retained_run_name": strict.get("retained_run_name"),
        "retained_run_dir": str(run_dir),
        "run_rebalance_dates_file": str(run_dates),
        "run_rebalance_positions_file": str(run_positions),
        "forward_rebalance_dates_file": str(forward_dates),
        "forward_rebalance_positions_file": str(forward_positions),
        "experiment_rebalance_dates_file": str(experiment_dates),
        "experiment_rebalance_positions_file": str(experiment_positions),
        "all_scheduled_rebalance_dates_included": True,
        "no_trade_rebalance_dates_included": True,
        "empty_portfolio_rebalance_dates_included": True,
        "full_daily_audit_pruned": prune_full_audit,
        "removed_full_audit_files": removed,
        **report,
    }
    manifest_file = experiment_root / EXPERIMENT_MANIFEST_FILE
    write_json(manifest_file, manifest)
    replace_link_or_copy(
        manifest_file, forward_root / EXPERIMENT_MANIFEST_FILE
    )
    print(
        "[OK] forward rebalance positions: "
        f"experiment={experiment_root.name} dates={report['rebalance_date_count']} "
        f"positions={report['position_row_count']} empty_dates="
        f"{report['empty_portfolio_rebalance_date_count']}"
    )
    return manifest


def export_matrix(matrix_root: Path, prune_full_audit: bool) -> dict[str, Any]:
    matrix_root = matrix_root.expanduser().resolve()
    expected_file = matrix_root / "expected_experiments.txt"
    if expected_file.exists():
        experiments = [
            matrix_root / line.strip()
            for line in expected_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        experiments = sorted(
            path.parent
            for path in matrix_root.glob(
                "*/strict_oos_forward/01_close_auction_grid/strict_oos_manifest.json"
            )
        )
    if not experiments:
        raise RuntimeError(f"no strict-forward experiments found under {matrix_root}")

    manifests = [export_experiment(path, prune_full_audit) for path in experiments]
    rows = [
        {
            "experiment": Path(item["experiment_root"]).name,
            "rebalance_date_count": item["rebalance_date_count"],
            "position_row_count": item["position_row_count"],
            "empty_portfolio_rebalance_date_count": item[
                "empty_portfolio_rebalance_date_count"
            ],
            "first_rebalance_date": item["first_rebalance_date"],
            "last_rebalance_date": item["last_rebalance_date"],
            "positions_file": item["experiment_rebalance_positions_file"],
            "dates_file": item["experiment_rebalance_dates_file"],
            "manifest_file": str(
                Path(item["experiment_root"]) / EXPERIMENT_MANIFEST_FILE
            ),
        }
        for item in manifests
    ]
    inventory = pd.DataFrame(rows)
    inventory_file = matrix_root / "forward_rebalance_positions_inventory.csv"
    inventory.to_csv(inventory_file, index=False, encoding="utf-8-sig")
    payload = {
        "status": "ok",
        "experiment_count": len(rows),
        "all_scheduled_rebalance_dates_included": True,
        "snapshot_phase": "post_rebalance_close_1500",
        "inventory_file": str(inventory_file),
        "experiments": rows,
    }
    manifest_file = matrix_root / "forward_rebalance_positions_manifest.json"
    write_json(manifest_file, payload)

    matrix_manifest_file = matrix_root / "fixed_signal_matrix_manifest.json"
    if matrix_manifest_file.exists():
        matrix_manifest = read_json(matrix_manifest_file)
        matrix_manifest["forward_rebalance_positions"] = payload
        write_json(matrix_manifest_file, matrix_manifest)

    print(f"[PASS] matrix forward rebalance positions={inventory_file}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--experiment-root")
    scope.add_argument("--matrix-root")
    parser.add_argument("--prune-full-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment_root:
        export_experiment(Path(args.experiment_root), args.prune_full_audit)
    else:
        export_matrix(Path(args.matrix_root), args.prune_full_audit)


if __name__ == "__main__":
    main()
