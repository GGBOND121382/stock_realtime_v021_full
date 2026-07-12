#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run only the selected historical grid row with compact/full artifacts.

A large historical search should use ``OUTPUT_MODE=summary``.  This script then
materializes the single best row so plotting and audit retain one NAV curve
instead of thousands of duplicate per-configuration time series.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_model_selection import (  # noqa: E402
    find_summary_file,
    select_historical_signal,
)
from utils.as1455_strict_oos import finalize_strict_oos_grid  # noqa: E402


def find_prediction_file(root: Path) -> Path:
    preferred = root / "00_predictions" / "test_preds.h5"
    if preferred.exists() and preferred.stat().st_size > 0:
        return preferred
    matches = sorted((root / "00_predictions").glob("*.h5"))
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot uniquely resolve prediction HDF under {root / '00_predictions'}: "
            f"{matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the best AS1455 historical grid row"
    )
    parser.add_argument("--backtest-root", required=True)
    parser.add_argument("--raw-daily-cache-dir", required=True)
    parser.add_argument("--rank-metric", default="sharpe")
    parser.add_argument("--capacity-mode", default="none")
    parser.add_argument("--output-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.backtest_root).expanduser().resolve()
    summary_file, grid_dir = find_summary_file(root)
    selection = select_historical_signal(
        backtest_root=root,
        rank_metric=args.rank_metric,
    )
    run_name = selection.run_name
    final_run_dir = grid_dir / "01_runs" / run_name
    final_nav = final_run_dir / "close_auction_nav.csv"
    if final_nav.exists() and final_nav.stat().st_size > 0 and not args.force:
        print(f"[SKIP] selected run already materialized: {final_nav}")
        return

    prediction_file = find_prediction_file(root)
    rebalance_every = int(selection.historical_rebalance_every or 0)
    offset = int(selection.historical_rebalance_offset or 0)
    max_positions = int(selection.historical_max_positions or 0)
    sell_rank = int(selection.historical_sell_rank or 0)
    if min(rebalance_every, max_positions, sell_rank) <= 0:
        raise RuntimeError(
            "selected historical row has incomplete trading parameters: "
            f"{selection.to_dict()}"
        )

    temp_out = root / "03_materialize_best_tmp"
    if temp_out.exists():
        shutil.rmtree(temp_out)
    temp_grid = temp_out / "01_close_auction_grid"
    command = common.build_grid_command(
        python_bin=sys.executable or "python3",
        grid_script=PROJECT_DIR
        / "code"
        / "backtest"
        / "run_as1455_close_auction_grid_inprocess.py",
        grid_out=temp_grid,
        prediction_file=prediction_file,
        raw_daily_cache_dir=Path(args.raw_daily_cache_dir),
        profile="close_auction_skip_limit",
        capacity_mode=args.capacity_mode,
        output_mode=args.output_mode,
        offset_mode="zero" if offset == 0 else "full",
        rebalance_every=rebalance_every,
        max_positions_list=str(max_positions),
        sell_rank_list=str(sell_rank),
        model_family="AS1455 materialized historical best",
        model_run=(
            f"source_root={root}; metric={args.rank_metric}; "
            f"source_summary={summary_file}"
        ),
        force_grid=True,
    )
    command.extend(["--signal-spec", selection.signal_spec])
    common.run_command(command, dry_run=args.dry_run)
    if args.dry_run:
        return

    strict_manifest = finalize_strict_oos_grid(temp_out, selection)
    materialized = temp_grid / "01_runs" / run_name
    if not (materialized / "close_auction_nav.csv").exists():
        raise RuntimeError(f"materialized selected run is missing NAV: {materialized}")

    if final_run_dir.exists():
        shutil.rmtree(final_run_dir)
    final_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(materialized, final_run_dir)

    payload = {
        "rank_metric": args.rank_metric,
        "source_summary": str(summary_file),
        "selection": selection.to_dict(),
        "prediction_file": str(prediction_file),
        "output_mode": args.output_mode,
        "materialized_run_dir": str(final_run_dir),
        "temporary_strict_manifest": strict_manifest,
    }
    (root / "materialized_best_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.rmtree(temp_out)
    print(f"[OK] materialized selected run: {final_run_dir}")


if __name__ == "__main__":
    main()
