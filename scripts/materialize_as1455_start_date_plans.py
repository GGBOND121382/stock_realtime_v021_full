#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute all existing 14:55 plans for the configured tracking start date.

This command is intentionally run only when the user changes/rebuilds the
tracking start date.  It reuses saved predictions/ranks and execution snapshots;
it does not run TensorFlow and does not recompute historical Fold/Grid results.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dashboard.as1455_plan_preview import compute_nine_strategy_day  # noqa: E402
from utils.as1455_materialized_plan import (  # noqa: E402
    PLAN_CACHE_VERSION,
    atomic_json,
    matrix_manifest_path,
    write_materialized_day,
)
from utils.as1455_tracking import TRACKING_SEMANTICS_VERSION, tracking_start_date  # noqa: E402

DATE_RE = re.compile(r"^\d{8}$")


def discover_source_dates(live_root: Path, start: pd.Timestamp) -> list[str]:
    dates: list[str] = []
    if not live_root.is_dir():
        return dates
    for item in live_root.iterdir():
        if not item.is_dir() or not DATE_RE.fullmatch(item.name):
            continue
        date = pd.to_datetime(item.name, format="%Y%m%d").normalize()
        if date < start:
            continue
        nine = item / "nine_strategy"
        # Saved predictions + sidecar are the actual expensive/live inputs needed
        # by the lightweight replay.  A canonical plan manifest is not required:
        # this also lets us recover a day whose old planner failed after inference.
        required = [
            item / "08_live_execution_sidecar.csv",
            item / "05_execution_calendar.csv",
            nine / "shared_predictions" / "r01" / "top5_live_predictions.csv",
            nine / "shared_predictions" / "r05" / "top5_live_predictions.csv",
            nine / "shared_predictions" / "r21" / "top5_live_predictions.csv",
        ]
        if all(path.is_file() for path in required):
            dates.append(item.name)
    return sorted(dates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix-root",
        default="saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1",
    )
    ap.add_argument("--live-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--tracking-start-date", default=None)
    ap.add_argument("--feature-preset", default="rotation_addon_onehot")
    args = ap.parse_args()

    matrix_root = Path(args.matrix_root).expanduser().resolve()
    live_root = Path(args.live_root).expanduser().resolve()
    start = (
        pd.Timestamp(args.tracking_start_date).normalize()
        if args.tracking_start_date
        else tracking_start_date(matrix_root)
    )
    if start is None:
        raise RuntimeError("tracking_start_date is not configured")

    dates = discover_source_dates(live_root, start)
    completed: list[str] = []
    print(
        f"[MATERIALIZE] start={start:%Y-%m-%d} source_dates={len(dates)} "
        "model_inference=no historical_grid=no"
    )
    for token in dates:
        selected = pd.to_datetime(token, format="%Y%m%d").normalize()
        preview = compute_nine_strategy_day(
            matrix_root,
            live_root,
            start,
            selected,
            feature_preset=args.feature_preset,
        )
        if preview.get("status") != "ok" or len(preview.get("details") or {}) != 9:
            raise RuntimeError(
                f"cannot materialize {token}: status={preview.get('status')} "
                f"detail_count={len(preview.get('details') or {})}"
            )
        write_materialized_day(
            live_root,
            token,
            start,
            preview,
            TRACKING_SEMANTICS_VERSION,
        )
        completed.append(token)
        print(f"[MATERIALIZE] {token} 9/9 complete")

    atomic_json(
        matrix_manifest_path(matrix_root),
        {
            "status": "ok",
            "protocol": "as1455_materialized_start_date_plan_matrix_v1",
            "plan_cache_version": PLAN_CACHE_VERSION,
            "tracking_start_date": start.strftime("%Y-%m-%d"),
            "tracking_semantics_version": TRACKING_SEMANTICS_VERSION,
            "completed_date_count": len(completed),
            "dates": completed,
            "model_inference_rerun": False,
            "historical_fold_grid_rerun": False,
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "tracking_start_date": start.strftime("%Y-%m-%d"),
                "completed_date_count": len(completed),
                "dates": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
