#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_fold_calendar import fold_window_from_dates  # noqa: E402


def main() -> None:
    dates = pd.bdate_range("2018-01-01", periods=1700)
    report = {
        "common_target_end": dates[-1].strftime("%Y-%m-%d"),
        "target_valid_ends": {
            "r01_fwd": dates[-1].strftime("%Y-%m-%d"),
            "r05_fwd": dates[-1].strftime("%Y-%m-%d"),
            "r21_fwd": dates[-1].strftime("%Y-%m-%d"),
        },
    }
    for fold in range(7):
        windows = {
            target: fold_window_from_dates(dates, fold, target, report)
            for target in ("r01_fwd", "r05_fwd", "r21_fwd")
        }
        starts = {window.fold_start for window in windows.values()}
        ends = {window.fold_end for window in windows.values()}
        assert len(starts) == 1 and len(ends) == 1, windows
        assert windows["r01_fwd"].train_end > windows["r21_fwd"].train_end
        if fold:
            newer = fold_window_from_dates(dates, fold - 1, "r01_fwd", report)
            assert windows["r01_fwd"].fold_end < newer.fold_start
    print("[PASS] r1/r5/r21 fold0..fold6 share identical market-date windows")


if __name__ == "__main__":
    main()
