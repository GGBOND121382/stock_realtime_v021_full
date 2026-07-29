#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight checkpoint coverage for the requested seven global experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_ch17_common import default_fold_dir_template  # noqa: E402


def saved_checkpoint_count(path: Path) -> int:
    table = pd.read_csv(path)
    if "checkpoint_saved" in table.columns:
        saved = table["checkpoint_saved"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        table = table.loc[saved]
    return int(len(table))


def main() -> None:
    problems: list[str] = []
    checks = {
        "r01_fwd": range(1, 7),
        "r21_fwd": range(1, 7),
        "r05_fwd": (0,),
    }
    for target_col, source_folds in checks.items():
        template = default_fold_dir_template("rotation_addon_onehot", target_col)
        for source_fold in source_folds:
            root = Path(template.format(fold=source_fold)).expanduser().resolve()
            table_file = root / "search_best_checkpoints.csv"
            if not root.is_dir():
                problems.append(
                    f"{target_col} source_fold{source_fold}: missing directory {root}"
                )
                continue
            if not table_file.exists():
                problems.append(
                    f"{target_col} source_fold{source_fold}: missing {table_file}"
                )
                continue
            count = saved_checkpoint_count(table_file)
            if count < 5:
                problems.append(
                    f"{target_col} source_fold{source_fold}: "
                    f"need 5 saved checkpoints, got {count}"
                )
    if problems:
        print("[BLOCKED] checkpoint preflight failed:")
        for problem in problems:
            print(f"  - {problem}")
        print("For a missing r21 source_fold6, try:")
        print(
            "  TARGET_COL=r21_fwd FEATURE_PRESETS=rotation_addon_onehot "
            "FOLDS=6 bash scripts/run_as1455_target_search_all.sh"
        )
        raise SystemExit(3)
    print(
        "[PASS] checkpoint preflight: r01/r21 source_fold1..6 and "
        "r05 source_fold0 each provide at least five checkpoints"
    )


if __name__ == "__main__":
    main()
