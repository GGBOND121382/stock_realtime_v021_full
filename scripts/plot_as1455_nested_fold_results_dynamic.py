#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the existing nested-fold plotting pipeline for a dynamic segment count."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts import plot_as1455_nested_fold_results as base


def load_segments_dynamic(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    table = base.read_csv(root / "nested_fold_target_results.csv")
    required = {"segment", "source_fold", "target_fold"}
    if not required.issubset(table.columns):
        raise RuntimeError(
            f"target result table missing: {sorted(required - set(table.columns))}"
        )
    table["source_fold"] = pd.to_numeric(
        table["source_fold"], errors="raise"
    ).astype(int)
    table = table.sort_values("source_fold", ascending=False).reset_index(drop=True)
    segments: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        source_fold = int(row["source_fold"])
        target_fold = row["target_fold"]
        segment_root = (
            root / f"source_fold{source_fold}" / "forward"
            if pd.isna(target_fold)
            else root / f"source_fold{source_fold}" / f"target_fold{int(target_fold)}"
        )
        run_dir = base.retained_run(segment_root)
        config = base.read_json(run_dir / "config.json")
        initial_cash = float(config.get("initial_cash", 200000.0))
        nav = base.load_nav(run_dir / "close_auction_nav.csv")
        nav["return_pct"] = (nav["nav"] / initial_cash - 1.0) * 100.0
        segments.append(
            {
                "label": str(row["segment"]),
                "run_name": run_dir.name,
                "run_dir": run_dir,
                "curve": nav[["date", "nav", "return_pct"]],
            }
        )
    if not segments:
        raise RuntimeError("no nested target/forward segments were found")
    return table, segments


base.load_segments = load_segments_dynamic

if __name__ == "__main__":
    base.main()
