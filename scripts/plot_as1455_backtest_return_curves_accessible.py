#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accessible AS1455 return-curve plotting.

Reuses the existing root/run selection logic, but distinguishes curves with both
line styles and markers so the figures do not rely on color perception alone.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in [PROJECT_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_as1455_backtest_return_curves as base  # noqa: E402

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def plot_frequency(
    curves: list[dict[str, Any]],
    frequency: str,
    out_file: Path,
    title: str,
) -> pd.DataFrame:
    base.plt.figure(figsize=(12, 6))
    rows = []
    for i, item in enumerate(curves):
        sampled = base.sample_curve(item["curve"], frequency)
        markevery = max(1, len(sampled) // 12)
        base.plt.plot(
            sampled["date"],
            sampled["return_pct"],
            linewidth=1.9,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=markevery,
            markersize=5.0,
            fillstyle="none",
            label=item["label"],
        )
        tmp = sampled.copy()
        tmp.insert(0, "label", item["label"])
        tmp.insert(1, "run_name", item["run_name"])
        tmp.insert(2, "frequency", frequency)
        tmp.insert(3, "line_style", LINE_STYLES[i % len(LINE_STYLES)])
        tmp.insert(4, "marker", MARKERS[i % len(MARKERS)])
        rows.append(tmp)
    base.plt.axhline(0.0, linewidth=1.0)
    base.plt.title(title)
    base.plt.xlabel("Date")
    base.plt.ylabel("Cumulative return (%)")
    base.plt.legend(loc="best", fontsize=9)
    base.plt.grid(True, alpha=0.3)
    base.plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(out_file, dpi=160)
    base.plt.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    base.plot_frequency = plot_frequency
    base.main()


if __name__ == "__main__":
    main()
