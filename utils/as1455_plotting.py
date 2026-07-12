#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plotting helpers for AS1455 backtest figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")


def curve_style(index: int) -> dict[str, Any]:
    return {
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)],
    }


def plot_frequency(
    *,
    curves: list[dict[str, Any]],
    frequency: str,
    out_file: Path,
    title: str,
    sample_curve: Callable[[pd.DataFrame, str], pd.DataFrame],
    plt: Any,
) -> pd.DataFrame:
    """Plot curves with both line styles and sparse markers.

    The chart remains distinguishable without relying on color alone.  The
    exported CSV records the selected style for each curve.
    """
    plt.figure(figsize=(12, 6))
    rows: list[pd.DataFrame] = []
    for index, item in enumerate(curves):
        sampled = sample_curve(item["curve"], frequency)
        style = curve_style(index)
        markevery = max(1, len(sampled) // 12)
        plt.plot(
            sampled["date"],
            sampled["return_pct"],
            linewidth=1.9,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=markevery,
            markersize=5.0,
            fillstyle="none",
            label=item["label"],
        )
        exported = sampled.copy()
        exported.insert(0, "label", item["label"])
        exported.insert(1, "run_name", item["run_name"])
        exported.insert(2, "frequency", frequency)
        exported.insert(3, "line_style", style["linestyle"])
        exported.insert(4, "marker", style["marker"])
        rows.append(exported)

    plt.axhline(0.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative return (%)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=160)
    plt.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
