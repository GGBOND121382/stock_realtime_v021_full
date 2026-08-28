#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add scheduled rebalance-day markers to global-fold experiment plots.

The marker source is the retained run's ``close_auction_nav.csv``. When the
backtest exported ``is_rebalance_day`` it is used directly. Older artifacts may
fall back to ``day_index`` plus the frozen rebalance period and offset. This
script only redraws figures and writes a marker audit CSV; it never changes
predictions, selected parameters, trades, or NAV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_model_selection import (  # noqa: E402
    find_summary_file,
    read_csv_auto,
    select_historical_signal,
)

FIXED_SIGNAL_SPEC = "ensemble_first3_mean:0,1,2:mean"
TRUE_VALUES = {"1", "true", "t", "yes", "y"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def selected_run_dir(backtest_root: Path, run_name: str) -> Path:
    _summary_file, grid_dir = find_summary_file(backtest_root)
    run_dir = grid_dir / "01_runs" / run_name
    if run_dir.is_dir():
        return run_dir
    matches = sorted(backtest_root.glob(f"**/01_runs/{run_name}"))
    if not matches:
        raise FileNotFoundError(run_dir)
    return matches[0]


def parse_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def load_nav_with_rebalance(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    nav = read_csv_auto(path).copy()
    required = {"date", "nav"}
    missing = required - set(nav.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = (
        nav.dropna(subset=["date", "nav"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if len(nav) < 2:
        raise RuntimeError(f"NAV has fewer than two rows: {path}")

    if "is_rebalance_day" in nav.columns:
        nav["is_rebalance_day"] = parse_bool_series(nav["is_rebalance_day"])
        nav["rebalance_marker_source"] = "nav.is_rebalance_day"
        return nav

    index_column = next(
        (
            name
            for name in ("day_index", "global_day_index", "engine_day_index")
            if name in nav.columns
        ),
        None,
    )
    if index_column is None:
        raise RuntimeError(
            f"{path} has no is_rebalance_day or day-index column; "
            "cannot mark rebalance dates"
        )
    every = int(config.get("rebalance_every", 0))
    offset = int(config.get("rebalance_offset", 0))
    if every <= 0 or not 0 <= offset < every:
        raise RuntimeError(
            f"invalid frozen rebalance config: every={every} "
            f"offset={offset} file={path}"
        )
    day_index = pd.to_numeric(nav[index_column], errors="coerce")
    nav["is_rebalance_day"] = (
        day_index.notna()
        & day_index.ge(offset)
        & ((day_index.astype("Int64") - offset) % every).eq(0)
    )
    nav["rebalance_marker_source"] = f"derived:{index_column}"
    return nav


def rebalance_rows(nav: pd.DataFrame, phase: str) -> pd.DataFrame:
    rows = nav.loc[
        nav["is_rebalance_day"],
        ["date", "nav", "rebalance_marker_source"],
    ].copy()
    rows.insert(0, "phase", phase)
    rows = rows.rename(columns={"nav": "nav_on_rebalance_day"})
    return rows.reset_index(drop=True)


def add_rebalance_markers(
    ax: Any,
    curve: pd.DataFrame,
    y_column: str,
    *,
    line_color: Any | None = None,
) -> int:
    marked = curve.loc[curve["is_rebalance_day"], ["date", y_column]].copy()
    if marked.empty:
        return 0
    for date in marked["date"]:
        ax.axvline(
            date,
            linestyle=":",
            linewidth=0.8,
            alpha=0.28,
            color="0.45",
        )
    ax.scatter(
        marked["date"],
        marked[y_column],
        s=22,
        marker="o",
        facecolors="none",
        edgecolors=line_color,
        linewidths=1.1,
        zorder=5,
        label="Rebalance day",
    )
    return int(len(marked))


def plot_historical(
    nav: pd.DataFrame,
    config: dict[str, Any],
    segment_returns: pd.DataFrame,
    out_file: Path,
) -> int:
    initial_cash = float(config.get("initial_cash", 200000.0))
    curve = nav.copy()
    curve["return_pct"] = (curve["nav"] / initial_cash - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(15, 7))
    (line,) = ax.plot(
        curve["date"],
        curve["return_pct"],
        linewidth=1.6,
        label="Cumulative return",
    )
    marker_count = add_rebalance_markers(
        ax,
        curve,
        "return_pct",
        line_color=line.get_color(),
    )
    ax.axhline(0.0, linewidth=0.8)

    y_min = float(curve["return_pct"].min())
    y_max = float(curve["return_pct"].max())
    y_span = max(y_max - y_min, 1.0)
    label_y = y_max + 0.06 * y_span
    ordered = segment_returns.copy()
    ordered["start"] = pd.to_datetime(
        ordered["start"], errors="raise"
    ).dt.normalize()
    ordered["end"] = pd.to_datetime(
        ordered["end"], errors="raise"
    ).dt.normalize()
    ordered = ordered.sort_values("start")
    for row in ordered.itertuples(index=False):
        ax.axvline(
            row.start,
            linestyle="--",
            linewidth=0.9,
            alpha=0.68,
            color="0.2",
        )
        midpoint = row.start + (row.end - row.start) / 2
        ax.text(
            midpoint,
            label_y,
            f"fold{row.target_fold}\n{float(row.segment_return_pct):+.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axvline(
        ordered.iloc[-1]["end"],
        linestyle="--",
        linewidth=0.9,
        alpha=0.68,
        color="0.2",
    )
    ax.set_ylim(y_min - 0.08 * y_span, label_y + 0.12 * y_span)
    ax.set_title(
        "Historical best configuration with target-fold returns and "
        "rebalance days\n"
        f"signal={FIXED_SIGNAL_SPEC}; "
        f"max_positions={config.get('max_positions')}; "
        f"sell_rank={config.get('sell_rank')}; "
        f"offset={config.get('rebalance_offset')}"
    )
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    return marker_count


def plot_forward(
    nav: pd.DataFrame,
    config: dict[str, Any],
    summary: dict[str, Any],
    out_file: Path,
) -> int:
    initial_cash = float(config.get("initial_cash", 200000.0))
    curve = nav.copy()
    curve["return_pct"] = (curve["nav"] / initial_cash - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(12, 6))
    (line,) = ax.plot(
        curve["date"],
        curve["return_pct"],
        linewidth=1.8,
        label="Cumulative return",
    )
    marker_count = add_rebalance_markers(
        ax,
        curve,
        "return_pct",
        line_color=line.get_color(),
    )
    ax.axhline(0.0, linewidth=0.8)
    total_return = float(
        summary.get(
            "total_return",
            curve.iloc[-1]["nav"] / initial_cash - 1.0,
        )
    )
    ax.annotate(
        f"{total_return * 100:+.2f}%\n{curve.iloc[-1]['date']:%Y-%m-%d}",
        xy=(curve.iloc[-1]["date"], curve.iloc[-1]["return_pct"]),
        xytext=(-85, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    ax.set_title(
        "Strict forward cumulative return with rebalance days\n"
        f"{curve.iloc[0]['date']:%Y-%m-%d} to "
        f"{curve.iloc[-1]['date']:%Y-%m-%d}"
    )
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    return marker_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark scheduled rebalance days on global-fold plots"
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--plots-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    history_root = out_root / "historical_fold0_to_fold5_selection"
    forward_root = out_root / "strict_oos_forward"
    plots_dir = (
        Path(args.plots_dir).expanduser().resolve()
        if args.plots_dir
        else out_root / "plots"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    selection = select_historical_signal(
        backtest_root=history_root,
        rank_metric="sharpe",
    )
    if selection.signal_spec != FIXED_SIGNAL_SPEC:
        raise RuntimeError(f"unexpected selected signal: {selection.signal_spec}")
    history_run = selected_run_dir(history_root, selection.run_name)
    history_config = read_json(history_run / "config.json")
    history_nav = load_nav_with_rebalance(
        history_run / "close_auction_nav.csv",
        history_config,
    )

    segment_file = out_root / "historical_fold_segment_returns.csv"
    segment_returns = read_csv_auto(segment_file)
    required_segment_columns = {
        "target_fold",
        "start",
        "end",
        "segment_return_pct",
    }
    missing = required_segment_columns - set(segment_returns.columns)
    if missing:
        raise RuntimeError(f"{segment_file} missing columns: {sorted(missing)}")

    strict = read_json(
        forward_root / "01_close_auction_grid" / "strict_oos_manifest.json"
    )
    forward_run_name = str(strict["retained_run_name"])
    forward_run = (
        forward_root
        / "01_close_auction_grid"
        / "01_runs"
        / forward_run_name
    )
    if not forward_run.is_dir():
        matches = sorted(forward_root.glob(f"**/01_runs/{forward_run_name}"))
        if not matches:
            raise FileNotFoundError(forward_run)
        forward_run = matches[0]
    forward_config = read_json(forward_run / "config.json")
    forward_summary = read_json(forward_run / "summary.json")
    forward_nav = load_nav_with_rebalance(
        forward_run / "close_auction_nav.csv",
        forward_config,
    )

    historical_count = plot_historical(
        history_nav,
        history_config,
        segment_returns,
        plots_dir / "historical_best_with_fold_returns.png",
    )
    forward_count = plot_forward(
        forward_nav,
        forward_config,
        forward_summary,
        plots_dir / "strict_forward_return_curve_latest.png",
    )
    # The standard plotter writes this daily filename. Overwrite only the daily
    # chart; weekly/monthly plots intentionally remain uncluttered.
    plot_forward(
        forward_nav,
        forward_config,
        forward_summary,
        plots_dir / "return_curve_daily.png",
    )

    audit = pd.concat(
        [
            rebalance_rows(history_nav, "historical_fold0_to_fold5"),
            rebalance_rows(forward_nav, "strict_oos_forward"),
        ],
        ignore_index=True,
    )
    audit_file = out_root / "rebalance_dates_audit.csv"
    audit.to_csv(audit_file, index=False, encoding="utf-8-sig")

    plot_manifest_file = plots_dir / "global_results_plot_manifest.json"
    plot_manifest = (
        read_json(plot_manifest_file) if plot_manifest_file.exists() else {}
    )
    plot_manifest.update(
        {
            "rebalance_markers": True,
            "rebalance_marker_semantics": (
                "scheduled strategy rebalance day from retained NAV"
            ),
            "historical_rebalance_day_count": historical_count,
            "strict_forward_rebalance_day_count": forward_count,
            "rebalance_dates_audit_file": str(audit_file),
            "rebalance_marked_plots": [
                str(plots_dir / "historical_best_with_fold_returns.png"),
                str(plots_dir / "strict_forward_return_curve_latest.png"),
                str(plots_dir / "return_curve_daily.png"),
            ],
        }
    )
    write_json(plot_manifest_file, plot_manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "historical_rebalance_days": historical_count,
                "strict_forward_rebalance_days": forward_count,
                "audit_file": str(audit_file),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
