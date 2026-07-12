#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot AS1455 backtest return curves for multiple backtest roots.

For each backtest root, select the best grid run by ``--rank-metric``, load its
NAV, and draw daily/weekly/monthly cumulative-return curves. Curves are always
distinguished by both line style and marker, so the figures do not rely on
color perception alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_plotting import plot_frequency as plot_frequency_shared  # noqa: E402

DEFAULT_BACKTEST_ROOTS = [
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_rotation_one_lag_daily_backtest_20260706",
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_rotation_addon_one_lag_daily_backtest_20260707",
]
DEFAULT_LABELS = [
    "A: rotation + one-hot best grid",
    "B: rotation + compact add-on + one-hot best grid",
]
FREQ_RULES = {
    "daily": None,
    "weekly": "W-FRI",
    "monthly": "M",
}
HIGHER_IS_BETTER = {
    "total_return",
    "annual_return",
    "sharpe",
    "calmar",
    "max_drawdown",
    "daily_win_rate",
    "monthly_win_rate",
    "trade_win_rate",
    "round_trip_win_rate",
}
LOWER_IS_BETTER = {
    "avg_turnover",
    "annualized_turnover",
    "gross_trade_amount",
    "total_fee",
    "fee_to_initial_cash",
    "n_orders",
    "n_rejections",
}
SUMMARY_CANDIDATES = [
    "01_close_auction_daily_grid/02_summary/grid_summary_compact.csv",
    "01_close_auction_grid/02_summary/grid_summary_compact.csv",
    "01_close_auction_daily_grid/02_summary/grid_summary.csv",
    "01_close_auction_grid/02_summary/grid_summary.csv",
]
GRID_DIR_CANDIDATES = [
    "01_close_auction_daily_grid",
    "01_close_auction_grid",
]


def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def find_summary_file(root: Path) -> tuple[Path, Path]:
    for relative in SUMMARY_CANDIDATES:
        path = root / relative
        if path.exists():
            return path, path.parents[1]
    matches = sorted(root.glob("**/02_summary/grid_summary_compact.csv"))
    if matches:
        path = matches[0]
        return path, path.parents[1]
    matches = sorted(root.glob("**/02_summary/grid_summary.csv"))
    if matches:
        path = matches[0]
        return path, path.parents[1]
    raise FileNotFoundError(f"cannot find grid summary under {root}")


def select_best_run(summary: pd.DataFrame, metric: str) -> pd.Series:
    frame = summary.copy()
    if "status" in frame.columns:
        ok = frame["status"].astype(str).str.lower().eq("ok")
        if ok.any():
            frame = frame.loc[ok].copy()
    if metric not in frame.columns:
        raise RuntimeError(
            f"rank metric {metric!r} not found in summary columns: "
            f"{list(frame.columns)}"
        )
    if "run_name" not in frame.columns:
        raise RuntimeError("summary does not contain run_name column")
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])
    if frame.empty:
        raise RuntimeError(f"no valid rows for metric {metric!r}")
    ascending = metric in LOWER_IS_BETTER and metric not in HIGHER_IS_BETTER
    return frame.sort_values(metric, ascending=ascending).iloc[0]


def find_nav_file(root: Path, grid_dir: Path, run_name: str) -> Path:
    candidates = [grid_dir / "01_runs" / run_name / "close_auction_nav.csv"]
    for grid_relative in GRID_DIR_CANDIDATES:
        candidates.append(
            root
            / grid_relative
            / "01_runs"
            / run_name
            / "close_auction_nav.csv"
        )
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.glob(f"**/01_runs/{run_name}/close_auction_nav.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"cannot find close_auction_nav.csv for run_name={run_name!r} "
        f"under {root}"
    )


def load_curve(nav_file: Path) -> pd.DataFrame:
    nav = read_csv_auto(nav_file)
    required = {"date", "nav"}
    missing = required - set(nav.columns)
    if missing:
        raise RuntimeError(f"{nav_file} missing columns: {sorted(missing)}")
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date").drop_duplicates("date", keep="last")
    start_nav = None
    if (
        "nav_before_trade" in nav.columns
        and pd.notna(nav.iloc[0].get("nav_before_trade"))
    ):
        start_nav = float(nav.iloc[0]["nav_before_trade"])
    if not start_nav or start_nav <= 0:
        start_nav = float(nav.iloc[0]["nav"])
    nav["return_pct"] = (
        nav["nav"].astype(float) / start_nav - 1.0
    ) * 100.0
    return nav[["date", "nav", "return_pct"]]


def sample_curve(curve: pd.DataFrame, frequency: str) -> pd.DataFrame:
    frequency = frequency.lower()
    if frequency not in FREQ_RULES:
        raise RuntimeError(
            f"unsupported frequency={frequency!r}; expected {sorted(FREQ_RULES)}"
        )
    if frequency == "daily":
        return curve.copy()
    rule = FREQ_RULES[frequency]
    return (
        curve.set_index("date")[["nav", "return_pct"]]
        .resample(rule)
        .last()
        .dropna()
        .reset_index()
    )


def metric_snapshot(row: pd.Series) -> dict[str, Any]:
    keep = [
        "run_name",
        "status",
        "signal_name",
        "signal_cols",
        "signal_mode",
        "max_positions",
        "sell_rank",
        "rebalance_every",
        "rebalance_offset",
        "total_return",
        "annual_return",
        "sharpe",
        "calmar",
        "max_drawdown",
        "daily_win_rate",
        "monthly_win_rate",
        "trade_win_rate",
        "round_trip_win_rate",
        "avg_turnover",
        "annualized_turnover",
        "gross_trade_amount",
        "total_fee",
        "fee_to_initial_cash",
        "avg_positions",
        "n_orders",
        "n_rejections",
    ]
    return {key: row[key] for key in keep if key in row.index}


def plot_frequency(
    curves: list[dict[str, Any]],
    frequency: str,
    out_file: Path,
    title: str,
) -> pd.DataFrame:
    return plot_frequency_shared(
        curves=curves,
        frequency=frequency,
        out_file=out_file,
        title=title,
        sample_curve=sample_curve,
        plt=plt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AS1455 cumulative return curves for best grid runs"
    )
    parser.add_argument(
        "--backtest-root",
        action="append",
        default=None,
        help="Backtest output root. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Curve label. Repeat in the same order as --backtest-root.",
    )
    parser.add_argument(
        "--rank-metric",
        default="sharpe",
        help="Metric used to choose the best grid row; default: sharpe",
    )
    parser.add_argument(
        "--frequencies",
        default="daily,weekly,monthly",
        help="Comma list from daily,weekly,monthly",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--title-prefix", default="AS1455 best-grid cumulative return"
    )
    parser.add_argument(
        "--show-selected",
        action="store_true",
        help="Print selected runs to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = (
        [Path(value).expanduser() for value in args.backtest_root]
        if args.backtest_root
        else DEFAULT_BACKTEST_ROOTS
    )
    roots = [
        root if root.is_absolute() else PROJECT_DIR / root for root in roots
    ]
    labels = args.label if args.label else DEFAULT_LABELS[: len(roots)]
    if len(labels) < len(roots):
        labels = labels + [root.name for root in roots[len(labels) :]]
    if len(labels) != len(roots):
        raise SystemExit(
            "number of --label values must match --backtest-root values"
        )
    frequencies = [
        value.strip().lower()
        for value in args.frequencies.split(",")
        if value.strip()
    ]
    if not frequencies:
        raise SystemExit("--frequencies is empty")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_backtest_plots"
        / f"plot_{stamp}"
    )
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for root, label in zip(roots, labels):
        if not root.exists():
            raise FileNotFoundError(root)
        summary_file, grid_dir = find_summary_file(root)
        summary = read_csv_auto(summary_file)
        best = select_best_run(summary, args.rank_metric)
        run_name = str(best["run_name"])
        nav_file = find_nav_file(root, grid_dir, run_name)
        curve = load_curve(nav_file)
        selected_rows.append(
            {
                "label": label,
                "backtest_root": str(root),
                "summary_file": str(summary_file),
                "grid_dir": str(grid_dir),
                "nav_file": str(nav_file),
                "rank_metric": args.rank_metric,
                **metric_snapshot(best),
            }
        )
        metric_value = best.get(args.rank_metric)
        curve_label = (
            f"{label} | {run_name} | "
            f"{args.rank_metric}={float(metric_value):.4g}"
        )
        curves.append(
            {"label": curve_label, "run_name": run_name, "curve": curve}
        )

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(
        out_dir / "selected_best_grids.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out_dir / "selected_best_grids.json").write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if args.show_selected:
        print(selected_df.to_string(index=False))

    for frequency in frequencies:
        out_png = out_dir / f"return_curve_{frequency}.png"
        curve_csv = out_dir / f"return_curve_{frequency}.csv"
        title = f"{args.title_prefix} ({frequency})"
        curve_df = plot_frequency(curves, frequency, out_png, title)
        curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] {frequency}: {out_png}")
        print(f"[OK] {frequency} csv: {curve_csv}")
    print(f"[OK] selected grids: {out_dir / 'selected_best_grids.csv'}")
    print(f"[DONE] out_dir={out_dir}")


if __name__ == "__main__":
    main()
