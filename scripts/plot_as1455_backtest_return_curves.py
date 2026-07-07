#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot AS1455 backtest return curves for multiple backtest roots.

Default comparison:
  saved_data/ashare_ml4t/ch17_as1455_rotation_one_lag_daily_backtest_20260706
  saved_data/ashare_ml4t/ch17_as1455_rotation_addon_one_lag_daily_backtest_20260707

For each backtest root, the script selects the best grid run from
02_summary/grid_summary_compact.csv by --rank-metric, then loads that run's
close_auction_nav.csv and draws cumulative return curves.

It writes three figures by default: daily, weekly, and monthly sampled curves.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_BACKTEST_ROOTS = [
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_rotation_one_lag_daily_backtest_20260706",
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_rotation_addon_one_lag_daily_backtest_20260707",
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
    for rel in SUMMARY_CANDIDATES:
        p = root / rel
        if p.exists():
            return p, p.parents[1]
    matches = sorted(root.glob("**/02_summary/grid_summary_compact.csv"))
    if matches:
        p = matches[0]
        return p, p.parents[1]
    matches = sorted(root.glob("**/02_summary/grid_summary.csv"))
    if matches:
        p = matches[0]
        return p, p.parents[1]
    raise FileNotFoundError(f"cannot find grid summary under {root}")


def select_best_run(summary: pd.DataFrame, metric: str) -> pd.Series:
    df = summary.copy()
    if "status" in df.columns:
        ok = df["status"].astype(str).str.lower().eq("ok")
        if ok.any():
            df = df.loc[ok].copy()
    if metric not in df.columns:
        raise RuntimeError(f"rank metric {metric!r} not found in summary columns: {list(df.columns)}")
    if "run_name" not in df.columns:
        raise RuntimeError("summary does not contain run_name column")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        raise RuntimeError(f"no valid rows for metric {metric!r}")
    ascending = metric in LOWER_IS_BETTER and metric not in HIGHER_IS_BETTER
    return df.sort_values(metric, ascending=ascending).iloc[0]


def find_nav_file(root: Path, grid_dir: Path, run_name: str) -> Path:
    candidates = [
        grid_dir / "01_runs" / run_name / "close_auction_nav.csv",
    ]
    for grid_rel in GRID_DIR_CANDIDATES:
        candidates.append(root / grid_rel / "01_runs" / run_name / "close_auction_nav.csv")
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(root.glob(f"**/01_runs/{run_name}/close_auction_nav.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"cannot find close_auction_nav.csv for run_name={run_name!r} under {root}")


def load_curve(nav_file: Path) -> pd.DataFrame:
    nav = read_csv_auto(nav_file)
    required = {"date", "nav"}
    missing = required - set(nav.columns)
    if missing:
        raise RuntimeError(f"{nav_file} missing columns: {sorted(missing)}")
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date").drop_duplicates("date", keep="last")
    start_nav = None
    if "nav_before_trade" in nav.columns and pd.notna(nav.iloc[0].get("nav_before_trade")):
        start_nav = float(nav.iloc[0]["nav_before_trade"])
    if not start_nav or start_nav <= 0:
        start_nav = float(nav.iloc[0]["nav"])
    nav["return_pct"] = (nav["nav"].astype(float) / start_nav - 1.0) * 100.0
    return nav[["date", "nav", "return_pct"]]


def sample_curve(curve: pd.DataFrame, frequency: str) -> pd.DataFrame:
    frequency = frequency.lower()
    if frequency not in FREQ_RULES:
        raise RuntimeError(f"unsupported frequency={frequency!r}; expected {sorted(FREQ_RULES)}")
    if frequency == "daily":
        return curve.copy()
    rule = FREQ_RULES[frequency]
    sampled = curve.set_index("date")[["nav", "return_pct"]].resample(rule).last().dropna().reset_index()
    return sampled


def metric_snapshot(row: pd.Series) -> dict[str, Any]:
    keep = [
        "run_name", "status", "signal_name", "signal_cols", "signal_mode",
        "max_positions", "sell_rank", "rebalance_every", "rebalance_offset",
        "total_return", "annual_return", "sharpe", "calmar", "max_drawdown",
        "daily_win_rate", "monthly_win_rate", "trade_win_rate", "round_trip_win_rate",
        "avg_turnover", "annualized_turnover", "gross_trade_amount", "total_fee",
        "fee_to_initial_cash", "avg_positions", "n_orders", "n_rejections",
    ]
    return {k: row[k] for k in keep if k in row.index}


def plot_frequency(curves: list[dict[str, Any]], frequency: str, out_file: Path, title: str) -> pd.DataFrame:
    plt.figure(figsize=(12, 6))
    rows = []
    for item in curves:
        sampled = sample_curve(item["curve"], frequency)
        plt.plot(sampled["date"], sampled["return_pct"], linewidth=1.8, label=item["label"])
        tmp = sampled.copy()
        tmp.insert(0, "label", item["label"])
        tmp.insert(1, "run_name", item["run_name"])
        tmp.insert(2, "frequency", frequency)
        rows.append(tmp)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AS1455 cumulative return curves for best backtest grid runs")
    p.add_argument("--backtest-root", action="append", default=None, help="Backtest output root. Repeat for multiple roots.")
    p.add_argument("--label", action="append", default=None, help="Curve label. Repeat in the same order as --backtest-root.")
    p.add_argument("--rank-metric", default="sharpe", help="Metric used to choose the best grid row; default: sharpe")
    p.add_argument("--frequencies", default="daily,weekly,monthly", help="Comma list from daily,weekly,monthly")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--title-prefix", default="AS1455 best-grid cumulative return")
    p.add_argument("--show-selected", action="store_true", help="Print selected runs to stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p).expanduser() for p in args.backtest_root] if args.backtest_root else DEFAULT_BACKTEST_ROOTS
    roots = [p if p.is_absolute() else (PROJECT_DIR / p) for p in roots]
    labels = args.label if args.label else DEFAULT_LABELS[: len(roots)]
    if len(labels) < len(roots):
        labels = labels + [p.name for p in roots[len(labels):]]
    if len(labels) != len(roots):
        raise SystemExit("number of --label values must match --backtest-root values")
    frequencies = [x.strip().lower() for x in args.frequencies.split(",") if x.strip()]
    if not frequencies:
        raise SystemExit("--frequencies is empty")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_backtest_plots" / f"plot_{stamp}"
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    curves = []
    for root, label in zip(roots, labels):
        if not root.exists():
            raise FileNotFoundError(root)
        summary_file, grid_dir = find_summary_file(root)
        summary = read_csv_auto(summary_file)
        best = select_best_run(summary, args.rank_metric)
        run_name = str(best["run_name"])
        nav_file = find_nav_file(root, grid_dir, run_name)
        curve = load_curve(nav_file)
        selected = {
            "label": label,
            "backtest_root": str(root),
            "summary_file": str(summary_file),
            "grid_dir": str(grid_dir),
            "nav_file": str(nav_file),
            "rank_metric": args.rank_metric,
            **metric_snapshot(best),
        }
        selected_rows.append(selected)
        metric_value = best.get(args.rank_metric)
        curve_label = f"{label} | {run_name} | {args.rank_metric}={float(metric_value):.4g}"
        curves.append({"label": curve_label, "run_name": run_name, "curve": curve})

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "selected_best_grids.csv", index=False, encoding="utf-8-sig")
    (out_dir / "selected_best_grids.json").write_text(json.dumps(selected_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.show_selected:
        print(selected_df.to_string(index=False))

    for freq in frequencies:
        out_png = out_dir / f"return_curve_{freq}.png"
        curve_csv = out_dir / f"return_curve_{freq}.csv"
        title = f"{args.title_prefix} ({freq})"
        curve_df = plot_frequency(curves, freq, out_png, title)
        curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] {freq}: {out_png}")
        print(f"[OK] {freq} csv: {curve_csv}")
    print(f"[OK] selected grids: {out_dir / 'selected_best_grids.csv'}")
    print(f"[DONE] out_dir={out_dir}")


if __name__ == "__main__":
    main()
