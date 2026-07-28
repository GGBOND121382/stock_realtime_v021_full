#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize and plot the global-fold first3-ensemble forward experiment.

The historical best run is one continuous account selected on concatenated
one-fold-lag target_fold5..target_fold0 predictions. This script partitions its
NAV by the recorded fold date boundaries, computes each fold's continuous-account
return, and plots both the historical fold contributions and the strict-forward
result. It also rewrites the top-level CSV/JSON summaries after a forward refresh.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_model_selection import (  # noqa: E402
    find_summary_file,
    read_csv_auto,
    select_historical_signal,
    successful_rows,
)

FIXED_SIGNAL_SPEC = "ensemble_first3_mean:0,1,2:mean"


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


def load_nav(path: Path) -> pd.DataFrame:
    nav = read_csv_auto(path)
    required = {"date", "nav"}
    missing = required - set(nav.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    nav = nav.copy()
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
    return nav


def selected_run_dir(backtest_root: Path, run_name: str) -> Path:
    _summary_file, grid_dir = find_summary_file(backtest_root)
    run_dir = grid_dir / "01_runs" / run_name
    if not run_dir.is_dir():
        matches = sorted(backtest_root.glob(f"**/01_runs/{run_name}"))
        if not matches:
            raise FileNotFoundError(run_dir)
        run_dir = matches[0]
    return run_dir


def daily_returns(nav: pd.DataFrame, initial_cash: float) -> pd.Series:
    values = nav["nav"].astype(float)
    returns = values.pct_change()
    returns.iloc[0] = values.iloc[0] / float(initial_cash) - 1.0
    return returns


def annualized_sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(math.sqrt(252.0) * values.mean() / std)


def local_max_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    curve = pd.concat(
        [pd.Series([1.0]), (1.0 + values).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def load_segments(path: Path) -> pd.DataFrame:
    segments = read_csv_auto(path)
    required = {"source_fold", "target_fold", "start", "end", "n_days"}
    missing = required - set(segments.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    segments = segments.copy()
    segments["start"] = pd.to_datetime(segments["start"], errors="coerce").dt.normalize()
    segments["end"] = pd.to_datetime(segments["end"], errors="coerce").dt.normalize()
    segments["source_fold"] = pd.to_numeric(
        segments["source_fold"], errors="raise"
    ).astype(int)
    segments["target_fold"] = pd.to_numeric(
        segments["target_fold"], errors="raise"
    ).astype(int)
    segments["n_days"] = pd.to_numeric(segments["n_days"], errors="raise").astype(int)
    segments = (
        segments.dropna(subset=["start", "end"])
        .sort_values("start")
        .reset_index(drop=True)
    )
    if len(segments) != 6:
        raise RuntimeError(
            f"expected six historical target-fold segments, got {len(segments)}"
        )
    return segments


def build_segment_returns(
    nav: pd.DataFrame,
    segments: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    work = nav.copy()
    work["daily_return_rebuilt"] = daily_returns(work, initial_cash)
    rows: list[dict[str, Any]] = []
    for segment in segments.itertuples(index=False):
        mask = work["date"].between(segment.start, segment.end)
        part = work.loc[mask].copy()
        if len(part) != int(segment.n_days):
            raise RuntimeError(
                f"target_fold{segment.target_fold} NAV coverage mismatch: "
                f"expected={segment.n_days} actual={len(part)} "
                f"dates={segment.start:%Y-%m-%d}..{segment.end:%Y-%m-%d}"
            )
        first_index = int(part.index[0])
        start_nav_before = (
            float(work.loc[first_index - 1, "nav"])
            if first_index > 0
            else float(initial_cash)
        )
        end_nav = float(part.iloc[-1]["nav"])
        segment_return = end_nav / start_nav_before - 1.0
        returns = part["daily_return_rebuilt"]
        compounded = float((1.0 + returns).prod() - 1.0)
        if not np.isclose(segment_return, compounded, atol=1e-10, rtol=1e-8):
            raise RuntimeError(
                f"segment return parity failed for target_fold{segment.target_fold}: "
                f"nav={segment_return} compounded={compounded}"
            )
        max_drawdown = local_max_drawdown(returns)
        rows.append(
            {
                "segment": f"target_fold{segment.target_fold}",
                "source_fold": int(segment.source_fold),
                "target_fold": int(segment.target_fold),
                "start": segment.start.strftime("%Y-%m-%d"),
                "end": segment.end.strftime("%Y-%m-%d"),
                "n_days": int(len(part)),
                "start_nav_before_segment": start_nav_before,
                "end_nav": end_nav,
                "segment_return": segment_return,
                "segment_return_pct": segment_return * 100.0,
                "annualized_sharpe": annualized_sharpe(returns),
                "segment_max_drawdown": max_drawdown,
                "segment_max_drawdown_pct": max_drawdown * 100.0,
            }
        )
    return pd.DataFrame(rows)


def plot_historical_curve(
    nav: pd.DataFrame,
    segment_returns: pd.DataFrame,
    initial_cash: float,
    out_file: Path,
    selected_config: dict[str, Any],
) -> None:
    curve = nav.copy()
    curve["return_pct"] = (curve["nav"] / float(initial_cash) - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(curve["date"], curve["return_pct"], linewidth=1.6)
    ax.axhline(0.0, linewidth=0.8)
    y_min = float(curve["return_pct"].min())
    y_max = float(curve["return_pct"].max())
    y_span = max(y_max - y_min, 1.0)
    label_y = y_max + 0.06 * y_span
    for row in segment_returns.itertuples(index=False):
        start = pd.Timestamp(row.start)
        end = pd.Timestamp(row.end)
        ax.axvline(start, linestyle="--", linewidth=0.8, alpha=0.65)
        midpoint = start + (end - start) / 2
        ax.text(
            midpoint,
            label_y,
            f"fold{row.target_fold}\n{row.segment_return_pct:+.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axvline(
        pd.Timestamp(segment_returns.iloc[-1]["end"]),
        linestyle="--",
        linewidth=0.8,
        alpha=0.65,
    )
    ax.set_ylim(y_min - 0.08 * y_span, label_y + 0.12 * y_span)
    ax.set_title(
        "Historical best configuration with target-fold returns\n"
        f"signal={FIXED_SIGNAL_SPEC}; "
        f"max_positions={selected_config.get('max_positions')}; "
        f"sell_rank={selected_config.get('sell_rank')}; "
        f"offset={selected_config.get('rebalance_offset')}"
    )
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_segment_bars(segment_returns: pd.DataFrame, out_file: Path) -> None:
    ordered = segment_returns.sort_values("start")
    labels = [f"fold{value}" for value in ordered["target_fold"]]
    values = ordered["segment_return_pct"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values)
    ax.axhline(0.0, linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    ax.set_title("Historical return by target fold (continuous account)")
    ax.set_xlabel("Target fold")
    ax.set_ylabel("Segment return (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_forward_curve(
    nav: pd.DataFrame,
    initial_cash: float,
    out_file: Path,
    summary: dict[str, Any],
) -> None:
    curve = nav.copy()
    curve["return_pct"] = (curve["nav"] / float(initial_cash) - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(curve["date"], curve["return_pct"], linewidth=1.8)
    ax.axhline(0.0, linewidth=0.8)
    total_return = float(
        summary.get("total_return", curve.iloc[-1]["nav"] / initial_cash - 1.0)
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
        "Strict forward cumulative return\n"
        f"{curve.iloc[0]['date']:%Y-%m-%d} to "
        f"{curve.iloc[-1]['date']:%Y-%m-%d}"
    )
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize fold returns and latest strict-forward plots"
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--prediction-source-root", default=None)
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
    history_nav = load_nav(history_run / "close_auction_nav.csv")
    history_config = read_json(history_run / "config.json")
    history_initial_cash = float(history_config.get("initial_cash", 200000.0))
    segments = load_segments(
        history_root / "00_predictions" / "prediction_segments.csv"
    )
    segment_returns = build_segment_returns(
        history_nav,
        segments,
        history_initial_cash,
    )
    segment_returns.to_csv(
        out_root / "historical_fold_segment_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_historical_curve(
        history_nav,
        segment_returns,
        history_initial_cash,
        plots_dir / "historical_best_with_fold_returns.png",
        history_config,
    )
    plot_segment_bars(
        segment_returns,
        plots_dir / "historical_fold_returns_bar.png",
    )

    strict_file = (
        forward_root / "01_close_auction_grid" / "strict_oos_manifest.json"
    )
    strict = read_json(strict_file)
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
    forward_nav = load_nav(forward_run / "close_auction_nav.csv")
    forward_summary = read_json(forward_run / "summary.json")
    forward_config = read_json(forward_run / "config.json")
    forward_initial_cash = float(forward_config.get("initial_cash", 200000.0))
    plot_forward_curve(
        forward_nav,
        forward_initial_cash,
        plots_dir / "strict_forward_return_curve_latest.png",
        forward_summary,
    )

    forward_result = {
        "signal_spec": selection.signal_spec,
        "historical_rank_metric": selection.rank_metric,
        "historical_rank_metric_value": selection.rank_metric_value,
        "max_positions": selection.historical_max_positions,
        "sell_rank": selection.historical_sell_rank,
        "historical_offset": selection.historical_rebalance_offset,
        "effective_forward_offset": forward_config.get("rebalance_offset"),
        "forward_start": forward_nav.iloc[0]["date"].strftime("%Y-%m-%d"),
        "forward_end": forward_nav.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "forward_n_days": int(len(forward_nav)),
        **forward_summary,
    }
    pd.DataFrame([forward_result]).to_csv(
        out_root / "strict_forward_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_file, _grid_dir = find_summary_file(history_root)
    historical_grid = successful_rows(read_csv_auto(summary_file)).copy()
    historical_grid["sharpe"] = pd.to_numeric(
        historical_grid["sharpe"], errors="coerce"
    )
    historical_grid.sort_values("sharpe", ascending=False).head(20).to_csv(
        out_root / "historical_grid_top20.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_file = out_root / "global_fold0_to_fold5_forward_manifest.json"
    manifest = read_json(manifest_file) if manifest_file.exists() else {}
    manifest.update(
        {
            "protocol": (
                "global_fold0_to_fold5_fixed_first3_ensemble_then_strict_forward"
            ),
            "fixed_signal_spec": FIXED_SIGNAL_SPEC,
            "historical_target_folds": [5, 4, 3, 2, 1, 0],
            "historical_source_folds": [6, 5, 4, 3, 2, 1],
            "historical_grid_count": int(len(historical_grid)),
            "forward_grid_count": 0,
            "forward_fixed_backtest_count": 1,
            "historical_folds_used_for_selection": True,
            "forward_results_used_for_selection": False,
            "pure_nested_historical_evaluation": False,
            "forward_account_starts_empty": True,
            "prediction_source_root": args.prediction_source_root,
            "selection": selection.to_dict(),
            "historical_selected_run_dir": str(history_run),
            "historical_segment_returns": segment_returns.to_dict(
                orient="records"
            ),
            "strict_oos_manifest": strict,
            "strict_forward_run_dir": str(forward_run),
            "strict_forward_start": forward_result["forward_start"],
            "strict_forward_end": forward_result["forward_end"],
            "strict_forward_n_days": forward_result["forward_n_days"],
            "strict_forward_summary": forward_summary,
            "plots": {
                "historical_with_fold_returns": str(
                    plots_dir / "historical_best_with_fold_returns.png"
                ),
                "historical_fold_return_bars": str(
                    plots_dir / "historical_fold_returns_bar.png"
                ),
                "strict_forward_latest": str(
                    plots_dir / "strict_forward_return_curve_latest.png"
                ),
            },
            "model_training": False,
            "data_refresh_during_finalization": False,
            "model_data_rebuild_during_finalization": False,
        }
    )
    write_json(manifest_file, manifest)
    write_json(
        plots_dir / "global_results_plot_manifest.json",
        {
            "status": "ok",
            "out_root": str(out_root),
            "historical_run": selection.run_name,
            "forward_run": forward_run_name,
            "forward_start": forward_result["forward_start"],
            "forward_end": forward_result["forward_end"],
            "fold_returns_file": str(
                out_root / "historical_fold_segment_returns.csv"
            ),
            "plots": manifest["plots"],
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "historical_selection": selection.to_dict(),
                "fold_returns": segment_returns[
                    ["segment", "segment_return_pct"]
                ].to_dict(orient="records"),
                "forward_start": forward_result["forward_start"],
                "forward_end": forward_result["forward_end"],
                "forward_summary": forward_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
