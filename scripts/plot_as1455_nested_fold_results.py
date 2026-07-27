#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot completed AS1455 nested-fold results without rerunning any work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_plotting import plot_frequency  # noqa: E402

RULES = {"daily": None, "weekly": "W-FRI", "monthly": "M"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def load_nav(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    if not {"date", "nav"}.issubset(frame.columns):
        raise RuntimeError(f"NAV lacks date/nav columns: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = (
        frame.dropna(subset=["date", "nav"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        raise RuntimeError(f"NAV has fewer than two rows: {path}")
    return frame


def sample_curve(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "daily":
        return frame.copy()
    if frequency not in RULES:
        raise ValueError(f"unsupported frequency={frequency}")
    return (
        frame.set_index("date")[["nav", "return_pct"]]
        .resample(RULES[frequency])
        .last()
        .dropna()
        .reset_index()
    )


def retained_run(segment_root: Path) -> Path:
    manifest_path = segment_root / "01_close_auction_grid" / "strict_oos_manifest.json"
    manifest = read_json(manifest_path)
    run_name = manifest.get("retained_run_name")
    if not run_name:
        raise RuntimeError(f"retained_run_name missing: {manifest_path}")
    run_dir = segment_root / "01_close_auction_grid" / "01_runs" / str(run_name)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir


def load_segments(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    table = read_csv(root / "nested_fold_target_results.csv")
    required = {"segment", "source_fold", "target_fold"}
    if not required.issubset(table.columns):
        raise RuntimeError(f"target result table missing: {sorted(required - set(table.columns))}")
    table["source_fold"] = pd.to_numeric(table["source_fold"], errors="raise").astype(int)
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
        run_dir = retained_run(segment_root)
        config = read_json(run_dir / "config.json")
        initial_cash = float(config.get("initial_cash", 200000.0))
        nav = load_nav(run_dir / "close_auction_nav.csv")
        nav["return_pct"] = (nav["nav"] / initial_cash - 1.0) * 100.0
        segments.append(
            {
                "label": str(row["segment"]),
                "run_name": run_dir.name,
                "run_dir": run_dir,
                "curve": nav[["date", "nav", "return_pct"]],
            }
        )
    if len(segments) != 7:
        raise RuntimeError(f"expected seven target/forward segments, got {len(segments)}")
    return table, segments


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_segment_comparison(segments: list[dict[str, Any]], out: Path) -> list[Path]:
    curves = [
        {"label": item["label"], "run_name": item["run_name"], "curve": item["curve"]}
        for item in segments
    ]
    generated: list[Path] = []
    for frequency in RULES:
        png = out / f"segment_comparison_{frequency}.png"
        csv = out / f"segment_comparison_{frequency}.csv"
        exported = plot_frequency(
            curves=curves,
            frequency=frequency,
            out_file=png,
            title=f"AS1455 nested fold independent segments ({frequency})",
            sample_curve=sample_curve,
            plt=plt,
        )
        exported.to_csv(csv, index=False, encoding="utf-8-sig")
        generated.extend([png, csv])
    return generated


def plot_segment_metrics(table: pd.DataFrame, out: Path) -> list[Path]:
    table = table.copy()
    table["segment"] = table["segment"].astype(str)
    generated: list[Path] = []
    specs = (
        ("total_return", "Total return", 100.0, "%"),
        ("annual_return", "Annualized return", 100.0, "%"),
        ("sharpe", "Sharpe ratio", 1.0, ""),
        ("max_drawdown", "Maximum drawdown", 100.0, "%"),
    )
    for column, title, scale, unit in specs:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce") * scale
        valid = values.notna()
        if not valid.any():
            continue
        path = out / f"segment_metric_{column}.png"
        plt.figure(figsize=(11, 5.5))
        plt.bar(table.loc[valid, "segment"], values.loc[valid])
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"AS1455 nested fold {title.lower()}")
        plt.xlabel("Evaluation segment")
        plt.ylabel(f"{title}{f' ({unit})' if unit else ''}")
        plt.xticks(rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        save(path)
        generated.append(path)

    selected = [
        column
        for column in (
            "segment",
            "source_fold",
            "target_fold",
            "selection_signal",
            "selection_max_positions",
            "selection_sell_rank",
            "selection_offset",
        )
        if column in table.columns
    ]
    csv = out / "selected_configuration_by_segment.csv"
    table[selected].to_csv(csv, index=False, encoding="utf-8-sig")
    generated.append(csv)
    for column, title in (
        ("selection_max_positions", "Selected maximum positions"),
        ("selection_sell_rank", "Selected sell rank"),
        ("selection_offset", "Selected validation rebalance offset"),
    ):
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        valid = values.notna()
        path = out / f"selected_{column}.png"
        plt.figure(figsize=(11, 5.5))
        plt.bar(table.loc[valid, "segment"], values.loc[valid])
        plt.title(title)
        plt.xlabel("Evaluation segment")
        plt.ylabel(column)
        plt.xticks(rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        save(path)
        generated.append(path)
    return generated


def plot_per_segment(segments: list[dict[str, Any]], out: Path) -> list[Path]:
    generated: list[Path] = []
    for item in segments:
        curve = item["curve"]
        label = item["label"]
        base = out / "per_segment" / label
        ret = base / "return_daily.png"
        plt.figure(figsize=(11, 5.5))
        plt.plot(curve["date"], curve["return_pct"], linewidth=1.9)
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"{label} cumulative return")
        plt.xlabel("Date")
        plt.ylabel("Cumulative return (%)")
        plt.grid(True, alpha=0.3)
        save(ret)

        dd = curve[["date", "nav"]].copy()
        dd["drawdown_pct"] = (dd["nav"] / dd["nav"].cummax() - 1.0) * 100.0
        dd_png = base / "drawdown_daily.png"
        dd_csv = base / "drawdown_daily.csv"
        plt.figure(figsize=(11, 4.8))
        plt.fill_between(dd["date"], dd["drawdown_pct"], 0.0, alpha=0.35)
        plt.plot(dd["date"], dd["drawdown_pct"], linewidth=1.5)
        plt.axhline(0.0, linewidth=1.0)
        plt.title(f"{label} drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        save(dd_png)
        dd.to_csv(dd_csv, index=False, encoding="utf-8-sig")
        generated.extend([ret, dd_png, dd_csv])
    return generated


def load_boundaries(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    if not {"segment", "start", "end"}.issubset(frame.columns):
        raise RuntimeError(f"protocol segment table incomplete: {path}")
    frame["start"] = pd.to_datetime(frame["start"], errors="coerce").dt.normalize()
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce").dt.normalize()
    return frame.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)


def mark_boundaries(ax: Any, boundaries: pd.DataFrame) -> None:
    top = ax.get_ylim()[1]
    for index, row in boundaries.iterrows():
        start = pd.Timestamp(row["start"])
        if index:
            ax.axvline(start, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(start, top, str(row["segment"]), rotation=90, va="top", ha="right", fontsize=7)


def plot_continuous(root: Path, out: Path) -> list[Path]:
    base = root / "continuous_target_folds_plus_forward"
    nav = load_nav(base / "close_auction_nav.csv")
    initial_cash = float(read_json(base / "config.json").get("initial_cash", 200000.0))
    nav["return_pct"] = (nav["nav"] / initial_cash - 1.0) * 100.0
    boundaries = load_boundaries(base / "protocol_segments.csv")
    generated: list[Path] = []

    for column, title, ylabel, filename in (
        ("nav", "AS1455 nested fold continuous account NAV", "NAV", "continuous_nav_daily.png"),
        ("return_pct", "AS1455 nested fold continuous cumulative return", "Cumulative return (%)", "continuous_return_daily.png"),
    ):
        path = out / filename
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(nav["date"], nav[column], linewidth=1.9)
        if column == "return_pct":
            ax.axhline(0.0, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        mark_boundaries(ax, boundaries)
        save(path)
        generated.append(path)

    dd = nav[["date", "nav"]].copy()
    dd["drawdown_pct"] = (dd["nav"] / dd["nav"].cummax() - 1.0) * 100.0
    dd_png = out / "continuous_drawdown_daily.png"
    dd_csv = out / "continuous_drawdown_daily.csv"
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.fill_between(dd["date"], dd["drawdown_pct"], 0.0, alpha=0.35)
    ax.plot(dd["date"], dd["drawdown_pct"], linewidth=1.5)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("AS1455 nested fold continuous drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    mark_boundaries(ax, boundaries)
    save(dd_png)
    dd.to_csv(dd_csv, index=False, encoding="utf-8-sig")
    generated.extend([dd_png, dd_csv])

    for frequency in ("weekly", "monthly"):
        sampled = sample_curve(nav[["date", "nav", "return_pct"]], frequency)
        png = out / f"continuous_return_{frequency}.png"
        csv = out / f"continuous_return_{frequency}.csv"
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(sampled["date"], sampled["return_pct"], linewidth=1.9, marker="o")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"AS1455 nested fold continuous cumulative return ({frequency})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (%)")
        ax.grid(True, alpha=0.3)
        mark_boundaries(ax, boundaries)
        save(png)
        sampled.to_csv(csv, index=False, encoding="utf-8-sig")
        generated.extend([png, csv])
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot existing AS1455 nested-fold results")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-per-segment", action="store_true")
    parser.add_argument("--skip-continuous", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = Path(args.plots_dir).expanduser().resolve() if args.plots_dir else root / "plots"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f"plots directory is not empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    table, segments = load_segments(root)
    generated = plot_segment_comparison(segments, out)
    generated += plot_segment_metrics(table, out)
    if not args.skip_per_segment:
        generated += plot_per_segment(segments, out)
    if not args.skip_continuous:
        generated += plot_continuous(root, out)

    manifest = {
        "mode": "plot_only_existing_nested_fold_results",
        "out_root": str(root),
        "plots_root": str(out),
        "backtest_executed": False,
        "prediction_executed": False,
        "validation_grid_executed": False,
        "training_executed": False,
        "segments": [item["label"] for item in segments],
        "generated_file_count": len(generated),
        "generated_files": [str(path) for path in generated],
    }
    manifest_path = out / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "plots_root": str(out), "generated_file_count": len(generated), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
