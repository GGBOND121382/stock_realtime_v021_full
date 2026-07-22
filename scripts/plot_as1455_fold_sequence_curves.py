#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot source-fold6..fold0 AS1455 return curves on aligned date windows.

fold6..fold1 are the one-fold-lag historical segments in which source fold k is
evaluated on target fold k-1. fold0 is the strict-OOS forward segment. Each
fold chart contains all requested target/feature strategies and is normalized
at the first common execution date of that fold.
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
    select_best_run,
)
from utils.as1455_plotting import plot_frequency  # noqa: E402

FREQ_RULES = {"daily": None, "weekly": "W-FRI", "monthly": "M"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aligned fold6..fold0 AS1455 strategy curves"
    )
    parser.add_argument("--historical-root", action="append", required=True)
    parser.add_argument("--forward-root", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--rank-metric", default="sharpe")
    parser.add_argument("--frequencies", default="daily,weekly,monthly")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def find_nav_file(root: Path, grid_dir: Path, run_name: str) -> Path:
    candidates = [
        grid_dir / "01_runs" / run_name / "close_auction_nav.csv",
        root / "01_close_auction_grid" / "01_runs" / run_name / "close_auction_nav.csv",
        root / "01_close_auction_daily_grid" / "01_runs" / run_name / "close_auction_nav.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.glob(f"**/01_runs/{run_name}/close_auction_nav.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"NAV not found root={root} run={run_name}")


def load_nav(path: Path) -> pd.DataFrame:
    nav = read_csv_auto(path)
    missing = {"date", "nav"} - set(nav.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    if "nav_before_trade" in nav.columns:
        nav["nav_before_trade"] = pd.to_numeric(
            nav["nav_before_trade"], errors="coerce"
        )
    nav = nav.dropna(subset=["date", "nav"]).sort_values("date")
    return nav.drop_duplicates("date", keep="last")


def select_nav(root: Path, rank_metric: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_file, grid_dir = find_summary_file(root)
    best = select_best_run(read_csv_auto(summary_file), rank_metric)
    run_name = str(best["run_name"])
    nav_file = find_nav_file(root, grid_dir, run_name)
    meta = {
        "root": str(root),
        "summary_file": str(summary_file),
        "nav_file": str(nav_file),
        "run_name": run_name,
        "rank_metric": rank_metric,
        "rank_metric_value": float(best[rank_metric]),
    }
    for key in (
        "signal_name",
        "signal_cols",
        "signal_mode",
        "max_positions",
        "sell_rank",
        "rebalance_every",
        "rebalance_offset",
        "annual_return",
        "sharpe",
        "max_drawdown",
    ):
        if key in best.index:
            meta[key] = best[key]
    return load_nav(nav_file), meta


def load_fold_mapping(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "00_predictions" / "one_lag_prediction_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mappings: dict[int, dict[str, Any]] = {}
    for row in payload.get("fold_mapping", []):
        source_fold = int(row["source_fold"])
        start = row.get("target_fold_start") or row.get("target_validation_start") or row.get("target_test_start")
        end = row.get("target_fold_end") or row.get("target_validation_end") or row.get("target_test_end")
        if not start or not end:
            raise RuntimeError(f"fold mapping lacks date boundary: {row}")
        mappings[source_fold] = {**row, "start": start, "end": end}
    missing = [fold for fold in range(1, 7) if fold not in mappings]
    if missing:
        raise RuntimeError(f"historical manifest missing source folds: {missing}")
    return mappings


def normalize_slice(nav: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = nav.loc[(nav["date"] >= start) & (nav["date"] <= end)].copy()
    if out.empty:
        raise RuntimeError(f"empty NAV slice {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    start_nav = None
    if "nav_before_trade" in out.columns:
        value = out.iloc[0].get("nav_before_trade")
        if pd.notna(value) and float(value) > 0:
            start_nav = float(value)
    if not start_nav:
        start_nav = float(out.iloc[0]["nav"])
    out["return_pct"] = (out["nav"].astype(float) / start_nav - 1.0) * 100.0
    return out[["date", "nav", "return_pct"]]


def sample_curve(curve: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency not in FREQ_RULES:
        raise RuntimeError(f"unsupported frequency={frequency}")
    if frequency == "daily":
        return curve.copy()
    return (
        curve.set_index("date")[["nav", "return_pct"]]
        .resample(FREQ_RULES[frequency])
        .last()
        .dropna()
        .reset_index()
    )


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value.item() if hasattr(value, "item") else str(value)


def main() -> None:
    args = parse_args()
    if not (
        len(args.historical_root) == len(args.forward_root) == len(args.label)
    ):
        raise SystemExit("historical-root, forward-root and label counts must match")
    frequencies = [item.strip().lower() for item in args.frequencies.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for historical_text, forward_text, label in zip(
        args.historical_root, args.forward_root, args.label
    ):
        historical_root = Path(historical_text)
        forward_root = Path(forward_text)
        historical_nav, historical_meta = select_nav(historical_root, args.rank_metric)
        forward_nav, forward_meta = select_nav(forward_root, args.rank_metric)
        mappings = load_fold_mapping(historical_root)
        strategies.append(
            {
                "label": label,
                "historical_nav": historical_nav,
                "forward_nav": forward_nav,
                "mappings": mappings,
            }
        )
        selected_rows.append(
            {
                "label": label,
                "historical": historical_meta,
                "forward": forward_meta,
            }
        )

    fold_manifest: dict[str, Any] = {
        "fold_semantics": {
            "fold6_to_fold1": "source fold k evaluated on aligned target fold k-1; historical portfolio state is continuous and each chart is renormalized at its fold boundary",
            "fold0": "fold0 checkpoint strict-OOS forward segment from empty initial portfolio",
        },
        "strategies": selected_rows,
        "folds": {},
    }

    for source_fold in range(6, -1, -1):
        raw_curves: list[dict[str, Any]] = []
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for strategy in strategies:
            if source_fold == 0:
                nav = strategy["forward_nav"]
                start = pd.Timestamp(nav["date"].min()).normalize()
                end = pd.Timestamp(nav["date"].max()).normalize()
            else:
                mapping = strategy["mappings"][source_fold]
                nav = strategy["historical_nav"]
                start = pd.Timestamp(mapping["start"]).normalize()
                end = pd.Timestamp(mapping["end"]).normalize()
            starts.append(start)
            ends.append(end)
            raw_curves.append({"label": strategy["label"], "nav": nav})

        common_start = max(starts)
        common_end = min(ends)
        if common_start > common_end:
            raise RuntimeError(
                f"fold{source_fold} has no common date interval: starts={starts} ends={ends}"
            )
        curves = []
        for item in raw_curves:
            curve = normalize_slice(item["nav"], common_start, common_end)
            curves.append(
                {
                    "label": item["label"],
                    "run_name": f"fold{source_fold}",
                    "curve": curve,
                }
            )

        fold_dir = out_dir / f"fold{source_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest["folds"][f"fold{source_fold}"] = {
            "common_start": common_start.strftime("%Y-%m-%d"),
            "common_end": common_end.strftime("%Y-%m-%d"),
            "n_strategies": len(curves),
        }
        for frequency in frequencies:
            png = fold_dir / f"return_curve_{frequency}.png"
            csv = fold_dir / f"return_curve_{frequency}.csv"
            frame = plot_frequency(
                curves=curves,
                frequency=frequency,
                out_file=png,
                title=f"AS1455 source fold{source_fold} return ({frequency})",
                sample_curve=sample_curve,
                plt=plt,
            )
            frame.to_csv(csv, index=False, encoding="utf-8-sig")
            print(f"[OK] fold{source_fold} {frequency}: {png}")

    pd.DataFrame(
        [
            {
                "label": row["label"],
                "historical_root": row["historical"]["root"],
                "historical_run": row["historical"]["run_name"],
                "forward_root": row["forward"]["root"],
                "forward_run": row["forward"]["run_name"],
            }
            for row in selected_rows
        ]
    ).to_csv(out_dir / "selected_runs.csv", index=False, encoding="utf-8-sig")
    (out_dir / "fold_sequence_manifest.json").write_text(
        json.dumps(fold_manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"[DONE] fold sequence plots: {out_dir}")


if __name__ == "__main__":
    main()
