#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize next-day model search outputs into one leaderboard.

This script recursively reads summary_*bps.csv files produced by
search_walk_forward_model_complexity.py and adds metadata from nearby
search_run_manifest.json / pipeline_summary.json when available.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


SORT_COLS = [
    "compound_return",
    "avg_return",
    "profit_factor",
    "win_rate",
    "trades",
    "windows",
]
PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float_from_name(path: Path) -> Optional[float]:
    m = re.search(r"summary_(\d+(?:\.\d+)?)bps\.csv$", path.name)
    return float(m.group(1)) if m else None


def infer_label_mode(search_dir: Path, manifest: Dict) -> str:
    if manifest.get("label_mode"):
        return str(manifest["label_mode"])
    name = search_dir.name.lower()
    if "close" in name:
        return "close_profit"
    if "hit" in name:
        return "hit"
    return "unknown"




def infer_entry_policy(search_dir: Path, manifest: Dict) -> str:
    if manifest.get("entry_policy"):
        return str(manifest["entry_policy"])
    name = search_dir.name.lower()
    if "all_days" in name or "all_dates" in name:
        return "all_days"
    if "vwap_low" in name or "low_vwap" in name or "below_vwap" in name:
        return "vwap_low"
    return "unknown"


def find_summary_files(pipeline_out: Optional[Path], search_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for d in search_dirs:
        if d.exists():
            files.extend(sorted(d.rglob("summary_*bps.csv")))
    if pipeline_out and pipeline_out.exists():
        files.extend(sorted(pipeline_out.rglob("summary_*bps.csv")))
    # de-duplicate while preserving order
    seen = set()
    out = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # trade_metrics in the existing project usually returns these columns.
    # Keep this mapping defensive for older outputs.
    aliases = {
        "n_trades": "trades",
        "num_trades": "trades",
        "mean_return": "avg_return",
        "compound_ret": "compound_return",
        "max_dd": "max_drawdown",
    }
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def put_front(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    """Add metadata column at the front, replacing an existing column safely.

    Newer search outputs may already contain metadata such as entry_policy.
    DataFrame.insert raises `ValueError: cannot insert ..., already exists`;
    this helper makes the summarizer backward/forward compatible.
    """
    if col in df.columns:
        df = df.drop(columns=[col])
    df.insert(0, col, value)
    return df


def load_one_summary(path: Path, pipeline_meta: Dict) -> pd.DataFrame:
    search_dir = path.parent
    manifest = read_json(search_dir / "search_run_manifest.json")
    df = pd.read_csv(path)
    df = normalize_metric_columns(df)

    symbol_meta = manifest.get("symbol") or pipeline_meta.get("symbol") or {}
    if isinstance(symbol_meta, dict):
        stock_code = symbol_meta.get("stock_code") or symbol_meta.get("input_symbol") or symbol_meta.get("raw_code")
        raw_code = symbol_meta.get("raw_code")
    else:
        stock_code = str(symbol_meta) if symbol_meta else None
        raw_code = None

    file_bps = as_float_from_name(path)
    label_mode = infer_label_mode(search_dir, manifest)
    entry_policy = infer_entry_policy(search_dir, manifest)
    target_hit_bps = manifest.get("target_hit_bps", file_bps)

    # Add/overwrite metadata defensively. Some newer search outputs already
    # include entry_policy in summary_*.csv.
    for col, value in [
        ("summary_file", str(path)),
        ("search_dir", str(search_dir)),
        ("entry_policy", entry_policy),
        ("label_mode", label_mode),
        ("stock_code", stock_code),
        ("raw_code", raw_code),
    ]:
        df = put_front(df, col, value)
    if "target_hit_bps" not in df.columns:
        df["target_hit_bps"] = target_hit_bps
    else:
        df["target_hit_bps"] = df["target_hit_bps"].fillna(target_hit_bps)
    df["sample_file"] = manifest.get("sample_file") or pipeline_meta.get("final_samples")
    df["intraday_bars"] = manifest.get("intraday_bars") or pipeline_meta.get("intraday_bars")
    df["feature_pipeline"] = ",".join(manifest.get("feature_pipeline", []) or pipeline_meta.get("feature_pipeline_effective", []) or [])
    df["external"] = ",".join(manifest.get("external", []) or pipeline_meta.get("external_effective", []) or [])
    df["entry_vwap_premium_bps"] = manifest.get("entry_vwap_premium_bps") or pipeline_meta.get("entry_vwap_premium_bps")
    return df


def rank_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in SORT_COLS + ["max_drawdown"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # A simple robust score for first-pass sorting; keep raw metrics for real decisions.
    # It rewards average return and trade count while penalizing drawdown.
    if {"avg_return", "trades", "max_drawdown"}.issubset(out.columns):
        trades_term = out["trades"].clip(lower=0).pow(0.5)
        dd_penalty = out["max_drawdown"].abs().fillna(0.0)
        out["rank_score"] = out["avg_return"].fillna(-999.0) * trades_term - 0.25 * dd_penalty
    else:
        out["rank_score"] = pd.NA

    existing_sort = [c for c in ["rank_score", *SORT_COLS] if c in out.columns]
    ascending = [False] * len(existing_sort)
    if existing_sort:
        out = out.sort_values(existing_sort, ascending=ascending, na_position="last")
    return out.reset_index(drop=True)


def aggregate_best_by_target(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = [c for c in ["stock_code", "entry_policy", "label_mode", "target_hit_bps"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    ranked = rank_leaderboard(df)
    return ranked.groupby(keys, dropna=False).head(5).reset_index(drop=True)


def write_outputs(leaderboard: pd.DataFrame, out_dir: Path, excel: bool = False) -> Dict[str, str]:
    ensure_dir(out_dir)
    final_csv = out_dir / "final_leaderboard.csv"
    best_csv = out_dir / "best_by_target_top5.csv"
    compare_csv = out_dir / "best_by_entry_policy_top5.csv"
    leaderboard.to_csv(final_csv, index=False, encoding="utf-8-sig")
    best = aggregate_best_by_target(leaderboard)
    best.to_csv(best_csv, index=False, encoding="utf-8-sig")
    # A convenience view: top candidates are grouped by entry policy as well as target.
    best.to_csv(compare_csv, index=False, encoding="utf-8-sig")
    outputs = {
        "final_leaderboard": str(final_csv),
        "best_by_target_top5": str(best_csv),
        "best_by_entry_policy_top5": str(compare_csv),
    }
    if excel:
        xlsx = out_dir / "final_leaderboard.xlsx"
        try:
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                leaderboard.to_excel(writer, sheet_name="leaderboard", index=False)
                best.to_excel(writer, sheet_name="best_by_target_top5", index=False)
                best.to_excel(writer, sheet_name="best_by_entry_policy", index=False)
            outputs["excel"] = str(xlsx)
        except Exception as e:
            outputs["excel_error"] = f"{type(e).__name__}: {e}"
    summary_json = out_dir / "summary_manifest.json"
    summary = {
        "rows": int(len(leaderboard)),
        "outputs": outputs,
        "columns": list(leaderboard.columns),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    outputs["summary_manifest"] = str(summary_json)
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize next-day model search outputs")
    p.add_argument("--pipeline-out", default=None, help="Pipeline root directory, e.g. saved_data/600176_pipeline_out")
    p.add_argument("--search-dir", action="append", default=[], help="Specific search dir; may be repeated")
    p.add_argument("--out-dir", default=None, help="Output dir; default: <pipeline-out>/99_summary or saved_data/nextday_summary_out")
    p.add_argument("--excel", action="store_true", help="Also write xlsx if openpyxl is installed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_out = Path(args.pipeline_out) if args.pipeline_out else None
    search_dirs = [Path(x) for x in args.search_dir]
    if not pipeline_out and not search_dirs:
        raise SystemExit("provide --pipeline-out or at least one --search-dir")

    pipeline_meta = read_json(pipeline_out / "pipeline_summary.json") if pipeline_out else {}
    summary_files = find_summary_files(pipeline_out, search_dirs)
    if not summary_files:
        raise SystemExit("no summary_*bps.csv files found")

    frames = [load_one_summary(path, pipeline_meta) for path in summary_files]
    leaderboard = rank_leaderboard(pd.concat(frames, ignore_index=True))
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif pipeline_out:
        out_dir = pipeline_out / "99_summary"
    else:
        out_dir = SAVED_DATA_DIR / "nextday_summary_out"
    outputs = write_outputs(leaderboard, out_dir, excel=args.excel)
    print(json.dumps({"rows": int(len(leaderboard)), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
