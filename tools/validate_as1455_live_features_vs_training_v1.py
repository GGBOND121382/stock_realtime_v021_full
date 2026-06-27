#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate one live AS1455 feature file against training model_data HDF.

This script is a hard check, not a model/inference command. It compares the 31
feature columns in live_as1455/YYYYMMDD/11_live_model_features_for_prediction.csv
against saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 for the same date
and same symbols when that date exists in the HDF.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dollar_vol", "dollar_vol_rank", "rsi", "bb_high", "bb_low",
    "NATR", "ATR", "PPO", "MACD", "sector",
    "r01", "r05", "r10", "r21", "r42", "r63",
    "r01dec", "r05dec", "r10dec", "r21dec", "r42dec", "r63dec",
    "r01q_sector", "r05q_sector", "r10q_sector", "r21q_sector", "r42q_sector", "r63q_sector",
    "year", "month", "weekday",
]


def normalize_symbol(value: object) -> str:
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        return ""
    if "." in s:
        a, b = s.split(".", 1)
        if a.isalpha():
            market, code = a.upper(), re.sub(r"\D", "", b)[:6]
        else:
            code, market = re.sub(r"\D", "", a)[:6], b.upper()
        market = market.replace("XSHE", "SZ").replace("XSHG", "SH")
        if market not in {"SH", "SZ"}:
            market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code.zfill(6)}.{market}"
    code = re.sub(r"\D", "", s)[:6].zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def parse_trade_date(value: str) -> str:
    s = str(value).strip().replace("-", "")
    if s.lower() == "today":
        return pd.Timestamp.today().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", s):
        raise ValueError(f"invalid trade date: {value!r}")
    return s


def find_live_feature_file(live_dir: Path) -> Path:
    candidates = [
        live_dir / "11_live_model_features_for_prediction.csv",
        live_dir / "11_live_model_features_usable.csv",
        live_dir / "11_live_model_features.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("no live feature file found under " + str(live_dir))


def numeric_diff_summary(m: pd.DataFrame, col: str, atol: float) -> tuple[dict, pd.DataFrame]:
    a = pd.to_numeric(m[f"{col}_live"], errors="coerce")
    b = pd.to_numeric(m[f"{col}_train"], errors="coerce")
    both = a.notna() & b.notna()
    diff = (a - b).abs()
    changed = both & (diff > atol)
    row = {
        "field": col,
        "n_live_nonnull": int(a.notna().sum()),
        "n_train_nonnull": int(b.notna().sum()),
        "n_both": int(both.sum()),
        "n_changed": int(changed.sum()),
        "changed_rate": float(changed.sum() / max(int(both.sum()), 1)),
        "max_abs_diff": None if not both.any() else float(diff[both].max()),
        "mean_abs_diff": None if not both.any() else float(diff[both].mean()),
        "p95_abs_diff": None if not both.any() else float(diff[both].quantile(0.95)),
    }
    if changed.any():
        top = m.loc[changed, ["symbol", f"{col}_live", f"{col}_train"]].copy()
        top["field"] = col
        top["abs_diff"] = diff.loc[changed].to_numpy()
        top = top.sort_values("abs_diff", ascending=False).head(50)
    else:
        top = pd.DataFrame()
    return row, top


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare live AS1455 features with training model_data on the same date")
    ap.add_argument("--trade-date", required=True, help="YYYYMMDD / YYYY-MM-DD / today")
    ap.add_argument("--out-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--model-data", default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5")
    ap.add_argument("--hdf-key", default="model_data")
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    target_date = pd.Timestamp(f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}").normalize()
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    live_path = find_live_feature_file(live_dir)
    out_dir = live_dir / "audit_feature_vs_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    live = pd.read_csv(live_path, dtype={"symbol": str}, encoding="utf-8-sig")
    if "symbol" not in live.columns:
        raise RuntimeError(f"{live_path} missing symbol")
    live["symbol"] = live["symbol"].map(normalize_symbol)
    live_missing = [c for c in FEATURE_COLUMNS if c not in live.columns]
    live_order = [c for c in live.columns if c in FEATURE_COLUMNS]
    live_feature_order_matches = live_order == FEATURE_COLUMNS
    for c in live_missing:
        live[c] = np.nan
    live = live[["symbol"] + FEATURE_COLUMNS].drop_duplicates("symbol", keep="last").copy()

    model_data_path = Path(args.model_data)
    md = pd.read_hdf(model_data_path, args.hdf_key)
    if list(md.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected model_data index names: {md.index.names}")
    dates = pd.to_datetime(md.index.get_level_values("date"), errors="coerce").normalize()
    if not dates.eq(target_date).any():
        report = {
            "status": "cannot_compare",
            "reason": "target_date_not_in_model_data",
            "target_date": str(target_date.date()),
            "model_data_min": str(dates.min().date()),
            "model_data_max": str(dates.max().date()),
            "live_path": str(live_path),
            "live_rows": int(len(live)),
            "live_missing_feature_columns": live_missing,
            "live_feature_order_matches_reference": bool(live_feature_order_matches),
        }
        (out_dir / "feature_vs_training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    train = md.loc[dates == target_date].reset_index().copy()
    train["symbol"] = train["symbol"].map(normalize_symbol)
    train_missing = [c for c in FEATURE_COLUMNS if c not in train.columns]
    if train_missing:
        raise RuntimeError(f"model_data missing feature columns: {train_missing}")
    train = train[["symbol"] + FEATURE_COLUMNS].drop_duplicates("symbol", keep="last").copy()

    m = live.merge(train, on="symbol", how="outer", suffixes=("_live", "_train"), indicator=True)
    rows: list[dict] = []
    tops: list[pd.DataFrame] = []
    for c in FEATURE_COLUMNS:
        row, top = numeric_diff_summary(m, c, args.atol)
        rows.append(row)
        if not top.empty:
            tops.append(top)
    summary = pd.DataFrame(rows).sort_values(["n_changed", "max_abs_diff"], ascending=False)

    summary_path = out_dir / "feature_vs_training_summary.csv"
    merged_path = out_dir / "feature_vs_training_merged.csv"
    topdiff_path = out_dir / "feature_vs_training_topdiff.csv"
    report_path = out_dir / "feature_vs_training_report.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    m.to_csv(merged_path, index=False, encoding="utf-8-sig")
    (pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()).to_csv(topdiff_path, index=False, encoding="utf-8-sig")

    sector_changed = int(summary.loc[summary["field"].eq("sector"), "n_changed"].iloc[0]) if "sector" in summary["field"].values else None
    report = {
        "status": "done",
        "target_date": str(target_date.date()),
        "live_path": str(live_path),
        "model_data_path": str(model_data_path),
        "live_rows": int(len(live)),
        "train_rows": int(len(train)),
        "merge_counts": {str(k): int(v) for k, v in m["_merge"].value_counts(dropna=False).to_dict().items()},
        "feature_columns": FEATURE_COLUMNS,
        "live_missing_feature_columns": live_missing,
        "live_feature_order_matches_reference": bool(live_feature_order_matches),
        "fields_with_changed_rows": int((summary["n_changed"] > 0).sum()),
        "total_changed_cells": int(summary["n_changed"].sum()),
        "max_field_changed_rate": float(summary["changed_rate"].max()),
        "sector_changed_rows": sector_changed,
        "passed_exact_with_atol": bool(not live_missing and live_feature_order_matches and int(summary["n_changed"].sum()) == 0 and m["_merge"].eq("both").all()),
        "summary_path": str(summary_path),
        "merged_path": str(merged_path),
        "topdiff_path": str(topdiff_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
