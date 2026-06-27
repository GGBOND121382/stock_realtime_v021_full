#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build T-day AS1455 live Ch12 feature matrix.

AS1455 live feature builder aligned v4.

Outputs both a full audit file and a strictly non-null usable file for prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    DEFAULT_CH12_DIR,
    EXPECTED_MODEL_COLUMNS,
    compute_ch12_features,
    enrich_universe_meta_with_sector_reference,
    ensure_dir,
    load_sector_reference_from_model_data,
    normalize_symbol,
    parse_trade_date,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)

DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"
DEFAULT_SECTOR_REFERENCE = DEFAULT_CH12_DIR / "model_data_as1455.h5"


def read_table_with_fallback(base: Path) -> pd.DataFrame:
    candidates = []
    if base.suffix:
        candidates.append(base)
        candidates.append(base.with_suffix(".csv"))
    else:
        candidates.extend([base.with_suffix(".parquet"), base.with_suffix(".csv")])
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, dtype={"symbol": str}, encoding="utf-8-sig")
    raise FileNotFoundError("none found: " + ", ".join(str(p) for p in candidates))


def write_table_with_fallback(df: pd.DataFrame, parquet_path: Path) -> str:
    ensure_dir(parquet_path.parent)
    try:
        df.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    except Exception as exc:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] parquet write failed for {parquet_path.name}; wrote {csv_path.name}: {type(exc).__name__}: {exc}", flush=True)
        return str(csv_path)


def load_feature_columns(path: str | None) -> list[str]:
    if not path:
        return EXPECTED_MODEL_COLUMNS.copy()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            for k in ["feature_columns", "columns", "model_columns"]:
                if k in obj:
                    return [str(x) for x in obj[k]]
    except Exception:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def live_raw_to_qfq_row(raw: pd.DataFrame, events: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    df = raw.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    if not events.empty:
        ev = events[["symbol", "event_ratio", "is_factor_event_today"]].copy()
        ev["symbol"] = ev["symbol"].map(normalize_symbol)
        df = df.merge(ev, on="symbol", how="left")
    else:
        df["event_ratio"] = 1.0
        df["is_factor_event_today"] = False
    return pd.DataFrame({
        "date": yyyymmdd_to_dash(trade_date),
        "symbol": df["symbol"],
        "open": pd.to_numeric(df["raw_open_as1455"], errors="coerce"),
        "high": pd.to_numeric(df["raw_high_as1455"], errors="coerce"),
        "low": pd.to_numeric(df["raw_low_as1455"], errors="coerce"),
        "close": pd.to_numeric(df["raw_close_as1455"], errors="coerce"),
        "volume": pd.to_numeric(df["raw_volume_as1455"], errors="coerce"),
        "event_ratio": pd.to_numeric(df.get("event_ratio", 1.0), errors="coerce").fillna(1.0),
        "is_factor_event_today": df.get("is_factor_event_today", False),
        "raw_open_as1455": df["raw_open_as1455"],
        "raw_high_as1455": df["raw_high_as1455"],
        "raw_low_as1455": df["raw_low_as1455"],
        "raw_close_as1455": df["raw_close_as1455"],
        "raw_volume_as1455": df["raw_volume_as1455"],
        "raw_amount_as1455": df.get("raw_amount_as1455", np.nan),
        "live_preclose": df.get("live_preclose", np.nan),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AS1455 T-day live Ch12 feature matrix")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--live-dir", default=None, help="defaults to out-root/YYYYMMDD")
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--training-feature-columns", default=None)
    ap.add_argument("--sector-reference", default=str(DEFAULT_SECTOR_REFERENCE),
                    help="training model_data HDF used to load symbol->sector mapping")
    ap.add_argument("--allow-sector-fallback", action="store_true",
                    help="allow factorized live industry fallback if training sector reference is missing")
    ap.add_argument("--min-feature-rows", type=int, default=980)
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    ensure_dir(live_dir)

    universe_raw = pd.read_csv(live_dir / "01_universe.csv", dtype={"symbol": str}, encoding="utf-8-sig")
    sector_ref, sector_ref_report = load_sector_reference_from_model_data(args.sector_reference)
    universe, sector_enrich_report = enrich_universe_meta_with_sector_reference(universe_raw, sector_ref)
    if (not sector_ref_report.get("loaded")) and (not args.allow_sector_fallback):
        raise RuntimeError(
            "training sector reference is required for live feature alignment; "
            f"failed to load {args.sector_reference}: {sector_ref_report}"
        )

    qfq_tail = read_table_with_fallback(live_dir / "05_history_tail_qfq_livebase.parquet")
    raw_live = pd.read_csv(live_dir / "08_live_raw_row_as1455.csv", dtype={"symbol": str}, encoding="utf-8-sig")
    events_path = live_dir / "03_adjustment_events.csv"
    events = pd.read_csv(events_path, dtype={"symbol": str}, encoding="utf-8-sig") if events_path.exists() else pd.DataFrame()

    live_qfq = live_raw_to_qfq_row(raw_live, events, trade_date)
    write_csv(live_dir / "09_live_qfq_row_as1455.csv", live_qfq)

    panel_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    panel = pd.concat([qfq_tail[panel_cols], live_qfq[panel_cols]], ignore_index=True, sort=False)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["symbol"] = panel["symbol"].map(normalize_symbol)
    for c in ["open", "high", "low", "close", "volume"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel = panel.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    panel = panel.sort_values(["date", "symbol"])
    feature_panel_path = write_table_with_fallback(panel, live_dir / "10_live_feature_panel_tail.parquet")

    prices = panel.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]].sort_index()
    features, outliers = compute_ch12_features(prices, universe, include_forward_labels=False)
    t_date = pd.Timestamp(yyyymmdd_to_dash(trade_date)).normalize()
    live_features = features.xs(t_date, level="date", drop_level=False).copy() if t_date in features.index.get_level_values("date") else pd.DataFrame()
    live_features = live_features.reset_index()

    training_cols = load_feature_columns(args.training_feature_columns)
    missing_cols = [c for c in training_cols if c not in live_features.columns]
    extra_expected_cols = [c for c in live_features.columns if c in EXPECTED_MODEL_COLUMNS and c not in training_cols]
    for c in missing_cols:
        live_features[c] = np.nan

    ordered = ["date", "symbol"] + training_cols
    live_model_features = live_features[ordered].copy() if not live_features.empty else pd.DataFrame(columns=ordered)

    # Full file is retained for audit.  It may contain a few NaNs in sector-quantile
    # features when a training sector has too few live candidates to qcut into five bins.
    # Do not fill those values: filling would change the training feature semantics.
    write_csv(live_dir / "11_live_model_features.csv", live_model_features)

    feature_matrix = live_model_features[training_cols] if len(live_model_features) else pd.DataFrame(columns=training_cols)
    row_has_nan = feature_matrix.isna().any(axis=1) if len(feature_matrix) else pd.Series(dtype=bool)
    live_model_features_usable = live_model_features.loc[~row_has_nan].copy() if len(live_model_features) else pd.DataFrame(columns=ordered)
    write_csv(live_dir / "11_live_model_features_usable.csv", live_model_features_usable)
    # Alias used by downstream inference scripts; intentionally contains only complete rows.
    write_csv(live_dir / "11_live_model_features_for_prediction.csv", live_model_features_usable)

    if len(live_model_features) and row_has_nan.any():
        dropped = live_model_features.loc[row_has_nan, ["date", "symbol"]].copy()
        dropped["missing_feature_columns"] = feature_matrix.loc[row_has_nan].apply(
            lambda r: ",".join([c for c in training_cols if pd.isna(r.get(c))]),
            axis=1,
        )
        write_csv(live_dir / "11_live_model_features_dropped_rows.csv", dropped)
    else:
        dropped = pd.DataFrame(columns=["date", "symbol", "missing_feature_columns"])

    if not outliers.empty:
        write_csv(live_dir / "11_live_feature_outlier_symbols.csv", outliers)

    nan_by_column = {c: int(feature_matrix[c].isna().sum()) for c in training_cols if c in feature_matrix.columns}
    nan_cols = [c for c, n in nan_by_column.items() if n > 0]
    nan_rows = int(row_has_nan.sum()) if len(feature_matrix) else 0
    usable_rows = int(len(live_model_features_usable))
    usable_matrix = live_model_features_usable[training_cols] if len(live_model_features_usable) else pd.DataFrame(columns=training_cols)
    usable_nan_by_column = {c: int(usable_matrix[c].isna().sum()) for c in training_cols if c in usable_matrix.columns}
    usable_nan_cols = [c for c, n in usable_nan_by_column.items() if n > 0]

    sector_values = pd.to_numeric(live_model_features.get("sector", pd.Series(dtype=float)), errors="coerce")
    sector_unmapped_rows = int((sector_values < 0).sum()) if len(live_model_features) and "sector" in live_model_features.columns else int(len(live_model_features))
    usable_sector_values = pd.to_numeric(live_model_features_usable.get("sector", pd.Series(dtype=float)), errors="coerce")
    usable_sector_unmapped_rows = int((usable_sector_values < 0).sum()) if len(live_model_features_usable) and "sector" in live_model_features_usable.columns else int(len(live_model_features_usable))
    dollar_vol_rank_nonnull = int(live_model_features["dollar_vol_rank"].notna().sum()) if "dollar_vol_rank" in live_model_features.columns else 0
    usable_dollar_vol_rank_nonnull = int(live_model_features_usable["dollar_vol_rank"].notna().sum()) if "dollar_vol_rank" in live_model_features_usable.columns else 0

    feature_passed = (
        usable_rows >= args.min_feature_rows
        and not missing_cols
        and not usable_nan_cols
        and usable_dollar_vol_rank_nonnull == usable_rows
        and sector_unmapped_rows == 0
        and usable_sector_unmapped_rows == 0
    )

    report = {
        "trade_date": trade_date,
        "live_dir": str(live_dir),
        "history_tail_rows": int(len(qfq_tail)),
        "live_raw_rows": int(len(raw_live)),
        "live_qfq_rows": int(len(live_qfq)),
        "feature_panel_rows": int(len(panel)),
        "feature_panel_path": feature_panel_path,
        "live_feature_rows": int(len(live_model_features)),
        "training_feature_columns": training_cols,
        "n_training_feature_columns": int(len(training_cols)),
        "missing_training_columns": missing_cols,
        "extra_expected_columns_not_used": extra_expected_cols,
        "nan_feature_rows": nan_rows,
        "nan_by_column": nan_by_column,
        "nan_columns": nan_cols,
        "usable_feature_rows": usable_rows,
        "usable_feature_path": str(live_dir / "11_live_model_features_usable.csv"),
        "prediction_feature_path": str(live_dir / "11_live_model_features_for_prediction.csv"),
        "dropped_feature_rows": int(len(dropped)),
        "dropped_rows_path": str(live_dir / "11_live_model_features_dropped_rows.csv") if len(dropped) else None,
        "usable_nan_by_column": usable_nan_by_column,
        "usable_nan_columns": usable_nan_cols,
        "outlier_symbols": int(len(outliers)),
        "sector_reference": sector_ref_report,
        "sector_enrichment": sector_enrich_report,
        "sector_unmapped_rows": sector_unmapped_rows,
        "usable_sector_unmapped_rows": usable_sector_unmapped_rows,
        "sector_unique_count": int(sector_values.nunique(dropna=True)) if len(sector_values) else 0,
        "dollar_vol_rank_nonnull": dollar_vol_rank_nonnull,
        "usable_dollar_vol_rank_nonnull": usable_dollar_vol_rank_nonnull,
        "dollar_vol_rank_min": float(pd.to_numeric(live_model_features.get("dollar_vol_rank", pd.Series(dtype=float)), errors="coerce").min()) if dollar_vol_rank_nonnull else None,
        "dollar_vol_rank_max": float(pd.to_numeric(live_model_features.get("dollar_vol_rank", pd.Series(dtype=float)), errors="coerce").max()) if dollar_vol_rank_nonnull else None,
        "feature_passed": bool(feature_passed),
    }
    write_json(live_dir / "12_feature_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not feature_passed:
        raise SystemExit("live feature alignment failed; see 12_feature_build_report.json")


if __name__ == "__main__":
    main()
