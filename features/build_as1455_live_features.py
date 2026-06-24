#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build T-day AS1455 live Ch12 feature matrix.

Inputs from the same live directory:
  05_history_tail_qfq_livebase.parquet/.csv
  08_live_raw_row_as1455.csv
  03_adjustment_events.csv
  01_universe.csv

Outputs:
  09_live_qfq_row_as1455.csv
  10_live_feature_panel_tail.parquet/.csv
  11_live_model_features.csv
  12_feature_build_report.json

This does not run model inference.
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
    EXPECTED_MODEL_COLUMNS,
    compute_ch12_features,
    ensure_dir,
    normalize_symbol,
    parse_trade_date,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)

DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"


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
    out = pd.DataFrame({
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AS1455 T-day live Ch12 feature matrix")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--live-dir", default=None, help="defaults to out-root/YYYYMMDD")
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--training-feature-columns", default=None)
    ap.add_argument("--min-feature-rows", type=int, default=980)
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    ensure_dir(live_dir)

    universe = pd.read_csv(live_dir / "01_universe.csv", dtype={"symbol": str}, encoding="utf-8-sig")
    universe["symbol"] = universe["symbol"].map(normalize_symbol)
    if "code" not in universe.columns:
        universe["code"] = universe["symbol"].str.slice(0, 6)
    if "industry" not in universe.columns:
        universe["industry"] = "unknown"

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
    if t_date not in features.index.get_level_values("date"):
        live_features = pd.DataFrame()
    else:
        live_features = features.xs(t_date, level="date", drop_level=False).copy()
    live_features = live_features.reset_index()

    training_cols = load_feature_columns(args.training_feature_columns)
    missing_cols = [c for c in training_cols if c not in live_features.columns]
    extra_expected_cols = [c for c in live_features.columns if c in EXPECTED_MODEL_COLUMNS and c not in training_cols]
    for c in missing_cols:
        live_features[c] = np.nan
    ordered = ["date", "symbol"] + training_cols
    live_model_features = live_features[ordered].copy() if not live_features.empty else pd.DataFrame(columns=ordered)
    write_csv(live_dir / "11_live_model_features.csv", live_model_features)
    if not outliers.empty:
        write_csv(live_dir / "11_live_feature_outlier_symbols.csv", outliers)

    nan_rows = int(live_model_features[training_cols].isna().any(axis=1).sum()) if len(live_model_features) else 0
    usable_rows = int(len(live_model_features) - nan_rows)
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
        "usable_feature_rows": usable_rows,
        "outlier_symbols": int(len(outliers)),
        "feature_passed": bool(usable_rows >= args.min_feature_rows and not missing_cols),
    }
    write_json(live_dir / "12_feature_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
