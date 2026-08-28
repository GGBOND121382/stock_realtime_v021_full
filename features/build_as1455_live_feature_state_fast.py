#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the compact pre-14:55 state used by AS1455 live inference.

This preserves the latest master fast-path design but delegates sector mapping
to the clean, already-audited live common module. Only data available before
14:55 is stored. No model inference is performed here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from features.as1455_live_common import (
    DEFAULT_CH12_DIR,
    DEFAULT_LIVE_ROOT,
    EXPECTED_MODEL_COLUMNS,
    ensure_dir,
    load_sector_reference_from_model_data,
    normalize_symbol,
    parse_trade_date,
    write_json,
)

DEFAULT_SECTOR_REFERENCE = DEFAULT_CH12_DIR / "model_data_as1455.h5"


def read_table_with_fallback(base: Path) -> tuple[pd.DataFrame, Path]:
    candidates = [base] if base.suffix else [base.with_suffix(".parquet"), base.with_suffix(".csv")]
    if base.suffix:
        candidates.append(base.with_suffix(".csv"))
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            if path.suffix == ".parquet":
                return pd.read_parquet(path), path
            return pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig", low_memory=False), path
    raise FileNotFoundError("none found: " + ", ".join(str(p) for p in candidates))


def load_feature_columns(path: str | None) -> list[str]:
    if not path:
        return list(EXPECTED_MODEL_COLUMNS)
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("feature_columns", "columns", "model_columns"):
            if key in obj:
                return [str(x) for x in obj[key]]
    raise RuntimeError(f"cannot read feature columns from {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AS1455 fast feature state before 14:55")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--sector-reference", default=str(DEFAULT_SECTOR_REFERENCE))
    ap.add_argument("--training-feature-columns", default=None)
    ap.add_argument("--tail-days", type=int, default=252)
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--allow-sector-fallback", action="store_true")
    args = ap.parse_args()

    started = time.time()
    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    state_path = Path(args.state_file) if args.state_file else live_dir / "06_live_feature_state_fast.npz"

    tail, source_path = read_table_with_fallback(live_dir / "05_history_tail_qfq_livebase")
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in tail.columns]
    if missing:
        raise RuntimeError(f"history qfq tail missing columns: {missing}")
    tail = tail[required].copy()
    tail["date"] = pd.to_datetime(tail["date"], errors="coerce").dt.normalize()
    tail["symbol"] = tail["symbol"].map(normalize_symbol)
    for c in ["open", "high", "low", "close", "volume"]:
        tail[c] = pd.to_numeric(tail[c], errors="coerce")
    tail = tail.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    tail = tail.sort_values(["symbol", "date"])

    sector_ref, sector_report = load_sector_reference_from_model_data(args.sector_reference)
    if sector_ref.empty and not args.allow_sector_fallback:
        raise RuntimeError(f"no usable training sector reference: {sector_report}")
    sector_map = {}
    if not sector_ref.empty:
        sector_ref = sector_ref.copy()
        sector_ref["symbol"] = sector_ref["symbol"].map(normalize_symbol)
        sector_map = sector_ref.drop_duplicates("symbol").set_index("symbol")["sector"].astype(int).to_dict()

    symbols = sorted(tail["symbol"].dropna().unique().tolist())
    if not symbols:
        raise RuntimeError("empty symbols in history tail")
    T = int(args.tail_days)
    if T < 65:
        raise ValueError("tail-days must be at least 65")
    N = len(symbols)
    arrays = {name: np.full((N, T), np.nan, dtype=np.float64) for name in ["open", "high", "low", "close", "volume"]}
    last_dates = np.array(["" for _ in range(N)], dtype=object)
    sectors = np.full(N, -1, dtype=np.int64)
    row_counts = np.zeros(N, dtype=np.int64)
    for i, sym in enumerate(symbols):
        g = tail[tail["symbol"].eq(sym)].sort_values("date").tail(T)
        k = len(g)
        row_counts[i] = k
        if k:
            for name in arrays:
                arrays[name][i, -k:] = g[name].to_numpy(dtype=float)
            last_dates[i] = pd.Timestamp(g["date"].iloc[-1]).strftime("%Y-%m-%d")
        sectors[i] = int(sector_map.get(sym, -1))

    missing_sector = int((sectors < 0).sum())
    if missing_sector and not args.allow_sector_fallback:
        examples = [symbols[i] for i in np.where(sectors < 0)[0][:10]]
        raise RuntimeError(f"training sector missing for {missing_sector} symbols; examples={examples}")

    feature_columns = load_feature_columns(args.training_feature_columns)
    base_expected = list(EXPECTED_MODEL_COLUMNS)
    if feature_columns != base_expected:
        raise RuntimeError(
            "prefast state must describe the 31 base live features; rotation/addon "
            f"is built later by clean common. expected={base_expected}, got={feature_columns}"
        )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_path,
        symbols=np.asarray(symbols, dtype=object),
        sectors=sectors,
        dates_last=last_dates,
        row_counts=row_counts,
        feature_columns=np.asarray(feature_columns, dtype=object),
        trade_date=np.asarray([trade_date], dtype=object),
        source_path=np.asarray([str(source_path)], dtype=object),
        **arrays,
    )
    report = {
        "prefast_passed": bool(N > 0 and int((row_counts >= 65).sum()) >= max(1, int(N * 0.98)) and (missing_sector == 0 or args.allow_sector_fallback)),
        "trade_date": trade_date,
        "state_file": str(state_path),
        "history_tail_source": str(source_path),
        "symbols": N,
        "tail_days": T,
        "symbols_with_65_rows": int((row_counts >= 65).sum()),
        "row_count_min": int(row_counts.min()),
        "row_count_median": float(np.median(row_counts)),
        "missing_sector": missing_sector,
        "sector_reference": sector_report,
        "feature_columns": feature_columns,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(live_dir / "06_live_feature_state_fast_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["prefast_passed"]:
        raise SystemExit("prefast quality gate failed")


if __name__ == "__main__":
    main()
