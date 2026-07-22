#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a current-day execution panel for the clean v7 live planner.

Model features remain isolated from these execution-only fields. The sidecar is
constructed from the <=14:55 raw snapshot plus T-1 raw-daily metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from features.as1455_live_common import normalize_symbol, parse_trade_date, raw_daily_path, yyyymmdd_to_dash


def infer_mainboard(symbol: str) -> bool:
    code = re.search(r"(\d{6})", str(symbol)).group(1)
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t", "ok", "正常", "交易"}:
        return True
    if text in {"0", "false", "no", "n", "f", "停牌"}:
        return False
    try:
        return float(text) != 0.0
    except Exception:
        return default


def load_universe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
    if "symbol" not in df.columns and "code" in df.columns:
        df["symbol"] = df["code"]
    if "symbol" not in df.columns:
        return pd.DataFrame()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df.drop_duplicates("symbol", keep="last")


def latest_raw_daily(cache_dir: Path, symbol: str, trade_date: pd.Timestamp) -> dict:
    path = raw_daily_path(cache_dir, symbol)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return {}
    if "date" not in df.columns:
        return {}
    df = df.copy()
    df["date_ts"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    candidates = df[df["date_ts"] <= trade_date].sort_values("date_ts")
    return {} if candidates.empty else candidates.iloc[-1].to_dict()


def first_value(row: pd.Series, names: list[str], default=np.nan):
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AS1455 live v7 execution sidecar")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--out-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--raw-daily-cache-dir", default="saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache")
    ap.add_argument("--universe", default="saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv")
    ap.add_argument("--input-raw-row", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--normal-limit-pct", type=float, default=0.10)
    ap.add_argument("--st-limit-pct", type=float, default=0.05)
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    trade_ts = pd.Timestamp(yyyymmdd_to_dash(trade_date)).normalize()
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    raw_path = Path(args.input_raw_row) if args.input_raw_row else live_dir / "08_live_raw_row_as1455.csv"
    out_path = Path(args.output) if args.output else live_dir / "08_live_execution_sidecar.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    live = pd.read_csv(raw_path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
    if "symbol" not in live.columns and "code" in live.columns:
        live["symbol"] = live["code"]
    if "symbol" not in live.columns:
        raise RuntimeError(f"{raw_path} has no symbol/code")
    live["symbol"] = live["symbol"].map(normalize_symbol)
    live = live.drop_duplicates("symbol", keep="last")
    universe = load_universe(Path(args.universe) if args.universe else None)
    if not universe.empty:
        keep = [c for c in ["symbol", "name", "board", "industry", "is_mainboard", "trade_allowed_mainboard"] if c in universe.columns]
        live = live.merge(universe[keep], on="symbol", how="left", suffixes=("", "_universe"))

    rows: list[dict] = []
    raw_cache = Path(args.raw_daily_cache_dir)
    for _, row in live.iterrows():
        symbol = row["symbol"]
        historical = latest_raw_daily(raw_cache, symbol, trade_ts)
        price = pd.to_numeric(first_value(row, ["raw_close_as1455", "close_as1455", "last_price", "price", "close"]), errors="coerce")
        preclose = pd.to_numeric(first_value(row, ["live_preclose", "prev_close", "preclose", "raw_preclose"]), errors="coerce")
        if not np.isfinite(preclose) or preclose <= 0:
            if historical:
                hist_date = pd.to_datetime(historical.get("date_ts"), errors="coerce")
                if pd.notna(hist_date) and hist_date == trade_ts and "preclose" in historical:
                    preclose = pd.to_numeric(historical.get("preclose"), errors="coerce")
                else:
                    preclose = pd.to_numeric(historical.get("close"), errors="coerce")
        name = str(first_value(row, ["name", "name_universe"], ""))
        is_st = "ST" in name.upper()
        if historical and "isST" in historical:
            is_st = parse_bool(historical.get("isST"), is_st)
        is_mainboard = parse_bool(first_value(row, ["is_mainboard", "trade_allowed_mainboard", "is_mainboard_universe"], None), infer_mainboard(symbol))
        core_complete = parse_bool(row.get("core_complete", True), True)
        tradestatus = pd.to_numeric(historical.get("tradestatus", np.nan), errors="coerce") if historical else np.nan
        tradable = bool((tradestatus > 0 if np.isfinite(tradestatus) else core_complete) and np.isfinite(price) and price > 0)
        limit_pct = args.st_limit_pct if is_st else args.normal_limit_pct
        up_limit = round(float(preclose) * (1 + limit_pct), 2) if np.isfinite(preclose) and preclose > 0 else np.nan
        down_limit = round(float(preclose) * (1 - limit_pct), 2) if np.isfinite(preclose) and preclose > 0 else np.nan
        rows.append({
            "date": trade_ts,
            "symbol": symbol,
            "raw_close_1500": price,
            "qfq_close_1500": price,
            "raw_preclose": preclose,
            "prev_raw_close_1500": preclose,
            "event_ratio": 1.0,
            "tradable": tradable,
            "is_st": is_st,
            "is_mainboard": is_mainboard,
            "up_limit": up_limit,
            "down_limit": down_limit,
            "last5_volume": np.nan,
            "last5_amount": np.nan,
            "raw_open_as1455": first_value(row, ["raw_open_as1455", "open_as1455", "open"]),
            "raw_high_as1455": first_value(row, ["raw_high_as1455", "high_as1455", "high"]),
            "raw_low_as1455": first_value(row, ["raw_low_as1455", "low_as1455", "low"]),
            "name": name,
            "board": first_value(row, ["board", "board_universe"], ""),
            "industry": first_value(row, ["industry", "industry_universe"], ""),
            "execution_sidecar_source": "live_as1455_snapshot+Tminus1_raw_daily",
        })
    sidecar = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(out_path, index=False, encoding="utf-8-sig")
    report = {
        "passed": bool(len(sidecar) and sidecar["raw_close_1500"].notna().mean() >= 0.95),
        "trade_date": trade_date,
        "input": str(raw_path),
        "output": str(out_path),
        "rows": int(len(sidecar)),
        "price_coverage": float(sidecar["raw_close_1500"].notna().mean()) if len(sidecar) else 0.0,
        "preclose_coverage": float(sidecar["raw_preclose"].notna().mean()) if len(sidecar) else 0.0,
        "tradable": int(sidecar["tradable"].sum()) if len(sidecar) else 0,
        "mainboard": int(sidecar["is_mainboard"].sum()) if len(sidecar) else 0,
        "is_st": int(sidecar["is_st"].sum()) if len(sidecar) else 0,
        "capacity_note": "last5 fields are unavailable at the 14:55 signal cutoff; strict live default is capacity_mode=none",
    }
    (live_dir / "08_live_execution_sidecar_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("execution sidecar quality gate failed")


if __name__ == "__main__":
    main()
