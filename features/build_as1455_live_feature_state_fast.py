#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build pre-14:55 fast feature state for AS1455 live prediction.

This script is for production live prediction, not model training. It reads the
T-1 qfq AS1455 price tail prepared by ``as1455_live_prepare.py`` and stores a
compact numeric state. The post-14:55 finalize step then appends only today's
1000 live rows and computes only the final prediction row.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    DEFAULT_CH12_DIR,
    DEFAULT_LIVE_ROOT,
    EXPECTED_MODEL_COLUMNS,
    ensure_dir,
    normalize_symbol,
    parse_trade_date,
    write_json,
)

DEFAULT_SECTOR_REFERENCE = DEFAULT_CH12_DIR / "model_data_as1455.h5"


def _code6(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE", "NULL"}:
        return None
    m = re.search(r"(\d{6})", s)
    if not m:
        return None
    return m.group(1)


def read_table_with_fallback(base: Path) -> pd.DataFrame:
    candidates: list[Path]
    if base.suffix:
        candidates = [base, base.with_suffix(".csv")]
    else:
        candidates = [base.with_suffix(".parquet"), base.with_suffix(".csv")]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, dtype={"symbol": str}, encoding="utf-8-sig", low_memory=False)
    raise FileNotFoundError("none found: " + ", ".join(str(p) for p in candidates))


def load_sector_reference_from_model_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load a stable code/symbol -> sector mapping from training model_data HDF.

    Handles the known pandas ambiguity where ``symbol`` can be both an index
    level and a column label. We never group by a possibly ambiguous label; we
    first reset index safely and then derive a canonical 6-digit code column.
    """
    p = Path(path)
    report = {"path": str(p), "loaded": False, "key": None, "rows": 0, "unique_sector": 0, "attempts": []}
    if not p.exists():
        report["reason"] = "missing file"
        return pd.DataFrame(columns=["code", "symbol", "sector"]), report
    try:
        with pd.HDFStore(p, mode="r") as store:
            keys = list(store.keys())
    except Exception as exc:
        report["reason"] = f"HDFStore open failed: {type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=["code", "symbol", "sector"]), report

    preferred = ["/model_data", "model_data", "/data", "data"]
    ordered_keys = []
    for k in preferred + keys:
        kk = k if str(k).startswith("/") else "/" + str(k)
        if kk in keys and kk not in ordered_keys:
            ordered_keys.append(kk)

    for key in ordered_keys:
        try:
            df = pd.read_hdf(p, key=key)
            # ``reset_index`` fails when an index level name collides with an
            # existing column, e.g. symbol is both index level and column.
            # Rename index levels before reset so both sources remain visible.
            if isinstance(df.index, pd.MultiIndex):
                new_names = []
                for j, name in enumerate(df.index.names):
                    base = str(name) if name is not None else f"level_{j}"
                    new_names.append(f"__idx_{j}_{base}" if base in df.columns else base)
                df = df.copy()
                df.index = df.index.set_names(new_names)
            else:
                base = str(df.index.name) if df.index.name is not None else "index"
                if base in df.columns:
                    df = df.copy()
                    df.index = df.index.rename(f"__idx_0_{base}")
            tmp = df.reset_index()
            tmp.columns = [str(c) for c in tmp.columns]
            cols = list(tmp.columns)
            sector_cols = [c for c in cols if c == "sector" or c.endswith("sector")]
            if "sector" in tmp.columns:
                sector_col = "sector"
            elif sector_cols:
                sector_col = sector_cols[0]
            else:
                report["attempts"].append({"key": key, "ok": False, "reason": "no sector column"})
                continue

            code_series = None
            # Prefer any symbol/code-like column that can yield six digits.
            for c in ["symbol", "code", "ts_code", "level_1", "level_0", "index"] + cols:
                if c in tmp.columns:
                    s = tmp[c].map(_code6)
                    if s.notna().sum() > 0:
                        code_series = s
                        break
            if code_series is None:
                report["attempts"].append({"key": key, "ok": False, "reason": "no code/symbol column"})
                continue
            out = pd.DataFrame({"code": code_series, "sector": pd.to_numeric(tmp[sector_col], errors="coerce")})
            out = out.dropna(subset=["code", "sector"])
            if out.empty:
                report["attempts"].append({"key": key, "ok": False, "reason": "empty mapping after dropna"})
                continue
            # Sector is time-invariant in this training table. Use the latest/non-null
            # observed sector per code; if duplicates disagree, mode is safer.
            def _mode_or_last(x: pd.Series):
                vc = x.value_counts(dropna=True)
                return vc.index[0] if len(vc) else x.iloc[-1]
            out = out.groupby("code", as_index=False)["sector"].agg(_mode_or_last)
            out["sector"] = out["sector"].astype(int)
            out["symbol"] = out["code"].map(normalize_symbol)
            report.update({
                "loaded": True,
                "key": key,
                "rows": int(len(out)),
                "unique_sector": int(out["sector"].nunique()),
            })
            return out[["code", "symbol", "sector"]], report
        except Exception as exc:
            report["attempts"].append({"key": key, "ok": False, "reason": f"{type(exc).__name__}: {exc}"})
    report["reason"] = "no usable HDF key"
    return pd.DataFrame(columns=["code", "symbol", "sector"]), report


def load_feature_columns(path: str | None) -> list[str]:
    if not path:
        return EXPECTED_MODEL_COLUMNS.copy()
    p = Path(path)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AS1455 fast live feature state before 14:55")
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

    tail = read_table_with_fallback(live_dir / "05_history_tail_qfq_livebase.parquet")
    for c in ["date", "symbol", "open", "high", "low", "close", "volume"]:
        if c not in tail.columns:
            raise ValueError(f"history qfq tail missing required column: {c}")
    tail = tail[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    tail["date"] = pd.to_datetime(tail["date"], errors="coerce").dt.normalize()
    tail["symbol"] = tail["symbol"].map(normalize_symbol)
    for c in ["open", "high", "low", "close", "volume"]:
        tail[c] = pd.to_numeric(tail[c], errors="coerce")
    tail = tail.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    tail = tail.sort_values(["symbol", "date"])

    symbols = sorted(tail["symbol"].dropna().unique().tolist())
    if not symbols:
        raise RuntimeError("empty symbols in history qfq tail")
    # Keep exactly the last tail_days per symbol, padded with NaN at the front.
    T = int(args.tail_days)
    N = len(symbols)
    open_arr = np.full((N, T), np.nan, dtype=np.float64)
    high_arr = np.full((N, T), np.nan, dtype=np.float64)
    low_arr = np.full((N, T), np.nan, dtype=np.float64)
    close_arr = np.full((N, T), np.nan, dtype=np.float64)
    volume_arr = np.full((N, T), np.nan, dtype=np.float64)
    last_dates = np.array(["" for _ in range(N)], dtype=object)
    for i, sym in enumerate(symbols):
        g = tail[tail["symbol"] == sym].sort_values("date").tail(T)
        k = len(g)
        if k:
            sl = slice(T - k, T)
            open_arr[i, sl] = g["open"].to_numpy(dtype=np.float64)
            high_arr[i, sl] = g["high"].to_numpy(dtype=np.float64)
            low_arr[i, sl] = g["low"].to_numpy(dtype=np.float64)
            close_arr[i, sl] = g["close"].to_numpy(dtype=np.float64)
            volume_arr[i, sl] = g["volume"].to_numpy(dtype=np.float64)
            last_dates[i] = pd.Timestamp(g["date"].iloc[-1]).strftime("%Y-%m-%d")

    sector_ref, sector_report = load_sector_reference_from_model_data(args.sector_reference)
    code_to_sector = dict(zip(sector_ref.get("code", []), sector_ref.get("sector", [])))
    sectors = np.array([code_to_sector.get(_code6(sym), -1) for sym in symbols], dtype=np.int32)
    if (sectors < 0).any() and not args.allow_sector_fallback:
        missing = [symbols[i] for i in np.where(sectors < 0)[0][:20]]
        raise RuntimeError(f"sector reference missing {int((sectors < 0).sum())} symbols, first={missing}; report={sector_report}")

    feature_columns = load_feature_columns(args.training_feature_columns)
    ensure_dir(state_path.parent)
    np.savez_compressed(
        state_path,
        symbols=np.array(symbols, dtype=object),
        codes=np.array([_code6(s) or "" for s in symbols], dtype=object),
        sectors=sectors,
        last_dates=last_dates,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
        feature_columns=np.array(feature_columns, dtype=object),
        trade_date=np.array([trade_date], dtype=object),
    )
    report = {
        "trade_date": trade_date,
        "live_dir": str(live_dir),
        "state_file": str(state_path),
        "n_symbols": int(N),
        "tail_days": int(T),
        "history_tail_rows": int(len(tail)),
        "state_shape": [int(N), int(T)],
        "last_date_min": str(pd.Series(last_dates).replace("", np.nan).dropna().min()) if N else "",
        "last_date_max": str(pd.Series(last_dates).replace("", np.nan).dropna().max()) if N else "",
        "sector_reference": sector_report,
        "sector_unmapped_symbols": int((sectors < 0).sum()),
        "unique_sector": int(len(set(sectors[sectors >= 0].tolist()))),
        "feature_columns": feature_columns,
        "n_feature_columns": int(len(feature_columns)),
        "prefast_passed": bool(N > 0 and (sectors >= 0).mean() >= 0.98 and len(feature_columns) > 0),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(live_dir / "06_live_feature_state_fast_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["prefast_passed"]:
        raise SystemExit("prefast state build failed; see 06_live_feature_state_fast_report.json")


if __name__ == "__main__":
    main()
