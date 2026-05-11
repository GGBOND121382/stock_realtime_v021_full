#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test Muyuan HK / hog HK proxy / hog futures context features.

This is a network diagnostic test, not a unit test with mocked data.  It checks
whether the data sources required by the saved 002714 all_no_ak model can be
fetched and whether representative features can be derived.

Run from project root:
    python3 tests/test_muyuan_realtime_context_features.py

Outputs:
    debug_muyuan_context_features/summary.json
    debug_muyuan_context_features/*.csv

Exit code:
    0: required feature groups are available
    1: one or more required groups failed

Use --allow-partial for diagnostic-only mode that never fails the shell job.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT_DIR = Path("debug_muyuan_context_features")
TIMEOUT = 25

MUYUAN_HK = "02714"
HK_PROXIES = {
    "01610": "hog_hk_cofco_joycome",
    "00288": "hog_hk_wh_group",
    "01068": "hog_hk_yurun_food",
    "01117": "hog_hk_modern_dairy",
}


def _run_child(fn, timeout: int = TIMEOUT) -> dict[str, Any]:
    def worker(q):
        try:
            q.put({"ok": True, "value": fn()})
        except Exception as e:
            q.put({"ok": False, "error": repr(e), "traceback": traceback.format_exc(limit=8)})

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "error": f"timeout>{timeout}s"}
    if q.empty():
        return {"ok": False, "error": "no result from child process"}
    return q.get()


def _save_df(name: str, df: pd.DataFrame) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_hk_daily(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _col(raw, ["date", "日期"])
    open_col = _col(raw, ["open", "开盘"])
    high_col = _col(raw, ["high", "最高"])
    low_col = _col(raw, ["low", "最低"])
    close_col = _col(raw, ["close", "收盘"])
    volume_col = _col(raw, ["volume", "成交量"])
    amount_col = _col(raw, ["amount", "成交额"])
    if date_col is None or close_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    out["open"] = _num(raw[open_col]) if open_col else np.nan
    out["high"] = _num(raw[high_col]) if high_col else np.nan
    out["low"] = _num(raw[low_col]) if low_col else np.nan
    out["close"] = _num(raw[close_col])
    out["volume"] = _num(raw[volume_col]) if volume_col else np.nan
    out["amount"] = _num(raw[amount_col]) if amount_col else np.nan
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def add_ret_features(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(df) < 21:
        return out
    close = _num(df["close"])
    volume = _num(df["volume"]) if "volume" in df else pd.Series(dtype=float)
    out[f"{prefix}_close"] = float(close.iloc[-1])
    out[f"{prefix}_close_ret1"] = float(close.iloc[-1] / close.iloc[-2] - 1.0)
    out[f"{prefix}_close_ret20"] = float(close.iloc[-1] / close.iloc[-21] - 1.0)
    if len(volume.dropna()) >= 2 and volume.iloc[-2] != 0:
        out[f"{prefix}_volume_ret1"] = float(volume.iloc[-1] / volume.iloc[-2] - 1.0)
    return out


def fetch_muyuan_hk_spot() -> tuple[pd.DataFrame, dict[str, Any]]:
    import akshare as ak
    raw = ak.stock_zh_ah_spot()
    mask = pd.Series(False, index=raw.index)
    for c in raw.columns:
        s = raw[c].astype(str)
        mask |= s.str.contains("02714|2714|牧原", regex=True, na=False)
    hit = raw[mask].copy()
    _save_df("muyuan_stock_zh_ah_spot_hit", hit)
    if hit.empty:
        raise RuntimeError("02714/牧原 not found in stock_zh_ah_spot")
    row = hit.iloc[0]
    features = {
        "hog_hk_muyuan_close": float(row["最新价"]),
        "hog_hk_muyuan_open": float(row["今开"]),
        "hog_hk_muyuan_high": float(row["最高"]),
        "hog_hk_muyuan_low": float(row["最低"]),
        "hog_hk_muyuan_volume": float(row["成交量"]),
        "hog_hk_muyuan_amount": float(row["成交额"]),
    }
    return hit, features


def fetch_hk_daily(symbol: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.stock_hk_daily(symbol=symbol)
    df = normalize_hk_daily(raw)
    if df.empty:
        raise RuntimeError(f"stock_hk_daily({symbol}) returned empty/unusable data")
    return df


def fetch_muyuan_hk_history() -> tuple[pd.DataFrame, dict[str, float]]:
    df = fetch_hk_daily(MUYUAN_HK)
    _save_df("muyuan_stock_hk_daily_02714", df)
    feats = add_ret_features(df, "hog_hk_muyuan")
    required = ["hog_hk_muyuan_close_ret1", "hog_hk_muyuan_close_ret20"]
    missing = [x for x in required if x not in feats or not np.isfinite(feats[x])]
    if missing:
        raise RuntimeError(f"missing derived muyuan history features: {missing}")
    return df, feats


def fetch_proxy_history() -> tuple[pd.DataFrame, dict[str, float]]:
    frames = []
    feats: dict[str, float] = {}
    for symbol, prefix in HK_PROXIES.items():
        try:
            df = fetch_hk_daily(symbol)
            f = add_ret_features(df, prefix)
            for k, v in f.items():
                feats[k] = v
            tmp = df[["date", "close", "volume", "amount"]].copy()
            tmp["symbol"] = symbol
            tmp["prefix"] = prefix
            frames.append(tmp)
            _save_df(f"proxy_{prefix}_{symbol}", df)
        except Exception as e:
            feats[f"{prefix}_ERROR"] = str(e)
    if not frames:
        raise RuntimeError("all HK proxy stock_hk_daily fetches failed")
    all_df = pd.concat(frames, ignore_index=True)
    # Aggregate close mean by date, then compute historical aggregate features.
    wide = all_df.pivot_table(index="date", columns="prefix", values="close", aggfunc="last").sort_index()
    proxy_mean = wide.mean(axis=1).dropna().reset_index(name="close")
    proxy_feats = add_ret_features(proxy_mean, "hog_hk_proxy_close_mean")
    feats.update(proxy_feats)
    _save_df("hog_hk_proxy_close_mean", proxy_mean)
    if "hog_hk_yurun_food_volume_ret1" not in feats:
        raise RuntimeError("missing hog_hk_yurun_food_volume_ret1; 01068 proxy unavailable or insufficient history")
    if "hog_hk_proxy_close_mean_close_ret1" in feats:
        # Backward-compatible aliases expected by historical builder naming.
        feats["hog_hk_proxy_close_mean_ret1"] = feats.pop("hog_hk_proxy_close_mean_close_ret1")
    if "hog_hk_proxy_close_mean_close_ret20" in feats:
        feats["hog_hk_proxy_close_mean_ret20"] = feats.pop("hog_hk_proxy_close_mean_close_ret20")
    required = ["hog_hk_proxy_close_mean_ret1", "hog_hk_proxy_close_mean_ret20", "hog_hk_yurun_food_volume_ret1"]
    missing = [x for x in required if x not in feats or not np.isfinite(feats[x])]
    if missing:
        raise RuntimeError(f"missing derived proxy features: {missing}")
    return all_df, feats


def normalize_futures_daily(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _col(raw, ["date", "日期"])
    close_col = _col(raw, ["close", "收盘价", "收盘"])
    hold_col = _col(raw, ["hold", "持仓量", "持仓"])
    if date_col is None or close_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    out["close"] = _num(raw[close_col])
    out["hold"] = _num(raw[hold_col]) if hold_col else np.nan
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def fetch_hog_futures_history() -> tuple[pd.DataFrame, dict[str, float]]:
    import akshare as ak
    errors = []
    for fn_name in ["futures_zh_daily_sina", "futures_main_sina"]:
        try:
            fn = getattr(ak, fn_name)
            raw = fn(symbol="LH0")
            df = normalize_futures_daily(raw)
            if len(df) < 61:
                raise RuntimeError(f"{fn_name}(LH0) insufficient rows={len(df)}")
            close = _num(df["close"])
            mean60 = close.shift(1).rolling(60, min_periods=30).mean().iloc[-1]
            std60 = close.shift(1).rolling(60, min_periods=30).std().iloc[-1]
            feats = {
                "hog_fut_close": float(close.iloc[-1]),
                "hog_fut_close_z60": float((close.iloc[-1] - mean60) / std60) if std60 and np.isfinite(std60) else np.nan,
            }
            hold = _num(df["hold"]) if "hold" in df else pd.Series(dtype=float)
            if len(hold.dropna()) >= 2 and hold.iloc[-2] != 0:
                feats["hog_fut_hold_ret1"] = float(hold.iloc[-1] / hold.iloc[-2] - 1.0)
            _save_df(f"hog_futures_{fn_name}_LH0", df)
            missing = [x for x in ["hog_fut_close_z60"] if x not in feats or not np.isfinite(feats[x])]
            if missing:
                raise RuntimeError(f"{fn_name}(LH0) missing {missing}")
            return df, feats
        except Exception as e:
            errors.append(f"{fn_name}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", help="Always exit 0; write failures to summary only.")
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tests = {
        "muyuan_hk_realtime_spot": fetch_muyuan_hk_spot,
        "muyuan_hk_history_daily": fetch_muyuan_hk_history,
        "hog_hk_proxy_history_daily": fetch_proxy_history,
        "hog_futures_history": fetch_hog_futures_history,
    }
    results = {}
    for name, fn in tests.items():
        print(f"[TEST] {name} ...", flush=True)
        r = _run_child(fn, timeout=args.timeout)
        if r.get("ok"):
            val = r.get("value")
            if isinstance(val, tuple) and len(val) == 2:
                df, feats = val
                results[name] = {
                    "ok": True,
                    "rows": int(len(df)) if isinstance(df, pd.DataFrame) else None,
                    "features": feats,
                }
            else:
                results[name] = {"ok": True, "value": str(type(val))}
            print(f"  OK")
        else:
            results[name] = {"ok": False, "error": r.get("error"), "traceback": r.get("traceback")}
            print(f"  FAIL: {r.get('error')}")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWROTE {summary_path}")

    failed = [k for k, v in results.items() if not v.get("ok")]
    if failed:
        print("FAILED GROUPS:", ", ".join(failed))
        return 0 if args.allow_partial else 1
    print("ALL REQUIRED MUYUAN CONTEXT FEATURE GROUPS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
