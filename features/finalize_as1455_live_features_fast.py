#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize only the current AS1455 row from a precomputed live feature state.

The output is the 31-column base feature contract used by the historical Ch12
model data. The clean Chapter-17 layer subsequently adds sector rotation,
compact addons and one-hot columns using its authoritative shared code.
"""
from __future__ import annotations

import argparse
import json
import sys
import math
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from features.as1455_live_common import (
    DEFAULT_LIVE_ROOT,
    EXPECTED_MODEL_COLUMNS,
    ensure_dir,
    normalize_symbol,
    parse_trade_date,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None

T_WINDOWS = [1, 5, 10, 21, 42, 63]


def _safe_std(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) >= 2 else float("nan")


def _last_zscore(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = _safe_std(x)
    return float((x[-1] - x.mean()) / sd) if np.isfinite(sd) and sd > 0 else float("nan")


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values, dtype="float64").ewm(span=span, adjust=False).mean().to_numpy()


def _rsi_fallback(close: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(close, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).to_numpy()


def _atr_fallback(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    h = pd.Series(high, dtype="float64")
    l = pd.Series(low, dtype="float64")
    c = pd.Series(close, dtype="float64")
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()


def _last_indicators(high: np.ndarray, low: np.ndarray, close: np.ndarray, *, allow_fallback: bool) -> dict[str, float]:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (high > 0) & (low > 0) & (close > 0)
    if valid.sum() < 65:
        return {k: float("nan") for k in ["rsi", "bb_high", "bb_low", "NATR", "ATR", "PPO", "MACD"]}
    first = int(np.argmax(valid))
    high, low, close = high[first:], low[first:], close[first:]
    if talib is None and not allow_fallback:
        raise RuntimeError("TA-Lib is required for production live feature parity; install requirements.txt or pass --allow-indicator-fallback only for tests")
    if talib is not None:
        rsi_arr = talib.RSI(close)
        upper, _mid, lower = talib.BBANDS(close, timeperiod=20)
        natr_arr = talib.NATR(high, low, close)
        atr_arr = talib.ATR(high, low, close, timeperiod=14)
        ppo_arr = talib.PPO(close)
        macd_arr = talib.MACD(close)[0]
    else:
        rsi_arr = _rsi_fallback(close)
        ma = pd.Series(close).rolling(20).mean().to_numpy()
        sd = pd.Series(close).rolling(20).std(ddof=0).to_numpy()
        upper, lower = ma + 2 * sd, ma - 2 * sd
        atr_arr = _atr_fallback(high, low, close)
        natr_arr = 100.0 * atr_arr / close
        ema_fast, ema_slow = _ema(close, 12), _ema(close, 26)
        ppo_arr = (ema_fast - ema_slow) / ema_slow * 100.0
        macd_arr = ema_fast - ema_slow
    cl, up, lo = close[-1], upper[-1], lower[-1]
    return {
        "rsi": float(rsi_arr[-1]) if np.isfinite(rsi_arr[-1]) else float("nan"),
        "bb_high": float(np.log1p((up - cl) / up)) if np.isfinite(up) and up != 0 else float("nan"),
        "bb_low": float(np.log1p((cl - lo) / cl)) if np.isfinite(lo) and cl != 0 else float("nan"),
        "NATR": float(natr_arr[-1]) if np.isfinite(natr_arr[-1]) else float("nan"),
        "ATR": _last_zscore(atr_arr),
        "PPO": float(ppo_arr[-1]) if np.isfinite(ppo_arr[-1]) else float("nan"),
        "MACD": _last_zscore(macd_arr),
    }


def qcut_safe(values: pd.Series, q: int) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna()
    if valid.nunique() < 2:
        return out
    try:
        out.loc[valid.index] = pd.qcut(valid.rank(method="first"), q=q, labels=False, duplicates="drop")
    except Exception:
        pass
    return out


def live_raw_to_qfq_row(raw: pd.DataFrame, events: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    df = raw.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    if not events.empty and "event_ratio" in events.columns:
        ev = events[[c for c in ["symbol", "event_ratio", "is_factor_event_today"] if c in events.columns]].copy()
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
        "raw_open_as1455": df.get("raw_open_as1455", np.nan),
        "raw_high_as1455": df.get("raw_high_as1455", np.nan),
        "raw_low_as1455": df.get("raw_low_as1455", np.nan),
        "raw_close_as1455": df.get("raw_close_as1455", np.nan),
        "raw_volume_as1455": df.get("raw_volume_as1455", np.nan),
        "raw_amount_as1455": df.get("raw_amount_as1455", np.nan),
        "live_preclose": df.get("live_preclose", np.nan),
    })
    return out.sort_values("symbol").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize AS1455 live base features after 14:55")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--training-feature-columns", default=None)
    ap.add_argument("--min-feature-rows", type=int, default=980)
    ap.add_argument("--max-elapsed-seconds", type=float, default=40.0)
    ap.add_argument("--warn-only-time", action="store_true")
    ap.add_argument("--allow-indicator-fallback", action="store_true")
    args = ap.parse_args()

    started = time.time()
    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    state_path = Path(args.state_file) if args.state_file else live_dir / "06_live_feature_state_fast.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"missing fast state: {state_path}")

    state = np.load(state_path, allow_pickle=True)
    symbols = [str(x) for x in state["symbols"].tolist()]
    sectors = state["sectors"].astype(int)
    high_hist = state["high"].astype(float)
    low_hist = state["low"].astype(float)
    close_hist = state["close"].astype(float)
    volume_hist = state["volume"].astype(float)
    state_cols = [str(x) for x in state["feature_columns"].tolist()]
    if state_cols != list(EXPECTED_MODEL_COLUMNS):
        raise RuntimeError(f"state feature contract mismatch: {state_cols}")

    raw_path = live_dir / "08_live_raw_row_as1455.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw_live = pd.read_csv(raw_path, dtype={"symbol": str}, encoding="utf-8-sig")
    events_path = live_dir / "03_adjustment_events.csv"
    events = pd.read_csv(events_path, dtype={"symbol": str}, encoding="utf-8-sig") if events_path.exists() else pd.DataFrame()
    live_qfq = live_raw_to_qfq_row(raw_live, events, trade_date)
    write_csv(live_dir / "09_live_qfq_row_as1455.csv", live_qfq)
    live_map = live_qfq.drop_duplicates("symbol", keep="last").set_index("symbol")

    today_high = np.full(len(symbols), np.nan)
    today_low = np.full(len(symbols), np.nan)
    today_close = np.full(len(symbols), np.nan)
    today_volume = np.full(len(symbols), np.nan)
    missing_live: list[str] = []
    for i, sym in enumerate(symbols):
        if sym not in live_map.index:
            missing_live.append(sym)
            continue
        row = live_map.loc[sym]
        today_high[i] = pd.to_numeric(row.get("high"), errors="coerce")
        today_low[i] = pd.to_numeric(row.get("low"), errors="coerce")
        today_close[i] = pd.to_numeric(row.get("close"), errors="coerce")
        today_volume[i] = pd.to_numeric(row.get("volume"), errors="coerce")

    hist_dollar_vol = close_hist * volume_hist / 1_000_000.0
    today_dollar_vol = today_close * today_volume / 1_000_000.0
    dv_21 = np.concatenate([hist_dollar_vol[:, -20:], today_dollar_vol[:, None]], axis=1)
    dv_ma = np.nanmean(dv_21, axis=1)
    dv_rank = pd.Series(dv_ma).rank(ascending=False, method="average").to_numpy()

    rows: list[dict] = []
    for i, sym in enumerate(symbols):
        h = np.concatenate([high_hist[i], [today_high[i]]])
        l = np.concatenate([low_hist[i], [today_low[i]]])
        c = np.concatenate([close_hist[i], [today_close[i]]])
        indicators = _last_indicators(h, l, c, allow_fallback=args.allow_indicator_fallback)
        row: dict[str, object] = {
            "date": yyyymmdd_to_dash(trade_date),
            "symbol": sym,
            "dollar_vol": today_dollar_vol[i],
            "dollar_vol_rank": dv_rank[i],
            **indicators,
            "sector": int(sectors[i]),
        }
        for t in T_WINDOWS:
            previous = c[-1 - t] if len(c) > t and np.isfinite(c[-1 - t]) else np.nan
            row[f"r{t:02}"] = c[-1] / previous - 1.0 if np.isfinite(c[-1]) and np.isfinite(previous) and previous > 0 else np.nan
        rows.append(row)

    features = pd.DataFrame(rows)
    for t in T_WINDOWS:
        col = f"r{t:02}"
        features[f"r{t:02}dec"] = qcut_safe(features[col], 10)
        features[f"r{t:02}q_sector"] = features.groupby("sector", group_keys=False, dropna=False)[col].apply(lambda x: qcut_safe(x, 5))
    date_ts = pd.Timestamp(yyyymmdd_to_dash(trade_date))
    features["year"], features["month"], features["weekday"] = date_ts.year, date_ts.month, date_ts.weekday()
    features = features[["date", "symbol", *EXPECTED_MODEL_COLUMNS]]
    for col in EXPECTED_MODEL_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    outlier_mask = features["r01"].gt(1)
    complete_mask = features[EXPECTED_MODEL_COLUMNS].notna().all(axis=1) & features["sector"].ge(0) & ~outlier_mask
    usable = features.loc[complete_mask].copy()
    dropped = features.loc[~complete_mask].copy()
    if not dropped.empty:
        dropped["drop_reason"] = ""
        dropped.loc[dropped["sector"].lt(0), "drop_reason"] += "missing_sector;"
        dropped.loc[outlier_mask, "drop_reason"] += "r01_gt_1;"
        missing_counts = features[EXPECTED_MODEL_COLUMNS].isna().sum(axis=1)
        dropped.loc[missing_counts.gt(0), "drop_reason"] += "missing_model_features;"

    write_csv(live_dir / "11_live_model_features.csv", features)
    write_csv(live_dir / "11_live_model_features_for_prediction.csv", usable)
    write_csv(live_dir / "11_live_model_features_dropped_rows.csv", dropped)
    elapsed = round(time.time() - started, 3)
    feature_passed = bool(len(usable) >= args.min_feature_rows)
    time_passed = bool(elapsed <= args.max_elapsed_seconds)
    report = {
        "feature_passed": feature_passed,
        "trade_date": trade_date,
        "state_file": str(state_path),
        "raw_live_rows": int(len(raw_live)),
        "state_symbols": int(len(symbols)),
        "missing_live_symbols": len(missing_live),
        "missing_live_examples": missing_live[:20],
        "feature_rows_all": int(len(features)),
        "feature_rows_usable": int(len(usable)),
        "feature_rows_dropped": int(len(dropped)),
        "min_feature_rows": int(args.min_feature_rows),
        "feature_columns": list(EXPECTED_MODEL_COLUMNS),
        "talib_used": talib is not None,
        "indicator_fallback_allowed": bool(args.allow_indicator_fallback),
        "elapsed_seconds": elapsed,
        "max_elapsed_seconds": float(args.max_elapsed_seconds),
        "time_passed": time_passed,
        "prediction_file": str(live_dir / "11_live_model_features_for_prediction.csv"),
    }
    write_json(live_dir / "12_feature_build_report.json", report)
    strict = {
        "passed": bool(feature_passed and (time_passed or args.warn_only_time)),
        "feature_rows_usable": int(len(usable)),
        "feature_columns_exact": list(usable.columns[2:]) == list(EXPECTED_MODEL_COLUMNS),
        "finite": bool(len(usable) and np.isfinite(usable[EXPECTED_MODEL_COLUMNS].to_numpy(dtype=float)).all()),
        "prediction_file": str(live_dir / "11_live_model_features_for_prediction.csv"),
        "elapsed_seconds": elapsed,
        "time_passed": time_passed,
        "warn_only_time": bool(args.warn_only_time),
        "talib_used": talib is not None,
    }
    strict["passed"] = bool(strict["passed"] and strict["feature_columns_exact"] and strict["finite"])
    write_json(live_dir / "13_live_feature_strict_validation_report.json", strict)
    print(json.dumps({"feature_report": report, "strict_report": strict}, ensure_ascii=False, indent=2), flush=True)
    if not strict["passed"]:
        raise SystemExit("live feature strict validation failed")


if __name__ == "__main__":
    main()
