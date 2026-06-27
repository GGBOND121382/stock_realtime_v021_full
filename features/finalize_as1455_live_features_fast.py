#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast post-14:55 AS1455 live feature finalizer.

Production goal: after 14:55, do NOT rebuild the 252-day feature panel. This
script reads the precomputed ``06_live_feature_state_fast.npz`` plus today's
``08_live_raw_row_as1455.csv`` and computes only the final T-day prediction
feature rows.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    DEFAULT_LIVE_ROOT,
    EXPECTED_MODEL_COLUMNS,
    ensure_dir,
    normalize_symbol,
    parse_trade_date,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)

try:  # Prefer exact TA-Lib semantics when available on the server.
    import talib  # type: ignore
except Exception:  # pragma: no cover - fallback is used in lightweight envs.
    talib = None

T_WINDOWS = [1, 5, 10, 21, 42, 63]
MONTH = 21


def _safe_std(x: np.ndarray) -> float:
    valid = x[np.isfinite(x)]
    if len(valid) < 2:
        return float("nan")
    return float(valid.std(ddof=1))


def _last_zscore(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) < 2:
        return float("nan")
    std = valid.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float((valid[-1] - valid.mean()) / std)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    # Fallback approximation. Server should use TA-Lib where installed.
    return pd.Series(values, dtype="float64").ewm(span=span, adjust=False).mean().to_numpy()


def _rsi_fallback(close: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(close, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder-style smoothing approximation.
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


def _last_indicators(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, float]:
    """Compute only final-row TA indicators for one symbol."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (high > 0) & (low > 0) & (close > 0)
    if valid.sum() < 65:
        return {k: float("nan") for k in ["rsi", "bb_high", "bb_low", "NATR", "ATR", "PPO", "MACD"]}
    # Keep the aligned valid suffix. Padding NaNs at the front are not useful.
    first = int(np.argmax(valid))
    high = high[first:]
    low = low[first:]
    close = close[first:]
    try:
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
            upper = ma + 2 * sd
            lower = ma - 2 * sd
            atr_arr = _atr_fallback(high, low, close)
            natr_arr = 100.0 * atr_arr / close
            ema_fast = _ema(close, 12)
            ema_slow = _ema(close, 26)
            ppo_arr = (ema_fast - ema_slow) / ema_slow * 100.0
            macd_arr = ema_fast - ema_slow
        cl = close[-1]
        up = upper[-1]
        lo = lower[-1]
        bb_high = float(np.log1p((up - cl) / up)) if np.isfinite(up) and up != 0 else float("nan")
        bb_low = float(np.log1p((cl - lo) / cl)) if np.isfinite(lo) and cl != 0 else float("nan")
        return {
            "rsi": float(rsi_arr[-1]) if np.isfinite(rsi_arr[-1]) else float("nan"),
            "bb_high": bb_high,
            "bb_low": bb_low,
            "NATR": float(natr_arr[-1]) if np.isfinite(natr_arr[-1]) else float("nan"),
            "ATR": _last_zscore(atr_arr),
            "PPO": float(ppo_arr[-1]) if np.isfinite(ppo_arr[-1]) else float("nan"),
            "MACD": _last_zscore(macd_arr),
        }
    except Exception:
        return {k: float("nan") for k in ["rsi", "bb_high", "bb_low", "NATR", "ATR", "PPO", "MACD"]}


def qcut_safe(x: pd.Series, q: int) -> pd.Series:
    valid = x.dropna()
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    if valid.nunique() < 2:
        return out
    try:
        out.loc[valid.index] = pd.qcut(valid.rank(method="first"), q=q, labels=False, duplicates="drop")
    except Exception:
        out.loc[valid.index] = np.nan
    return out


def load_feature_columns(path: str | None, state_cols: list[str] | None = None) -> list[str]:
    if path:
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
    if state_cols:
        return list(state_cols)
    return EXPECTED_MODEL_COLUMNS.copy()


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
    ap = argparse.ArgumentParser(description="Fast AS1455 post-14:55 prediction feature finalizer")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--training-feature-columns", default=None)
    ap.add_argument("--min-feature-rows", type=int, default=980)
    ap.add_argument("--max-elapsed-seconds", type=float, default=40.0)
    ap.add_argument("--warn-only-time", action="store_true", help="do not fail if elapsed exceeds max")
    args = ap.parse_args()

    started = time.time()
    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.live_dir) if args.live_dir else Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    state_path = Path(args.state_file) if args.state_file else live_dir / "06_live_feature_state_fast.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"missing fast feature state: {state_path}. Run prefast first.")

    state = np.load(state_path, allow_pickle=True)
    symbols = [str(x) for x in state["symbols"].tolist()]
    sectors = state["sectors"].astype(int)
    high_hist = state["high"].astype(float)
    low_hist = state["low"].astype(float)
    close_hist = state["close"].astype(float)
    volume_hist = state["volume"].astype(float)
    state_feature_cols = [str(x) for x in state.get("feature_columns", np.array(EXPECTED_MODEL_COLUMNS, dtype=object)).tolist()]
    training_cols = load_feature_columns(args.training_feature_columns, state_feature_cols)

    raw_live = pd.read_csv(live_dir / "08_live_raw_row_as1455.csv", dtype={"symbol": str}, encoding="utf-8-sig")
    events_path = live_dir / "03_adjustment_events.csv"
    events = pd.read_csv(events_path, dtype={"symbol": str}, encoding="utf-8-sig") if events_path.exists() else pd.DataFrame()
    live_qfq = live_raw_to_qfq_row(raw_live, events, trade_date)
    write_csv(live_dir / "09_live_qfq_row_as1455.csv", live_qfq)

    live_map = live_qfq.set_index("symbol")
    rows = []
    missing_live = []
    for i, sym in enumerate(symbols):
        if sym not in live_map.index:
            missing_live.append(sym)
            continue
        lr = live_map.loc[sym]
        today_open = float(lr["open"]) if pd.notna(lr["open"]) else np.nan
        today_high = float(lr["high"]) if pd.notna(lr["high"]) else np.nan
        today_low = float(lr["low"]) if pd.notna(lr["low"]) else np.nan
        today_close = float(lr["close"]) if pd.notna(lr["close"]) else np.nan
        today_volume = float(lr["volume"]) if pd.notna(lr["volume"]) else np.nan
        h = np.concatenate([high_hist[i], [today_high]])
        l = np.concatenate([low_hist[i], [today_low]])
        c = np.concatenate([close_hist[i], [today_close]])
        v = np.concatenate([volume_hist[i], [today_volume]])
        ind = _last_indicators(h, l, c)
        row = {
            "date": yyyymmdd_to_dash(trade_date),
            "symbol": sym,
            "sector": int(sectors[i]),
            "dollar_vol": today_close * (today_volume / 1e3) / 1e3 if np.isfinite(today_close) and np.isfinite(today_volume) else np.nan,
            **ind,
        }
        # 21-day dollar_vol rolling mean for ranking, in the same units as training.
        hist_dv = close_hist[i] * (volume_hist[i] / 1e3) / 1e3
        prev20 = hist_dv[np.isfinite(hist_dv)][-20:]
        row["_dollar_vol_ma21_today"] = np.nanmean(np.concatenate([prev20, [row["dollar_vol"]]])) if np.isfinite(row["dollar_vol"]) else np.nan
        for t in T_WINDOWS:
            lag = c[-(t + 1)] if len(c) >= t + 1 else np.nan
            row[f"r{t:02}"] = today_close / lag - 1.0 if np.isfinite(today_close) and np.isfinite(lag) and lag != 0 else np.nan
        rows.append(row)

    full = pd.DataFrame(rows)
    if full.empty:
        raise RuntimeError("no live feature rows were produced")
    # Cross-sectional ranks/quantiles are the only truly cross-symbol operations after 14:55.
    full["dollar_vol_rank"] = full["_dollar_vol_ma21_today"].rank(ascending=False, method="average")
    for t in T_WINDOWS:
        full[f"r{t:02}dec"] = qcut_safe(full[f"r{t:02}"], 10)
    for t in T_WINDOWS:
        full[f"r{t:02}q_sector"] = full.groupby("sector", group_keys=False)[f"r{t:02}"].apply(lambda x: qcut_safe(x, 5))
    dt = pd.Timestamp(yyyymmdd_to_dash(trade_date))
    full["year"] = dt.year
    full["month"] = dt.month
    full["weekday"] = dt.weekday()

    missing_cols = [c for c in training_cols if c not in full.columns]
    for c in missing_cols:
        full[c] = np.nan
    ordered = ["date", "symbol"] + training_cols
    full_out = full[ordered].copy()
    write_csv(live_dir / "11_live_model_features.csv", full_out)

    nan_mask = full_out[training_cols].isna().any(axis=1)
    usable = full_out.loc[~nan_mask].copy()
    dropped = full_out.loc[nan_mask, ["date", "symbol"]].copy()
    if len(dropped):
        reasons = []
        for idx in full_out.index[nan_mask]:
            cols = [c for c in training_cols if pd.isna(full_out.at[idx, c])]
            reasons.append(",".join(cols))
        dropped["missing_feature_columns"] = reasons
    else:
        dropped["missing_feature_columns"] = []
    write_csv(live_dir / "11_live_model_features_usable.csv", usable)
    write_csv(live_dir / "11_live_model_features_for_prediction.csv", usable)
    write_csv(live_dir / "11_live_model_features_dropped_rows.csv", dropped)

    nan_by_column = {c: int(full_out[c].isna().sum()) for c in training_cols}
    usable_nan_cols = [c for c in training_cols if len(usable) and usable[c].isna().any()]
    elapsed = time.time() - started
    report = {
        "trade_date": trade_date,
        "live_dir": str(live_dir),
        "state_file": str(state_path),
        "mode": "fast_finalize_final_row_only",
        "talib_available": bool(talib is not None),
        "live_raw_rows": int(len(raw_live)),
        "live_qfq_rows": int(len(live_qfq)),
        "state_symbols": int(len(symbols)),
        "missing_live_symbols": int(len(missing_live)),
        "live_feature_rows": int(len(full_out)),
        "usable_feature_rows": int(len(usable)),
        "dropped_feature_rows": int(len(dropped)),
        "training_feature_columns": training_cols,
        "n_training_feature_columns": int(len(training_cols)),
        "missing_training_columns": missing_cols,
        "nan_by_column": nan_by_column,
        "nan_columns": [c for c, n in nan_by_column.items() if n > 0],
        "usable_nan_columns": usable_nan_cols,
        "sector_unmapped_rows": int((full_out.get("sector", pd.Series(dtype=float)) < 0).sum()) if "sector" in full_out.columns else int(len(full_out)),
        "sector_unique_count": int(full_out["sector"].nunique()) if "sector" in full_out.columns else 0,
        "dollar_vol_rank_nonnull": int(full_out["dollar_vol_rank"].notna().sum()) if "dollar_vol_rank" in full_out.columns else 0,
        "dollar_vol_rank_min": None if "dollar_vol_rank" not in full_out or full_out["dollar_vol_rank"].dropna().empty else float(full_out["dollar_vol_rank"].min()),
        "dollar_vol_rank_max": None if "dollar_vol_rank" not in full_out or full_out["dollar_vol_rank"].dropna().empty else float(full_out["dollar_vol_rank"].max()),
        "elapsed_seconds": round(elapsed, 3),
        "max_elapsed_seconds": float(args.max_elapsed_seconds),
        "time_passed": bool(elapsed <= args.max_elapsed_seconds),
    }
    report["feature_passed"] = bool(
        len(usable) >= args.min_feature_rows
        and not missing_cols
        and not usable_nan_cols
        and report["sector_unmapped_rows"] == 0
        and report["dollar_vol_rank_nonnull"] >= args.min_feature_rows
    )
    write_json(live_dir / "12_feature_build_report.json", report)
    # 13 is intentionally similar but named as strict validation for downstream checks.
    validation = {
        "passed": bool(report["feature_passed"] and (report["time_passed"] or args.warn_only_time)),
        "feature_passed": bool(report["feature_passed"]),
        "time_passed": bool(report["time_passed"]),
        "elapsed_seconds": report["elapsed_seconds"],
        "feature_rows_full": int(len(full_out)),
        "feature_rows_usable": int(len(usable)),
        "dropped_rows_from_full": int(len(dropped)),
        "prediction_file": str(live_dir / "11_live_model_features_for_prediction.csv"),
        "prediction_file_matches_usable": True,
        "nan_columns": usable_nan_cols,
        "missing_training_columns": missing_cols,
        "dollar_vol_rank_bad_rows": 0 if report["dollar_vol_rank_nonnull"] == len(full_out) else int(len(full_out) - report["dollar_vol_rank_nonnull"]),
        "sector_unmapped_rows": report["sector_unmapped_rows"],
        "missing_live_symbols": int(len(missing_live)),
    }
    write_json(live_dir / "13_live_feature_strict_validation_report.json", validation)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    if not report["feature_passed"]:
        raise SystemExit("fast live feature finalize failed; see 12_feature_build_report.json")
    if elapsed > args.max_elapsed_seconds and not args.warn_only_time:
        raise SystemExit(f"fast live feature finalize exceeded time budget: {elapsed:.3f}s > {args.max_elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()
