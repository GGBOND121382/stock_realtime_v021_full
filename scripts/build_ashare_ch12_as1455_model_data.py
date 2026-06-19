#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Chapter 12-style A-share model_data from 14:55-or-earlier 5min bars.

This builder is data-construction only. It does not train models and does not
run backtests.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import time
from dataclasses import asdict, dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = PROJECT_DIR / "saved_data" / "ashare_static_universe" / "07_universe_allA_top1000_static.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455"
DEFAULT_BAR_ROOT = PROJECT_DIR / "saved_data"
DEFAULT_BAR_GLOB = "**/*_5m_raw.csv"
DEFAULT_QFQ_DAILY_CACHE = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce" / "baostock_qfq_daily_cache"
DEFAULT_BAOSTOCK_5M_CACHE = DEFAULT_OUT_DIR / "baostock_5m_cache"

MONTH = 21
YEAR = 12 * MONTH
MIN_OBS = 7 * YEAR
T = [1, 5, 10, 21, 42, 63]
FWD_T = [1, 5, 21]
CUTOFF = "14:55"

EXPECTED_COLUMNS = [
    "dollar_vol",
    "dollar_vol_rank",
    "rsi",
    "bb_high",
    "bb_low",
    "NATR",
    "ATR",
    "PPO",
    "MACD",
    "sector",
    "r01",
    "r05",
    "r10",
    "r21",
    "r42",
    "r63",
    "r01dec",
    "r05dec",
    "r10dec",
    "r21dec",
    "r42dec",
    "r63dec",
    "r01q_sector",
    "r05q_sector",
    "r10q_sector",
    "r21q_sector",
    "r42q_sector",
    "r63q_sector",
    "r01_fwd",
    "r05_fwd",
    "r21_fwd",
    "year",
    "month",
    "weekday",
]

FORBIDDEN_MODEL_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "board",
    "industry",
    "is_mainboard",
    "tradestatus",
    "isST",
    "raw_open_as1455",
    "raw_high_as1455",
    "raw_low_as1455",
    "raw_close_as1455",
    "raw_volume_as1455",
    "raw_amount_as1455",
    "last_bar_time",
    "open_limit_up",
    "open_limit_down",
}


@dataclass
class BuildSummary:
    universe_path: str
    bar_root: str
    bar_glob: str
    output_dir: str
    model_data_path: str
    start_date: str | None
    end_date: str | None
    cutoff: str
    timestamp_convention: str
    adjust_factor_mode: str
    min_obs: int
    universe_rows: int
    universe_symbols: int
    bar_files_found: int
    symbols_with_bar_files: int
    symbols_missing_bar_files: int
    raw_ohlcv_rows: int
    adj_ohlcv_rows: int
    symbols_after_min_obs: int
    symbols_after_industry_filter: int
    symbols_after_outlier_drop: int
    model_rows_before_dropna: int
    model_rows_after_dropna: int
    model_columns: int
    max_datetime_used: str
    used_after_cutoff_count: int
    volume_adjustment: str
    label_definition: str
    chapter17_smoke_passed: bool


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_symbol(value: Any) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 6:
        digits = digits.zfill(6)
    return digits[-6:]


def symbol_from_path(path: Path) -> str | None:
    match = re.search(r"(\d{6})", path.name)
    if match:
        return match.group(1)
    for part in reversed(path.parts):
        match = re.search(r"(\d{6})", part)
        if match:
            return match.group(1)
    return None


def require_runtime_deps() -> None:
    missing = []
    try:
        import talib  # noqa: F401
    except Exception:
        missing.append("TA-Lib/talib")
    try:
        import tables  # noqa: F401
    except Exception:
        missing.append("tables/PyTables")
    if missing:
        raise SystemExit("Missing required dependency: " + ", ".join(missing))


def qcut_codes(x: pd.Series, q: int) -> pd.Series:
    try:
        return pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=x.index)


def qcut_by_group(values: pd.Series, groupers: Any, q: int) -> pd.Series:
    return values.groupby(groupers, group_keys=False).apply(lambda x: qcut_codes(x, q)).reindex(values.index)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return s.sub(s.mean()).div(std)


def load_universe(path: Path, start_date: str | None, end_date: str | None, max_symbols: int | None) -> pd.DataFrame:
    universe = pd.read_csv(path, dtype={"code": str})
    universe["code"] = universe["code"].map(normalize_symbol)
    if "selected_for_train" in universe:
        universe = universe[universe["selected_for_train"].fillna(False).astype(bool)]
    if "industry" not in universe:
        raise RuntimeError("universe must contain industry")
    universe["industry"] = universe["industry"].fillna("").astype(str)
    if start_date:
        universe = universe.copy()
    if end_date:
        universe = universe.copy()
    if max_symbols is not None:
        universe = universe.head(max_symbols)
    return universe.drop_duplicates("code").reset_index(drop=True)


def discover_bar_files(bar_root: Path, bar_glob: str, symbols: set[str]) -> tuple[dict[str, Path], pd.DataFrame]:
    rows = []
    out: dict[str, Path] = {}
    for path in bar_root.glob(bar_glob):
        if not path.is_file():
            continue
        symbol = symbol_from_path(path)
        if symbol is None or symbol not in symbols:
            continue
        rows.append({"symbol": symbol, "path": str(path)})
        out.setdefault(symbol, path)
    return out, pd.DataFrame(rows)


def baostock_code(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}.{symbol}"


def parse_baostock_datetime(df: pd.DataFrame) -> pd.Series:
    vals = []
    for date_v, time_v in zip(df["date"].astype(str), df["time"].astype(str)):
        date_digits = "".join(ch for ch in date_v if ch.isdigit())
        time_digits = "".join(ch for ch in time_v if ch.isdigit())
        if len(time_digits) >= 14:
            raw = time_digits[:14]
        else:
            raw = date_digits[:8] + time_digits[:6].zfill(6)
        vals.append(raw)
    return pd.to_datetime(vals, format="%Y%m%d%H%M%S", errors="coerce")


def query_baostock_5m_logged_in(bs: Any, symbol: str, start_date: str, end_date: str, adjustflag: str) -> pd.DataFrame:
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    rs = bs.query_history_k_data_plus(
        baostock_code(symbol),
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="5",
        adjustflag=str(adjustflag),
    )
    if getattr(rs, "error_code", "") != "0":
        raise RuntimeError(f"BaoStock query failed for {symbol}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "datetime", "open", "high", "low", "close", "volume", "amount", "source", "bar_freq", "bar_label"])
    out = pd.DataFrame()
    out["symbol"] = normalize_symbol(symbol)
    out["trade_date"] = df["date"].astype(str).str.replace("-", "", regex=False)
    out["datetime"] = parse_baostock_datetime(df)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["source"] = f"baostock_5m_adjustflag_{adjustflag}"
    out["bar_freq"] = "5min"
    # BaoStock minute bars are handled as right-endpoint bars here; this is
    # also evidenced by the generated timestamp convention report.
    out["bar_label"] = "right"
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    if out.empty:
        return out
    t = out["datetime"].dt.time
    session = ((t >= dtime(9, 30)) & (t <= dtime(11, 30))) | ((t >= dtime(13, 0)) & (t <= dtime(15, 0)))
    return out.loc[session].sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)


def query_baostock_5m_isolated(symbol: str, start_date: str, end_date: str, adjustflag: str) -> pd.DataFrame:
    import baostock as bs  # type: ignore

    lg = bs.login()
    if getattr(lg, "error_code", "") != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
    try:
        return query_baostock_5m_logged_in(bs, symbol, start_date, end_date, adjustflag)
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def query_baostock_5m_worker(symbol: str, start_date: str, end_date: str, adjustflag: str, queue: Any) -> None:
    try:
        queue.put(("ok", query_baostock_5m_isolated(symbol, start_date, end_date, adjustflag), ""))
    except Exception as exc:
        queue.put(("error", None, f"{type(exc).__name__}: {exc}"))


def query_baostock_5m_with_timeout(symbol: str, start_date: str, end_date: str, adjustflag: str, timeout: float) -> pd.DataFrame:
    queue: Any = mp.Queue(maxsize=1)
    proc = mp.Process(target=query_baostock_5m_worker, args=(symbol, start_date, end_date, adjustflag, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        raise TimeoutError(f"timeout after {timeout:g}s")
    if queue.empty():
        raise RuntimeError(f"BaoStock worker exited with code {proc.exitcode} without returning data")
    status, df, error = queue.get()
    if status != "ok":
        raise RuntimeError(error)
    return df


def date_for_baostock(value: Any) -> str:
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%d")


def fetch_missing_baostock_5m(
    missing_symbols: list[str],
    universe: pd.DataFrame,
    cache_dir: Path,
    start_date: str | None,
    end_date: str | None,
    adjustflag: str,
    reports_dir: Path,
    retries: int,
    sleep_seconds: float,
    fetch_limit: int | None,
    query_timeout: float,
) -> tuple[dict[str, Path], pd.DataFrame]:
    ensure_dir(cache_dir)
    rows = []
    if not missing_symbols:
        return {}, pd.DataFrame(rows)
    try:
        import baostock as bs  # type: ignore
    except Exception as exc:
        raise RuntimeError("BaoStock is required to fetch missing 5min bars: pip install baostock") from exc

    meta = universe.set_index("code", drop=False)
    default_start = start_date or (pd.to_datetime(universe["history_start"], errors="coerce").min().strftime("%Y-%m-%d") if "history_start" in universe else None)
    default_end = end_date or (pd.to_datetime(universe["asof_date"], errors="coerce").max().strftime("%Y-%m-%d") if "asof_date" in universe else None)
    if not default_start or not default_end:
        raise RuntimeError("Cannot infer BaoStock start/end dates; pass --start-date and --end-date")

    if fetch_limit is not None:
        missing_symbols = missing_symbols[:fetch_limit]

    fetched: dict[str, Path] = {}
    for i, symbol in enumerate(missing_symbols, 1):
        start = start_date
        if start is None and symbol in meta.index and "history_start" in meta.columns and pd.notna(meta.at[symbol, "history_start"]):
            start = date_for_baostock(meta.at[symbol, "history_start"])
        start = start or default_start
        end = end_date or default_end
        out_path = cache_dir / f"{symbol}_5m_raw.csv"
        status = "ok"
        error = ""
        n_rows = 0
        for attempt in range(1, max(retries, 1) + 1):
            try:
                df = query_baostock_5m_with_timeout(symbol, start, end, adjustflag, query_timeout)
                n_rows = int(len(df))
                if n_rows > 0:
                    df.to_csv(out_path, index=False, encoding="utf-8-sig")
                    fetched[symbol] = out_path
                else:
                    status = "empty"
                error = ""
                break
            except TimeoutError as exc:
                status = "timeout"
                error = f"attempt {attempt}/{max(retries, 1)} {exc}"
            except Exception as exc:
                status = "error"
                error = f"attempt {attempt}/{max(retries, 1)} {type(exc).__name__}: {exc}"
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        rows.append({"symbol": symbol, "start_date": start, "end_date": end, "status": status, "rows": n_rows, "path": str(out_path if n_rows else ""), "error": error})
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(reports_dir / "as1455_baostock_5m_fetch_report.csv", index=False, encoding="utf-8-sig")
    report = pd.DataFrame(rows)
    report.to_csv(reports_dir / "as1455_baostock_5m_fetch_report.csv", index=False, encoding="utf-8-sig")
    return fetched, report


def read_5m_file(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise RuntimeError(f"{path} missing datetime column")
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["symbol", "date", "datetime", "open", "high", "low", "close", "volume", "amount"]].dropna(
        subset=["open", "high", "low", "close"]
    )


def write_coverage_report(reports_dir: Path, universe: pd.DataFrame, bar_files: dict[str, Path]) -> None:
    rows = []
    for symbol in universe["code"]:
        rows.append({"symbol": symbol, "has_5m_bar_file": symbol in bar_files, "path": str(bar_files.get(symbol, ""))})
    pd.DataFrame(rows).to_csv(reports_dir / "as1455_5min_coverage_check.csv", index=False, encoding="utf-8-sig")


def build_timestamp_convention_report(bars: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    rows = []
    if bars.empty:
        out = pd.DataFrame(columns=["date", "symbol", "last_bar_before_cutoff", "bar_times_observed", "has_14_55_bar", "suspected_timestamp_convention"])
        out.to_csv(reports_dir / "as1455_bar_timestamp_convention_check.csv", index=False, encoding="utf-8-sig")
        return out
    sample = bars.groupby(["symbol", "date"], sort=False).head(999999)
    for (symbol, date), g in sample.groupby(["symbol", "date"], sort=False):
        times = sorted(g["datetime"].dt.strftime("%H:%M").unique().tolist())
        before = [t for t in times if t <= CUTOFF]
        has_1455 = CUTOFF in times
        source_convention = ""
        if "bar_label" in g.columns:
            source_convention = ",".join(sorted(g["bar_label"].dropna().astype(str).unique()))
        suspected = "right_endpoint" if has_1455 or "right" in source_convention else "unknown"
        rows.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "symbol": symbol,
                "last_bar_before_cutoff": before[-1] if before else "",
                "bar_times_observed": " ".join(times),
                "has_14_55_bar": bool(has_1455),
                "suspected_timestamp_convention": suspected,
            }
        )
        if len(rows) >= 5000:
            break
    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / "as1455_bar_timestamp_convention_check.csv", index=False, encoding="utf-8-sig")
    return out


def aggregate_raw_as1455(
    bar_files: dict[str, Path],
    reports_dir: Path,
    start_date: str | None,
    end_date: str | None,
    timestamp_convention: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_time = pd.Timestamp(CUTOFF).time()
    rows = []
    missing = []
    convention_samples = []
    strict_lt = timestamp_convention == "left"
    for symbol, path in sorted(bar_files.items()):
        try:
            bars = read_5m_file(path, symbol)
        except Exception as exc:
            missing.append({"symbol": symbol, "date": "", "reason": f"read_error:{type(exc).__name__}:{exc}"})
            continue
        if start_date:
            bars = bars[bars["date"] >= pd.Timestamp(start_date)]
        if end_date:
            bars = bars[bars["date"] <= pd.Timestamp(end_date)]
        if len(convention_samples) < 100_000:
            convention_samples.append(bars.head(500))
        time_series = bars["datetime"].dt.time
        bars_asof = bars[time_series < cutoff_time] if strict_lt else bars[time_series <= cutoff_time]
        if bars_asof.empty:
            continue
        for date, g in bars_asof.sort_values("datetime").groupby("date", sort=False):
            if g.empty:
                missing.append({"symbol": symbol, "date": pd.Timestamp(date).strftime("%Y-%m-%d"), "reason": "missing_as1455"})
                continue
            last_dt = g["datetime"].iloc[-1]
            has_1455 = bool((g["datetime"].dt.strftime("%H:%M") == CUTOFF).any())
            if not has_1455:
                missing.append(
                    {
                        "symbol": symbol,
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "reason": "missing_14_55_bar_used_last_before_cutoff",
                        "last_bar_time": last_dt.strftime("%H:%M"),
                    }
                )
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(date),
                    "raw_open_as1455": float(g["open"].iloc[0]),
                    "raw_high_as1455": float(g["high"].max()),
                    "raw_low_as1455": float(g["low"].min()),
                    "raw_close_as1455": float(g["close"].iloc[-1]),
                    "raw_volume_as1455": float(g["volume"].fillna(0.0).sum()),
                    "raw_amount_as1455": float(g["amount"].fillna(0.0).sum()),
                    "last_bar_time": last_dt.strftime("%H:%M"),
                    "max_datetime_used": last_dt,
                    "has_14_55_bar": has_1455,
                    "used_after_cutoff": bool(last_dt.time() > cutoff_time),
                    "source_path": str(path),
                }
            )
    convention_bars = pd.concat(convention_samples, ignore_index=True) if convention_samples else pd.DataFrame()
    build_timestamp_convention_report(convention_bars, reports_dir)
    raw = pd.DataFrame(rows)
    if not raw.empty:
        raw = raw.set_index(["symbol", "date"]).sort_index()
        dist = raw.reset_index().groupby("last_bar_time").size().rename("count").reset_index()
    else:
        dist = pd.DataFrame(columns=["last_bar_time", "count"])
    dist.to_csv(reports_dir / "as1455_last_bar_time_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing).to_csv(reports_dir / "as1455_missing_bar_report.csv", index=False, encoding="utf-8-sig")
    cutoff_report = raw.reset_index()[["symbol", "date", "max_datetime_used", "last_bar_time", "has_14_55_bar", "used_after_cutoff"]] if not raw.empty else pd.DataFrame(columns=["symbol", "date", "max_datetime_used", "last_bar_time", "has_14_55_bar", "used_after_cutoff"])
    cutoff_report.to_csv(reports_dir / "as1455_cutoff_leakage_check.csv", index=False, encoding="utf-8-sig")
    return raw, pd.DataFrame(missing)


def read_daily_close(cache_dir: Path, symbol: str) -> pd.Series:
    candidates = [
        cache_dir / f"{symbol}_qfq_daily.csv",
        cache_dir / f"{symbol}_daily_raw.csv",
        cache_dir / f"{symbol}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "date" not in df.columns or "close" not in df.columns:
                continue
            s = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]).dt.normalize(), name=symbol)
            return s.dropna()
    return pd.Series(dtype=float, name=symbol)


def build_adjusted_ohlcv(
    raw: pd.DataFrame,
    reports_dir: Path,
    qfq_daily_cache: Path,
    raw_daily_cache: Path | None,
    adjust_factor_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adj = raw.copy()
    rows = []
    if adjust_factor_mode == "identity":
        adj_factor = pd.Series(1.0, index=raw.index)
        rows.append({"mode": "identity", "note": "explicit identity factor; assumes 5min bars are already adjusted"})
    else:
        if raw_daily_cache is None:
            raise RuntimeError("--raw-daily-cache-dir is required when --adjust-factor-mode daily_qfq_div_raw")
        factors = []
        for symbol in raw.index.get_level_values("symbol").unique():
            qfq = read_daily_close(qfq_daily_cache, symbol)
            raw_daily = read_daily_close(raw_daily_cache, symbol)
            if qfq.empty or raw_daily.empty:
                rows.append({"symbol": symbol, "factor_missing": True, "qfq_obs": int(len(qfq)), "raw_obs": int(len(raw_daily))})
                continue
            factor = qfq.div(raw_daily).replace([np.inf, -np.inf], np.nan).dropna().rename(symbol)
            factors.append(factor)
            rows.append(
                {
                    "symbol": symbol,
                    "factor_missing": False,
                    "qfq_obs": int(len(qfq)),
                    "raw_obs": int(len(raw_daily)),
                    "factor_obs": int(len(factor)),
                    "factor_min": float(factor.min()),
                    "factor_max": float(factor.max()),
                }
            )
        factor_panel = pd.concat(factors, axis=1).stack() if factors else pd.Series(dtype=float)
        factor_panel.index = factor_panel.index.set_names(["date", "symbol"]).swaplevel()
        adj_factor = factor_panel.reindex(raw.index)
        missing_factor = adj_factor.isna()
        if missing_factor.any():
            missing = raw.index[missing_factor]
            rows.append({"mode": "daily_qfq_div_raw", "missing_factor_rows": int(missing_factor.sum())})
            raise RuntimeError(f"missing adjustment factors for {int(missing_factor.sum())} rows; see as1455_adjust_factor_check.csv")
    adj["adj_factor"] = adj_factor.to_numpy()
    for col in ["open", "high", "low", "close"]:
        adj[f"adj_{col}_as1455"] = adj[f"raw_{col}_as1455"].mul(adj["adj_factor"])
    adj["adj_volume_as1455"] = adj["raw_volume_as1455"]
    pd.DataFrame(rows).to_csv(reports_dir / "as1455_adjust_factor_check.csv", index=False, encoding="utf-8-sig")
    return adj, pd.DataFrame(rows)


def make_prices_and_metadata(adj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.DataFrame(
        {
            "open": adj["adj_open_as1455"],
            "high": adj["adj_high_as1455"],
            "low": adj["adj_low_as1455"],
            "close": adj["adj_close_as1455"],
            "volume": adj["adj_volume_as1455"],
        },
        index=adj.index,
    ).sort_index()
    metadata = adj[
        [
            "raw_open_as1455",
            "raw_high_as1455",
            "raw_low_as1455",
            "raw_close_as1455",
            "raw_volume_as1455",
            "raw_amount_as1455",
            "last_bar_time",
            "has_14_55_bar",
            "used_after_cutoff",
        ]
    ].copy()
    return prices, metadata


def filter_prices_and_universe(prices: pd.DataFrame, universe: pd.DataFrame, min_obs: int, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_industry = universe[universe["industry"].astype(str).str.len().gt(0)].copy()
    nobs = prices.groupby(level="symbol").size().rename("nobs").reset_index()
    nobs.to_csv(reports_dir / "as1455_nobs_by_symbol.csv", index=False, encoding="utf-8-sig")
    keep_obs = set(nobs.loc[nobs["nobs"] > min_obs, "symbol"])
    keep_industry = set(valid_industry["code"])
    keep = keep_obs.intersection(keep_industry)
    prices = prices.loc[prices.index.get_level_values("symbol").isin(keep)].copy()
    metadata = valid_industry[valid_industry["code"].isin(keep)].copy()
    return prices, metadata, nobs


def compute_features(prices: pd.DataFrame, universe_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import talib
    from talib import ATR, BBANDS, MACD, RSI

    prices = prices.sort_index().copy()
    meta = universe_meta.copy()
    meta["sector"] = pd.factorize(meta["industry"])[0].astype(int)
    sector_map = meta.set_index("code")["sector"]

    # Chapter 12 A-share daily builder first scales volume by 1e3, then computes
    # dollar_vol = close * volume / 1e3. Equivalent to close * raw_volume / 1e6.
    prices["volume"] = prices["volume"].div(1e3)
    prices["dollar_vol"] = prices["close"].mul(prices["volume"]).div(1e3)
    dollar_vol_ma = prices["dollar_vol"].unstack("symbol").rolling(window=MONTH, min_periods=1).mean()
    prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().swaplevel()
    del dollar_vol_ma

    prices["rsi"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(RSI)

    def compute_bb(close: pd.Series) -> pd.DataFrame:
        upper, _mid, lower = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": upper, "bb_low": lower}, index=close.index)

    bb = prices.groupby(level="symbol", group_keys=False)["close"].apply(compute_bb)
    prices["bb_high"] = bb["bb_high"].sub(prices["close"]).div(bb["bb_high"]).apply(np.log1p)
    prices["bb_low"] = prices["close"].sub(bb["bb_low"]).div(prices["close"]).apply(np.log1p)
    del bb

    prices["NATR"] = prices.groupby(level="symbol", group_keys=False).apply(lambda x: talib.NATR(x["high"], x["low"], x["close"]))

    def compute_atr(g: pd.DataFrame) -> pd.Series:
        return zscore(ATR(g["high"], g["low"], g["close"], timeperiod=14))

    prices["ATR"] = prices.groupby(level="symbol", group_keys=False).apply(compute_atr)
    prices["PPO"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(talib.PPO)

    def compute_macd(close: pd.Series) -> pd.Series:
        return zscore(MACD(close)[0])

    prices["MACD"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(compute_macd)
    prices["sector"] = prices.index.get_level_values("symbol").map(sector_map).astype(int)

    by_symbol_close = prices.groupby(level="symbol")["close"]
    for t in T:
        prices[f"r{t:02}"] = by_symbol_close.pct_change(t)

    dates = prices.index.get_level_values("date")
    for t in T:
        prices[f"r{t:02}dec"] = qcut_by_group(prices[f"r{t:02}"], dates, 10)
    for t in T:
        prices[f"r{t:02}q_sector"] = qcut_by_group(prices[f"r{t:02}"], [dates, prices["sector"]], 5)
    for t in FWD_T:
        prices[f"r{t:02}_fwd"] = prices.groupby(level="symbol")[f"r{t:02}"].shift(-t)

    outliers = prices[prices["r01"] > 1].index.get_level_values("symbol").unique()
    outlier_df = pd.DataFrame({"symbol": list(outliers)})
    if len(outliers):
        prices = prices.drop(outliers, level="symbol")

    dates = prices.index.get_level_values("date")
    prices["year"] = dates.year
    prices["month"] = dates.month
    prices["weekday"] = dates.weekday
    return prices, outlier_df


def write_feature_column_check(model_data: pd.DataFrame, reports_dir: Path) -> None:
    check = pd.DataFrame(
        {
            "expected": EXPECTED_COLUMNS,
            "actual": list(model_data.columns),
            "matches_expected_position": [a == b for a, b in zip(EXPECTED_COLUMNS, model_data.columns)],
        }
    )
    check["actual_columns_match_expected_34"] = list(model_data.columns) == EXPECTED_COLUMNS
    check.to_csv(reports_dir / "as1455_feature_column_check.csv", index=False, encoding="utf-8-sig")


def make_label_alignment_samples(features_with_close: pd.DataFrame, reports_dir: Path, sample_n: int = 200) -> dict[str, float]:
    rows = []
    max_diff: dict[str, float] = {}
    for t in FWD_T:
        fwd_col = f"r{t:02}_fwd"
        diffs = []
        for symbol, g in features_with_close.groupby(level="symbol", sort=False):
            g = g.sort_index()
            dates = g.index.get_level_values("date")
            close = g["close"]
            manual = close.shift(-t).div(close).sub(1.0)
            diff = g[fwd_col].sub(manual)
            valid_diff = diff.abs().dropna()
            if not valid_diff.empty:
                diffs.append(float(valid_diff.max()))
            if len(rows) < sample_n:
                valid_pos = np.flatnonzero(diff.notna().to_numpy())
                for pos in valid_pos[: max(1, sample_n // 20)]:
                    rows.append(
                        {
                            "symbol": symbol,
                            "date_t": dates[pos].strftime("%Y-%m-%d"),
                            f"date_t{t}": dates[pos + t].strftime("%Y-%m-%d") if pos + t < len(g) else "",
                            "close_as1455_t": float(close.iloc[pos]),
                            f"close_as1455_t{t}": float(close.iloc[pos + t]) if pos + t < len(g) else np.nan,
                            f"{fwd_col}_t": float(g[fwd_col].iloc[pos]),
                            f"manual_{fwd_col}": float(manual.iloc[pos]),
                            "diff": float(diff.iloc[pos]),
                            "label_definition": f"{fwd_col}: t 14:55 to t+{t} 14:55, not full daily close-to-close",
                        }
                    )
        max_diff[f"max_abs_diff_{fwd_col}"] = max(diffs) if diffs else float("nan")
        if not math.isnan(max_diff[f"max_abs_diff_{fwd_col}"]) and max_diff[f"max_abs_diff_{fwd_col}"] >= 1e-10:
            raise RuntimeError(f"{fwd_col} alignment failed: {max_diff[f'max_abs_diff_{fwd_col}']}")
    pd.DataFrame(rows).head(sample_n).to_csv(reports_dir / "as1455_label_alignment_samples.csv", index=False, encoding="utf-8-sig")
    return max_diff


def write_daily_counts(model_data: pd.DataFrame, reports_dir: Path) -> pd.Series:
    before = model_data.groupby(level="date").size().rename("sample_count")
    clean_mask = ~model_data.isna().any(axis=1)
    after = clean_mask.groupby(level="date").sum().astype(int).rename("sample_count")
    before.to_csv(reports_dir / "as1455_daily_sample_count_before_dropna.csv", encoding="utf-8-sig")
    after.to_csv(reports_dir / "as1455_daily_sample_count_after_dropna.csv", encoding="utf-8-sig")
    return clean_mask


def write_model_data_hdf(path: Path, model_data: pd.DataFrame, chunk_rows: int = 250_000) -> None:
    if path.exists():
        path.unlink()
    with pd.HDFStore(path, mode="w") as store:
        for start in range(0, len(model_data), chunk_rows):
            stop = min(start + chunk_rows, len(model_data))
            store.append("model_data", model_data.iloc[start:stop], format="table", index=True)


def write_hdf(path: Path, key: str, df: pd.DataFrame) -> None:
    if path.exists():
        path.unlink()
    df.to_hdf(path, key, mode="w", format="table")


def run_chapter17_read_smoke(model_data_path: Path, reports_dir: Path) -> bool:
    data = pd.read_hdf(model_data_path, "model_data").dropna().sort_index()
    outcomes = data.filter(like="fwd").columns.tolist()
    ok = outcomes == ["r01_fwd", "r05_fwd", "r21_fwd"] and not data.empty
    X = data.drop(outcomes, axis=1) if ok else pd.DataFrame()
    y = data["r01_fwd"] if ok else pd.Series(dtype=float)
    report = {
        "ok": bool(ok),
        "rows_after_dropna": int(len(data)),
        "X_shape": list(X.shape),
        "y_rows": int(len(y)),
        "outcomes": outcomes,
        "date_min": data.index.get_level_values("date").min().strftime("%Y-%m-%d") if not data.empty else "",
        "date_max": data.index.get_level_values("date").max().strftime("%Y-%m-%d") if not data.empty else "",
    }
    (reports_dir / "as1455_chapter17_read_smoke_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return bool(ok)


def compare_with_daily_leakage(prices: pd.DataFrame, qfq_daily_cache: Path, reports_dir: Path, sample_n: int = 500) -> None:
    rows = []
    sample_index = prices.sample(min(sample_n, len(prices)), random_state=7).index if len(prices) else []
    for symbol, date in sample_index:
        daily_path = qfq_daily_cache / f"{symbol}_qfq_daily.csv"
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path)
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        row = daily[daily["date"].eq(pd.Timestamp(date))]
        if row.empty:
            continue
        d = row.iloc[0]
        p = prices.loc[(symbol, date)]
        rows.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "close_as1455": p["close"],
                "daily_close": d.get("close", np.nan),
                "high_as1455": p["high"],
                "daily_high": d.get("high", np.nan),
                "low_as1455": p["low"],
                "daily_low": d.get("low", np.nan),
                "volume_as1455": p["volume"] * 1e3,
                "daily_volume": d.get("volume", np.nan),
                "volume_as1455_le_daily_volume": bool((p["volume"] * 1e3) <= d.get("volume", np.nan)) if pd.notna(d.get("volume", np.nan)) else pd.NA,
            }
        )
    pd.DataFrame(rows).to_csv(reports_dir / "as1455_daily_leakage_sample_check.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Chapter 12-style model_data from 14:55 5min bars")
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--bar-root", default=str(DEFAULT_BAR_ROOT))
    p.add_argument("--bar-glob", default=DEFAULT_BAR_GLOB)
    p.add_argument("--baostock-5m-cache-dir", default=str(DEFAULT_BAOSTOCK_5M_CACHE))
    p.add_argument("--fetch-missing-baostock", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--baostock-adjustflag",
        default="3",
        help="BaoStock minute adjustflag for fetched 5min bars. Existing local project tools use 3.",
    )
    p.add_argument("--baostock-fetch-retries", type=int, default=3)
    p.add_argument("--baostock-fetch-sleep", type=float, default=1.0)
    p.add_argument("--baostock-fetch-limit", type=int, default=None, help="Fetch at most this many missing symbols in this run; useful for resumable batches")
    p.add_argument("--baostock-query-timeout", type=float, default=180.0, help="Per-symbol BaoStock query timeout in seconds")
    p.add_argument("--qfq-daily-cache-dir", default=str(DEFAULT_QFQ_DAILY_CACHE))
    p.add_argument("--raw-daily-cache-dir", default=None)
    p.add_argument("--adjust-factor-mode", choices=["daily_qfq_div_raw", "identity"], default="daily_qfq_div_raw")
    p.add_argument("--timestamp-convention", choices=["right", "left"], default="right")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--min-obs", type=int, default=MIN_OBS)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--allow-partial-coverage", action="store_true", help="Write partial sample output when 5min files do not cover the full universe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_deps()
    out_dir = ensure_dir(Path(args.out_dir))
    reports_dir = ensure_dir(out_dir / "reports")
    model_data_path = out_dir / "model_data_as1455.h5"

    universe = load_universe(Path(args.universe), args.start_date, args.end_date, args.max_symbols)
    universe_symbols = set(universe["code"])
    bar_files, bar_file_rows = discover_bar_files(Path(args.bar_root), args.bar_glob, universe_symbols)
    cache_files, cache_file_rows = discover_bar_files(Path(args.baostock_5m_cache_dir), "*_5m_raw.csv", universe_symbols)
    for symbol, path in cache_files.items():
        bar_files.setdefault(symbol, path)
    if not cache_file_rows.empty:
        cache_file_rows = cache_file_rows.assign(source="baostock_5m_cache")
    if not bar_file_rows.empty:
        bar_file_rows = bar_file_rows.assign(source="bar_root")
    bar_file_rows = pd.concat([bar_file_rows, cache_file_rows], ignore_index=True) if not cache_file_rows.empty else bar_file_rows
    bar_file_rows.to_csv(reports_dir / "as1455_bar_file_sources.csv", index=False, encoding="utf-8-sig")
    write_coverage_report(reports_dir, universe, bar_files)
    missing_symbols = sorted(universe_symbols.difference(bar_files))
    if missing_symbols and args.fetch_missing_baostock:
        fetched_files, _fetch_report = fetch_missing_baostock_5m(
            missing_symbols,
            universe,
            Path(args.baostock_5m_cache_dir),
            args.start_date,
            args.end_date,
            args.baostock_adjustflag,
            reports_dir,
            args.baostock_fetch_retries,
            args.baostock_fetch_sleep,
            args.baostock_fetch_limit,
            args.baostock_query_timeout,
        )
        for symbol, path in fetched_files.items():
            bar_files.setdefault(symbol, path)
        missing_symbols = sorted(universe_symbols.difference(bar_files))
        write_coverage_report(reports_dir, universe, bar_files)
    if missing_symbols and not args.allow_partial_coverage:
        summary = {
            "status": "blocked",
            "reason": "5min bar coverage is incomplete after local discovery and optional BaoStock fetch; rerun with --allow-partial-coverage only for smoke/sample builds",
            "universe_symbols": len(universe_symbols),
            "symbols_with_bar_files": len(bar_files),
            "symbols_missing_bar_files": len(missing_symbols),
            "coverage_report": str((reports_dir / "as1455_5min_coverage_check.csv").resolve()),
            "baostock_fetch_report": str((reports_dir / "as1455_baostock_5m_fetch_report.csv").resolve()),
        }
        (reports_dir / "as1455_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(json.dumps(summary, ensure_ascii=False))

    raw, missing = aggregate_raw_as1455(bar_files, reports_dir, args.start_date, args.end_date, args.timestamp_convention)
    if raw.empty:
        raise SystemExit("No as1455 raw OHLCV rows were constructed; see reports.")
    used_after_cutoff_count = int(raw["used_after_cutoff"].sum())
    if used_after_cutoff_count:
        raise RuntimeError(f"cutoff leakage detected: used_after_cutoff_count={used_after_cutoff_count}")

    raw_hdf = out_dir / "as1455_ohlcv_raw.h5"
    write_hdf(raw_hdf, "ohlcv", raw)

    adj, factor_check = build_adjusted_ohlcv(raw, reports_dir, Path(args.qfq_daily_cache_dir), Path(args.raw_daily_cache_dir) if args.raw_daily_cache_dir else None, args.adjust_factor_mode)
    adj_hdf = out_dir / "as1455_ohlcv_adj.h5"
    write_hdf(adj_hdf, "ohlcv", adj)

    prices, exec_meta = make_prices_and_metadata(adj)
    exec_meta_hdf = out_dir / "as1455_execution_metadata.h5"
    write_hdf(exec_meta_hdf, "metadata", exec_meta)

    prices, metadata, nobs = filter_prices_and_universe(prices, universe, args.min_obs, reports_dir)
    if prices.empty:
        raise SystemExit("No symbols satisfy min_obs and industry filters; see reports/as1455_nobs_by_symbol.csv")
    features_with_prices, outliers = compute_features(prices, metadata)
    outliers.to_csv(reports_dir / "as1455_outlier_symbols_r01_gt_1.csv", index=False, encoding="utf-8-sig")
    label_validation = make_label_alignment_samples(features_with_prices, reports_dir)
    compare_with_daily_leakage(prices, Path(args.qfq_daily_cache_dir), reports_dir)

    model_data = features_with_prices.drop(["open", "high", "low", "close", "volume"], axis=1)
    model_data = model_data[EXPECTED_COLUMNS]
    forbidden = sorted(FORBIDDEN_MODEL_COLUMNS.intersection(model_data.columns))
    if forbidden:
        raise RuntimeError(f"forbidden columns in model_data: {forbidden}")
    if list(model_data.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("model_data columns do not match expected 34-column schema")
    if model_data.shape[1] != 34:
        raise RuntimeError(f"model_data must have 34 columns, got {model_data.shape[1]}")
    write_feature_column_check(model_data, reports_dir)
    clean_mask = write_daily_counts(model_data, reports_dir)
    write_model_data_hdf(model_data_path, model_data)
    smoke_ok = run_chapter17_read_smoke(model_data_path, reports_dir)

    max_dt = pd.to_datetime(raw["max_datetime_used"]).max()
    summary = BuildSummary(
        universe_path=str(Path(args.universe).resolve()),
        bar_root=str(Path(args.bar_root).resolve()),
        bar_glob=args.bar_glob,
        output_dir=str(out_dir.resolve()),
        model_data_path=str(model_data_path.resolve()),
        start_date=args.start_date,
        end_date=args.end_date,
        cutoff=CUTOFF,
        timestamp_convention=args.timestamp_convention,
        adjust_factor_mode=args.adjust_factor_mode,
        min_obs=args.min_obs,
        universe_rows=int(len(universe)),
        universe_symbols=int(len(universe_symbols)),
        bar_files_found=int(len(bar_file_rows)),
        symbols_with_bar_files=int(len(bar_files)),
        symbols_missing_bar_files=int(len(missing_symbols)),
        raw_ohlcv_rows=int(len(raw)),
        adj_ohlcv_rows=int(len(adj)),
        symbols_after_min_obs=int(prices.index.get_level_values("symbol").nunique()),
        symbols_after_industry_filter=int(metadata["code"].nunique()),
        symbols_after_outlier_drop=int(model_data.index.get_level_values("symbol").nunique()),
        model_rows_before_dropna=int(len(model_data)),
        model_rows_after_dropna=int(clean_mask.sum()),
        model_columns=int(model_data.shape[1]),
        max_datetime_used=max_dt.strftime("%Y-%m-%d %H:%M:%S"),
        used_after_cutoff_count=used_after_cutoff_count,
        volume_adjustment="none",
        label_definition="as1455 r01_fwd is t 14:55 to t+1 14:55, not full daily close-to-close",
        chapter17_smoke_passed=smoke_ok,
    )
    (reports_dir / "as1455_build_summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
