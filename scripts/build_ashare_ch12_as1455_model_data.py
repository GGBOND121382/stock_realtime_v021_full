#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Chapter 12-style A-share model_data from 14:55-or-earlier 5min bars.

This builder is data-construction only. It does not train models and does not
run backtests.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
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
DEFAULT_AS1455_DAILY_CACHE = DEFAULT_OUT_DIR / "as1455_daily_cache"

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


def log_memory(enabled: bool, label: str) -> None:
    if not enabled:
        return
    try:
        import psutil

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f"[memory] {label}: rss_mb={rss_mb:.1f}", flush=True)
    except Exception as exc:
        print(f"[memory] {label}: unavailable {type(exc).__name__}: {exc}", flush=True)


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
    if isinstance(groupers, (list, tuple)):
        group_index = pd.MultiIndex.from_arrays(groupers)
        group_codes, _ = pd.factorize(group_index, sort=False)
    else:
        group_codes, _ = pd.factorize(groupers, sort=False)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if not len(values):
        return pd.Series(result, index=values.index)

    order = np.argsort(group_codes, kind="stable")
    ordered_codes = group_codes[order]
    boundaries = np.flatnonzero(np.r_[True, ordered_codes[1:] != ordered_codes[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        positions = order[start:stop]
        if ordered_codes[start] < 0:
            continue
        cut = qcut_codes(values.iloc[positions], q)
        result[positions] = cut.to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.Series(result, index=values.index)


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


def query_baostock_5m_worker(symbol: str, start_date: str, end_date: str, adjustflag: str, out_path: str, queue: Any) -> None:
    try:
        df = query_baostock_5m_isolated(symbol, start_date, end_date, adjustflag)
        n_rows = int(len(df))
        if n_rows > 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            queue.put(("ok", n_rows, out_path, ""))
        else:
            queue.put(("empty", 0, "", ""))
    except Exception as exc:
        queue.put(("error", 0, "", f"{type(exc).__name__}: {exc}"))


def query_baostock_5m_with_timeout(symbol: str, start_date: str, end_date: str, adjustflag: str, out_path: Path, timeout: float) -> tuple[str, int, str]:
    queue: Any = mp.Queue(maxsize=1)
    proc = mp.Process(target=query_baostock_5m_worker, args=(symbol, start_date, end_date, adjustflag, str(out_path), queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        raise TimeoutError(f"timeout after {timeout:g}s")
    if queue.empty():
        raise RuntimeError(f"BaoStock worker exited with code {proc.exitcode} without returning data")
    status, n_rows, path, error = queue.get()
    if status == "error":
        raise RuntimeError(error)
    return status, int(n_rows), str(path)


def query_baostock_qfq_daily_worker(symbol: str, start_date: str, end_date: str, out_path: str, queue: Any) -> None:
    try:
        import baostock as bs  # type: ignore

        lg = bs.login()
        if getattr(lg, "error_code", "") != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
        try:
            fields = "date,code,open,high,low,close,volume"
            rs = bs.query_history_k_data_plus(
                baostock_code(symbol),
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            if getattr(rs, "error_code", "") != "0":
                raise RuntimeError(f"BaoStock qfq daily query failed: {rs.error_code} {rs.error_msg}")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            if not df.empty:
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
            queue.put(("ok" if len(df) else "empty", int(len(df)), out_path if len(df) else "", ""))
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    except Exception as exc:
        queue.put(("error", 0, "", f"{type(exc).__name__}: {exc}"))


def query_baostock_qfq_daily_with_timeout(symbol: str, start_date: str, end_date: str, out_path: Path, timeout: float) -> tuple[str, int, str]:
    queue: Any = mp.Queue(maxsize=1)
    proc = mp.Process(target=query_baostock_qfq_daily_worker, args=(symbol, start_date, end_date, str(out_path), queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        raise TimeoutError(f"timeout after {timeout:g}s")
    if queue.empty():
        raise RuntimeError(f"BaoStock qfq daily worker exited with code {proc.exitcode} without returning data")
    status, n_rows, path, error = queue.get()
    if status == "error":
        raise RuntimeError(error)
    return status, int(n_rows), str(path)


def materialize_qfq_daily_cache(
    symbols: list[str],
    cache_dir: Path,
    reports_dir: Path,
    start_date: str,
    end_date: str,
    retries: int,
    sleep_seconds: float,
    query_timeout: float,
) -> pd.DataFrame:
    ensure_dir(cache_dir)
    report_rows = []
    for i, symbol in enumerate(symbols, 1):
        out_path = cache_dir / f"{symbol}_qfq_daily.csv"
        status = "cached"
        n_rows = 0
        error = ""
        needs_fetch = not out_path.exists() or out_path.stat().st_size == 0
        if not needs_fetch:
            try:
                cached_dates = pd.to_datetime(pd.read_csv(out_path, usecols=["date"])["date"], errors="coerce").dropna()
                needs_fetch = cached_dates.empty or cached_dates.min() > pd.Timestamp(start_date) + pd.Timedelta(days=10) or cached_dates.max() < pd.Timestamp(end_date) - pd.Timedelta(days=10)
                n_rows = int(len(cached_dates))
            except Exception:
                needs_fetch = True
        if needs_fetch:
            for attempt in range(1, max(1, retries) + 1):
                try:
                    status, n_rows, _ = query_baostock_qfq_daily_with_timeout(symbol, start_date, end_date, out_path, query_timeout)
                    error = ""
                    break
                except Exception as exc:
                    status = "timeout" if isinstance(exc, TimeoutError) else "error"
                    error = f"attempt {attempt}/{max(1, retries)} {type(exc).__name__}: {exc}"
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
        report_rows.append({"symbol": symbol, "status": status, "rows": n_rows, "path": str(out_path if out_path.exists() else ""), "error": error})
        print(f"[qfq-daily] {i}/{len(symbols)} {symbol} status={status} rows={n_rows} error={error}", flush=True)
        pd.DataFrame(report_rows).to_csv(reports_dir / "as1455_qfq_daily_fetch_report.csv", index=False, encoding="utf-8-sig")
        if sleep_seconds > 0 and status != "cached":
            time.sleep(sleep_seconds)
    return pd.DataFrame(report_rows)


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
                status, n_rows, written_path = query_baostock_5m_with_timeout(symbol, start, end, adjustflag, out_path, query_timeout)
                if status == "ok" and n_rows > 0:
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
        print(f"[baostock] {i}/{len(missing_symbols)} {symbol} status={status} rows={n_rows} path={out_path if n_rows else ''} error={error}", flush=True)
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


def aggregate_symbol_as1455(
    path: Path,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    timestamp_convention: str,
) -> pd.DataFrame:
    cutoff_time = pd.Timestamp(CUTOFF).time()
    strict_lt = timestamp_convention == "left"
    bars = read_5m_file(path, symbol)
    if start_date:
        bars = bars[bars["date"] >= pd.Timestamp(start_date)]
    if end_date:
        bars = bars[bars["date"] <= pd.Timestamp(end_date)]
    observed_bar_times = " ".join(sorted(bars["datetime"].dt.strftime("%H:%M").dropna().unique().tolist()))
    full_day_close = bars.sort_values("datetime").groupby("date", sort=False, observed=True)["close"].last()
    time_series = bars["datetime"].dt.time
    bars = bars[time_series < cutoff_time] if strict_lt else bars[time_series <= cutoff_time]
    if bars.empty:
        return pd.DataFrame()

    bars.sort_values("datetime", inplace=True)
    bars["has_14_55_bar"] = bars["datetime"].dt.strftime("%H:%M").eq(CUTOFF)
    grouped = bars.groupby("date", sort=False, observed=True)
    daily = grouped.agg(
        raw_open_as1455=("open", "first"),
        raw_high_as1455=("high", "max"),
        raw_low_as1455=("low", "min"),
        raw_close_as1455=("close", "last"),
        raw_volume_as1455=("volume", "sum"),
        raw_amount_as1455=("amount", "sum"),
        max_datetime_used=("datetime", "last"),
        has_14_55_bar=("has_14_55_bar", "any"),
    ).reset_index()
    daily.insert(0, "symbol", symbol)
    daily["raw_daily_close"] = daily["date"].map(full_day_close)
    daily["last_bar_time"] = daily["max_datetime_used"].dt.strftime("%H:%M")
    daily["used_after_cutoff"] = daily["max_datetime_used"].dt.time.gt(cutoff_time)
    daily["source_path"] = str(path)
    daily.attrs["bar_times_observed"] = observed_bar_times
    return daily


def materialize_as1455_daily_cache(
    bar_files: dict[str, Path],
    cache_dir: Path,
    reports_dir: Path,
    start_date: str | None,
    end_date: str | None,
    timestamp_convention: str,
    rebuild: bool,
) -> tuple[dict[str, Path], pd.DataFrame]:
    ensure_dir(cache_dir)
    report_rows = []
    cache_files: dict[str, Path] = {}
    total = len(bar_files)
    for i, (symbol, path) in enumerate(sorted(bar_files.items()), 1):
        cache_path = cache_dir / f"{symbol}_as1455_daily.csv"
        timestamp_meta_path = cache_dir / f"{symbol}_as1455_daily.timestamp.json"
        status = "cached"
        error = ""
        rows = 0
        bar_times_observed = ""
        if rebuild or not cache_path.exists() or cache_path.stat().st_size == 0:
            try:
                daily = aggregate_symbol_as1455(path, symbol, start_date, end_date, timestamp_convention)
                rows = int(len(daily))
                bar_times_observed = str(daily.attrs.get("bar_times_observed", ""))
                if daily.empty:
                    status = "empty"
                else:
                    daily.to_csv(cache_path, index=False, encoding="utf-8-sig")
                    timestamp_meta_path.write_text(
                        json.dumps({"symbol": symbol, "bar_times_observed": bar_times_observed}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    status = "ok"
                del daily
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                with cache_path.open("r", encoding="utf-8-sig") as fh:
                    rows = max(0, sum(1 for _ in fh) - 1)
                if timestamp_meta_path.exists():
                    bar_times_observed = str(json.loads(timestamp_meta_path.read_text(encoding="utf-8")).get("bar_times_observed", ""))
            except Exception:
                rows = -1
        if cache_path.exists() and cache_path.stat().st_size > 0:
            cache_files[symbol] = cache_path
        report_rows.append(
            {
                "symbol": symbol,
                "source_path": str(path),
                "daily_cache_path": str(cache_path),
                "timestamp_meta_path": str(timestamp_meta_path),
                "bar_times_observed": bar_times_observed,
                "status": status,
                "rows": rows,
                "error": error,
            }
        )
        if i == 1 or i % 25 == 0 or i == total or status in {"error", "empty"}:
            print(f"[as1455-daily] {i}/{total} {symbol} status={status} rows={rows} error={error}", flush=True)
            pd.DataFrame(report_rows).to_csv(reports_dir / "as1455_daily_cache_build_report.csv", index=False, encoding="utf-8-sig")
        if i % 25 == 0:
            gc.collect()
    report = pd.DataFrame(report_rows)
    report.to_csv(reports_dir / "as1455_daily_cache_build_report.csv", index=False, encoding="utf-8-sig")
    return cache_files, report


def load_as1455_daily_panel(cache_files: dict[str, Path], reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    missing_rows = []
    convention_rows = []
    for i, (symbol, path) in enumerate(sorted(cache_files.items()), 1):
        try:
            timestamp_meta_path = path.with_name(path.name.replace(".csv", ".timestamp.json"))
            bar_times_observed = ""
            if timestamp_meta_path.exists():
                bar_times_observed = str(json.loads(timestamp_meta_path.read_text(encoding="utf-8")).get("bar_times_observed", ""))
            daily = pd.read_csv(path, dtype={"symbol": str, "last_bar_time": str})
            daily["symbol"] = daily["symbol"].map(normalize_symbol)
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
            daily["max_datetime_used"] = pd.to_datetime(daily["max_datetime_used"], errors="coerce")
            daily["has_14_55_bar"] = daily["has_14_55_bar"].astype(str).str.lower().eq("true")
            daily["used_after_cutoff"] = daily["used_after_cutoff"].astype(str).str.lower().eq("true")
            daily.dropna(subset=["date", "max_datetime_used"], inplace=True)
            for col in [
                "raw_open_as1455",
                "raw_high_as1455",
                "raw_low_as1455",
                "raw_close_as1455",
                "raw_volume_as1455",
                "raw_amount_as1455",
                "raw_daily_close",
            ]:
                daily[col] = pd.to_numeric(daily[col], errors="coerce")
            absent = daily.loc[~daily["has_14_55_bar"], ["symbol", "date", "last_bar_time"]]
            if not absent.empty:
                absent = absent.assign(reason="missing_14_55_bar_used_last_before_cutoff")
                missing_rows.append(absent)
            if len(convention_rows) < 5000:
                sample = daily.head(5000 - len(convention_rows))
                convention_rows.extend(
                    {
                        "date": row.date.strftime("%Y-%m-%d"),
                        "symbol": row.symbol,
                        "last_bar_before_cutoff": row.last_bar_time,
                        "bar_times_observed": bar_times_observed,
                        "has_14_55_bar": bool(row.has_14_55_bar),
                        "suspected_timestamp_convention": "right_endpoint" if row.has_14_55_bar else "unknown",
                    }
                    for row in sample.itertuples(index=False)
                )
            frames.append(daily)
        except Exception as exc:
            missing_rows.append(pd.DataFrame([{"symbol": symbol, "date": "", "last_bar_time": "", "reason": f"read_error:{type(exc).__name__}:{exc}"}]))
        if i % 100 == 0 or i == len(cache_files):
            print(f"[as1455-panel] {i}/{len(cache_files)} loaded", flush=True)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True, copy=False)
    frames.clear()
    raw.drop_duplicates(["symbol", "date"], keep="last", inplace=True)
    raw.set_index(["symbol", "date"], inplace=True)
    raw.sort_index(inplace=True)
    missing = pd.concat(missing_rows, ignore_index=True) if missing_rows else pd.DataFrame(columns=["symbol", "date", "last_bar_time", "reason"])
    pd.DataFrame(convention_rows).to_csv(reports_dir / "as1455_bar_timestamp_convention_check.csv", index=False, encoding="utf-8-sig")
    dist = raw.groupby("last_bar_time", observed=True).size().rename("count").reset_index()
    dist.to_csv(reports_dir / "as1455_last_bar_time_distribution.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(reports_dir / "as1455_missing_bar_report.csv", index=False, encoding="utf-8-sig")
    cutoff_report = raw.reset_index()[["symbol", "date", "max_datetime_used", "last_bar_time", "has_14_55_bar", "used_after_cutoff"]]
    cutoff_report.to_csv(reports_dir / "as1455_cutoff_leakage_check.csv", index=False, encoding="utf-8-sig")
    del cutoff_report
    return raw, missing


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
        factors = []
        for symbol in raw.index.get_level_values("symbol").unique():
            qfq = read_daily_close(qfq_daily_cache, symbol)
            if raw_daily_cache is not None:
                raw_daily = read_daily_close(raw_daily_cache, symbol)
                raw_source = "raw_daily_cache"
            elif "raw_daily_close" in raw.columns:
                raw_daily = raw.xs(symbol, level="symbol")["raw_daily_close"].dropna()
                raw_daily.index = pd.to_datetime(raw_daily.index).normalize()
                raw_source = "5min_full_day_close"
            else:
                raw_daily = pd.Series(dtype=float, name=symbol)
                raw_source = "missing"
            if qfq.empty or raw_daily.empty:
                rows.append({"symbol": symbol, "factor_missing": True, "qfq_obs": int(len(qfq)), "raw_obs": int(len(raw_daily)), "raw_source": raw_source})
                continue
            factor = qfq.div(raw_daily).replace([np.inf, -np.inf], np.nan).dropna().rename(symbol)
            factors.append(factor)
            rows.append(
                {
                    "symbol": symbol,
                    "factor_missing": False,
                    "qfq_obs": int(len(qfq)),
                    "raw_obs": int(len(raw_daily)),
                    "raw_source": raw_source,
                    "factor_obs": int(len(factor)),
                    "factor_min": float(factor.min()),
                    "factor_max": float(factor.max()),
                }
            )
        if factors:
            factor_panel = pd.concat(factors, axis=1).stack()
            factor_panel.index = factor_panel.index.set_names(["date", "symbol"]).swaplevel()
            adj_factor = factor_panel.reindex(raw.index)
        else:
            adj_factor = pd.Series(np.nan, index=raw.index, dtype=float)
        missing_factor = adj_factor.isna()
        if missing_factor.any():
            rows.append({"mode": "daily_qfq_div_raw", "missing_factor_rows": int(missing_factor.sum())})
            pd.DataFrame(rows).to_csv(reports_dir / "as1455_adjust_factor_check.csv", index=False, encoding="utf-8-sig")
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


def compute_features(prices: pd.DataFrame, universe_meta: pd.DataFrame, profile_memory: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    import talib
    from talib import ATR, BBANDS, MACD, RSI

    log_memory(profile_memory, "compute_features:start")
    if not prices.index.is_monotonic_increasing:
        prices.sort_index(inplace=True)
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
    gc.collect()
    log_memory(profile_memory, "compute_features:after_dollar_vol_rank")

    prices["rsi"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(RSI)

    def compute_bb(close: pd.Series) -> pd.DataFrame:
        upper, _mid, lower = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": upper, "bb_low": lower}, index=close.index)

    bb = prices.groupby(level="symbol", group_keys=False)["close"].apply(compute_bb)
    prices["bb_high"] = bb["bb_high"].sub(prices["close"]).div(bb["bb_high"]).apply(np.log1p)
    prices["bb_low"] = prices["close"].sub(bb["bb_low"]).div(prices["close"]).apply(np.log1p)
    del bb
    gc.collect()
    log_memory(profile_memory, "compute_features:after_bbands")

    def compute_natr(g: pd.DataFrame) -> pd.Series:
        return pd.Series(talib.NATR(g["high"], g["low"], g["close"]), index=g.index)

    natr_parts = [compute_natr(g) for _symbol, g in prices.groupby(level="symbol", sort=False)]
    prices["NATR"] = pd.concat(natr_parts).reindex(prices.index)
    del natr_parts

    def compute_atr(g: pd.DataFrame) -> pd.Series:
        return zscore(ATR(g["high"], g["low"], g["close"], timeperiod=14))

    atr_parts = [compute_atr(g) for _symbol, g in prices.groupby(level="symbol", sort=False)]
    prices["ATR"] = pd.concat(atr_parts).reindex(prices.index)
    del atr_parts
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
        gc.collect()
        log_memory(profile_memory, f"compute_features:after_r{t:02}dec")
    for t in T:
        prices[f"r{t:02}q_sector"] = qcut_by_group(prices[f"r{t:02}"], [dates, prices["sector"]], 5)
        gc.collect()
        log_memory(profile_memory, f"compute_features:after_r{t:02}q_sector")
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
    log_memory(profile_memory, "compute_features:end")
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
    p.add_argument("--as1455-daily-cache-dir", default=str(DEFAULT_AS1455_DAILY_CACHE))
    p.add_argument("--rebuild-as1455-daily-cache", action="store_true", help="Rebuild per-symbol as1455 daily CSV cache from 5min files")
    p.add_argument("--daily-cache-only", action="store_true", help="Build per-symbol as1455 daily CSV cache and exit before panel/features/HDF")
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
    p.add_argument("--fetch-missing-qfq-daily", action=argparse.BooleanOptionalAction, default=True, help="Automatically fetch missing BaoStock qfq daily files needed for adjustment factors")
    p.add_argument("--adjust-factor-mode", choices=["daily_qfq_div_raw", "identity"], default="daily_qfq_div_raw")
    p.add_argument("--timestamp-convention", choices=["right", "left"], default="right")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--min-obs", type=int, default=MIN_OBS)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--allow-partial-coverage", action="store_true", help="Write partial sample output when 5min files do not cover the full universe")
    p.add_argument("--profile-memory", action="store_true", help="Print process RSS checkpoints during the build")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_deps()
    out_dir = ensure_dir(Path(args.out_dir))
    reports_dir = ensure_dir(out_dir / "reports")
    model_data_path = out_dir / "model_data_as1455.h5"
    if str(args.baostock_5m_cache_dir) == str(DEFAULT_BAOSTOCK_5M_CACHE):
        args.baostock_5m_cache_dir = str(out_dir / "baostock_5m_cache")
    if str(args.as1455_daily_cache_dir) == str(DEFAULT_AS1455_DAILY_CACHE):
        args.as1455_daily_cache_dir = str(out_dir / "as1455_daily_cache")

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

    daily_cache_files, daily_cache_report = materialize_as1455_daily_cache(
        bar_files,
        Path(args.as1455_daily_cache_dir),
        reports_dir,
        args.start_date,
        args.end_date,
        args.timestamp_convention,
        bool(args.rebuild_as1455_daily_cache),
    )
    expected_daily_symbols = set(bar_files) if args.allow_partial_coverage else universe_symbols
    daily_cache_missing = sorted(expected_daily_symbols.difference(daily_cache_files))
    if daily_cache_missing:
        summary = {
            "status": "blocked",
            "reason": "per-symbol as1455 daily cache is incomplete; inspect as1455_daily_cache_build_report.csv",
            "expected_daily_symbols": len(expected_daily_symbols),
            "symbols_with_daily_cache": len(daily_cache_files),
            "symbols_missing_daily_cache": len(daily_cache_missing),
            "daily_cache_dir": str(Path(args.as1455_daily_cache_dir).resolve()),
        }
        (reports_dir / "as1455_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(json.dumps(summary, ensure_ascii=False))
    if args.daily_cache_only:
        summary = {
            "status": "daily_cache_complete",
            "expected_daily_symbols": len(expected_daily_symbols),
            "symbols_with_daily_cache": len(daily_cache_files),
            "daily_cache_rows": int(pd.to_numeric(daily_cache_report["rows"], errors="coerce").clip(lower=0).sum()),
            "daily_cache_dir": str(Path(args.as1455_daily_cache_dir).resolve()),
        }
        (reports_dir / "as1455_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    log_memory(bool(args.profile_memory), "main:before_daily_panel_load")
    raw, missing = load_as1455_daily_panel(daily_cache_files, reports_dir)
    if raw.empty:
        raise SystemExit("No as1455 raw OHLCV rows were constructed; see reports.")
    del daily_cache_report
    gc.collect()
    log_memory(bool(args.profile_memory), "main:after_daily_panel_load")
    used_after_cutoff_count = int(raw["used_after_cutoff"].sum())
    if used_after_cutoff_count:
        raise RuntimeError(f"cutoff leakage detected: used_after_cutoff_count={used_after_cutoff_count}")
    raw_ohlcv_rows = int(len(raw))
    max_dt = pd.to_datetime(raw["max_datetime_used"]).max()

    if args.adjust_factor_mode == "daily_qfq_div_raw" and args.fetch_missing_qfq_daily:
        raw_dates = raw.index.get_level_values("date")
        qfq_symbols = sorted(raw.index.get_level_values("symbol").unique().tolist())
        materialize_qfq_daily_cache(
            qfq_symbols,
            Path(args.qfq_daily_cache_dir),
            reports_dir,
            pd.Timestamp(raw_dates.min()).strftime("%Y-%m-%d"),
            pd.Timestamp(raw_dates.max()).strftime("%Y-%m-%d"),
            args.baostock_fetch_retries,
            args.baostock_fetch_sleep,
            args.baostock_query_timeout,
        )
        missing_qfq = [symbol for symbol in qfq_symbols if not (Path(args.qfq_daily_cache_dir) / f"{symbol}_qfq_daily.csv").exists()]
        if missing_qfq:
            raise SystemExit(
                f"qfq daily cache remains incomplete for {len(missing_qfq)} symbols; rerun the same command to resume. "
                f"See {reports_dir / 'as1455_qfq_daily_fetch_report.csv'}"
            )
        gc.collect()
        log_memory(bool(args.profile_memory), "main:after_qfq_daily_cache")

    raw_hdf = out_dir / "as1455_ohlcv_raw.h5"
    write_hdf(raw_hdf, "ohlcv", raw)
    log_memory(bool(args.profile_memory), "main:after_raw_hdf")

    adj, factor_check = build_adjusted_ohlcv(raw, reports_dir, Path(args.qfq_daily_cache_dir), Path(args.raw_daily_cache_dir) if args.raw_daily_cache_dir else None, args.adjust_factor_mode)
    adj_ohlcv_rows = int(len(adj))
    adj_hdf = out_dir / "as1455_ohlcv_adj.h5"
    write_hdf(adj_hdf, "ohlcv", adj)
    del raw
    gc.collect()
    log_memory(bool(args.profile_memory), "main:after_adj_hdf_and_raw_release")

    prices, exec_meta = make_prices_and_metadata(adj)
    exec_meta_hdf = out_dir / "as1455_execution_metadata.h5"
    write_hdf(exec_meta_hdf, "metadata", exec_meta)
    del exec_meta, adj, factor_check
    gc.collect()
    log_memory(bool(args.profile_memory), "main:after_execution_metadata_and_adj_release")

    prices, metadata, nobs = filter_prices_and_universe(prices, universe, args.min_obs, reports_dir)
    if prices.empty:
        raise SystemExit("No symbols satisfy min_obs and industry filters; see reports/as1455_nobs_by_symbol.csv")
    symbols_after_min_obs = int(prices.index.get_level_values("symbol").nunique())
    symbols_after_industry_filter = int(metadata["code"].nunique())
    compare_with_daily_leakage(prices, Path(args.qfq_daily_cache_dir), reports_dir)
    features_with_prices, outliers = compute_features(prices, metadata, profile_memory=bool(args.profile_memory))
    del metadata, nobs
    gc.collect()
    outliers.to_csv(reports_dir / "as1455_outlier_symbols_r01_gt_1.csv", index=False, encoding="utf-8-sig")
    label_validation = make_label_alignment_samples(features_with_prices, reports_dir)

    features_with_prices.drop(columns=["open", "high", "low", "close", "volume"], inplace=True)
    model_data = features_with_prices
    if list(model_data.columns) != EXPECTED_COLUMNS:
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
    model_rows_before_dropna = int(len(model_data))
    model_rows_after_dropna = int(clean_mask.sum())
    model_columns = int(model_data.shape[1])
    symbols_after_outlier_drop = int(model_data.index.get_level_values("symbol").nunique())
    write_model_data_hdf(model_data_path, model_data)
    del model_data, features_with_prices, clean_mask, prices
    gc.collect()
    log_memory(bool(args.profile_memory), "main:after_model_hdf_and_feature_release")
    smoke_ok = run_chapter17_read_smoke(model_data_path, reports_dir)

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
        raw_ohlcv_rows=raw_ohlcv_rows,
        adj_ohlcv_rows=adj_ohlcv_rows,
        symbols_after_min_obs=symbols_after_min_obs,
        symbols_after_industry_filter=symbols_after_industry_filter,
        symbols_after_outlier_drop=symbols_after_outlier_drop,
        model_rows_before_dropna=model_rows_before_dropna,
        model_rows_after_dropna=model_rows_after_dropna,
        model_columns=model_columns,
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
