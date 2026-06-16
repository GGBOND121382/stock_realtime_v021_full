#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an A-share Chapter 12 style model_data.h5 for Chapter 17.

This script intentionally mirrors the ML4T Chapter 12 -> Chapter 17 data shape:
  data = pd.read_hdf(model_data_path, "model_data").dropna().sort_index()
  outcomes = data.filter(like="fwd").columns.tolist()
  X = data.drop(outcomes, axis=1)
  y = data["r01_fwd"]

Only the Chapter 12 columns are written to /model_data. A-share metadata and
validation artifacts are written separately under reports/ and assets_ashare.h5.
"""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures as futures
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = PROJECT_DIR / "saved_data" / "ashare_static_universe" / "07_universe_allA_top1000_static.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce"
DEFAULT_CACHE_DIR = DEFAULT_OUT_DIR / "baostock_qfq_daily_cache"

MONTH = 21
YEAR = 12 * MONTH
MIN_OBS = 7 * YEAR
T = [1, 5, 10, 21, 42, 63]
FWD_T = [1, 5, 21]

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
    "board",
    "industry",
    "is_trade_universe",
    "is_mainboard",
    "is_chinext",
    "tradestatus",
    "isST",
    "entry_tradable_t1",
    "exit_tradable_t2",
    "entry_open_limit_up_t1",
    "exit_open_limit_down_t2",
    "r01_fwd_open",
    "open_to_open_return",
    "limit_up_count",
    "suspended_days",
}

_WORKER_BAOSTOCK = None


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
    output_dir: str
    model_data_path: str
    assets_path: str
    start_date: str
    end_date: str
    workers: int
    min_obs: int
    sample_mode: str
    industry_sample_size: int
    min_industry_sample_size: int
    source_cache_dir: Optional[str]
    source_cache_pattern: str
    source_cache_adjust: str
    universe_rows: int
    fetched_symbols: int
    symbols_after_min_obs: int
    symbols_after_outlier_drop: int
    symbols_after_dropna: int
    model_rows_before_dropna: int
    model_rows_after_dropna: int
    model_columns: int
    chapter17_outcomes: list[str]
    chapter17_X_rows: int
    chapter17_y_rows: int
    price_sources: dict[str, int]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits.zfill(6)


def baostock_code(code: str) -> str:
    code6 = normalize_code(code)
    return f"{'sh' if code6.startswith(('6', '9')) else 'sz'}.{code6}"


def sample_universe(
    universe: pd.DataFrame,
    max_symbols: Optional[int],
    sample_mode: str,
    industry_sample_size: int,
    min_industry_sample_size: int,
) -> pd.DataFrame:
    if not max_symbols:
        return universe.copy()
    limit = int(max_symbols)
    if str(sample_mode) == "head":
        return universe.head(limit).copy()
    if str(sample_mode) != "industry-balanced":
        raise ValueError(f"unknown sample_mode={sample_mode}")

    per_industry = max(1, int(industry_sample_size))
    min_per_industry = max(1, int(min_industry_sample_size))
    if per_industry < min_per_industry:
        raise ValueError("--industry-sample-size must be >= --min-industry-sample-size")

    src = universe.copy()
    src["industry"] = src["industry"].fillna("").astype(str).str.strip()
    src = src[src["industry"].ne("")]
    industry_counts = src.groupby("industry")["code"].transform("nunique")
    src = src[industry_counts >= min_per_industry].copy()

    selected = []
    seen_industries = []
    for industry in src["industry"]:
        if industry not in seen_industries:
            seen_industries.append(industry)
    for industry in seen_industries:
        remaining = limit - sum(len(x) for x in selected)
        if remaining < min_per_industry:
            break
        part = src[src["industry"].eq(industry)].head(min(per_industry, remaining))
        if len(part) >= min_per_industry:
            selected.append(part)
    if not selected:
        raise RuntimeError("industry-balanced sample selected no symbols; lower --min-industry-sample-size")
    return pd.concat(selected, ignore_index=True).head(limit).copy()


def result_to_df(rs: Any) -> pd.DataFrame:
    rows = []
    while getattr(rs, "error_code", "0") == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def init_baostock_worker() -> None:
    global _WORKER_BAOSTOCK
    import baostock as bs

    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock worker login failed: {lg.error_code} {lg.error_msg}")
    _WORKER_BAOSTOCK = bs


def fetch_qfq_daily(bs: Any, bs_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        bs_code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock qfq query failed for {bs_code}: {rs.error_code} {rs.error_msg}")
    return result_to_df(rs)


def merge_cache(cache_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if cache_path.exists():
        try:
            frames.append(pd.read_csv(cache_path, dtype={"code": str}))
        except Exception:
            pass
    if new_df is not None and not new_df.empty:
        frames.append(new_df)
    if not frames:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def load_cache_window(cache_path: Path, start_date: str, end_date: str, start_slack_days: int, end_slack_days: int) -> Optional[pd.DataFrame]:
    if not cache_path.exists():
        return None
    cached = pd.read_csv(cache_path, dtype={"code": str})
    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    have = cached[(cached["date"] >= pd.Timestamp(start_date)) & (cached["date"] <= pd.Timestamp(end_date))]
    if (
        not have.empty
        and have["date"].max() >= pd.Timestamp(end_date) - pd.Timedelta(days=end_slack_days)
    ):
        return have.copy()
    return None


def load_or_fetch_symbol(
    code: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    source_cache_dir: Optional[Path] = None,
    source_cache_pattern: str = "{code}_daily_raw.csv",
    fetch_missing_source_cache: bool = True,
) -> tuple[str, pd.DataFrame, str, str]:
    code6 = normalize_code(code)
    bs_code = baostock_code(code6)
    cache_path = cache_dir / f"{code6}_qfq_daily.csv"
    try:
        have = load_cache_window(cache_path, start_date, end_date, start_slack_days=21, end_slack_days=21)
        if have is not None:
            return code6, normalize_price_frame(code6, have), "", "primary_cache"

        if source_cache_dir is not None:
            source_name = source_cache_pattern.format(code=code6, bs_code=bs_code)
            source_path = source_cache_dir / source_name
            have = load_cache_window(source_path, start_date, end_date, start_slack_days=14, end_slack_days=14)
            if have is not None:
                return code6, normalize_price_frame(code6, have), "", "source_cache"
            if not fetch_missing_source_cache:
                return code6, pd.DataFrame(), f"source cache missing or incomplete: {source_path}", "missing_source_cache"

        if _WORKER_BAOSTOCK is None:
            init_baostock_worker()
        new_df = fetch_qfq_daily(_WORKER_BAOSTOCK, bs_code, start_date, end_date)
        merged = merge_cache(cache_path, new_df)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        have = merged[(merged["date"] >= pd.Timestamp(start_date)) & (merged["date"] <= pd.Timestamp(end_date))]
        return code6, normalize_price_frame(code6, have), "", "baostock_qfq"
    except Exception as exc:
        return code6, pd.DataFrame(), f"{type(exc).__name__}: {exc}", "error"


def normalize_price_frame(code6: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["symbol"] = code6
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out["volume"] > 0)]
    return out[["symbol", "date", "open", "high", "low", "close", "volume"]].sort_values(["symbol", "date"])


def fetch_prices(
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    workers: int,
    source_cache_dir: Optional[Path] = None,
    source_cache_pattern: str = "{code}_daily_raw.csv",
    fetch_missing_source_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(cache_dir)
    codes = universe["code"].astype(str).map(normalize_code).tolist()
    records = []
    errors = []
    sources = []
    if workers <= 1:
        for i, code in enumerate(codes, start=1):
            code6, df, err, source = load_or_fetch_symbol(
                code,
                start_date,
                end_date,
                cache_dir,
                source_cache_dir,
                source_cache_pattern,
                fetch_missing_source_cache,
            )
            sources.append({"code": code6, "source": source})
            if not df.empty:
                records.append(df)
            if err:
                errors.append({"code": code6, "source": source, "error": err})
            if i == 1 or i % 50 == 0 or i == len(codes):
                print(f"[prices] {i}/{len(codes)} {code6} source={source} rows={len(df)} error={err}", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futs = {
                executor.submit(
                    load_or_fetch_symbol,
                    code,
                    start_date,
                    end_date,
                    cache_dir,
                    source_cache_dir,
                    source_cache_pattern,
                    fetch_missing_source_cache,
                ): code
                for code in codes
            }
            for i, fut in enumerate(futures.as_completed(futs), start=1):
                code6, df, err, source = fut.result()
                sources.append({"code": code6, "source": source})
                if not df.empty:
                    records.append(df)
                if err:
                    errors.append({"code": code6, "source": source, "error": err})
                if i == 1 or i % 50 == 0 or i == len(codes):
                    print(f"[prices:{workers}w] {i}/{len(codes)} {code6} source={source} rows={len(df)} error={err}", flush=True)
    if source_cache_dir is not None and not fetch_missing_source_cache and errors:
        first = errors[0]
        raise RuntimeError(f"{len(errors)} source cache files missing or incomplete; first={first['code']} {first['error']}")
    if not records:
        raise RuntimeError("No BaoStock qfq daily rows fetched")
    prices = pd.concat(records, ignore_index=True)
    records.clear()
    prices.drop_duplicates(["symbol", "date"], keep="last", inplace=True)
    prices.sort_values(["symbol", "date"], inplace=True)
    prices.set_index(["symbol", "date"], inplace=True)
    errors_df = pd.DataFrame(errors, columns=["code", "source", "error"])
    sources_df = pd.DataFrame(sources, columns=["code", "source"])
    return prices, errors_df, sources_df


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
        raise SystemExit(
            "Missing required dependency for strict Chapter 12 reproduction: "
            + ", ".join(missing)
            + ". Install before running the builder."
        )


def qcut_codes(x: pd.Series, q: int) -> pd.Series:
    try:
        return pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=x.index)


def quantile_rank_codes(values: pd.Series, group_keys: Any, q: int) -> pd.Series:
    if isinstance(group_keys, (list, tuple)):
        groups = pd.MultiIndex.from_arrays(group_keys)
    else:
        groups = group_keys
    group_codes, _uniques = pd.factorize(groups, sort=False)
    ngroups = int(group_codes.max()) + 1 if len(group_codes) else 0
    if ngroups <= 0:
        return pd.Series(np.nan, index=values.index)

    valid = values.notna().to_numpy()
    counts = np.bincount(group_codes[valid], minlength=ngroups)
    nunique_by_group = values.groupby(group_codes, sort=False).nunique()
    nunique = np.zeros(ngroups, dtype=np.int64)
    nunique[nunique_by_group.index.to_numpy(dtype=np.int64)] = nunique_by_group.to_numpy(dtype=np.int64)
    rank = values.groupby(group_codes, sort=False).rank(method="average", na_option="keep").to_numpy()

    count_by_row = counts[group_codes]
    nunique_by_row = nunique[group_codes]
    result = np.full(len(values), np.nan, dtype=np.float64)
    mask = valid & (count_by_row > 1) & (nunique_by_row > 1)
    result[mask] = np.floor((rank[mask] - 1) * (q - 1) / (count_by_row[mask] - 1))
    return pd.Series(result, index=values.index)


def compute_features(prices: pd.DataFrame, metadata: pd.DataFrame, profile_memory: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    import talib
    from talib import ATR, BBANDS, MACD, RSI

    log_memory(profile_memory, "compute_features:start")
    if not prices.index.is_monotonic_increasing:
        prices.sort_index(inplace=True)
        log_memory(profile_memory, "compute_features:after_sort_index")
    metadata = metadata.copy()
    metadata["sector"] = pd.factorize(metadata["industry"])[0].astype(int)
    metadata = metadata.set_index("code", drop=False)

    prices["dollar_vol"] = prices["close"].mul(prices["volume"]).div(1e3)
    dollar_vol_ma = prices.dollar_vol.unstack("symbol").rolling(window=MONTH, min_periods=1).mean()
    prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().swaplevel()
    del dollar_vol_ma
    log_memory(profile_memory, "compute_features:after_dollar_vol_rank")

    prices["rsi"] = prices.groupby(level="symbol", group_keys=False).close.apply(RSI)
    log_memory(profile_memory, "compute_features:after_rsi")

    def compute_bb(close: pd.Series) -> pd.DataFrame:
        high, _mid, low = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": high, "bb_low": low}, index=close.index)

    bbands = prices.groupby(level="symbol", group_keys=False).close.apply(compute_bb)
    prices["bb_high"] = bbands["bb_high"]
    prices["bb_low"] = bbands["bb_low"]
    del bbands
    prices["bb_high"] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
    prices["bb_low"] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)
    log_memory(profile_memory, "compute_features:after_bbands")

    prices["NATR"] = prices.groupby(level="symbol", group_keys=False).apply(lambda x: talib.NATR(x.high, x.low, x.close))
    log_memory(profile_memory, "compute_features:after_natr")

    def compute_atr(stock_data: pd.DataFrame) -> pd.Series:
        s = ATR(stock_data.high, stock_data.low, stock_data.close, timeperiod=14)
        return s.sub(s.mean()).div(s.std())

    prices["ATR"] = prices.groupby("symbol", group_keys=False).apply(compute_atr)
    prices["PPO"] = prices.groupby(level="symbol", group_keys=False).close.apply(talib.PPO)
    log_memory(profile_memory, "compute_features:after_atr_ppo")

    def compute_macd(close: pd.Series) -> pd.Series:
        macd = MACD(close)[0]
        return (macd - np.mean(macd)) / np.std(macd)

    prices["MACD"] = prices.groupby("symbol", group_keys=False).close.apply(compute_macd)
    sector_map = metadata["sector"]
    prices["sector"] = prices.index.get_level_values("symbol").map(sector_map).astype(int)
    log_memory(profile_memory, "compute_features:after_macd_sector")

    by_sym = prices.groupby(level="symbol").close
    for t in T:
        prices[f"r{t:02}"] = by_sym.pct_change(t)
    log_memory(profile_memory, "compute_features:after_returns")

    dates = prices.index.get_level_values("date")
    for t in T:
        prices[f"r{t:02}dec"] = quantile_rank_codes(prices[f"r{t:02}"], dates, 10)
        log_memory(profile_memory, f"compute_features:after_r{t:02}dec")

    for t in T:
        prices[f"r{t:02}q_sector"] = quantile_rank_codes(prices[f"r{t:02}"], [dates, prices["sector"]], 5)
        log_memory(profile_memory, f"compute_features:after_r{t:02}q_sector")

    for t in FWD_T:
        prices[f"r{t:02}_fwd"] = prices.groupby(level="symbol")[f"r{t:02}"].shift(-t)
    log_memory(profile_memory, "compute_features:after_forward_returns")

    outliers = prices[prices.r01 > 1].index.get_level_values("symbol").unique()
    outlier_df = pd.DataFrame({"symbol": list(outliers)})
    if len(outliers):
        prices.drop(outliers, level="symbol", inplace=True)
        log_memory(profile_memory, "compute_features:after_outlier_drop")

    dates = prices.index.get_level_values("date")
    prices["year"] = dates.year
    prices["month"] = dates.month
    prices["weekday"] = dates.weekday
    log_memory(profile_memory, "compute_features:end")
    return prices, outlier_df


def make_label_alignment_samples(prices_with_close: pd.DataFrame, sample_n: int = 200) -> pd.DataFrame:
    rows = []
    for symbol, g in prices_with_close.groupby(level="symbol"):
        g = g.sort_index()
        dates = g.index.get_level_values("date")
        close = g["close"]
        for pos in range(0, max(0, len(g) - 21), max(1, len(g) // 20)):
            row: Dict[str, Any] = {"symbol": symbol, "date_t": dates[pos].strftime("%Y-%m-%d"), "close_t": float(close.iloc[pos])}
            for t in FWD_T:
                if pos + t >= len(g):
                    continue
                fwd_col = f"r{t:02}_fwd"
                ret_col = f"r{t:02}"
                row[f"date_t{t}"] = dates[pos + t].strftime("%Y-%m-%d")
                row[f"close_t{t}"] = float(close.iloc[pos + t])
                row[f"{ret_col}_t{t}"] = float(g[ret_col].iloc[pos + t])
                row[f"{fwd_col}_t"] = float(g[fwd_col].iloc[pos])
                row[f"manual_{fwd_col}"] = float(close.iloc[pos + t] / close.iloc[pos] - 1.0)
                row[f"diff_{fwd_col}"] = row[f"{fwd_col}_t"] - row[f"manual_{fwd_col}"]
            rows.append(row)
            if len(rows) >= sample_n:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def validate_forward_label_alignment(prices_with_close: pd.DataFrame, tolerance: float = 1e-10) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in FWD_T:
        fwd_col = f"r{t:02}_fwd"
        max_diff = float("nan")
        for _symbol, g in prices_with_close.groupby(level="symbol", sort=False):
            close = g["close"]
            manual = close.shift(-t).div(close).sub(1.0)
            diff = g[fwd_col].sub(manual).abs().dropna()
            if diff.empty:
                continue
            group_max = float(diff.max())
            if math.isnan(max_diff) or group_max > max_diff:
                max_diff = group_max
        out[f"max_abs_diff_{fwd_col}"] = max_diff
        if not math.isnan(max_diff) and max_diff >= tolerance:
            raise RuntimeError(f"{fwd_col} label alignment failed: max_abs_diff={max_diff} >= {tolerance}")
    return out


def compute_na_report_and_clean_mask(model_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean_mask = pd.Series(True, index=model_data.index)
    rows = []
    total = len(model_data)
    for column in model_data.columns:
        is_na = model_data[column].isna()
        na_count = int(is_na.sum())
        clean_mask &= ~is_na
        rows.append({"column": column, "na_count": na_count, "na_ratio": na_count / total})
    na_report = pd.DataFrame(rows).set_index("column")
    return na_report, clean_mask


def write_reports(
    reports_dir: Path,
    universe: pd.DataFrame,
    metadata: pd.DataFrame,
    nobs_by_symbol: pd.DataFrame,
    unique_symbols_with_prices: int,
    model_data: pd.DataFrame,
    na_report: pd.DataFrame,
    clean_mask: pd.Series,
    outliers: pd.DataFrame,
    fetch_errors: pd.DataFrame,
    price_sources: pd.DataFrame,
    label_samples: pd.DataFrame,
    label_validation: dict[str, float],
) -> None:
    ensure_dir(reports_dir)
    pd.DataFrame(
        {
            "column": EXPECTED_COLUMNS,
            "actual": list(model_data.columns),
            "matches_expected_position": [a == b for a, b in zip(EXPECTED_COLUMNS, model_data.columns)],
        }
    ).to_csv(reports_dir / "column_check.csv", index=False, encoding="utf-8-sig")
    na_report.to_csv(reports_dir / "na_report_before_dropna.csv", encoding="utf-8-sig")
    model_data.groupby(level="date").size().rename("sample_count").to_csv(
        reports_dir / "daily_sample_count_before_dropna.csv", encoding="utf-8-sig"
    )
    clean_mask.groupby(level="date").sum().astype(int).rename("sample_count").to_csv(
        reports_dir / "daily_sample_count_after_dropna.csv", encoding="utf-8-sig"
    )
    label_samples.to_csv(reports_dir / "label_alignment_samples.csv", index=False, encoding="utf-8-sig")
    (reports_dir / "label_alignment_validation.json").write_text(
        json.dumps(label_validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outliers.to_csv(reports_dir / "outlier_symbols_r01_gt_1.csv", index=False, encoding="utf-8-sig")
    fetch_errors.to_csv(reports_dir / "fetch_errors.csv", index=False, encoding="utf-8-sig")
    price_sources.to_csv(reports_dir / "price_sources.csv", index=False, encoding="utf-8-sig")
    universe["board"].value_counts().rename_axis("board").rename("count").to_csv(
        reports_dir / "board_distribution.csv", encoding="utf-8-sig"
    )
    nobs_by_symbol.to_csv(reports_dir / "nobs_by_symbol.csv", index=False, encoding="utf-8-sig")
    clean_index = model_data.index[clean_mask.to_numpy()]
    daily_after = clean_mask.groupby(level="date").sum()
    pool = {
        "unique_symbols_before": int(universe["code"].nunique()),
        "unique_symbols_with_prices": int(unique_symbols_with_prices),
        "unique_symbols_after_outlier_drop": int(model_data.index.get_level_values("symbol").nunique()),
        "unique_symbols_after_dropna": int(clean_index.get_level_values("symbol").nunique()),
        "daily_symbol_count_before_dropna_min": int(model_data.groupby(level="date").size().min()),
        "daily_symbol_count_after_dropna_min": int(daily_after[daily_after > 0].min()),
    }
    (reports_dir / "pool_validation.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata.to_csv(reports_dir / "metadata_used.csv", index=False, encoding="utf-8-sig")


def summarize_chapter17_data(model_data: pd.DataFrame, clean_mask: pd.Series) -> dict[str, Any]:
    outcomes = model_data.filter(like="fwd").columns.tolist()
    if outcomes != ["r01_fwd", "r05_fwd", "r21_fwd"]:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")
    feature_cols = [c for c in model_data.columns if c not in outcomes]
    if any("fwd" in c for c in feature_cols):
        raise RuntimeError("X contains fwd columns")
    clean_rows = int(clean_mask.sum())
    if clean_rows <= 0:
        raise RuntimeError("model_data.dropna() is empty")
    clean_index = model_data.index[clean_mask.to_numpy()]
    return {
        "outcomes": outcomes,
        "X_shape": [clean_rows, len(feature_cols)],
        "y_rows": clean_rows,
        "symbols": int(clean_index.get_level_values("symbol").nunique()),
        "date_start": clean_index.get_level_values("date").min().strftime("%Y-%m-%d"),
        "date_end": clean_index.get_level_values("date").max().strftime("%Y-%m-%d"),
    }


def validate_model_data_file(model_data_path: Path, expected_shape: tuple[int, int]) -> None:
    with pd.HDFStore(model_data_path, mode="r") as store:
        if "/model_data" not in store.keys():
            raise RuntimeError("model_data.h5 does not contain /model_data")
        storer = store.get_storer("model_data")
        storer_shape = storer.shape
        if np.isscalar(storer_shape):
            actual_shape = (int(storer_shape), int(expected_shape[1]))
        else:
            actual_shape = tuple(storer_shape)
    if actual_shape != tuple(expected_shape):
        raise RuntimeError(f"model_data.h5 shape mismatch: {actual_shape} != {tuple(expected_shape)}")


def write_model_data_hdf(model_data_path: Path, model_data: pd.DataFrame, chunk_rows: int = 250_000) -> None:
    with pd.HDFStore(model_data_path, mode="w") as store:
        for start in range(0, len(model_data), chunk_rows):
            stop = min(start + chunk_rows, len(model_data))
            store.append("model_data", model_data.iloc[start:stop], format="table", index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict ML4T Chapter 12 style A-share model_data.h5")
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--start-date", default="2010-01-01", help="Default is long enough for strict nobs > 1764")
    p.add_argument("--end-date", default=None, help="Default: universe asof_date max, or today")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-symbols", type=int, default=None, help="Smoke-test limit")
    p.add_argument(
        "--sample-mode",
        default="head",
        choices=["head", "industry-balanced"],
        help="How --max-symbols selects a smoke-test subset.",
    )
    p.add_argument(
        "--industry-sample-size",
        type=int,
        default=3,
        help="With --sample-mode industry-balanced, select up to this many symbols per industry.",
    )
    p.add_argument(
        "--min-industry-sample-size",
        type=int,
        default=2,
        help="With --sample-mode industry-balanced, skip industries with fewer selected symbols than this.",
    )
    p.add_argument("--min-obs", type=int, default=MIN_OBS)
    p.add_argument(
        "--source-cache-dir",
        default=None,
        help="Optional existing daily CSV cache to read before fetching BaoStock qfq data, e.g. ashare_static_universe/baostock_daily_cache",
    )
    p.add_argument(
        "--source-cache-pattern",
        default="{code}_daily_raw.csv",
        help="Filename pattern under --source-cache-dir. Available placeholders: {code}, {bs_code}",
    )
    p.add_argument(
        "--source-cache-adjust",
        default="raw",
        choices=["raw", "qfq"],
        help="Only metadata for reports; raw means BaoStock adjustflag=3 and is not strict qfq.",
    )
    p.add_argument(
        "--no-fetch-missing-source-cache",
        action="store_true",
        help="When --source-cache-dir is set, fail missing/incomplete source cache files instead of fetching BaoStock qfq fallback data.",
    )
    p.add_argument("--profile-memory", action="store_true", help="Print RSS checkpoints during the build.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_deps()
    out_dir = ensure_dir(Path(args.out_dir))
    reports_dir = ensure_dir(out_dir / "reports")
    model_data_path = out_dir / "model_data.h5"
    assets_path = out_dir / "assets_ashare.h5"

    universe = pd.read_csv(args.universe, dtype={"code": str})
    universe["code"] = universe["code"].map(normalize_code)
    universe = sample_universe(
        universe,
        args.max_symbols,
        str(args.sample_mode),
        int(args.industry_sample_size),
        int(args.min_industry_sample_size),
    )
    end_date = args.end_date
    if end_date is None:
        end_date = pd.to_datetime(universe.get("asof_date", pd.Series([pd.Timestamp.today()])).dropna().max()).strftime("%Y-%m-%d")
    start_date = pd.to_datetime(args.start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    if str(args.cache_dir) == str(DEFAULT_CACHE_DIR):
        args.cache_dir = str(out_dir / "baostock_qfq_daily_cache")

    metadata = universe[["code", "name", "board", "industry", "circ_mv"]].copy()
    metadata = metadata.rename(columns={"circ_mv": "marketcap"})
    metadata["industry"] = metadata["industry"].fillna("").astype(str).str.strip()
    metadata = metadata[metadata["industry"].ne("")]

    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    if source_cache_dir is not None and str(args.source_cache_adjust) != "qfq":
        print(
            f"[warning] using source cache with adjust={args.source_cache_adjust}; "
            "this avoids BaoStock downloads but is not strict qfq-adjusted pricing.",
            flush=True,
        )
    raw_prices, fetch_errors, price_sources = fetch_prices(
        metadata,
        start_date,
        end_date,
        Path(args.cache_dir),
        max(1, int(args.workers)),
        source_cache_dir=source_cache_dir,
        source_cache_pattern=str(args.source_cache_pattern),
        fetch_missing_source_cache=not bool(args.no_fetch_missing_source_cache),
    )
    log_memory(bool(args.profile_memory), "main:after_fetch_prices")
    nobs = raw_prices.groupby(level="symbol").size()
    nobs_by_symbol = nobs.rename("nobs").reset_index()
    unique_symbols_with_prices = int(raw_prices.index.get_level_values("symbol").nunique())
    keep_symbols = nobs[nobs > int(args.min_obs)].index
    drop_symbols = nobs.index.difference(keep_symbols)
    if len(drop_symbols):
        raw_prices.drop(drop_symbols, level="symbol", inplace=True)
    prices = raw_prices
    del raw_prices, drop_symbols, nobs
    log_memory(bool(args.profile_memory), "main:after_min_obs_filter")
    metadata = metadata[metadata["code"].isin(keep_symbols)].copy()
    if prices.empty:
        raise RuntimeError(f"No symbols passed nobs > {args.min_obs}; try an earlier --start-date")

    prices, outliers = compute_features(prices, metadata, profile_memory=bool(args.profile_memory))
    label_samples = make_label_alignment_samples(prices)
    label_validation = validate_forward_label_alignment(prices)
    log_memory(bool(args.profile_memory), "main:after_label_validation")
    prices.drop(columns=["open", "close", "low", "high", "volume"], inplace=True)
    model_data = prices
    log_memory(bool(args.profile_memory), "main:after_drop_ohlcv")
    if list(model_data.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("actual_columns != expected_columns")
    forbidden = sorted(FORBIDDEN_MODEL_COLUMNS.intersection(model_data.columns))
    if forbidden:
        raise RuntimeError(f"forbidden columns in model_data: {forbidden}")

    metadata_assets = metadata.copy()
    metadata_assets["sector"] = pd.factorize(metadata_assets["industry"])[0].astype(int)
    for path in (assets_path, model_data_path):
        if path.exists():
            path.unlink()
    metadata_assets.to_hdf(assets_path, "/ashare/metadata", mode="w")
    write_model_data_hdf(model_data_path, model_data)
    validate_model_data_file(model_data_path, tuple(model_data.shape))
    log_memory(bool(args.profile_memory), "main:after_hdf_write")

    na_report, clean_mask = compute_na_report_and_clean_mask(model_data)
    log_memory(bool(args.profile_memory), "main:after_clean_mask")

    write_reports(
        reports_dir,
        universe,
        metadata_assets,
        nobs_by_symbol,
        unique_symbols_with_prices,
        model_data,
        na_report,
        clean_mask,
        outliers,
        fetch_errors,
        price_sources,
        label_samples,
        label_validation,
    )
    chapter17 = summarize_chapter17_data(model_data, clean_mask)
    log_memory(bool(args.profile_memory), "main:after_reports")
    (reports_dir / "chapter17_read_smoke_test.json").write_text(json.dumps(chapter17, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = BuildSummary(
        universe_path=str(Path(args.universe).resolve()),
        output_dir=str(out_dir.resolve()),
        model_data_path=str(model_data_path.resolve()),
        assets_path=str(assets_path.resolve()),
        start_date=start_date,
        end_date=end_date,
        workers=int(args.workers),
        min_obs=int(args.min_obs),
        sample_mode=str(args.sample_mode),
        industry_sample_size=int(args.industry_sample_size),
        min_industry_sample_size=int(args.min_industry_sample_size),
        source_cache_dir=str(source_cache_dir.resolve()) if source_cache_dir is not None else None,
        source_cache_pattern=str(args.source_cache_pattern),
        source_cache_adjust=str(args.source_cache_adjust),
        universe_rows=int(len(universe)),
        fetched_symbols=int(unique_symbols_with_prices),
        symbols_after_min_obs=int(len(keep_symbols)),
        symbols_after_outlier_drop=int(model_data.index.get_level_values("symbol").nunique()),
        symbols_after_dropna=int(model_data.index[clean_mask.to_numpy()].get_level_values("symbol").nunique()),
        model_rows_before_dropna=int(len(model_data)),
        model_rows_after_dropna=int(clean_mask.sum()),
        model_columns=int(model_data.shape[1]),
        chapter17_outcomes=chapter17["outcomes"],
        chapter17_X_rows=int(chapter17["X_shape"][0]),
        chapter17_y_rows=int(chapter17["y_rows"]),
        price_sources={str(k): int(v) for k, v in price_sources["source"].value_counts().sort_index().items()},
    )
    (out_dir / "build_summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
