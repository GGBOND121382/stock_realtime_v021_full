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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits.zfill(6)


def baostock_code(code: str) -> str:
    code6 = normalize_code(code)
    return f"{'sh' if code6.startswith(('6', '9')) else 'sz'}.{code6}"


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


def load_or_fetch_symbol(code: str, start_date: str, end_date: str, cache_dir: Path) -> tuple[str, pd.DataFrame, str]:
    if _WORKER_BAOSTOCK is None:
        init_baostock_worker()
    code6 = normalize_code(code)
    bs_code = baostock_code(code6)
    cache_path = cache_dir / f"{code6}_qfq_daily.csv"
    try:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, dtype={"code": str})
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            have = cached[(cached["date"] >= pd.Timestamp(start_date)) & (cached["date"] <= pd.Timestamp(end_date))]
            if (
                not have.empty
                and have["date"].min() <= pd.Timestamp(start_date) + pd.Timedelta(days=21)
                and have["date"].max() >= pd.Timestamp(end_date) - pd.Timedelta(days=21)
            ):
                return code6, normalize_price_frame(code6, have), ""
        new_df = fetch_qfq_daily(_WORKER_BAOSTOCK, bs_code, start_date, end_date)
        merged = merge_cache(cache_path, new_df)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        have = merged[(merged["date"] >= pd.Timestamp(start_date)) & (merged["date"] <= pd.Timestamp(end_date))]
        return code6, normalize_price_frame(code6, have), ""
    except Exception as exc:
        return code6, pd.DataFrame(), f"{type(exc).__name__}: {exc}"


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


def fetch_prices(universe: pd.DataFrame, start_date: str, end_date: str, cache_dir: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(cache_dir)
    codes = universe["code"].astype(str).map(normalize_code).tolist()
    records = []
    errors = []
    if workers <= 1:
        init_baostock_worker()
        for i, code in enumerate(codes, start=1):
            code6, df, err = load_or_fetch_symbol(code, start_date, end_date, cache_dir)
            if not df.empty:
                records.append(df)
            if err:
                errors.append({"code": code6, "error": err})
            if i == 1 or i % 50 == 0 or i == len(codes):
                print(f"[qfq] {i}/{len(codes)} {code6} rows={len(df)} error={err}", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=workers, initializer=init_baostock_worker) as executor:
            futs = {executor.submit(load_or_fetch_symbol, code, start_date, end_date, cache_dir): code for code in codes}
            for i, fut in enumerate(futures.as_completed(futs), start=1):
                code6, df, err = fut.result()
                if not df.empty:
                    records.append(df)
                if err:
                    errors.append({"code": code6, "error": err})
                if i == 1 or i % 50 == 0 or i == len(codes):
                    print(f"[qfq:{workers}w] {i}/{len(codes)} {code6} rows={len(df)} error={err}", flush=True)
    if not records:
        raise RuntimeError("No BaoStock qfq daily rows fetched")
    prices = pd.concat(records, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    prices = prices.sort_values(["symbol", "date"]).set_index(["symbol", "date"])
    errors_df = pd.DataFrame(errors, columns=["code", "error"])
    return prices, errors_df


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


def compute_features(prices: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import talib
    from talib import ATR, BBANDS, MACD, RSI

    prices = prices.sort_index().copy()
    metadata = metadata.copy()
    metadata["sector"] = pd.factorize(metadata["industry"])[0].astype(int)
    metadata = metadata.set_index("code", drop=False)

    prices["dollar_vol"] = prices[["close", "volume"]].prod(axis=1).div(1e3)
    dollar_vol_ma = prices.dollar_vol.unstack("symbol").rolling(window=MONTH, min_periods=1).mean()
    prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().swaplevel()

    prices["rsi"] = prices.groupby(level="symbol", group_keys=False).close.apply(RSI)

    def compute_bb(close: pd.Series) -> pd.DataFrame:
        high, _mid, low = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": high, "bb_low": low}, index=close.index)

    prices = prices.join(prices.groupby(level="symbol", group_keys=False).close.apply(compute_bb))
    prices["bb_high"] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
    prices["bb_low"] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)

    prices["NATR"] = prices.groupby(level="symbol", group_keys=False).apply(lambda x: talib.NATR(x.high, x.low, x.close))

    def compute_atr(stock_data: pd.DataFrame) -> pd.Series:
        s = ATR(stock_data.high, stock_data.low, stock_data.close, timeperiod=14)
        return s.sub(s.mean()).div(s.std())

    prices["ATR"] = prices.groupby("symbol", group_keys=False).apply(compute_atr)
    prices["PPO"] = prices.groupby(level="symbol", group_keys=False).close.apply(talib.PPO)

    def compute_macd(close: pd.Series) -> pd.Series:
        macd = MACD(close)[0]
        return (macd - np.mean(macd)) / np.std(macd)

    prices["MACD"] = prices.groupby("symbol", group_keys=False).close.apply(compute_macd)
    prices = prices.join(metadata[["sector"]], on="symbol")

    by_sym = prices.groupby(level="symbol").close
    for t in T:
        prices[f"r{t:02}"] = by_sym.pct_change(t)

    for t in T:
        prices[f"r{t:02}dec"] = prices[f"r{t:02}"].groupby(level="date", group_keys=False).apply(lambda x: qcut_codes(x, 10))

    for t in T:
        prices[f"r{t:02}q_sector"] = prices.groupby(["date", "sector"])[f"r{t:02}"].transform(lambda x: qcut_codes(x, 5))

    for t in FWD_T:
        prices[f"r{t:02}_fwd"] = prices.groupby(level="symbol")[f"r{t:02}"].shift(-t)

    outliers = prices[prices.r01 > 1].index.get_level_values("symbol").unique()
    outlier_df = pd.DataFrame({"symbol": list(outliers)})
    if len(outliers):
        prices = prices.drop(outliers, level="symbol")

    dates = prices.index.get_level_values("date")
    prices["year"] = dates.year
    prices["month"] = dates.month
    prices["weekday"] = dates.weekday
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
        manual = prices_with_close.groupby(level="symbol")["close"].shift(-t).div(prices_with_close["close"]).sub(1.0)
        diff = prices_with_close[fwd_col].sub(manual).abs().dropna()
        max_diff = float(diff.max()) if not diff.empty else float("nan")
        out[f"max_abs_diff_{fwd_col}"] = max_diff
        if not diff.empty and max_diff >= tolerance:
            raise RuntimeError(f"{fwd_col} label alignment failed: max_abs_diff={max_diff} >= {tolerance}")
    return out


def write_reports(
    reports_dir: Path,
    universe: pd.DataFrame,
    metadata: pd.DataFrame,
    raw_prices: pd.DataFrame,
    prices: pd.DataFrame,
    model_data: pd.DataFrame,
    outliers: pd.DataFrame,
    fetch_errors: pd.DataFrame,
) -> None:
    ensure_dir(reports_dir)
    pd.DataFrame(
        {
            "column": EXPECTED_COLUMNS,
            "actual": list(model_data.columns),
            "matches_expected_position": [a == b for a, b in zip(EXPECTED_COLUMNS, model_data.columns)],
        }
    ).to_csv(reports_dir / "column_check.csv", index=False, encoding="utf-8-sig")
    model_data.isna().sum().rename("na_count").to_frame().assign(na_ratio=lambda x: x.na_count / len(model_data)).to_csv(
        reports_dir / "na_report_before_dropna.csv", encoding="utf-8-sig"
    )
    model_data.groupby(level="date").size().rename("sample_count").to_csv(
        reports_dir / "daily_sample_count_before_dropna.csv", encoding="utf-8-sig"
    )
    model_data.dropna().groupby(level="date").size().rename("sample_count").to_csv(
        reports_dir / "daily_sample_count_after_dropna.csv", encoding="utf-8-sig"
    )
    make_label_alignment_samples(prices).to_csv(reports_dir / "label_alignment_samples.csv", index=False, encoding="utf-8-sig")
    label_validation = validate_forward_label_alignment(prices)
    (reports_dir / "label_alignment_validation.json").write_text(
        json.dumps(label_validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outliers.to_csv(reports_dir / "outlier_symbols_r01_gt_1.csv", index=False, encoding="utf-8-sig")
    fetch_errors.to_csv(reports_dir / "fetch_errors.csv", index=False, encoding="utf-8-sig")
    universe["board"].value_counts().rename_axis("board").rename("count").to_csv(
        reports_dir / "board_distribution.csv", encoding="utf-8-sig"
    )
    nobs = raw_prices.groupby(level="symbol").size().rename("nobs").reset_index()
    nobs.to_csv(reports_dir / "nobs_by_symbol.csv", index=False, encoding="utf-8-sig")
    pool = {
        "unique_symbols_before": int(universe["code"].nunique()),
        "unique_symbols_with_prices": int(raw_prices.index.get_level_values("symbol").nunique()),
        "unique_symbols_after_outlier_drop": int(prices.index.get_level_values("symbol").nunique()),
        "unique_symbols_after_dropna": int(model_data.dropna().index.get_level_values("symbol").nunique()),
        "daily_symbol_count_before_dropna_min": int(model_data.groupby(level="date").size().min()),
        "daily_symbol_count_after_dropna_min": int(model_data.dropna().groupby(level="date").size().min()),
    }
    (reports_dir / "pool_validation.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata.to_csv(reports_dir / "metadata_used.csv", index=False, encoding="utf-8-sig")


def validate_model_data(model_data_path: Path) -> dict[str, Any]:
    data = pd.read_hdf(model_data_path, "model_data").dropna().sort_index()
    outcomes = data.filter(like="fwd").columns.tolist()
    X = data.drop(outcomes, axis=1)
    y = data["r01_fwd"]
    if outcomes != ["r01_fwd", "r05_fwd", "r21_fwd"]:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")
    if any("fwd" in c for c in X.columns):
        raise RuntimeError("X contains fwd columns")
    if y.isna().any():
        raise RuntimeError("y contains NA")
    if X.shape[0] != y.shape[0]:
        raise RuntimeError("X/y row mismatch")
    return {
        "outcomes": outcomes,
        "X_shape": list(X.shape),
        "y_rows": int(y.shape[0]),
        "symbols": int(data.index.get_level_values("symbol").nunique()),
        "date_start": data.index.get_level_values("date").min().strftime("%Y-%m-%d"),
        "date_end": data.index.get_level_values("date").max().strftime("%Y-%m-%d"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict ML4T Chapter 12 style A-share model_data.h5")
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--start-date", default="2010-01-01", help="Default is long enough for strict nobs > 1764")
    p.add_argument("--end-date", default=None, help="Default: universe asof_date max, or today")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-symbols", type=int, default=None, help="Smoke-test limit")
    p.add_argument("--min-obs", type=int, default=MIN_OBS)
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
    if args.max_symbols:
        universe = universe.head(args.max_symbols).copy()
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

    raw_prices, fetch_errors = fetch_prices(metadata, start_date, end_date, Path(args.cache_dir), max(1, int(args.workers)))
    nobs = raw_prices.groupby(level="symbol").size()
    keep_symbols = nobs[nobs > int(args.min_obs)].index
    prices = raw_prices.loc[raw_prices.index.get_level_values("symbol").isin(keep_symbols)].copy()
    metadata = metadata[metadata["code"].isin(keep_symbols)].copy()
    if prices.empty:
        raise RuntimeError(f"No symbols passed nobs > {args.min_obs}; try an earlier --start-date")

    prices, outliers = compute_features(prices, metadata)
    model_data = prices.drop(["open", "close", "low", "high", "volume"], axis=1)
    model_data = model_data[EXPECTED_COLUMNS]
    if list(model_data.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("actual_columns != expected_columns")
    forbidden = sorted(FORBIDDEN_MODEL_COLUMNS.intersection(model_data.columns))
    if forbidden:
        raise RuntimeError(f"forbidden columns in model_data: {forbidden}")

    metadata_assets = metadata.copy()
    metadata_assets["sector"] = pd.factorize(metadata_assets["industry"])[0].astype(int)
    metadata_assets.to_hdf(assets_path, "/ashare/metadata", mode="w")
    model_data.to_hdf(model_data_path, "model_data", mode="w")

    write_reports(reports_dir, universe, metadata_assets, raw_prices, prices, model_data, outliers, fetch_errors)
    chapter17 = validate_model_data(model_data_path)
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
        universe_rows=int(len(universe)),
        fetched_symbols=int(raw_prices.index.get_level_values("symbol").nunique()),
        symbols_after_min_obs=int(len(keep_symbols)),
        symbols_after_outlier_drop=int(prices.index.get_level_values("symbol").nunique()),
        symbols_after_dropna=int(model_data.dropna().index.get_level_values("symbol").nunique()),
        model_rows_before_dropna=int(len(model_data)),
        model_rows_after_dropna=int(len(model_data.dropna())),
        model_columns=int(model_data.shape[1]),
        chapter17_outcomes=chapter17["outcomes"],
        chapter17_X_rows=int(chapter17["X_shape"][0]),
        chapter17_y_rows=int(chapter17["y_rows"]),
    )
    (out_dir / "build_summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
