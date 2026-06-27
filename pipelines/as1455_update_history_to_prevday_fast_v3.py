#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append-only fast incremental AS1455 history updater.

Drop-in replacement for pipelines/as1455_update_history_to_prevday.py.

Compared with fast_v2, this version fixes the remaining expensive path:
normal first-run daily updates append newly downloaded 5m rows to the existing
5m CSV instead of reading/merging/rewriting the full multi-year 5m file.

Design goals:
- Never rewrite raw 5m cache when it is already current.
- For a normal non-overlapping incremental update, append-only for raw 5m,
  raw daily, and AS1455 daily caches.
- Fall back to full merge only when explicitly requested or when overlap is
  detected.
- Keep the existing cache file formats and report file names.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    aggregate_as1455_from_5m,
    as1455_daily_path,
    ensure_dir,
    get_last_cached_date,
    load_universe,
    merge_dedup_csv,
    normalize_symbol,
    parse_trade_date,
    raw_5m_path,
    raw_daily_path,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)
from pipelines.as1455_update_history_to_prevday import (  # noqa: E402
    DEFAULT_AS1455_DAILY_CACHE,
    DEFAULT_LIVE_ROOT,
    DEFAULT_RAW_5M_CACHE,
    DEFAULT_RAW_DAILY_CACHE,
    OLD_5M_COLUMNS,
    fetch_raw_5m,
    fetch_raw_daily,
    import_baostock,
    merge_old_5m_cache,
    next_query_start,
    read_old_5m_cache,
    resolve_history_end_date,
    standardize_5m_to_old_schema,
    write_old_5m_cache,
)


def _date_str(ts: Optional[pd.Timestamp]) -> str:
    return "" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def _parse_date(value: object) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", "null"}:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if "symbol" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df.drop_duplicates("symbol", keep="last")


def manifest_get(manifest: pd.DataFrame, symbol: str, column: str, path: Path) -> Optional[pd.Timestamp]:
    if manifest.empty or not path.exists() or path.stat().st_size == 0:
        return None
    rows = manifest[manifest["symbol"].eq(normalize_symbol(symbol))]
    if rows.empty or column not in rows.columns:
        return None
    return _parse_date(rows.iloc[-1].get(column))


def get_last_date_fast(
    path: Path,
    date_col: str,
    symbol: str,
    manifest: pd.DataFrame,
    manifest_col: str,
    manifest_mode: str,
) -> Optional[pd.Timestamp]:
    """Resolve last cached date.

    manifest_mode:
    - trust: use manifest when available, otherwise scan date columns.
    - scan: ignore manifest and scan date columns.
    - auto: use manifest only when the file exists and the manifest has a value;
      otherwise scan. This is the default.
    """
    if manifest_mode in {"trust", "auto"}:
        m = manifest_get(manifest, symbol, manifest_col, path)
        if m is not None:
            return m
        if manifest_mode == "trust":
            return None
    return get_last_cached_date(path, date_col=date_col)


def max_trade_date_from_5m(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    if "trade_date" in df.columns:
        s = df["trade_date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
        dates = pd.to_datetime(s, format="%Y%m%d", errors="coerce").dropna()
    elif "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    else:
        return None
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def max_date_from_column(df: pd.DataFrame, col: str = "date") -> Optional[pd.Timestamp]:
    if df is None or df.empty or col not in df.columns:
        return None
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        if last not in {b"\n", b"\r"}:
            f.write(b"\n")


def align_to_existing_header(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return df.copy()
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        cols = list(header.columns)
    except Exception:
        return df.copy()
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    # Keep existing column order. Drop extra columns to avoid corrupt CSV shape.
    return out[cols]


def append_csv(path: Path, df: pd.DataFrame, encoding: str = "utf-8-sig") -> None:
    ensure_dir(path.parent)
    if df is None or df.empty:
        return
    if path.exists() and path.stat().st_size > 0:
        out = align_to_existing_header(path, df)
        ensure_trailing_newline(path)
        out.to_csv(path, index=False, header=False, mode="a", encoding=encoding)
    else:
        df.to_csv(path, index=False, encoding=encoding)


def append_5m_cache(path: Path, new5: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Append standardized 5m rows and return the standardized appended rows."""
    std = standardize_5m_to_old_schema(new5, symbol=symbol)
    if std.empty:
        return std
    std = std[OLD_5M_COLUMNS].copy()
    append_csv(path, std)
    return std


def append_daily_cache(path: Path, newd: pd.DataFrame) -> pd.DataFrame:
    if newd is None or newd.empty:
        return pd.DataFrame()
    out = newd.copy()
    append_csv(path, out)
    return out


def append_as1455_cache(path: Path, daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    out = daily.copy()
    append_csv(path, out)
    return out


def read_5m_range(path: Path, symbol: str, start_date: str, end_date: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Read only the requested date range from a 5m CSV using chunks.

    Used only when AS1455 is stale but no new 5m rows were fetched for the stale
    interval. It avoids materializing the entire multi-year cache.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    parts: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            path,
            dtype={"symbol": str, "trade_date": str, "code": str},
            encoding="utf-8-sig",
            low_memory=False,
            chunksize=chunksize,
        )
    except Exception:
        return pd.DataFrame()
    for chunk in reader:
        try:
            std = standardize_5m_to_old_schema(chunk, symbol=symbol)
        except Exception:
            continue
        if std.empty:
            continue
        dates = pd.to_datetime(std["trade_date"], format="%Y%m%d", errors="coerce")
        keep = dates.ge(start_ts) & dates.le(end_ts)
        if keep.any():
            parts.append(std.loc[keep].copy())
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    return standardize_5m_to_old_schema(out, symbol=symbol)


def update_manifest(path: Path, manifest: pd.DataFrame, rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if not new.empty and "symbol" in new.columns:
        new["symbol"] = new["symbol"].map(normalize_symbol)
    if manifest.empty:
        out = new
    else:
        out = pd.concat([manifest, new], ignore_index=True, sort=False)
    out = out.drop_duplicates("symbol", keep="last").sort_values("symbol")
    ensure_dir(path.parent)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def is_strict_increment(last: Optional[pd.Timestamp], start: str) -> bool:
    if last is None:
        return False
    return pd.Timestamp(start).normalize() == (pd.Timestamp(last).normalize() + pd.Timedelta(days=1))


def update_one_symbol_fast(symbol: str, args, history_end_dash: str, manifest: pd.DataFrame, bs=None) -> tuple[dict, dict]:
    symbol = normalize_symbol(symbol)
    raw5m_p = raw_5m_path(Path(args.raw_5m_cache_dir), symbol)
    rawdaily_p = raw_daily_path(Path(args.raw_daily_cache_dir), symbol)
    as1455_p = as1455_daily_path(Path(args.as1455_daily_cache_dir), symbol)

    default_start = args.history_start_date
    history_end_ts = pd.Timestamp(history_end_dash).normalize()

    last5 = get_last_date_fast(raw5m_p, "trade_date", symbol, manifest, "raw_5m_last_date", args.manifest_mode)
    lastd = get_last_date_fast(rawdaily_p, "date", symbol, manifest, "raw_daily_last_date", args.manifest_mode)
    lasta = get_last_date_fast(as1455_p, "date", symbol, manifest, "as1455_last_date", args.manifest_mode)

    start5 = next_query_start(last5, default_start)
    startd = next_query_start(lastd, default_start)
    starta = next_query_start(lasta, default_start)

    row = {
        "symbol": symbol,
        "history_end_date": history_end_dash,
        "raw_5m_path": str(raw5m_p),
        "raw_daily_path": str(rawdaily_p),
        "as1455_daily_path": str(as1455_p),
        "raw_5m_status": "skip",
        "raw_daily_status": "skip",
        "as1455_status": "skip",
        "raw_5m_new_rows": 0,
        "raw_daily_new_rows": 0,
        "as1455_new_rows": 0,
        "raw_5m_append_only": False,
        "raw_daily_append_only": False,
        "as1455_append_only": False,
        "raw_5m_last_cached_date": _date_str(last5),
        "raw_daily_last_cached_date": _date_str(lastd),
        "as1455_last_cached_date": _date_str(lasta),
        "raw_5m_query_start": start5,
        "raw_daily_query_start": startd,
        "as1455_query_start": starta,
        "used_manifest": bool(args.manifest_mode in {"trust", "auto"} and not manifest.empty),
        "error": "",
    }

    std_new5 = pd.DataFrame(columns=OLD_5M_COLUMNS)

    # 1) raw 5m cache: fetch only missing range; append-only for normal increment.
    if pd.Timestamp(start5) <= history_end_ts:
        if args.dry_run:
            row["raw_5m_status"] = "would_fetch"
        else:
            if bs is None:
                raise RuntimeError("BaoStock session is not initialized")
            new5 = fetch_raw_5m(bs, symbol, start5, history_end_dash)
            std_new5 = standardize_5m_to_old_schema(new5, symbol=symbol)
            row["raw_5m_new_rows"] = int(len(std_new5))
            if std_new5.empty:
                row["raw_5m_status"] = "empty"
            else:
                can_append = (not args.force_full_5m_merge) and (not raw5m_p.exists() or raw5m_p.stat().st_size == 0 or is_strict_increment(last5, start5))
                if can_append:
                    append_5m_cache(raw5m_p, std_new5, symbol=symbol)
                    row["raw_5m_status"] = "updated_append"
                    row["raw_5m_append_only"] = True
                else:
                    # Slow fallback for one-time repair / overlap cases only.
                    existing = read_old_5m_cache(raw5m_p, symbol)
                    merged5 = merge_old_5m_cache(existing, std_new5, symbol)
                    write_old_5m_cache(raw5m_p, merged5)
                    row["raw_5m_status"] = "updated_full_merge"
                new_last5 = max_trade_date_from_5m(std_new5)
                if new_last5 is not None:
                    last5 = max(last5, new_last5) if last5 is not None else new_last5
    else:
        row["raw_5m_status"] = "cached"

    # 2) raw daily cache: fetch missing range; append-only for normal increment.
    if pd.Timestamp(startd) <= history_end_ts:
        if args.dry_run:
            row["raw_daily_status"] = "would_fetch"
        else:
            if bs is None:
                raise RuntimeError("BaoStock session is not initialized")
            newd = fetch_raw_daily(bs, symbol, startd, history_end_dash)
            row["raw_daily_new_rows"] = int(len(newd))
            if newd.empty:
                row["raw_daily_status"] = "empty"
            else:
                can_append = (not args.force_full_daily_merge) and (not rawdaily_p.exists() or rawdaily_p.stat().st_size == 0 or is_strict_increment(lastd, startd))
                if can_append:
                    append_daily_cache(rawdaily_p, newd)
                    row["raw_daily_status"] = "updated_append"
                    row["raw_daily_append_only"] = True
                else:
                    merged = merge_dedup_csv(rawdaily_p, newd, subset=["symbol", "date"])
                    write_csv(rawdaily_p, merged)
                    row["raw_daily_status"] = "updated_full_merge"
                new_lastd = max_date_from_column(newd, "date")
                if new_lastd is not None:
                    lastd = max(lastd, new_lastd) if lastd is not None else new_lastd
    else:
        row["raw_daily_status"] = "cached"

    # 3) AS1455 daily cache: skip if current; aggregate only missing range.
    if args.skip_as1455_aggregate:
        row["as1455_status"] = "skipped_by_arg"
    elif lasta is not None and lasta >= history_end_ts:
        row["as1455_status"] = "cached"
    elif args.dry_run:
        row["as1455_status"] = "would_aggregate"
    else:
        agg_start = starta
        bars = pd.DataFrame(columns=OLD_5M_COLUMNS)
        if not std_new5.empty:
            min_new_date = pd.to_datetime(std_new5["trade_date"], format="%Y%m%d", errors="coerce").min()
            if pd.notna(min_new_date) and pd.Timestamp(agg_start) >= pd.Timestamp(min_new_date).normalize():
                bars = std_new5.copy()
            else:
                bars = read_5m_range(raw5m_p, symbol, agg_start, history_end_dash)
        else:
            bars = read_5m_range(raw5m_p, symbol, agg_start, history_end_dash)

        if bars.empty:
            row["as1455_status"] = "no_bars_for_missing_range"
        else:
            daily = aggregate_as1455_from_5m(bars, symbol=symbol, start_date=agg_start, end_date=history_end_dash)
            if daily.empty:
                row["as1455_status"] = "empty"
            else:
                before = len(daily)
                daily = daily[daily["has_14_55_bar"].astype(bool)].copy()
                row["as1455_missing_1455_rows"] = int(before - len(daily))
                row["as1455_new_rows"] = int(len(daily))
                if daily.empty:
                    row["as1455_status"] = "empty_after_1455_filter"
                else:
                    can_append = (not args.force_full_as1455_merge) and (not as1455_p.exists() or as1455_p.stat().st_size == 0 or is_strict_increment(lasta, agg_start))
                    if can_append:
                        append_as1455_cache(as1455_p, daily)
                        row["as1455_status"] = "updated_append"
                        row["as1455_append_only"] = True
                    else:
                        merged = merge_dedup_csv(as1455_p, daily, subset=["symbol", "date"])
                        write_csv(as1455_p, merged)
                        row["as1455_status"] = "updated_full_merge"
                    new_lasta = max_date_from_column(daily, "date")
                    if new_lasta is not None:
                        lasta = max(lasta, new_lasta) if lasta is not None else new_lasta

    manifest_row = {
        "symbol": symbol,
        "raw_5m_path": str(raw5m_p),
        "raw_daily_path": str(rawdaily_p),
        "as1455_daily_path": str(as1455_p),
        "raw_5m_last_date": _date_str(last5),
        "raw_daily_last_date": _date_str(lastd),
        "as1455_last_date": _date_str(lasta),
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update({
        "raw_5m_last_cached_date_after": _date_str(last5),
        "raw_daily_last_cached_date_after": _date_str(lastd),
        "as1455_last_cached_date_after": _date_str(lasta),
    })
    return row, manifest_row


def numeric_sum(report: pd.DataFrame, column: str) -> int:
    if report.empty or column not in report.columns:
        return 0
    return int(pd.to_numeric(report[column], errors="coerce").fillna(0).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Append-only fast incremental AS1455 historical cache update to previous trading day")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--history-end-date", default="auto")
    ap.add_argument("--history-start-date", default="2020-01-01")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--raw-5m-cache-dir", default=str(DEFAULT_RAW_5M_CACHE))
    ap.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY_CACHE))
    ap.add_argument("--as1455-daily-cache-dir", default=str(DEFAULT_AS1455_DAILY_CACHE))
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-as1455-aggregate", action="store_true")
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--no-baostock-calendar", action="store_true")
    ap.add_argument("--cache-manifest", default=None)
    ap.add_argument("--manifest-mode", choices=["auto", "trust", "scan"], default="auto")
    ap.add_argument("--force-full-5m-merge", action="store_true", help="slow repair mode: read/merge/rewrite full 5m cache when new bars arrive")
    ap.add_argument("--force-full-daily-merge", action="store_true", help="slow repair mode for raw daily cache")
    ap.add_argument("--force-full-as1455-merge", action="store_true", help="slow repair mode for AS1455 daily cache")
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    history_end = resolve_history_end_date(trade_date, args.history_end_date, not args.no_baostock_calendar)
    history_end_dash = yyyymmdd_to_dash(history_end)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    for p in [args.raw_5m_cache_dir, args.raw_daily_cache_dir, args.as1455_daily_cache_dir]:
        ensure_dir(Path(p))

    manifest_path = Path(args.cache_manifest) if args.cache_manifest else Path(args.as1455_daily_cache_dir).parent / "cache_manifest_as1455_fast_v3.csv"
    manifest = load_manifest(manifest_path)

    universe = load_universe(args.universe, args.max_symbols)
    write_csv(live_dir / "01_universe.csv", universe)
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    started = time.time()

    bs_session = None
    if not args.dry_run:
        bs_session = import_baostock()
        lg = bs_session.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        for i, symbol in enumerate(universe["symbol"].tolist(), 1):
            try:
                r, mr = update_one_symbol_fast(symbol, args, history_end_dash, manifest=manifest, bs=bs_session)
            except Exception as exc:
                r = {"symbol": symbol, "history_end_date": history_end_dash, "error": f"{type(exc).__name__}: {exc}"}
                mr = {"symbol": symbol, "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
            rows.append(r)
            manifest_rows.append(mr)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            if i % 50 == 0:
                print(f"[INFO] processed {i}/{len(universe)} symbols", flush=True)
    finally:
        if bs_session is not None:
            try:
                bs_session.logout()
            except Exception:
                pass

    update_manifest(manifest_path, manifest, manifest_rows)
    report = pd.DataFrame(rows)
    write_csv(live_dir / "00_history_update_by_symbol.csv", report)

    summary = {
        "trade_date": trade_date,
        "history_end_date": history_end_dash,
        "n_symbols": int(len(universe)),
        "dry_run": bool(args.dry_run),
        "fast_v3_append_only": True,
        "cache_manifest": str(manifest_path),
        "manifest_mode": args.manifest_mode,
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": int(report.get("error", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()) if not report.empty else 0,
        "raw_5m_status_counts": report.get("raw_5m_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "raw_daily_status_counts": report.get("raw_daily_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "as1455_status_counts": report.get("as1455_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "raw_5m_new_rows_sum": numeric_sum(report, "raw_5m_new_rows"),
        "raw_daily_new_rows_sum": numeric_sum(report, "raw_daily_new_rows"),
        "as1455_new_rows_sum": numeric_sum(report, "as1455_new_rows"),
        "raw_5m_append_only_symbols": int(report.get("raw_5m_append_only", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not report.empty else 0,
        "raw_daily_append_only_symbols": int(report.get("raw_daily_append_only", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not report.empty else 0,
        "as1455_append_only_symbols": int(report.get("as1455_append_only", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not report.empty else 0,
        "expected_raw_5m_rows_for_one_full_day_per_symbol": 48,
        "expected_raw_5m_rows_for_one_full_day_all_symbols": int(len(universe) * 48),
        "expected_raw_daily_rows_for_one_full_day_all_symbols": int(len(universe)),
        "expected_as1455_rows_for_one_full_day_all_symbols": int(len(universe)),
    }
    write_json(live_dir / "00_history_update_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
