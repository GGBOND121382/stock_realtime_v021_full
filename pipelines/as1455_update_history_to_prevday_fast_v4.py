#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 history updater, fast_v4.

Purpose:
- Fix fast_v3 field-contract bug: AS1455 aggregation requires `date`, while
  old 5m cache schema provides `trade_date` + `datetime`.
- Preserve per-stage status even if a later stage fails.
- Avoid manifest trust by default; scan actual cache files for last dates.
- Support repair mode that skips raw downloads and only rebuilds AS1455 daily
  from existing raw 5m cache.

This script keeps the same default cache file formats as the original updater.
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


def max_trade_date_from_5m(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    if "trade_date" in df.columns:
        s = df["trade_date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
        dates = pd.to_datetime(s, format="%Y%m%d", errors="coerce").dropna()
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
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


def add_aggregate_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return bars compatible with aggregate_as1455_from_5m.

    Required downstream contract:
      date: normalized pandas Timestamp
      datetime: pandas datetime64
      open/high/low/close/volume/amount numeric-compatible

    Old 5m cache contract:
      trade_date: YYYYMMDD string
      datetime: timestamp
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" not in out.columns:
        raise ValueError("5m bars must contain datetime before AS1455 aggregation")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if "date" not in out.columns:
        if "trade_date" in out.columns:
            s = out["trade_date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
            out["date"] = pd.to_datetime(s, format="%Y%m%d", errors="coerce").dt.normalize()
        else:
            out["date"] = out["datetime"].dt.normalize()
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["datetime", "date"]).copy()


def append_5m_cache(path: Path, new5: pd.DataFrame, symbol: str) -> pd.DataFrame:
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


def read_5m_range(path: Path, symbol: str, start_date: str, end_date: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Read only requested date range from old-schema 5m cache.

    This never rewrites raw 5m cache. It is used for AS1455 repair/update.
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
            std = add_aggregate_date_column(std)
        except Exception:
            continue
        if std.empty:
            continue
        keep = std["date"].ge(start_ts) & std["date"].le(end_ts)
        if keep.any():
            parts.append(std.loc[keep].copy())
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = standardize_5m_to_old_schema(out, symbol=symbol)
    out = add_aggregate_date_column(out)
    out.sort_values(["symbol", "datetime"], inplace=True)
    out.drop_duplicates(["symbol", "datetime"], keep="last", inplace=True)
    return out.reset_index(drop=True)


def safe_append_allowed(last_cached: Optional[pd.Timestamp], start: str, path: Path) -> bool:
    """Append is safe when query start is strictly after actual last cached date.

    Calendar gaps such as weekends are valid. The old fast_v3 used exact
    last+1 calendar day, which unnecessarily fell back to full merge after
    weekends. Here we only forbid overlap.
    """
    if not path.exists() or path.stat().st_size == 0:
        return True
    if last_cached is None:
        return False
    return pd.Timestamp(start).normalize() > pd.Timestamp(last_cached).normalize()


def update_one_symbol_v4(symbol: str, args, history_end_dash: str, bs=None) -> dict:
    symbol = normalize_symbol(symbol)
    raw5m_p = raw_5m_path(Path(args.raw_5m_cache_dir), symbol)
    rawdaily_p = raw_daily_path(Path(args.raw_daily_cache_dir), symbol)
    as1455_p = as1455_daily_path(Path(args.as1455_daily_cache_dir), symbol)

    history_end_ts = pd.Timestamp(history_end_dash).normalize()
    default_start = args.history_start_date

    last5 = get_last_cached_date(raw5m_p, date_col="trade_date")
    lastd = get_last_cached_date(rawdaily_p, date_col="date")
    lasta = get_last_cached_date(as1455_p, date_col="date")

    start5 = next_query_start(last5, default_start)
    startd = next_query_start(lastd, default_start)
    starta = next_query_start(lasta, default_start)

    row: dict[str, object] = {
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
        "as1455_written": False,
        "raw_5m_last_cached_date": _date_str(last5),
        "raw_daily_last_cached_date": _date_str(lastd),
        "as1455_last_cached_date": _date_str(lasta),
        "raw_5m_query_start": start5,
        "raw_daily_query_start": startd,
        "as1455_query_start": starta,
        "raw_5m_error": "",
        "raw_daily_error": "",
        "as1455_error": "",
        "error": "",
    }

    # Stage 1: raw 5m. Independent failure must not erase later/earlier states.
    try:
        if args.skip_raw_5m:
            row["raw_5m_status"] = "skipped_by_arg"
        elif pd.Timestamp(start5) <= history_end_ts:
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
                elif safe_append_allowed(last5, start5, raw5m_p) and not args.force_full_5m_merge:
                    append_5m_cache(raw5m_p, std_new5, symbol=symbol)
                    row["raw_5m_status"] = "updated_append"
                    row["raw_5m_append_only"] = True
                    new_last5 = max_trade_date_from_5m(std_new5)
                    if new_last5 is not None:
                        last5 = max(last5, new_last5) if last5 is not None else new_last5
                else:
                    existing = read_old_5m_cache(raw5m_p, symbol)
                    merged5 = merge_old_5m_cache(existing, std_new5, symbol)
                    write_old_5m_cache(raw5m_p, merged5)
                    row["raw_5m_status"] = "updated_full_merge"
                    new_last5 = max_trade_date_from_5m(merged5)
                    if new_last5 is not None:
                        last5 = new_last5
        else:
            row["raw_5m_status"] = "cached"
    except Exception as exc:
        row["raw_5m_status"] = "error"
        row["raw_5m_error"] = f"{type(exc).__name__}: {exc}"

    # Stage 2: raw daily.
    try:
        if args.skip_raw_daily:
            row["raw_daily_status"] = "skipped_by_arg"
        elif pd.Timestamp(startd) <= history_end_ts:
            if args.dry_run:
                row["raw_daily_status"] = "would_fetch"
            else:
                if bs is None:
                    raise RuntimeError("BaoStock session is not initialized")
                newd = fetch_raw_daily(bs, symbol, startd, history_end_dash)
                row["raw_daily_new_rows"] = int(len(newd))
                if newd.empty:
                    row["raw_daily_status"] = "empty"
                elif safe_append_allowed(lastd, startd, rawdaily_p) and not args.force_full_daily_merge:
                    append_daily_cache(rawdaily_p, newd)
                    row["raw_daily_status"] = "updated_append"
                    row["raw_daily_append_only"] = True
                    new_lastd = max_date_from_column(newd, "date")
                    if new_lastd is not None:
                        lastd = max(lastd, new_lastd) if lastd is not None else new_lastd
                else:
                    merged = merge_dedup_csv(rawdaily_p, newd, subset=["symbol", "date"])
                    write_csv(rawdaily_p, merged)
                    row["raw_daily_status"] = "updated_full_merge"
                    new_lastd = max_date_from_column(merged, "date")
                    if new_lastd is not None:
                        lastd = new_lastd
        else:
            row["raw_daily_status"] = "cached"
    except Exception as exc:
        row["raw_daily_status"] = "error"
        row["raw_daily_error"] = f"{type(exc).__name__}: {exc}"

    # Re-scan AS1455 start after possible previous partial repair.
    try:
        lasta = get_last_cached_date(as1455_p, date_col="date")
        starta = next_query_start(lasta, default_start)
        row["as1455_last_cached_date_before_stage"] = _date_str(lasta)
        row["as1455_query_start_before_stage"] = starta
    except Exception:
        pass

    # Stage 3: AS1455 daily. Read local raw 5m range and explicitly add `date`.
    try:
        if args.skip_as1455_aggregate:
            row["as1455_status"] = "skipped_by_arg"
        elif lasta is not None and lasta >= history_end_ts:
            row["as1455_status"] = "cached"
        elif args.dry_run:
            row["as1455_status"] = "would_aggregate"
        else:
            bars = read_5m_range(raw5m_p, symbol, starta, history_end_dash)
            row["as1455_source_5m_rows"] = int(len(bars))
            if bars.empty:
                row["as1455_status"] = "no_bars_for_missing_range"
            else:
                # Field contract assertion: aggregate requires `date`.
                if "date" not in bars.columns:
                    raise RuntimeError("internal contract violation: bars missing date before aggregate")
                daily = aggregate_as1455_from_5m(bars, symbol=symbol, start_date=starta, end_date=history_end_dash)
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
                        merged = merge_dedup_csv(as1455_p, daily, subset=["symbol", "date"])
                        write_csv(as1455_p, merged)
                        row["as1455_status"] = "updated_merge_dedup"
                        row["as1455_written"] = True
                        new_lasta = max_date_from_column(daily, "date")
                        if new_lasta is not None:
                            lasta = max(lasta, new_lasta) if lasta is not None else new_lasta
    except Exception as exc:
        row["as1455_status"] = "error"
        row["as1455_error"] = f"{type(exc).__name__}: {exc}"

    row.update({
        "raw_5m_last_cached_date_after": _date_str(last5),
        "raw_daily_last_cached_date_after": _date_str(lastd),
        "as1455_last_cached_date_after": _date_str(lasta),
    })
    stage_errors = [str(row.get(k, "")) for k in ["raw_5m_error", "raw_daily_error", "as1455_error"] if str(row.get(k, "")).strip()]
    row["error"] = "; ".join(stage_errors)
    return row


def numeric_sum(report: pd.DataFrame, column: str) -> int:
    if report.empty or column not in report.columns:
        return 0
    return int(pd.to_numeric(report[column], errors="coerce").fillna(0).sum())


def bool_sum(report: pd.DataFrame, column: str) -> int:
    if report.empty or column not in report.columns:
        return 0
    return int(report[column].fillna(False).astype(bool).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="AS1455 fast_v4 incremental history updater with explicit field contracts")
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
    ap.add_argument("--skip-raw-5m", action="store_true", help="repair mode: do not download or write raw 5m cache")
    ap.add_argument("--skip-raw-daily", action="store_true", help="repair mode: do not download or write raw daily cache")
    ap.add_argument("--skip-as1455-aggregate", action="store_true")
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--no-baostock-calendar", action="store_true")
    ap.add_argument("--force-full-5m-merge", action="store_true")
    ap.add_argument("--force-full-daily-merge", action="store_true")
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    history_end = resolve_history_end_date(trade_date, args.history_end_date, not args.no_baostock_calendar)
    history_end_dash = yyyymmdd_to_dash(history_end)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    for p in [args.raw_5m_cache_dir, args.raw_daily_cache_dir, args.as1455_daily_cache_dir]:
        ensure_dir(Path(p))

    universe = load_universe(args.universe, args.max_symbols)
    write_csv(live_dir / "01_universe.csv", universe)

    rows: list[dict] = []
    started = time.time()
    bs_session = None
    need_baostock = (not args.dry_run) and (not args.skip_raw_5m or not args.skip_raw_daily)
    if need_baostock:
        bs_session = import_baostock()
        lg = bs_session.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        symbols = universe["symbol"].map(normalize_symbol).tolist()
        for i, symbol in enumerate(symbols, 1):
            r = update_one_symbol_v4(symbol, args, history_end_dash, bs=bs_session)
            rows.append(r)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            if i % 50 == 0:
                print(f"[INFO] processed {i}/{len(symbols)} symbols", flush=True)
    finally:
        if bs_session is not None:
            try:
                bs_session.logout()
            except Exception:
                pass

    report = pd.DataFrame(rows)
    write_csv(live_dir / "00_history_update_by_symbol.csv", report)
    summary = {
        "trade_date": trade_date,
        "history_end_date": history_end_dash,
        "n_symbols": int(len(universe)),
        "dry_run": bool(args.dry_run),
        "fast_v4": True,
        "manifest_used": False,
        "skip_raw_5m": bool(args.skip_raw_5m),
        "skip_raw_daily": bool(args.skip_raw_daily),
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": int(report["error"].fillna("").ne("").sum()) if "error" in report.columns else int(len(report)),
        "raw_5m_status_counts": report.get("raw_5m_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "raw_daily_status_counts": report.get("raw_daily_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "as1455_status_counts": report.get("as1455_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "raw_5m_new_rows_sum": numeric_sum(report, "raw_5m_new_rows"),
        "raw_daily_new_rows_sum": numeric_sum(report, "raw_daily_new_rows"),
        "as1455_new_rows_sum": numeric_sum(report, "as1455_new_rows"),
        "raw_5m_append_only_symbols": bool_sum(report, "raw_5m_append_only"),
        "raw_daily_append_only_symbols": bool_sum(report, "raw_daily_append_only"),
        "as1455_written_symbols": bool_sum(report, "as1455_written"),
        "expected_raw_5m_rows_for_one_full_day_per_symbol": 48,
        "expected_raw_5m_rows_for_one_full_day_all_symbols": int(len(universe) * 48),
        "expected_raw_daily_rows_for_one_full_day_all_symbols": int(len(universe)),
        "expected_as1455_rows_for_one_full_day_all_symbols": int(len(universe)),
        "by_symbol_report": str(live_dir / "00_history_update_by_symbol.csv"),
    }
    write_json(live_dir / "00_history_update_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
