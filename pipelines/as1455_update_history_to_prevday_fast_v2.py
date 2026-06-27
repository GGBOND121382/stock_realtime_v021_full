#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast incremental AS1455 history updater.

This is a drop-in replacement for pipelines/as1455_update_history_to_prevday.py
with three intentionally conservative fixes:

1. If raw 5m/raw daily/AS1455 caches are already up to history_end, do not
   rewrite large CSV files.
2. Use lightweight date-column scans and an optional manifest to determine
   cached end dates, instead of loading and standardizing full 5m caches.
3. Skip AS1455 aggregation when the AS1455 daily cache is already current;
   when aggregation is needed, aggregate only the missing date range.

It keeps the original cache file formats and report file names.
"""
from __future__ import annotations

import argparse
import json
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
    fetch_raw_5m,
    fetch_raw_daily,
    import_baostock,
    merge_old_5m_cache,
    next_query_start,
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
    if not s or s.lower() in {"nan", "nat", "none"}:
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


def manifest_get(manifest: pd.DataFrame, symbol: str, column: str, path: Path, trust: bool) -> Optional[pd.Timestamp]:
    if not trust or manifest.empty or not path.exists() or path.stat().st_size == 0:
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
    trust_manifest: bool,
) -> Optional[pd.Timestamp]:
    m = manifest_get(manifest, symbol, manifest_col, path, trust_manifest)
    if m is not None:
        return m
    return get_last_cached_date(path, date_col=date_col)


def read_5m_range(path: Path, symbol: str, start_date: str, end_date: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Read only the requested date range from a 5m CSV using chunks.

    This still scans the file if AS1455 is stale while 5m is cached, but avoids
    building a full multi-year DataFrame in memory.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    parts: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(path, dtype={"symbol": str, "trade_date": str, "code": str}, encoding="utf-8-sig", low_memory=False, chunksize=chunksize)
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
    if not new.empty:
        new["symbol"] = new["symbol"].map(normalize_symbol)
    if manifest.empty:
        out = new
    else:
        out = pd.concat([manifest, new], ignore_index=True, sort=False)
    out = out.drop_duplicates("symbol", keep="last").sort_values("symbol")
    ensure_dir(path.parent)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def update_one_symbol_fast(symbol: str, args, history_end_dash: str, manifest: pd.DataFrame, bs=None) -> tuple[dict, dict]:
    symbol = normalize_symbol(symbol)
    raw5m_p = raw_5m_path(Path(args.raw_5m_cache_dir), symbol)
    rawdaily_p = raw_daily_path(Path(args.raw_daily_cache_dir), symbol)
    as1455_p = as1455_daily_path(Path(args.as1455_daily_cache_dir), symbol)

    trust_manifest = args.manifest_mode == "trust"
    default_start = args.history_start_date
    history_end_ts = pd.Timestamp(history_end_dash).normalize()

    last5 = get_last_date_fast(raw5m_p, "trade_date", symbol, manifest, "raw_5m_last_date", trust_manifest)
    lastd = get_last_date_fast(rawdaily_p, "date", symbol, manifest, "raw_daily_last_date", trust_manifest)
    lasta = get_last_date_fast(as1455_p, "date", symbol, manifest, "as1455_last_date", trust_manifest)

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
        "raw_5m_last_cached_date": _date_str(last5),
        "raw_daily_last_cached_date": _date_str(lastd),
        "as1455_last_cached_date": _date_str(lasta),
        "raw_5m_query_start": start5,
        "raw_daily_query_start": startd,
        "as1455_aggregate_start": starta,
        "used_manifest": bool(trust_manifest and not manifest.empty),
        "error": "",
    }
    new5 = pd.DataFrame()

    # 1) raw 5m cache: fetch only missing range; do not rewrite if cached.
    if pd.Timestamp(start5) <= history_end_ts:
        if args.dry_run:
            row["raw_5m_status"] = "would_fetch"
        else:
            if bs is None:
                raise RuntimeError("BaoStock session is not initialized")
            new5 = fetch_raw_5m(bs, symbol, start5, history_end_dash)
            row["raw_5m_new_rows"] = int(len(new5))
            if new5.empty:
                row["raw_5m_status"] = "empty"
            else:
                if raw5m_p.exists() and raw5m_p.stat().st_size > 0:
                    # Full read/write only when there are actual new rows to merge.
                    existing = read_5m_range(raw5m_p, symbol, default_start, history_end_dash) if args.rewrite_5m_on_update else pd.read_csv(raw5m_p, dtype={"symbol": str, "trade_date": str, "code": str}, encoding="utf-8-sig", low_memory=False)
                    merged5 = merge_old_5m_cache(existing, new5, symbol)
                else:
                    merged5 = standardize_5m_to_old_schema(new5, symbol=symbol)
                write_old_5m_cache(raw5m_p, merged5)
                row["raw_5m_status"] = "updated"
                last5 = get_last_cached_date(raw5m_p, date_col="trade_date")
    else:
        row["raw_5m_status"] = "cached"

    # 2) raw daily cache: fetch only missing range; do not rewrite if cached.
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
                merged = merge_dedup_csv(rawdaily_p, newd, subset=["symbol", "date"])
                write_csv(rawdaily_p, merged)
                row["raw_daily_status"] = "updated"
                lastd = get_last_cached_date(rawdaily_p, date_col="date")
    else:
        row["raw_daily_status"] = "cached"

    # 3) AS1455 daily cache: skip if already current; aggregate missing range only.
    if args.skip_as1455_aggregate:
        row["as1455_status"] = "skipped_by_arg"
    elif lasta is not None and lasta >= history_end_ts:
        row["as1455_status"] = "cached"
    elif args.dry_run:
        row["as1455_status"] = "would_aggregate"
    else:
        agg_start = starta
        bars = pd.DataFrame()
        if not new5.empty:
            bars = new5.copy()
            # If AS1455 was behind earlier than fetched raw5m start, fall back to chunks.
            min_new_date = pd.to_datetime(bars.get("trade_date", pd.Series(dtype=str)), format="%Y%m%d", errors="coerce").min()
            if pd.isna(min_new_date) or pd.Timestamp(agg_start) < pd.Timestamp(min_new_date).normalize():
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
                    merged = merge_dedup_csv(as1455_p, daily, subset=["symbol", "date"])
                    write_csv(as1455_p, merged)
                    row["as1455_status"] = "updated"
                    lasta = get_last_cached_date(as1455_p, date_col="date")

    # Refresh lightweight dates after operations only when needed.
    if last5 is None and raw5m_p.exists():
        last5 = get_last_cached_date(raw5m_p, date_col="trade_date")
    if lastd is None and rawdaily_p.exists():
        lastd = get_last_cached_date(rawdaily_p, date_col="date")
    if lasta is None and as1455_p.exists():
        lasta = get_last_cached_date(as1455_p, date_col="date")

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast incremental AS1455 historical cache update to previous trading day")
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
    ap.add_argument("--manifest-mode", choices=["trust", "scan"], default="trust", help="trust existing manifest dates when available, or rescan date columns")
    ap.add_argument("--rewrite-5m-on-update", action="store_true", help="rewrite old 5m cache through full standardizer when new bars arrive; slower but useful for one-time migration")
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    history_end = resolve_history_end_date(trade_date, args.history_end_date, not args.no_baostock_calendar)
    history_end_dash = yyyymmdd_to_dash(history_end)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    for p in [args.raw_5m_cache_dir, args.raw_daily_cache_dir, args.as1455_daily_cache_dir]:
        ensure_dir(Path(p))

    manifest_path = Path(args.cache_manifest) if args.cache_manifest else Path(args.as1455_daily_cache_dir).parent / "cache_manifest_as1455_fast_v2.csv"
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
        "fast_v2": True,
        "cache_manifest": str(manifest_path),
        "manifest_mode": args.manifest_mode,
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": int(report.get("error", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()) if not report.empty else 0,
        "raw_5m_status_counts": report.get("raw_5m_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "raw_daily_status_counts": report.get("raw_daily_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "as1455_status_counts": report.get("as1455_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not report.empty else {},
        "raw_5m_new_rows_sum": int(pd.to_numeric(report.get("raw_5m_new_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not report.empty else 0,
        "raw_daily_new_rows_sum": int(pd.to_numeric(report.get("raw_daily_new_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not report.empty else 0,
        "as1455_new_rows_sum": int(pd.to_numeric(report.get("as1455_new_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not report.empty else 0,
    }
    write_json(live_dir / "00_history_update_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
