#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin bounded-process dispatcher for the canonical AS1455 fast_v4 updater."""
from __future__ import annotations

import argparse
import atexit
import csv
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd

from features.as1455_live_common import (
    ensure_dir,
    load_universe,
    normalize_symbol,
    parse_trade_date,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)
from pipelines import as1455_update_history_to_prevday_fast_v4 as fast_v4
from pipelines.as1455_update_history_to_prevday import (
    DEFAULT_AS1455_DAILY_CACHE,
    DEFAULT_LIVE_ROOT,
    DEFAULT_RAW_5M_CACHE,
    DEFAULT_RAW_DAILY_CACHE,
    import_baostock,
    resolve_history_end_date,
)

_WORKER_BS = None
_WORKER_NEED_BAOSTOCK = False
_ORIGINAL_LAST_DATE = fast_v4.get_last_cached_date
_ORIGINAL_READ_RANGE = fast_v4.read_5m_range
_AS1455_SUCCESS_STATUSES = {"cached", "updated_merge_dedup", "skipped_by_arg"}


def _parse_cached_date(value: object, column: str) -> Optional[pd.Timestamp]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    if column in {"trade_date", "time"}:
        digits = "".join(ch for ch in text if ch.isdigit())[:8]
        ts = pd.to_datetime(digits, format="%Y%m%d", errors="coerce") if len(digits) == 8 else pd.NaT
    else:
        ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _read_tail_bytes(path: Path, size: int) -> tuple[int, bytes]:
    file_size = path.stat().st_size
    offset = max(0, file_size - size)
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    if offset > 0:
        first_newline = data.find(b"\n")
        data = data[first_newline + 1 :] if first_newline >= 0 else b""
    return offset, data


def get_last_cached_date_tail(path: Path, date_col: str = "date", *, tail_bytes: int = 256 * 1024) -> Optional[pd.Timestamp]:
    """Read the newest valid date from the CSV tail, with full-scan fallback."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    candidates: list[str] = []
    for column in [date_col, "date", "trade_date", "datetime", "time"]:
        if column not in candidates:
            candidates.append(column)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        indexes = [(column, header.index(column)) for column in candidates if column in header]
        if not indexes:
            return None
        _offset, data = _read_tail_bytes(path, tail_bytes)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8", errors="replace"))))
        for row in reversed(rows):
            for column, index in indexes:
                if index >= len(row):
                    continue
                parsed = _parse_cached_date(row[index], column)
                if parsed is not None:
                    return parsed
    except Exception:
        pass
    return _ORIGINAL_LAST_DATE(path, date_col=date_col)


def read_5m_range_tail(path: Path, symbol: str, start_date: str, end_date: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Read recent 5m ranges from the file tail, with canonical fallback."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header_line = handle.readline().rstrip("\r\n")
        if not header_line:
            return pd.DataFrame()
        file_size = path.stat().st_size
        window = min(file_size, 1024 * 1024)
        while window > 0:
            offset, data = _read_tail_bytes(path, window)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            csv_text = text if offset == 0 else header_line + "\n" + text
            frame = pd.read_csv(io.StringIO(csv_text), dtype={"symbol": str, "trade_date": str, "code": str}, low_memory=False)
            std = fast_v4.standardize_5m_to_old_schema(frame, symbol=symbol)
            std = fast_v4.add_aggregate_date_column(std)
            if not std.empty:
                earliest = pd.Timestamp(std["date"].min()).normalize()
                if offset == 0 or earliest <= start_ts:
                    keep = std["date"].ge(start_ts) & std["date"].le(end_ts)
                    out = std.loc[keep].copy()
                    if not out.empty:
                        out.sort_values(["symbol", "datetime"], inplace=True)
                        out.drop_duplicates(["symbol", "datetime"], keep="last", inplace=True)
                        return out.reset_index(drop=True)
                    return _ORIGINAL_READ_RANGE(path, symbol, start_date, end_date, chunksize=chunksize)
            if offset == 0:
                break
            window = min(file_size, window * 2)
    except Exception:
        pass
    return _ORIGINAL_READ_RANGE(path, symbol, start_date, end_date, chunksize=chunksize)


def _logout_worker() -> None:
    global _WORKER_BS
    if _WORKER_BS is not None:
        try:
            _WORKER_BS.logout()
        except Exception:
            pass
        _WORKER_BS = None


def _login_worker() -> None:
    global _WORKER_BS
    _logout_worker()
    bs = import_baostock()
    login = bs.login()
    if login is None or getattr(login, "error_code", None) != "0":
        raise RuntimeError(f"baostock worker login failed: {getattr(login, 'error_msg', 'empty login result')}")
    _WORKER_BS = bs


def _worker_init(need_baostock: bool) -> None:
    global _WORKER_NEED_BAOSTOCK
    _WORKER_NEED_BAOSTOCK = need_baostock
    fast_v4.get_last_cached_date = get_last_cached_date_tail
    fast_v4.read_5m_range = read_5m_range_tail
    atexit.register(_logout_worker)


def _error_row(symbol: str, history_end_dash: str, exc: Exception) -> dict[str, object]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "symbol": normalize_symbol(symbol),
        "history_end_date": history_end_dash,
        "raw_5m_status": "error",
        "raw_daily_status": "error",
        "as1455_status": "error",
        "raw_5m_new_rows": 0,
        "raw_daily_new_rows": 0,
        "as1455_new_rows": 0,
        "raw_5m_error": message,
        "raw_daily_error": message,
        "as1455_error": message,
        "error": message,
    }


def _network_retry_reason(row: dict[str, object], args: SimpleNamespace) -> str:
    if str(row.get("error", "")).strip():
        return "explicit_error"
    if not args.skip_raw_5m and str(row.get("raw_5m_status", "")) == "empty":
        return "raw_5m_empty"
    return ""


def _run_symbol(payload: tuple[int, str, dict[str, Any], str, int]) -> tuple[int, dict[str, object]]:
    index, symbol, args_dict, history_end_dash, retries = payload
    args = SimpleNamespace(**args_dict)
    last_row: dict[str, object] | None = None
    for attempt in range(1, retries + 2):
        try:
            if _WORKER_NEED_BAOSTOCK and _WORKER_BS is None:
                _login_worker()
            last_row = fast_v4.update_one_symbol_v4(symbol, args, history_end_dash, bs=_WORKER_BS)
        except Exception as exc:
            last_row = _error_row(symbol, history_end_dash, exc)
        last_row["worker_attempt"] = attempt
        retry_reason = _network_retry_reason(last_row, args)
        last_row["worker_retry_reason"] = retry_reason
        if not retry_reason:
            break
        if attempt <= retries and _WORKER_BS is not None:
            try:
                _login_worker()
            except Exception as exc:
                last_row["worker_relogin_error"] = f"{type(exc).__name__}: {exc}"
        if attempt <= retries:
            time.sleep(max(0.2, float(args.sleep_seconds)))
    if args.sleep_seconds > 0:
        time.sleep(args.sleep_seconds)
    return index, last_row or _error_row(symbol, history_end_dash, RuntimeError("empty worker result"))


def _repair_symbol_from_full_cache(payload: tuple[int, str, dict[str, Any], str]) -> tuple[int, dict[str, object]]:
    index, symbol, args_dict, history_end_dash = payload
    args = SimpleNamespace(**args_dict)
    args.dry_run = False
    args.skip_raw_5m = True
    args.skip_raw_daily = True
    args.skip_as1455_aggregate = False
    current_reader = fast_v4.read_5m_range
    try:
        fast_v4.read_5m_range = _ORIGINAL_READ_RANGE
        row = fast_v4.update_one_symbol_v4(symbol, args, history_end_dash, bs=None)
    except Exception as exc:
        row = _error_row(symbol, history_end_dash, exc)
    finally:
        fast_v4.read_5m_range = current_reader
    row["as1455_full_scan_repair"] = True
    return index, row


def _as1455_output_path(args: argparse.Namespace | SimpleNamespace, symbol: str) -> Path:
    return fast_v4.as1455_daily_path(Path(args.as1455_daily_cache_dir), normalize_symbol(symbol))


def _needs_as1455_repair(row: dict[str, object], args: argparse.Namespace, symbol: str) -> bool:
    if args.skip_as1455_aggregate or args.dry_run:
        return False
    path = _as1455_output_path(args, symbol)
    status = str(row.get("as1455_status", ""))
    return status not in _AS1455_SUCCESS_STATUSES or not path.exists() or path.stat().st_size == 0


def _merge_repair_row(original: dict[str, object], repaired: dict[str, object]) -> dict[str, object]:
    merged = dict(original)
    merged["as1455_initial_status"] = original.get("as1455_status", "")
    for key, value in repaired.items():
        if key.startswith("as1455_") or key in {"error"}:
            merged[key] = value
    merged["as1455_full_scan_repair"] = True
    return merged


def _finalize_row(row: dict[str, object], args: argparse.Namespace, symbol: str) -> dict[str, object]:
    out = dict(row)
    issues: list[str] = []
    explicit = str(out.get("error", "")).strip()
    if explicit:
        issues.append(explicit)
    if not args.skip_as1455_aggregate and not args.dry_run:
        path = _as1455_output_path(args, symbol)
        status = str(out.get("as1455_status", ""))
        if status not in _AS1455_SUCCESS_STATUSES:
            issues.append(f"as1455_status={status or '<missing>'}")
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"missing_as1455_output={path}")
    out["error"] = "; ".join(dict.fromkeys(issue for issue in issues if issue))
    return out


def numeric_sum(report: pd.DataFrame, column: str) -> int:
    if report.empty or column not in report.columns:
        return 0
    return int(pd.to_numeric(report[column], errors="coerce").fillna(0).sum())


def bool_sum(report: pd.DataFrame, column: str) -> int:
    if report.empty or column not in report.columns:
        return 0
    return int(report[column].fillna(False).astype(bool).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded-process dispatcher for the canonical AS1455 fast_v4 updater")
    parser.add_argument("--trade-date", default="today")
    parser.add_argument("--history-end-date", default="auto")
    parser.add_argument("--history-start-date", default="2020-01-01")
    parser.add_argument("--universe", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--raw-5m-cache-dir", default=str(DEFAULT_RAW_5M_CACHE))
    parser.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY_CACHE))
    parser.add_argument("--as1455-daily-cache-dir", default=str(DEFAULT_AS1455_DAILY_CACHE))
    parser.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-raw-5m", action="store_true")
    parser.add_argument("--skip-raw-daily", action="store_true")
    parser.add_argument("--skip-as1455-aggregate", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-baostock-calendar", action="store_true")
    parser.add_argument("--force-full-5m-merge", action="store_true")
    parser.add_argument("--force-full-daily-merge", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--symbol-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")
    if args.symbol_retries < 0:
        raise SystemExit("--symbol-retries must be non-negative")

    trade_date = parse_trade_date(args.trade_date)
    history_end = resolve_history_end_date(trade_date, args.history_end_date, not args.no_baostock_calendar)
    history_end_dash = yyyymmdd_to_dash(history_end)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    for directory in [args.raw_5m_cache_dir, args.raw_daily_cache_dir, args.as1455_daily_cache_dir]:
        ensure_dir(Path(directory))

    universe = load_universe(args.universe, args.max_symbols)
    write_csv(live_dir / "01_universe.csv", universe)
    symbols = universe["symbol"].map(normalize_symbol).tolist()
    workers = min(args.workers, max(1, len(symbols)))
    need_baostock = (not args.dry_run) and (not args.skip_raw_5m or not args.skip_raw_daily)
    args_dict = vars(args).copy()
    payloads = [(index, symbol, args_dict, history_end_dash, args.symbol_retries) for index, symbol in enumerate(symbols)]

    started = time.time()
    indexed_rows: dict[int, dict[str, object]] = {}
    print(f"[PARALLEL] workers={workers} symbols={len(symbols)} history_end={history_end_dash}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(need_baostock,)) as executor:
        future_map = {executor.submit(_run_symbol, payload): payload[:2] for payload in payloads}
        completed = 0
        for future in as_completed(future_map):
            index, symbol = future_map[future]
            try:
                result_index, row = future.result()
            except Exception as exc:
                result_index, row = index, _error_row(symbol, history_end_dash, exc)
            indexed_rows[result_index] = row
            completed += 1
            if completed % 10 == 0 or completed == len(symbols):
                explicit_errors = sum(bool(str(item.get("error", "")).strip()) for item in indexed_rows.values())
                print(f"[INFO] processed {completed}/{len(symbols)} symbols explicit_errors={explicit_errors}", flush=True)

    repair_indexes = [index for index, symbol in enumerate(symbols) if _needs_as1455_repair(indexed_rows[index], args, symbol)]
    repaired_count = 0
    if repair_indexes:
        print(f"[REPAIR] full-cache AS1455 repair candidates={len(repair_indexes)}", flush=True)
        repair_payloads = [(index, symbols[index], args_dict, history_end_dash) for index in repair_indexes]
        with ProcessPoolExecutor(max_workers=min(workers, len(repair_payloads)), initializer=_worker_init, initargs=(False,)) as executor:
            future_map = {executor.submit(_repair_symbol_from_full_cache, payload): payload[:2] for payload in repair_payloads}
            for future in as_completed(future_map):
                index, symbol = future_map[future]
                try:
                    result_index, repaired = future.result()
                except Exception as exc:
                    result_index, repaired = index, _error_row(symbol, history_end_dash, exc)
                merged = _merge_repair_row(indexed_rows[result_index], repaired)
                indexed_rows[result_index] = merged
                if not _needs_as1455_repair(merged, args, symbol):
                    repaired_count += 1
        print(f"[REPAIR] recovered={repaired_count}/{len(repair_indexes)}", flush=True)

    rows = [_finalize_row(indexed_rows[index], args, symbols[index]) for index in range(len(symbols))]
    report = pd.DataFrame(rows)
    write_csv(live_dir / "00_history_update_by_symbol.csv", report)
    unresolved_symbols = report.loc[report.get("error", pd.Series(dtype=object)).fillna("").ne(""), "symbol"].astype(str).tolist() if not report.empty else []
    missing_as1455_symbols = [symbol for symbol in symbols if not _as1455_output_path(args, symbol).exists() or _as1455_output_path(args, symbol).stat().st_size == 0] if not args.skip_as1455_aggregate and not args.dry_run else []
    summary = {
        "trade_date": trade_date,
        "history_end_date": history_end_dash,
        "n_symbols": int(len(universe)),
        "dry_run": bool(args.dry_run),
        "fast_v4": True,
        "parallel_dispatcher": True,
        "workers": int(workers),
        "symbol_retries": int(args.symbol_retries),
        "tail_last_date_scan": True,
        "tail_incremental_range_scan": True,
        "as1455_full_scan_repair_candidates": int(len(repair_indexes)),
        "as1455_full_scan_recovered": int(repaired_count),
        "manifest_used": False,
        "skip_raw_5m": bool(args.skip_raw_5m),
        "skip_raw_daily": bool(args.skip_raw_daily),
        "elapsed_seconds": round(time.time() - started, 3),
        "errors": int(len(unresolved_symbols)),
        "unresolved_symbols": unresolved_symbols,
        "missing_as1455_symbols": missing_as1455_symbols,
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
