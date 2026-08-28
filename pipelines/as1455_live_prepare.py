#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare AS1455 live state before the 14:55 snapshot.

This is the last master-branch live-prepare design: it snapshots the universe
and preclose values, detects current-day adjustment events, and builds a compact
raw/qfq history tail rebased to the current live date. It performs neither the
14:55 collection nor model inference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    as1455_daily_path,
    collect_sina_quotes,
    ensure_dir,
    load_universe,
    normalize_symbol,
    parse_trade_date,
    raw_daily_path,
    write_csv,
    write_json,
    yyyymmdd_to_dash,
)

DEFAULT_CH12_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455"
DEFAULT_RAW_DAILY_CACHE = DEFAULT_CH12_DIR / "baostock_raw_daily_cache"
DEFAULT_AS1455_DAILY_CACHE = DEFAULT_CH12_DIR / "as1455_daily_cache"
DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"


def load_raw_daily_for_symbol(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
    if "date" not in df.columns or "close" not in df.columns or "preclose" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["raw_daily_close"] = pd.to_numeric(df["close"], errors="coerce")
    df["raw_daily_preclose"] = pd.to_numeric(df["preclose"], errors="coerce")
    df = df.dropna(subset=["date", "raw_daily_close", "raw_daily_preclose"])
    return df.sort_values("date").drop_duplicates("date", keep="last")


def compute_factor_to_history_end(raw_daily: pd.DataFrame) -> pd.DataFrame:
    if raw_daily.empty:
        return raw_daily
    out = raw_daily.copy().sort_values("date").reset_index(drop=True)
    out["prev_raw_daily_close"] = out["raw_daily_close"].shift(1)
    out["event_ratio"] = out["raw_daily_preclose"] / out["prev_raw_daily_close"]
    out.loc[~np.isfinite(out["event_ratio"]), "event_ratio"] = np.nan
    out["event_ratio"] = out["event_ratio"].fillna(1.0)
    future_event_ratio = out["event_ratio"].shift(-1).fillna(1.0)
    out["factor_to_history_end"] = future_event_ratio.iloc[::-1].cumprod().iloc[::-1].to_numpy()
    return out


def load_as1455_tail_for_symbol(path: Path, history_end: pd.Timestamp, tail_days: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"] <= history_end].sort_values("date")
    return df.tail(int(tail_days)).copy() if tail_days > 0 else df


def build_adjustment_events(
    universe: pd.DataFrame,
    preclose: pd.DataFrame,
    raw_daily_cache_dir: Path,
    trade_date: str,
    threshold_pct: float,
) -> tuple[pd.DataFrame, dict]:
    required = ["symbol", "prev_close", "source_trade_date", "source_trade_time", "core_complete", "missing_core_fields"]
    missing = [c for c in required if c not in preclose.columns]
    if missing:
        raise RuntimeError(f"preclose snapshot missing columns: {missing}")
    pre = preclose[required].copy()
    pre["symbol"] = pre["symbol"].map(normalize_symbol)
    pre = pre.drop_duplicates("symbol", keep="last")
    rows: list[dict] = []
    for symbol in universe["symbol"]:
        rd = load_raw_daily_for_symbol(raw_daily_path(raw_daily_cache_dir, symbol))
        raw_close_prev = np.nan
        raw_close_prev_date = ""
        if not rd.empty:
            last = rd.iloc[-1]
            raw_close_prev = float(last["raw_daily_close"])
            raw_close_prev_date = pd.Timestamp(last["date"]).strftime("%Y-%m-%d")
        p = pre[pre["symbol"] == symbol]
        live_preclose = np.nan
        status = "ok"
        if p.empty:
            status = "missing_preclose_snapshot"
        else:
            live_preclose = pd.to_numeric(p["prev_close"].iloc[0], errors="coerce")
        if not np.isfinite(raw_close_prev) or raw_close_prev <= 0:
            status = "missing_raw_close_prev"
        if not np.isfinite(live_preclose) or live_preclose <= 0:
            status = "missing_live_preclose"
        event_ratio = np.nan
        abs_event_diff_pct = np.nan
        is_factor_event_today = False
        if status == "ok":
            event_ratio = float(live_preclose) / float(raw_close_prev)
            abs_event_diff_pct = abs(event_ratio - 1.0) * 100.0
            is_factor_event_today = bool(abs_event_diff_pct > threshold_pct)
        rows.append({
            "symbol": symbol,
            "trade_date": yyyymmdd_to_dash(trade_date),
            "raw_close_prev_date": raw_close_prev_date,
            "raw_close_prev": raw_close_prev,
            "live_preclose": live_preclose,
            "event_ratio": event_ratio,
            "abs_event_diff_pct": abs_event_diff_pct,
            "is_factor_event_today": is_factor_event_today,
            "status": status,
        })
    events = pd.DataFrame(rows)
    return events, {
        "event_symbols": int(events["is_factor_event_today"].sum()),
        "status_counts": events["status"].value_counts(dropna=False).to_dict(),
        "max_abs_event_diff_pct": None if events["abs_event_diff_pct"].dropna().empty else float(events["abs_event_diff_pct"].max()),
    }


def latest_date_before_from_csv(path: Path, trade_ts: pd.Timestamp) -> pd.Timestamp | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c == "date", dtype={"date": str}, encoding="utf-8-sig", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, dtype={"date": str}, encoding="utf-8-sig", low_memory=False)
        except Exception:
            return None
    if "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    dates = dates[(dates.notna()) & (dates < trade_ts)]
    return None if dates.empty else pd.Timestamp(dates.max()).normalize()


def resolve_history_end_date(
    history_end_date: str,
    trade_date: str,
    universe: pd.DataFrame,
    raw_daily_cache_dir: Path,
    as1455_daily_cache_dir: Path,
) -> tuple[str, dict]:
    value = str(history_end_date).strip()
    if value.lower() not in {"auto", "prev", "prev_trade_date"}:
        resolved = yyyymmdd_to_dash(value)
        return resolved, {"mode": "explicit", "input": value, "resolved_history_end_date": resolved}
    trade_ts = pd.Timestamp(yyyymmdd_to_dash(trade_date)).normalize()
    rows: list[dict] = []
    for symbol in universe["symbol"]:
        d = latest_date_before_from_csv(as1455_daily_path(as1455_daily_cache_dir, symbol), trade_ts)
        if d is not None:
            rows.append({"symbol": symbol, "source": "as1455_daily_cache", "date": d})
    if not rows:
        for symbol in universe["symbol"]:
            d = latest_date_before_from_csv(raw_daily_path(raw_daily_cache_dir, symbol), trade_ts)
            if d is not None:
                rows.append({"symbol": symbol, "source": "raw_daily_cache", "date": d})
    if not rows:
        raise SystemExit(
            "--history-end-date auto could not be resolved: no local AS1455/raw daily "
            f"cache dates before trade_date={yyyymmdd_to_dash(trade_date)} were found. "
            "Run the history stage first or pass a concrete date."
        )
    dates = pd.Series([r["date"] for r in rows])
    resolved = pd.Timestamp(dates.max()).normalize().strftime("%Y-%m-%d")
    return resolved, {
        "mode": "auto",
        "input": value,
        "resolved_history_end_date": resolved,
        "trade_date": yyyymmdd_to_dash(trade_date),
        "source_counts": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
        "date_counts": dates.dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
        "symbols_with_candidate_dates": int(len(rows)),
        "n_symbols": int(len(universe)),
    }


def build_history_tails(
    universe: pd.DataFrame,
    events: pd.DataFrame,
    raw_daily_cache_dir: Path,
    as1455_daily_cache_dir: Path,
    history_end_date: str,
    tail_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Build raw/qfq history tails and the raw-daily execution calendar.

    The execution calendar is the union of dates in the raw-daily cache through
    ``history_end_date``. This matches the calendar source used by the clean
    strict-OOS grid while avoiding a second full cache scan in the 14:55 path.
    """
    history_end = pd.Timestamp(history_end_date).normalize()
    event_map = events.set_index("symbol")["event_ratio"].to_dict()
    raw_parts: list[pd.DataFrame] = []
    qfq_parts: list[pd.DataFrame] = []
    report_rows: list[dict] = []
    execution_calendar_parts: list[np.ndarray] = []
    for symbol in universe["symbol"]:
        tail = load_as1455_tail_for_symbol(
            as1455_daily_path(as1455_daily_cache_dir, symbol), history_end, tail_days
        )
        rd = compute_factor_to_history_end(
            load_raw_daily_for_symbol(raw_daily_path(raw_daily_cache_dir, symbol))
        )
        if not rd.empty:
            execution_calendar_parts.append(
                rd.loc[rd["date"] <= history_end, "date"].to_numpy()
            )
        status = "ok"
        if tail.empty:
            status = "missing_as1455_tail"
        if rd.empty:
            status = "missing_raw_daily_factor"
        missing_factor_dates = 0
        if status == "ok":
            qfq = tail.merge(
                rd[["date", "factor_to_history_end"]], on="date", how="left"
            )
            missing_factor_dates = int(qfq["factor_to_history_end"].isna().sum())
            today_ratio = event_map.get(symbol, 1.0)
            if not np.isfinite(today_ratio) or today_ratio <= 0:
                today_ratio = 1.0
            qfq["event_ratio_today"] = float(today_ratio)
            qfq["factor_to_live_date"] = (
                qfq["factor_to_history_end"].fillna(1.0) * float(today_ratio)
            )
            for dst, src in [
                ("open", "raw_open_as1455"),
                ("high", "raw_high_as1455"),
                ("low", "raw_low_as1455"),
                ("close", "raw_close_as1455"),
            ]:
                qfq[dst] = qfq[src] * qfq["factor_to_live_date"]
            qfq["volume"] = qfq["raw_volume_as1455"]
            qfq = qfq[[
                "date", "symbol", "open", "high", "low", "close",
                "volume", "factor_to_live_date", "factor_to_history_end",
                "event_ratio_today",
            ]]
            raw_parts.append(tail)
            qfq_parts.append(qfq)
        report_rows.append({
            "symbol": symbol,
            "status": status,
            "tail_rows": int(len(tail)),
            "raw_daily_rows": int(len(rd)),
            "first_tail_date": (
                "" if tail.empty else pd.Timestamp(tail["date"].min()).strftime("%Y-%m-%d")
            ),
            "last_tail_date": (
                "" if tail.empty else pd.Timestamp(tail["date"].max()).strftime("%Y-%m-%d")
            ),
            "missing_factor_dates": missing_factor_dates,
        })
    if execution_calendar_parts:
        execution_calendar = pd.DatetimeIndex(
            np.concatenate(execution_calendar_parts)
        ).dropna().normalize().unique().sort_values()
    else:
        execution_calendar = pd.DatetimeIndex([])
    return (
        pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame(),
        pd.concat(qfq_parts, ignore_index=True) if qfq_parts else pd.DataFrame(),
        pd.DataFrame(report_rows),
        execution_calendar,
    )


def write_table_with_fallback(df: pd.DataFrame, parquet_path: Path) -> str:
    ensure_dir(parquet_path.parent)
    try:
        df.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    except Exception as exc:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] parquet write failed for {parquet_path.name}; wrote CSV fallback {csv_path.name}: {type(exc).__name__}: {exc}", flush=True)
        return str(csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare AS1455 live feature state before 14:55")
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--history-end-date", required=True)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY_CACHE))
    ap.add_argument("--as1455-daily-cache-dir", default=str(DEFAULT_AS1455_DAILY_CACHE))
    ap.add_argument("--out-root", default=str(DEFAULT_LIVE_ROOT))
    ap.add_argument("--history-tail-days", type=int, default=252)
    ap.add_argument("--event-threshold-pct", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--timeout-seconds", type=float, default=8.0)
    ap.add_argument("--batch-sleep-seconds", type=float, default=0.2)
    ap.add_argument("--skip-preclose-fetch", action="store_true")
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    live_dir = Path(args.out_root) / trade_date
    ensure_dir(live_dir)
    started = time.time()
    universe = load_universe(args.universe, args.max_symbols)
    write_csv(live_dir / "01_universe.csv", universe)
    history_end, resolution = resolve_history_end_date(
        args.history_end_date, trade_date, universe,
        Path(args.raw_daily_cache_dir), Path(args.as1455_daily_cache_dir),
    )
    write_json(live_dir / "01_history_end_resolution.json", resolution)
    print(f"[INFO] resolved history_end_date={history_end}", flush=True)

    pre_path = live_dir / "02_preclose_snapshot_0935.csv"
    if args.skip_preclose_fetch and pre_path.exists():
        preclose = pd.read_csv(pre_path, dtype={"symbol": str}, encoding="utf-8-sig")
        errors = pd.DataFrame()
    else:
        preclose, errors = collect_sina_quotes(
            universe, batch_size=args.batch_size,
            timeout=args.timeout_seconds, batch_sleep=args.batch_sleep_seconds,
        )
        write_csv(pre_path, preclose)
        if not errors.empty:
            write_csv(live_dir / "02_preclose_fetch_errors.csv", errors)

    events, event_summary = build_adjustment_events(
        universe, preclose, Path(args.raw_daily_cache_dir),
        trade_date, args.event_threshold_pct,
    )
    write_csv(live_dir / "03_adjustment_events.csv", events)
    raw_tail, qfq_tail, tail_report, execution_calendar = build_history_tails(
        universe, events, Path(args.raw_daily_cache_dir),
        Path(args.as1455_daily_cache_dir), history_end, args.history_tail_days,
    )
    raw_tail_path = write_table_with_fallback(raw_tail, live_dir / "04_history_tail_raw.parquet") if not raw_tail.empty else ""
    qfq_tail_path = write_table_with_fallback(qfq_tail, live_dir / "05_history_tail_qfq_livebase.parquet") if not qfq_tail.empty else ""
    write_csv(live_dir / "05_history_tail_report.csv", tail_report)
    write_csv(
        live_dir / "05_execution_calendar.csv",
        pd.DataFrame({"date": execution_calendar.strftime("%Y-%m-%d")}),
    )
    summary = {
        "trade_date": trade_date,
        "history_end_date": history_end,
        "history_end_resolution": resolution,
        "n_symbols": int(len(universe)),
        "preclose_rows": int(len(preclose)),
        "preclose_core_complete_rate": float(preclose["core_complete"].mean()) if "core_complete" in preclose and len(preclose) else None,
        "preclose_error_batches": int(len(errors)),
        "adjustment_event_summary": event_summary,
        "raw_tail_rows": int(len(raw_tail)),
        "qfq_tail_rows": int(len(qfq_tail)),
        "raw_tail_path": raw_tail_path,
        "qfq_tail_path": qfq_tail_path,
        "tail_status_counts": tail_report["status"].value_counts(dropna=False).to_dict() if not tail_report.empty else {},
        "tail_missing_factor_symbols": int((tail_report.get("missing_factor_dates", pd.Series(dtype=int)).fillna(0) > 0).sum()) if not tail_report.empty else 0,
        "execution_calendar_file": str(live_dir / "05_execution_calendar.csv"),
        "execution_calendar_days": int(len(execution_calendar)),
        "execution_calendar_start": None if execution_calendar.empty else execution_calendar[0].strftime("%Y-%m-%d"),
        "execution_calendar_end": None if execution_calendar.empty else execution_calendar[-1].strftime("%Y-%m-%d"),
        "prepare_passed": bool(
            len(universe) > 0 and len(qfq_tail) > 0 and len(execution_calendar) > 0
            and (tail_report["status"].eq("ok").mean() if not tail_report.empty else 0) >= 0.98
            and (preclose["core_complete"].mean() if "core_complete" in preclose and len(preclose) else 0) >= 0.98
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(live_dir / "05_prepare_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
