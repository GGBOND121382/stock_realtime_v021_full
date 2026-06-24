#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect live A-share quotes for the AS1455 NN pipeline.

Scope of this first version
---------------------------
This script only builds the live *data collection* layer for the AS1455 NN
pipeline.  It does not build Ch12 features, does not load a neural network,
and does not produce trading orders.

Target output for one trading day::

    saved_data/ashare_ml4t/live_as1455/YYYYMMDD/
        effective_universe.csv
        snapshots_raw.csv
        snapshots_latest_asof1455.csv
        as1455_live_raw_panel.csv
        as1455_live_collection_report.csv
        run_summary.json

Design notes
------------
* Use the full AS1455 training universe for collection.  Do not restrict to
  mainboard at collection time, because Ch12 cross-sectional features need the
  same broad cross-section as training.
* Use the quote source's own date/time when available.  `collected_at` is kept
  for audit, but cutoff filtering uses source quote time first.
* Normalize volume to shares by checking `amount / volume / price`.  If the
  quote source already reports shares, multiplier is 1; if it reports lots,
  multiplier is 100.
* Keep invalid/stale rows visible in reports instead of silently filling them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"
DEFAULT_CUTOFF_TIME = "14:55:00"
DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_BATCH_SIZE = 250

DEFAULT_UNIVERSE_CANDIDATES = [
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "effective_universe.csv",
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "universe" / "effective_universe.csv",
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "universe" / "07_universe_allA_top1000_static.csv",
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "07_universe_allA_top1000_static.csv",
    PROJECT_DIR / "data" / "universe" / "07_universe_allA_top1000_static.csv",
]

SYMBOL_COLUMNS = [
    "symbol",
    "code",
    "ts_code",
    "股票代码",
    "证券代码",
    "代码",
]

BOARD_COLUMNS = ["board", "板块", "market_board", "listing_board"]
INDUSTRY_COLUMNS = ["industry", "行业", "sector", "申万行业", "中信行业"]
NAME_COLUMNS = ["name", "股票简称", "证券简称", "名称"]

SNAPSHOT_COLUMNS = [
    "symbol", "code", "exchange", "name", "trade_date", "source_trade_date",
    "source_trade_time", "quote_datetime", "collected_at", "source", "source_status",
    "last_price", "open", "high", "low", "prev_close", "volume_raw",
    "amount_raw", "volume_unit_inferred", "volume_unit_multiplier", "volume_shares",
    "vwap_ratio_raw", "pct_chg", "bid_price_1", "ask_price_1", "bid_volume_1",
    "ask_volume_1", "core_complete", "missing_core_fields", "raw_payload",
]

RAW_PANEL_COLUMNS = [
    "symbol", "date", "name", "exchange", "board", "industry",
    "raw_open_as1455", "raw_high_as1455", "raw_low_as1455", "raw_close_as1455",
    "raw_volume_as1455", "raw_amount_as1455", "live_preclose",
    "snapshot_time", "snapshot_age_seconds", "source_used", "source_status",
    "core_complete", "quality_status", "missing_core_fields",
    "volume_unit_inferred", "volume_unit_multiplier", "vwap_ratio_raw",
    "is_mainboard", "trade_allowed_mainboard",
]


@dataclass
class LiveSnapshot:
    symbol: str
    code: str
    exchange: str
    name: Optional[str]
    trade_date: str
    source_trade_date: Optional[str]
    source_trade_time: Optional[str]
    quote_datetime: Optional[str]
    collected_at: str
    source: str
    source_status: Optional[str]
    last_price: Optional[float]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    prev_close: Optional[float]
    volume_raw: Optional[float]
    amount_raw: Optional[float]
    volume_unit_inferred: str
    volume_unit_multiplier: Optional[float]
    volume_shares: Optional[float]
    vwap_ratio_raw: Optional[float]
    pct_chg: Optional[float]
    bid_price_1: Optional[float]
    ask_price_1: Optional[float]
    bid_volume_1: Optional[float]
    ask_volume_1: Optional[float]
    core_complete: bool
    missing_core_fields: str
    raw_payload: Optional[str]


def to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "None", "nan", "NaN", "null"}:
        return None
    try:
        x = float(text)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def normalize_symbol(value: str) -> str:
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        raise ValueError(f"invalid symbol: {value!r}")
    if "." in s:
        a, b = s.split(".", 1)
        # Accept both 000001.SZ and SZ.000001 / sh.600000.
        if a.isalpha() and b[:6].isdigit():
            market, code = a.upper(), re.sub(r"\D", "", b)[:6]
        else:
            code, market = re.sub(r"\D", "", a)[:6], b.upper()
        if market in {"XSHE", "SZSE"}:
            market = "SZ"
        if market in {"XSHG", "SSE"}:
            market = "SH"
        if market not in {"SH", "SZ"}:
            market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code.zfill(6)}.{market}"
    digits = re.sub(r"\D", "", s)
    if not digits:
        raise ValueError(f"invalid symbol: {value!r}")
    code = digits[:6].zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def sina_symbol(symbol: str) -> str:
    s = normalize_symbol(symbol)
    code, market = s.split(".", 1)
    return f"{market.lower()}{code}"


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def parse_time(value: str) -> dtime:
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        hh, mm = parts
        ss = 0
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        raise ValueError(f"invalid HH:MM[:SS] time: {value!r}")
    return dtime(int(hh), int(mm), int(ss))


def yyyymmdd_to_dash(value: str) -> str:
    s = str(value).replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def load_universe(path: Optional[str], max_symbols: Optional[int] = None) -> pd.DataFrame:
    if path:
        p = Path(path)
    else:
        p = next((x for x in DEFAULT_UNIVERSE_CANDIDATES if x.exists()), None)
        if p is None:
            candidates = "\n".join(str(x) for x in DEFAULT_UNIVERSE_CANDIDATES)
            raise FileNotFoundError(
                "universe file not found; pass --universe explicitly. Tried:\n" + candidates
            )
    df = pd.read_csv(p, encoding="utf-8-sig")
    sym_col = next((c for c in SYMBOL_COLUMNS if c in df.columns), None)
    if sym_col is None:
        raise ValueError(f"cannot find symbol column in {p}; columns={list(df.columns)}")
    out = df.copy()
    out["symbol"] = out[sym_col].map(normalize_symbol)
    out["code"] = out["symbol"].str.slice(0, 6)
    out["exchange"] = out["symbol"].str.split(".").str[1]

    name_col = next((c for c in NAME_COLUMNS if c in out.columns), None)
    board_col = next((c for c in BOARD_COLUMNS if c in out.columns), None)
    industry_col = next((c for c in INDUSTRY_COLUMNS if c in out.columns), None)
    out["name"] = out[name_col] if name_col else ""
    out["board"] = out[board_col] if board_col else ""
    out["industry"] = out[industry_col] if industry_col else ""
    out["is_mainboard"] = out.apply(infer_mainboard, axis=1)
    out["trade_allowed_mainboard"] = out["is_mainboard"]
    keep_cols = ["symbol", "code", "exchange", "name", "board", "industry", "is_mainboard", "trade_allowed_mainboard"]
    extra_cols = [c for c in out.columns if c not in keep_cols]
    out = out[keep_cols + extra_cols]
    out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    if max_symbols is not None:
        out = out.head(int(max_symbols)).copy()
    return out


def infer_mainboard(row: pd.Series) -> bool:
    board_text = " ".join(str(row.get(c, "")) for c in BOARD_COLUMNS if c in row.index).lower()
    symbol = str(row.get("symbol", ""))
    code = symbol[:6]
    if any(x in board_text for x in ["chinext", "创业", "star", "科创", "北交", "bj"]):
        return False
    if any(x in board_text for x in ["main", "主板", "sh_mainboard", "sz_mainboard"]):
        return True
    # Conservative fallback for current A-share universe.
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def request_sina_batch(symbols: Sequence[str], timeout: float = 8.0) -> dict[str, str]:
    market_symbols = [sina_symbol(s) for s in symbols]
    if not market_symbols:
        return {}
    url = "https://hq.sinajs.cn/list=" + ",".join(market_symbols)
    headers = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        text = resp.content.decode("gbk")
    except Exception:
        text = resp.text
    result: dict[str, str] = {}
    code_to_symbol = {sina_symbol(s): normalize_symbol(s) for s in symbols}
    for line in text.splitlines():
        m = re.search(r"hq_str_([a-z]{2}\d{6})=\"(.*)\";?", line.strip(), flags=re.I)
        if not m:
            continue
        market_code = m.group(1).lower()
        symbol = code_to_symbol.get(market_code)
        if symbol:
            result[symbol] = m.group(2)
    return result


def parse_sina_payload(symbol: str, raw: str, collected_at: datetime, include_raw: bool) -> LiveSnapshot:
    symbol_n = normalize_symbol(symbol)
    code, exchange = symbol_n.split(".", 1)
    parts = [x.strip() for x in str(raw).strip().strip('";').split(",")]
    # Sina A-share fields:
    # name, open, prev_close, current, high, low, bid, ask, volume, amount,
    # bid1_vol,bid1_price,...,bid5_vol,bid5_price,
    # ask1_vol,ask1_price,...,ask5_vol,ask5_price,date,time,status
    name = parts[0] if len(parts) > 0 and parts[0] else None
    open_px = to_float(parts[1]) if len(parts) > 1 else None
    prev_close = to_float(parts[2]) if len(parts) > 2 else None
    last_price = to_float(parts[3]) if len(parts) > 3 else None
    high = to_float(parts[4]) if len(parts) > 4 else None
    low = to_float(parts[5]) if len(parts) > 5 else None
    volume_raw = to_float(parts[8]) if len(parts) > 8 else None
    amount_raw = to_float(parts[9]) if len(parts) > 9 else None
    bid_volume_1 = to_float(parts[10]) if len(parts) > 10 else None
    bid_price_1 = to_float(parts[11]) if len(parts) > 11 else None
    ask_volume_1 = to_float(parts[20]) if len(parts) > 20 else None
    ask_price_1 = to_float(parts[21]) if len(parts) > 21 else None
    source_trade_date = normalize_source_date(parts[30]) if len(parts) > 30 else None
    source_trade_time = normalize_source_time(parts[31]) if len(parts) > 31 else None
    source_status = parts[32] if len(parts) > 32 and parts[32] else None
    quote_datetime = None
    if source_trade_date and source_trade_time:
        quote_datetime = f"{yyyymmdd_to_dash(source_trade_date)} {source_trade_time}"

    unit, multiplier, volume_shares, ratio = infer_volume_unit(volume_raw, amount_raw, last_price)
    pct_chg = None
    if last_price is not None and prev_close is not None and prev_close > 0:
        pct_chg = last_price / prev_close - 1.0

    missing = []
    for field, value in [
        ("last_price", last_price), ("open", open_px), ("high", high),
        ("low", low), ("prev_close", prev_close), ("volume", volume_shares),
        ("amount", amount_raw),
    ]:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            missing.append(field)
    if high is not None and low is not None and high < low:
        missing.append("high_lt_low")
    if last_price is not None and high is not None and low is not None and not (low <= last_price <= high):
        missing.append("last_outside_low_high")
    if open_px is not None and high is not None and low is not None and not (low <= open_px <= high):
        missing.append("open_outside_low_high")

    return LiveSnapshot(
        symbol=symbol_n,
        code=code,
        exchange=exchange,
        name=name,
        trade_date=source_trade_date or collected_at.strftime("%Y%m%d"),
        source_trade_date=source_trade_date,
        source_trade_time=source_trade_time,
        quote_datetime=quote_datetime,
        collected_at=collected_at.isoformat(timespec="seconds"),
        source="sina_batch",
        source_status=source_status,
        last_price=last_price,
        open=open_px,
        high=high,
        low=low,
        prev_close=prev_close,
        volume_raw=volume_raw,
        amount_raw=amount_raw,
        volume_unit_inferred=unit,
        volume_unit_multiplier=multiplier,
        volume_shares=volume_shares,
        vwap_ratio_raw=ratio,
        pct_chg=pct_chg,
        bid_price_1=bid_price_1,
        ask_price_1=ask_price_1,
        bid_volume_1=bid_volume_1,
        ask_volume_1=ask_volume_1,
        core_complete=(len(missing) == 0),
        missing_core_fields=",".join(missing),
        raw_payload=raw if include_raw else None,
    )


def normalize_source_date(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        return digits[:8]
    return None


def normalize_source_time(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if m:
        hh, mm, ss = m.group(1), m.group(2), m.group(3) or "00"
        return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 6:
        return f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"
    if len(digits) >= 4:
        return f"{digits[:2]}:{digits[2:4]}:00"
    return None


def infer_volume_unit(volume, amount, price) -> tuple[str, Optional[float], Optional[float], Optional[float]]:
    vol = to_float(volume)
    amt = to_float(amount)
    px = to_float(price)
    if vol is None or amt is None or px is None or vol <= 0 or px <= 0:
        return "unknown", None, None, None
    ratio = (amt / vol) / px
    if not math.isfinite(ratio):
        return "unknown", None, None, None
    if 0.5 <= ratio <= 2.0:
        return "shares", 1.0, vol, ratio
    if 50.0 <= ratio <= 150.0:
        return "lots", 100.0, vol * 100.0, ratio
    return "unknown", None, None, ratio


def chunks(seq: Sequence[str], n: int) -> Iterable[list[str]]:
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])


def output_day_dir(out_root: str | Path, trade_date: str) -> Path:
    return Path(out_root) / str(trade_date)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def collect_once(args: argparse.Namespace) -> None:
    trade_date = args.trade_date or today_yyyymmdd()
    universe = load_universe(args.universe, args.max_symbols)
    day_dir = output_day_dir(args.out_root, trade_date)
    day_dir.mkdir(parents=True, exist_ok=True)
    write_csv(day_dir / "effective_universe.csv", universe)

    symbols = universe["symbol"].tolist()
    collected_at = datetime.now()
    rows = []
    errors = []
    for batch_no, batch in enumerate(chunks(symbols, args.batch_size), start=1):
        try:
            got = request_sina_batch(batch, timeout=args.timeout_seconds)
        except Exception as exc:
            errors.append({
                "batch_no": batch_no,
                "symbols": ",".join(batch),
                "error": f"{type(exc).__name__}: {exc}",
            })
            got = {}
        for symbol in batch:
            raw = got.get(symbol)
            if raw is None:
                rows.append(error_snapshot(symbol, collected_at, trade_date, "missing_sina_payload", args.include_raw))
            else:
                rows.append(asdict(parse_sina_payload(symbol, raw, collected_at, args.include_raw)))
        if args.batch_sleep_seconds > 0 and batch_no < math.ceil(len(symbols) / args.batch_size):
            time.sleep(args.batch_sleep_seconds)

    df = pd.DataFrame(rows)
    for col in SNAPSHOT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SNAPSHOT_COLUMNS]
    append_csv(day_dir / "snapshots_raw.csv", df)
    if errors:
        append_csv(day_dir / "collection_errors.csv", pd.DataFrame(errors))
    summary = {
        "mode": "collect-once",
        "trade_date": trade_date,
        "collected_at": collected_at.isoformat(timespec="seconds"),
        "universe_symbols": int(len(universe)),
        "snapshot_rows_written": int(len(df)),
        "core_complete_rows_written": int(df["core_complete"].fillna(False).astype(bool).sum()) if len(df) else 0,
        "error_batches": int(len(errors)),
        "out_dir": str(day_dir),
    }
    (day_dir / "last_collect_once_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.finalize:
        finalize_day(args)


def error_snapshot(symbol: str, collected_at: datetime, trade_date: str, reason: str, include_raw: bool) -> dict:
    symbol_n = normalize_symbol(symbol)
    code, exchange = symbol_n.split(".", 1)
    row = asdict(LiveSnapshot(
        symbol=symbol_n,
        code=code,
        exchange=exchange,
        name=None,
        trade_date=trade_date,
        source_trade_date=None,
        source_trade_time=None,
        quote_datetime=None,
        collected_at=collected_at.isoformat(timespec="seconds"),
        source="sina_batch",
        source_status=reason,
        last_price=None,
        open=None,
        high=None,
        low=None,
        prev_close=None,
        volume_raw=None,
        amount_raw=None,
        volume_unit_inferred="unknown",
        volume_unit_multiplier=None,
        volume_shares=None,
        vwap_ratio_raw=None,
        pct_chg=None,
        bid_price_1=None,
        ask_price_1=None,
        bid_volume_1=None,
        ask_volume_1=None,
        core_complete=False,
        missing_core_fields="missing_payload",
        raw_payload=reason if include_raw else None,
    ))
    return row


def collect_loop(args: argparse.Namespace) -> None:
    until_time = parse_time(args.until)
    while True:
        now = datetime.now()
        if now.time() > until_time:
            print(f"stop: current time {now.strftime('%H:%M:%S')} > --until {args.until}")
            break
        collect_once(args)
        time.sleep(max(1, int(args.interval_seconds)))
    finalize_day(args)


def finalize_day(args: argparse.Namespace) -> None:
    trade_date = args.trade_date or today_yyyymmdd()
    day_dir = output_day_dir(args.out_root, trade_date)
    snap_path = day_dir / "snapshots_raw.csv"
    if not snap_path.exists():
        raise FileNotFoundError(snap_path)
    universe_path = day_dir / "effective_universe.csv"
    if universe_path.exists():
        universe = pd.read_csv(universe_path, encoding="utf-8-sig")
    else:
        universe = load_universe(args.universe, args.max_symbols)
        write_csv(universe_path, universe)
    df = pd.read_csv(snap_path, encoding="utf-8-sig")
    latest = select_latest_asof(df, trade_date, args.cutoff_time, args.max_snapshot_age_seconds)
    latest = universe[["symbol", "exchange", "name", "board", "industry", "is_mainboard", "trade_allowed_mainboard"]].merge(
        latest,
        on="symbol",
        how="left",
        suffixes=("_universe", ""),
    )
    # Prefer quote fields when available, otherwise universe metadata.
    for col in ["name", "exchange"]:
        ucol = f"{col}_universe"
        if col in latest.columns and ucol in latest.columns:
            latest[col] = latest[col].where(latest[col].notna() & latest[col].astype(str).ne(""), latest[ucol])
            latest = latest.drop(columns=[ucol])

    panel = build_raw_panel(latest, trade_date)
    report = build_collection_report(latest, panel)
    write_csv(day_dir / "snapshots_latest_asof1455.csv", latest)
    write_csv(day_dir / "as1455_live_raw_panel.csv", panel)
    write_csv(day_dir / "as1455_live_collection_report.csv", report)
    summary = build_run_summary(trade_date, universe, df, latest, panel, report, args)
    (day_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def select_latest_asof(df: pd.DataFrame, trade_date: str, cutoff_time: str, max_age_seconds: int) -> pd.DataFrame:
    work = df.copy()
    cutoff_dt = pd.Timestamp(f"{yyyymmdd_to_dash(trade_date)} {normalize_source_time(cutoff_time) or cutoff_time}")
    work["quote_datetime_parsed"] = pd.to_datetime(work.get("quote_datetime"), errors="coerce")
    collected = pd.to_datetime(work.get("collected_at"), errors="coerce")
    # Fallback to collected_at only when source quote time is missing.
    work["effective_datetime"] = work["quote_datetime_parsed"].where(work["quote_datetime_parsed"].notna(), collected)
    work["source_time_missing"] = work["quote_datetime_parsed"].isna()
    work["after_cutoff"] = work["effective_datetime"] > cutoff_dt
    before = work[(work["effective_datetime"].notna()) & (~work["after_cutoff"])].copy()
    before = before.sort_values(["symbol", "effective_datetime", "collected_at"])
    latest = before.groupby("symbol", as_index=False).tail(1).copy()
    latest["snapshot_age_seconds"] = (cutoff_dt - latest["effective_datetime"]).dt.total_seconds()
    latest["snapshot_too_old"] = latest["snapshot_age_seconds"] > float(max_age_seconds)
    latest["cutoff_datetime"] = str(cutoff_dt)
    return latest.reset_index(drop=True)


def build_raw_panel(latest: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    rows = []
    for _, row in latest.iterrows():
        quality = quality_status(row)
        rows.append({
            "symbol": row.get("symbol"),
            "date": yyyymmdd_to_dash(trade_date),
            "name": row.get("name"),
            "exchange": row.get("exchange"),
            "board": row.get("board"),
            "industry": row.get("industry"),
            "raw_open_as1455": to_float(row.get("open")),
            "raw_high_as1455": to_float(row.get("high")),
            "raw_low_as1455": to_float(row.get("low")),
            "raw_close_as1455": to_float(row.get("last_price")),
            "raw_volume_as1455": to_float(row.get("volume_shares")),
            "raw_amount_as1455": to_float(row.get("amount_raw")),
            "live_preclose": to_float(row.get("prev_close")),
            "snapshot_time": row.get("quote_datetime"),
            "snapshot_age_seconds": to_float(row.get("snapshot_age_seconds")),
            "source_used": row.get("source"),
            "source_status": row.get("source_status"),
            "core_complete": as_bool(row.get("core_complete")),
            "quality_status": quality,
            "missing_core_fields": row.get("missing_core_fields"),
            "volume_unit_inferred": row.get("volume_unit_inferred"),
            "volume_unit_multiplier": to_float(row.get("volume_unit_multiplier")),
            "vwap_ratio_raw": to_float(row.get("vwap_ratio_raw")),
            "is_mainboard": bool(row.get("is_mainboard")) if pd.notna(row.get("is_mainboard")) else False,
            "trade_allowed_mainboard": bool(row.get("trade_allowed_mainboard")) if pd.notna(row.get("trade_allowed_mainboard")) else False,
        })
    panel = pd.DataFrame(rows)
    for col in RAW_PANEL_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
    return panel[RAW_PANEL_COLUMNS]


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def quality_status(row: pd.Series) -> str:
    if pd.isna(row.get("last_price")):
        return "missing_snapshot"
    if as_bool(row.get("after_cutoff", False)):
        return "after_cutoff"
    if as_bool(row.get("snapshot_too_old", False)):
        return "stale_snapshot"
    if not as_bool(row.get("core_complete", False)):
        return "core_incomplete"
    values = [row.get("open"), row.get("high"), row.get("low"), row.get("last_price"), row.get("prev_close")]
    nums = [to_float(x) for x in values]
    if any(x is None or x <= 0 for x in nums):
        return "nonpositive_price"
    op, hi, lo, close, _pre = nums
    if hi < lo:
        return "high_lt_low"
    if not (lo <= close <= hi):
        return "close_outside_low_high"
    if not (lo <= op <= hi):
        return "open_outside_low_high"
    vol = to_float(row.get("volume_shares"))
    amt = to_float(row.get("amount_raw"))
    if vol is None or amt is None or vol <= 0 or amt <= 0:
        return "nonpositive_volume_amount"
    return "ok"


def build_collection_report(latest: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    report = panel[[
        "symbol", "date", "name", "board", "industry", "quality_status", "core_complete",
        "missing_core_fields", "snapshot_time", "snapshot_age_seconds", "source_status",
        "volume_unit_inferred", "vwap_ratio_raw", "is_mainboard", "trade_allowed_mainboard",
    ]].copy()
    return report


def build_run_summary(
    trade_date: str,
    universe: pd.DataFrame,
    snapshots: pd.DataFrame,
    latest: pd.DataFrame,
    panel: pd.DataFrame,
    report: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    expected = int(len(universe))
    latest_symbols = int(latest["symbol"].nunique()) if len(latest) and "symbol" in latest.columns else 0
    ok_rows = int((panel["quality_status"] == "ok").sum()) if len(panel) else 0
    mainboard_ok = int(((panel["quality_status"] == "ok") & panel["trade_allowed_mainboard"].fillna(False).astype(bool)).sum()) if len(panel) else 0
    status_counts = report["quality_status"].fillna("missing_snapshot").value_counts().to_dict() if len(report) else {}
    return {
        "mode": getattr(args, "cmd", None),
        "trade_date": trade_date,
        "cutoff_time": args.cutoff_time,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_symbols": expected,
        "snapshot_raw_rows": int(len(snapshots)),
        "symbols_with_asof_snapshot": latest_symbols,
        "valid_panel_rows": ok_rows,
        "valid_panel_rate": ok_rows / expected if expected else None,
        "mainboard_valid_rows": mainboard_ok,
        "quality_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "collection_passed": bool(expected > 0 and ok_rows >= math.ceil(expected * args.min_valid_rate)),
        "min_valid_rate": float(args.min_valid_rate),
        "out_dir": str(output_day_dir(args.out_root, trade_date)),
        "notes": [
            "This is collection-only output. It is not Ch12 live features and not NN scores.",
            "Use full AS1455 universe for collection; apply mainboard trading filter after scoring.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect live quotes and build AS1455 raw live panel")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--universe", help="AS1455 universe CSV. If omitted, try standard saved_data paths.")
        sp.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
        sp.add_argument("--trade-date", help="YYYYMMDD; defaults to today")
        sp.add_argument("--cutoff-time", default=DEFAULT_CUTOFF_TIME, help="HH:MM[:SS], default 14:55:00")
        sp.add_argument("--max-symbols", type=int, help="Debug only: restrict universe to first N symbols")
        sp.add_argument("--include-raw", action="store_true", help="Keep raw Sina payload in snapshots_raw.csv")
        sp.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        sp.add_argument("--timeout-seconds", type=float, default=8.0)
        sp.add_argument("--batch-sleep-seconds", type=float, default=0.2)
        sp.add_argument("--max-snapshot-age-seconds", type=int, default=180)
        sp.add_argument("--min-valid-rate", type=float, default=0.98)

    sp = sub.add_parser("collect-once", help="Collect one quote snapshot round")
    add_common(sp)
    sp.add_argument("--finalize", action="store_true", help="Also build latest/panel/report after this round")

    sp = sub.add_parser("collect-loop", help="Collect repeatedly until --until, then finalize")
    add_common(sp)
    sp.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    sp.add_argument("--until", default="14:55:05", help="HH:MM[:SS] local server time")
    sp.set_defaults(finalize=True)

    sp = sub.add_parser("finalize", help="Build latest/panel/report from existing snapshots_raw.csv")
    add_common(sp)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "collect-once":
        collect_once(args)
    elif args.cmd == "collect-loop":
        collect_loop(args)
    elif args.cmd == "finalize":
        finalize_day(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
