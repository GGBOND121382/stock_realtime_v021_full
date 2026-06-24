#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the AS1455 live data/feature pipeline.

This module intentionally contains no model inference or trading logic.  It is
shared by:
  * historical AS1455 cache updater,
  * 09:35 live prepare,
  * 14:55 live quote collector,
  * live Ch12 feature builder.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_ROOT = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "live_as1455"
DEFAULT_CH12_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455"
DEFAULT_UNIVERSE_CANDIDATES = [
    DEFAULT_CH12_DIR / "as1455_model_universe_from_h5.csv",
    DEFAULT_CH12_DIR / "effective_universe.csv",
    DEFAULT_CH12_DIR / "universe" / "effective_universe.csv",
    DEFAULT_CH12_DIR / "universe" / "07_universe_allA_top1000_static.csv",
    DEFAULT_CH12_DIR / "07_universe_allA_top1000_static.csv",
    PROJECT_DIR / "data" / "universe" / "07_universe_allA_top1000_static.csv",
]

MONTH = 21
YEAR = 12 * MONTH
T_WINDOWS = [1, 5, 10, 21, 42, 63]
FWD_T = [1, 5, 21]
CUTOFF_HHMM = "14:55"
EXPECTED_MODEL_COLUMNS = [
    "dollar_vol", "dollar_vol_rank", "rsi", "bb_high", "bb_low",
    "NATR", "ATR", "PPO", "MACD", "sector",
    "r01", "r05", "r10", "r21", "r42", "r63",
    "r01dec", "r05dec", "r10dec", "r21dec", "r42dec", "r63dec",
    "r01q_sector", "r05q_sector", "r10q_sector", "r21q_sector", "r42q_sector", "r63q_sector",
    "year", "month", "weekday",
]

SYMBOL_COLUMNS = ["symbol", "code", "ts_code", "股票代码", "证券代码", "代码"]
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float(value) -> Optional[float]:
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


def parse_clock(value: str) -> dtime:
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        return dtime(int(parts[0]), int(parts[1]), 0)
    if len(parts) == 3:
        return dtime(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"invalid clock time: {value!r}")


def parse_trade_date(value: str | None) -> str:
    if value is None or str(value).strip().lower() == "today":
        return datetime.now().strftime("%Y%m%d")
    s = str(value).strip().replace("-", "")
    if not re.fullmatch(r"\d{8}", s):
        raise ValueError(f"invalid trade date {value!r}; use YYYYMMDD or YYYY-MM-DD")
    return s


def yyyymmdd_to_dash(value: str) -> str:
    s = str(value).replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def dash_to_yyyymmdd(value: str) -> str:
    return str(value).replace("-", "")[:8]


def normalize_symbol(value: str) -> str:
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        raise ValueError(f"invalid symbol: {value!r}")
    if "." in s:
        a, b = s.split(".", 1)
        if a.isalpha() and re.sub(r"\D", "", b)[:6]:
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


def symbol_code(symbol: str) -> str:
    return normalize_symbol(symbol).split(".", 1)[0]


def baostock_code(symbol: str) -> str:
    s = normalize_symbol(symbol)
    code, market = s.split(".", 1)
    return f"{market.lower()}.{code}"


def sina_code(symbol: str) -> str:
    s = normalize_symbol(symbol)
    code, market = s.split(".", 1)
    return f"{market.lower()}{code}"


def infer_mainboard(row: pd.Series) -> bool:
    board_text = " ".join(str(row.get(c, "")) for c in BOARD_COLUMNS if c in row.index).lower()
    symbol = str(row.get("symbol", ""))
    code = symbol[:6]
    if any(x in board_text for x in ["chinext", "创业", "star", "科创", "北交", "bj"]):
        return False
    if any(x in board_text for x in ["main", "主板", "sh_mainboard", "sz_mainboard"]):
        return True
    return code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def load_universe(path: str | Path | None = None, max_symbols: int | None = None) -> pd.DataFrame:
    if path:
        p = Path(path)
    else:
        p = next((x for x in DEFAULT_UNIVERSE_CANDIDATES if x.exists()), None)
        if p is None:
            candidates = "\n".join(str(x) for x in DEFAULT_UNIVERSE_CANDIDATES)
            raise FileNotFoundError("universe file not found; pass --universe. Tried:\n" + candidates)
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
    out["industry"] = out[industry_col] if industry_col else "unknown"
    out["is_mainboard"] = out.apply(infer_mainboard, axis=1)
    out["trade_allowed_mainboard"] = out["is_mainboard"]
    keep = ["symbol", "code", "exchange", "name", "board", "industry", "is_mainboard", "trade_allowed_mainboard"]
    rest = [c for c in out.columns if c not in keep]
    out = out[keep + rest].drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    if max_symbols is not None:
        out = out.head(int(max_symbols)).copy()
    return out


def read_5m_csv(path: Path, symbol: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str, "symbol": str}, encoding="utf-8-sig", low_memory=False)
    if "symbol" not in df.columns or df["symbol"].isna().all():
        if symbol is None:
            m = re.search(r"(\d{6})", path.name)
            symbol = normalize_symbol(m.group(1)) if m else ""
        df["symbol"] = normalize_symbol(symbol)
    else:
        df["symbol"] = df["symbol"].map(normalize_symbol)
    if "datetime" not in df.columns:
        if "time" in df.columns:
            t = df["time"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 14)
            df["datetime"] = pd.to_datetime(t, format="%Y%m%d%H%M%S", errors="coerce")
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise ValueError(f"5m file has neither datetime nor time/date: {path}")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "date" not in df.columns:
        df["date"] = df["datetime"].dt.normalize()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["datetime", "date"])


def aggregate_as1455_from_5m(df: pd.DataFrame, symbol: str, start_date: str | None = None, end_date: str | None = None, cutoff: str = CUTOFF_HHMM) -> pd.DataFrame:
    bars = df.copy()
    if start_date:
        bars = bars[bars["date"] >= pd.Timestamp(start_date)]
    if end_date:
        bars = bars[bars["date"] <= pd.Timestamp(end_date)]
    if bars.empty:
        return pd.DataFrame()
    cutoff_time = pd.Timestamp(cutoff).time()
    price_cols = ["open", "high", "low", "close"]
    valid_price = bars[price_cols].notna().all(axis=1) & bars[price_cols].gt(0).all(axis=1)
    bars = bars.loc[valid_price].copy()
    bars = bars[bars["datetime"].dt.time <= cutoff_time]
    if bars.empty:
        return pd.DataFrame()
    bars.sort_values("datetime", inplace=True)
    bars["has_14_55_bar"] = bars["datetime"].dt.strftime("%H:%M").eq(cutoff)
    daily = bars.groupby("date", sort=False, observed=True).agg(
        raw_open_as1455=("open", "first"),
        raw_high_as1455=("high", "max"),
        raw_low_as1455=("low", "min"),
        raw_close_as1455=("close", "last"),
        raw_volume_as1455=("volume", "sum"),
        raw_amount_as1455=("amount", "sum"),
        max_datetime_used=("datetime", "last"),
        has_14_55_bar=("has_14_55_bar", "any"),
    ).reset_index()
    daily.insert(0, "symbol", normalize_symbol(symbol))
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily["last_bar_time"] = pd.to_datetime(daily["max_datetime_used"]).dt.strftime("%H:%M")
    daily["used_after_cutoff"] = pd.to_datetime(daily["max_datetime_used"]).dt.time.gt(cutoff_time)
    return daily


def merge_dedup_csv(existing_path: Path, new_df: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    if existing_path.exists() and existing_path.stat().st_size > 0:
        try:
            old = pd.read_csv(existing_path, dtype={"symbol": str, "code": str}, encoding="utf-8-sig")
            df = pd.concat([old, new_df], ignore_index=True, sort=False)
        except Exception:
            df = new_df.copy()
    else:
        df = new_df.copy()
    for c in subset:
        if c not in df.columns:
            df[c] = ""
    df = df.drop_duplicates(list(subset), keep="last")
    if "date" in df.columns:
        df = df.sort_values(["symbol", "date"] if "symbol" in df.columns else ["date"])
    return df.reset_index(drop=True)


def raw_daily_path(cache_dir: Path, symbol: str) -> Path:
    code = symbol_code(symbol)
    candidates = [cache_dir / f"{code}_daily_raw.csv", cache_dir / f"{code}_raw_daily.csv", cache_dir / f"{code}.csv"]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def raw_5m_path(cache_dir: Path, symbol: str) -> Path:
    code = symbol_code(symbol)
    candidates = [cache_dir / f"{code}_5m_raw.csv", cache_dir / f"{code}_raw_5m.csv", cache_dir / f"{code}.csv"]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def as1455_daily_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol_code(symbol)}_as1455_daily.csv"


def get_last_cached_date(path: Path, date_col: str = "date") -> Optional[pd.Timestamp]:
    """Return the latest cached calendar date from a local CSV cache.

    Historical cache files in this project are not perfectly uniform:

    - BaoStock daily/raw AS1455 caches usually have ``date``.
    - Existing 5m caches produced by earlier scripts may have ``trade_date``
      plus ``datetime`` instead of ``date``.
    - BaoStock native 5m rows may have ``time`` in ``YYYYMMDDHHMMSS...`` form.

    The history updater uses this function to decide whether a symbol needs
    more downloads.  If we only look for ``date``, old 5m caches are falsely
    treated as empty and the updater tries to refetch many years of 5m bars.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None

    candidate_cols = []
    for c in [date_col, "date", "trade_date", "datetime", "time"]:
        if c not in candidate_cols:
            candidate_cols.append(c)

    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        available = [c for c in candidate_cols if c in header.columns]
        if not available:
            return None
        df = pd.read_csv(
            path,
            usecols=available,
            dtype={c: str for c in available},
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception:
        return None

    dates = pd.Series(dtype="datetime64[ns]")
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    elif "trade_date" in df.columns:
        dates = pd.to_datetime(df["trade_date"], errors="coerce")
    elif "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"], errors="coerce")
    elif "time" in df.columns:
        t = df["time"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
        dates = pd.to_datetime(t, format="%Y%m%d", errors="coerce")

    dates = pd.to_datetime(dates, errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def request_sina_batch(symbols: Sequence[str], timeout: float = 8.0) -> dict[str, str]:
    if requests is None:
        raise RuntimeError("requests is not installed")
    market_symbols = [sina_code(s) for s in symbols]
    if not market_symbols:
        return {}
    url = "https://hq.sinajs.cn/list=" + ",".join(market_symbols)
    headers = {"Referer": "https://finance.sina.com.cn/", "User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    text = resp.content.decode("gbk", errors="replace")
    out: dict[str, str] = {}
    pattern = re.compile(r'var hq_str_(s[hz]\d{6})="(.*?)";')
    for m in pattern.finditer(text):
        out[m.group(1)] = m.group(2)
    return out


def infer_volume_unit(volume_raw: Optional[float], amount_raw: Optional[float], price: Optional[float]) -> tuple[str, Optional[float], Optional[float], Optional[float]]:
    if volume_raw is None or amount_raw is None or price is None or volume_raw <= 0 or amount_raw <= 0 or price <= 0:
        return "unknown", None, volume_raw, None
    ratio = amount_raw / volume_raw / price
    if 0.2 <= ratio <= 5.0:
        return "shares", 1.0, volume_raw, ratio
    if 20 <= ratio <= 500:
        return "lots", 100.0, volume_raw * 100.0, ratio
    return "unknown", 1.0, volume_raw, ratio


def parse_sina_payload(symbol: str, payload: str, collected_at: datetime) -> dict:
    symbol = normalize_symbol(symbol)
    code, exch = symbol.split(".")
    fields = payload.split(",") if payload is not None else []
    status = "ok" if len(fields) >= 32 and fields[0] else "empty_payload"
    name = fields[0] if len(fields) > 0 else ""
    open_px = safe_float(fields[1]) if len(fields) > 1 else None
    prev_close = safe_float(fields[2]) if len(fields) > 2 else None
    last_price = safe_float(fields[3]) if len(fields) > 3 else None
    high = safe_float(fields[4]) if len(fields) > 4 else None
    low = safe_float(fields[5]) if len(fields) > 5 else None
    bid = safe_float(fields[6]) if len(fields) > 6 else None
    ask = safe_float(fields[7]) if len(fields) > 7 else None
    volume_raw = safe_float(fields[8]) if len(fields) > 8 else None
    amount_raw = safe_float(fields[9]) if len(fields) > 9 else None
    bid_volume_1 = safe_float(fields[10]) if len(fields) > 10 else None
    bid_price_1 = safe_float(fields[11]) if len(fields) > 11 else None
    ask_volume_1 = safe_float(fields[20]) if len(fields) > 20 else None
    ask_price_1 = safe_float(fields[21]) if len(fields) > 21 else None
    source_trade_date = fields[30] if len(fields) > 30 else None
    source_trade_time = fields[31] if len(fields) > 31 else None
    quote_dt = None
    if source_trade_date and source_trade_time:
        q = pd.to_datetime(f"{source_trade_date} {source_trade_time}", errors="coerce")
        if pd.notna(q):
            quote_dt = q.strftime("%Y-%m-%d %H:%M:%S")
    pct_chg = None
    if last_price is not None and prev_close is not None and prev_close > 0:
        pct_chg = last_price / prev_close - 1.0
    unit, multiplier, volume_shares, vwap_ratio = infer_volume_unit(volume_raw, amount_raw, last_price)
    required = {
        "last_price": last_price,
        "open": open_px,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "volume": volume_shares,
        "amount": amount_raw,
    }
    missing = [k for k, v in required.items() if v is None or (isinstance(v, float) and (not math.isfinite(v) or v <= 0))]
    core_complete = status == "ok" and not missing
    return {
        "symbol": symbol,
        "code": code,
        "exchange": exch,
        "name": name,
        "trade_date": collected_at.strftime("%Y%m%d"),
        "source_trade_date": source_trade_date,
        "source_trade_time": source_trade_time,
        "quote_datetime": quote_dt,
        "collected_at": collected_at.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "sina_targeted",
        "source_status": status,
        "last_price": last_price,
        "open": open_px,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "volume_raw": volume_raw,
        "amount_raw": amount_raw,
        "volume_unit_inferred": unit,
        "volume_unit_multiplier": multiplier,
        "volume_shares": volume_shares,
        "vwap_ratio_raw": vwap_ratio,
        "pct_chg": pct_chg,
        "bid_price_1": bid_price_1,
        "ask_price_1": ask_price_1,
        "bid_volume_1": bid_volume_1,
        "ask_volume_1": ask_volume_1,
        "core_complete": core_complete,
        "missing_core_fields": ",".join(missing),
        "raw_payload": payload,
    }


def collect_sina_quotes(universe: pd.DataFrame, batch_size: int = 250, timeout: float = 8.0, batch_sleep: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    symbols = universe["symbol"].tolist()
    collected_at = datetime.now()
    for batch_idx, start in enumerate(range(0, len(symbols), int(batch_size)), 1):
        batch = symbols[start:start + int(batch_size)]
        try:
            payloads = request_sina_batch(batch, timeout=timeout)
            for symbol in batch:
                payload = payloads.get(sina_code(symbol), "")
                rows.append(parse_sina_payload(symbol, payload, collected_at))
        except Exception as exc:
            errors.append({"batch_idx": batch_idx, "n_symbols": len(batch), "error": f"{type(exc).__name__}: {exc}"})
            for symbol in batch:
                rows.append(parse_sina_payload(symbol, "", collected_at))
        if batch_sleep > 0 and start + batch_size < len(symbols):
            time.sleep(batch_sleep)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.merge(universe[["symbol", "board", "industry", "is_mainboard", "trade_allowed_mainboard"]], on="symbol", how="left")
    return df, pd.DataFrame(errors)


def select_latest_asof_snapshot(snapshots: pd.DataFrame, cutoff_time: str = "14:55:00") -> pd.DataFrame:
    if snapshots.empty:
        return snapshots.copy()
    df = snapshots.copy()
    cutoff = parse_clock(cutoff_time)
    if "quote_datetime" in df.columns:
        qdt = pd.to_datetime(df["quote_datetime"], errors="coerce")
    else:
        qdt = pd.Series(pd.NaT, index=df.index)
    if qdt.isna().all():
        qdt = pd.to_datetime(df["collected_at"], errors="coerce")
        df["source_time_uncertain"] = True
    else:
        df["source_time_uncertain"] = qdt.isna()
        qdt = qdt.fillna(pd.to_datetime(df["collected_at"], errors="coerce"))
    df["_quote_dt"] = qdt
    df["_time"] = qdt.dt.time
    df = df[(df["_time"] <= cutoff) & (df["core_complete"].astype(bool))].copy()
    if df.empty:
        return df.drop(columns=[c for c in ["_quote_dt", "_time"] if c in df.columns], errors="ignore")
    df.sort_values(["symbol", "_quote_dt", "collected_at"], inplace=True)
    latest = df.groupby("symbol", as_index=False, sort=False).tail(1).copy()
    return latest.drop(columns=["_quote_dt", "_time"], errors="ignore").sort_values("symbol").reset_index(drop=True)


def snapshots_to_raw_panel(latest: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    rows = []
    t_dash = yyyymmdd_to_dash(trade_date)
    now = datetime.now()
    for _, r in latest.iterrows():
        qdt = pd.to_datetime(r.get("quote_datetime"), errors="coerce")
        if pd.isna(qdt):
            qdt = pd.to_datetime(r.get("collected_at"), errors="coerce")
        age = None if pd.isna(qdt) else max(0.0, (now - qdt.to_pydatetime()).total_seconds())
        quality = "ok"
        miss = str(r.get("missing_core_fields", "") or "")
        try:
            lo = float(r["low"]); hi = float(r["high"]); op = float(r["open"]); cl = float(r["last_price"])
            if not (lo <= op <= hi and lo <= cl <= hi):
                quality = "price_order_invalid"
        except Exception:
            quality = "invalid_numeric"
        if miss:
            quality = "missing_core_fields"
        rows.append({
            "symbol": r.get("symbol"),
            "date": t_dash,
            "name": r.get("name", ""),
            "exchange": r.get("exchange", ""),
            "board": r.get("board", ""),
            "industry": r.get("industry", ""),
            "raw_open_as1455": r.get("open"),
            "raw_high_as1455": r.get("high"),
            "raw_low_as1455": r.get("low"),
            "raw_close_as1455": r.get("last_price"),
            "raw_volume_as1455": r.get("volume_shares"),
            "raw_amount_as1455": r.get("amount_raw"),
            "live_preclose": r.get("prev_close"),
            "snapshot_time": r.get("quote_datetime") or r.get("collected_at"),
            "snapshot_age_seconds": age,
            "source_used": r.get("source"),
            "source_status": r.get("source_status"),
            "core_complete": r.get("core_complete"),
            "quality_status": quality,
            "missing_core_fields": miss,
            "volume_unit_inferred": r.get("volume_unit_inferred"),
            "volume_unit_multiplier": r.get("volume_unit_multiplier"),
            "vwap_ratio_raw": r.get("vwap_ratio_raw"),
            "is_mainboard": r.get("is_mainboard"),
            "trade_allowed_mainboard": r.get("trade_allowed_mainboard"),
        })
    return pd.DataFrame(rows, columns=RAW_PANEL_COLUMNS)


def zscore(x: pd.Series) -> pd.Series:
    std = x.std()
    if std is None or not np.isfinite(std) or std == 0:
        return x * np.nan
    return (x - x.mean()) / std


def qcut_safe(x: pd.Series, q: int) -> pd.Series:
    valid = x.dropna()
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    if valid.nunique() < 2:
        return out
    try:
        out.loc[valid.index] = pd.qcut(valid.rank(method="first"), q=q, labels=False, duplicates="drop")
    except Exception:
        out.loc[valid.index] = np.nan
    return out


def qcut_by_group(values: pd.Series, groupers, q: int) -> pd.Series:
    return values.groupby(groupers, group_keys=False).apply(lambda x: qcut_safe(x, q))


def compute_ch12_features(prices: pd.DataFrame, universe_meta: pd.DataFrame, include_forward_labels: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the AS1455 Ch12 feature panel.

    Input `prices` must have MultiIndex (date, symbol) and columns open/high/low/close/volume.
    The implementation mirrors the patched offline AS1455 builder.  TA-Lib is
    imported inside the function so that non-feature commands can run without it.
    """
    import talib
    from talib import ATR, BBANDS, MACD, RSI

    prices = prices.copy()
    if not prices.index.is_monotonic_increasing:
        prices.sort_index(inplace=True)
    meta = universe_meta.copy()
    if "code" not in meta.columns:
        meta["code"] = meta["symbol"].map(lambda s: normalize_symbol(s)[:6])
    if "industry" not in meta.columns:
        meta["industry"] = "unknown"
    meta["sector"] = pd.factorize(meta["industry"].fillna("unknown"))[0].astype(int)
    sector_map = meta.set_index("code")["sector"]

    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").div(1e3)
    prices["dollar_vol"] = prices["close"].mul(prices["volume"]).div(1e3)
    dollar_vol_ma = prices["dollar_vol"].unstack("symbol").rolling(window=MONTH, min_periods=1).mean()
    prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().swaplevel()

    prices["rsi"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(RSI)

    def compute_bb(close: pd.Series) -> pd.DataFrame:
        upper, _mid, lower = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": upper, "bb_low": lower}, index=close.index)

    bb = prices.groupby(level="symbol", group_keys=False)["close"].apply(compute_bb)
    prices["bb_high"] = bb["bb_high"].sub(prices["close"]).div(bb["bb_high"]).apply(np.log1p)
    prices["bb_low"] = prices["close"].sub(bb["bb_low"]).div(prices["close"]).apply(np.log1p)

    def compute_natr(g: pd.DataFrame) -> pd.Series:
        return pd.Series(talib.NATR(g["high"], g["low"], g["close"]), index=g.index)

    prices["NATR"] = pd.concat([compute_natr(g) for _s, g in prices.groupby(level="symbol", sort=False)]).reindex(prices.index)

    def compute_atr(g: pd.DataFrame) -> pd.Series:
        return zscore(ATR(g["high"], g["low"], g["close"], timeperiod=14))

    prices["ATR"] = pd.concat([compute_atr(g) for _s, g in prices.groupby(level="symbol", sort=False)]).reindex(prices.index)
    prices["PPO"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(talib.PPO)

    def compute_macd(close: pd.Series) -> pd.Series:
        return zscore(MACD(close)[0])

    prices["MACD"] = prices.groupby(level="symbol", group_keys=False)["close"].apply(compute_macd)
    prices["sector"] = prices.index.get_level_values("symbol").astype(str).str.slice(0, 6).map(sector_map).fillna(-1).astype(int)

    by_symbol_close = prices.groupby(level="symbol")["close"]
    for t in T_WINDOWS:
        prices[f"r{t:02}"] = by_symbol_close.pct_change(t)

    dates = prices.index.get_level_values("date")
    for t in T_WINDOWS:
        prices[f"r{t:02}dec"] = qcut_by_group(prices[f"r{t:02}"], dates, 10)
    for t in T_WINDOWS:
        prices[f"r{t:02}q_sector"] = qcut_by_group(prices[f"r{t:02}"], [dates, prices["sector"]], 5)
    if include_forward_labels:
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
