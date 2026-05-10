#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect AKShare L1 / five-level quote data for selected stocks and ETFs.

The collector is a temporary daily inference layer for days not yet covered by
data88. It intentionally writes English column names and keeps AKShare data
under a separate pending directory so vendor data can later replace it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"

DEFAULT_OUT_DIR = SAVED_DATA_DIR / "akshare_realtime_cache"
DEFAULT_WATCHLIST = Path("PurchasedData/selected_watchlist.txt")
DEFAULT_INTERVAL_SECONDS = 30
TRADING_WINDOWS = (
    (dtime(9, 15), dtime(11, 30, 30)),
    (dtime(13, 0), dtime(15, 0, 30)),
)
SOURCE_HOSTS = {
    "em": ["push2.eastmoney.com", "70.push2.eastmoney.com"],
    "eastmoney": ["push2.eastmoney.com", "70.push2.eastmoney.com"],
    "sina": ["vip.stock.finance.sina.com.cn"],
    "tx": ["qt.gtimg.cn"],
    "tencent": ["qt.gtimg.cn"],
    "ths": ["q.10jqka.com.cn"],
    "xq": ["stock.xueqiu.com"],
}


@dataclass
class Snapshot:
    vendor: str
    source: str
    quote_source: str
    collected_at: str
    symbol: str
    exchange: str
    trade_date: str
    trade_time: str
    phase: str
    name: Optional[str] = None
    last_price: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    volume: Optional[float] = None
    amount: Optional[float] = None
    turnover: Optional[float] = None
    pct_chg: Optional[float] = None
    bid_price_1: Optional[float] = None
    bid_price_2: Optional[float] = None
    bid_price_3: Optional[float] = None
    bid_price_4: Optional[float] = None
    bid_price_5: Optional[float] = None
    ask_price_1: Optional[float] = None
    ask_price_2: Optional[float] = None
    ask_price_3: Optional[float] = None
    ask_price_4: Optional[float] = None
    ask_price_5: Optional[float] = None
    bid_volume_1: Optional[float] = None
    bid_volume_2: Optional[float] = None
    bid_volume_3: Optional[float] = None
    bid_volume_4: Optional[float] = None
    bid_volume_5: Optional[float] = None
    ask_volume_1: Optional[float] = None
    ask_volume_2: Optional[float] = None
    ask_volume_3: Optional[float] = None
    ask_volume_4: Optional[float] = None
    ask_volume_5: Optional[float] = None
    spread_1: Optional[float] = None
    mid_price_1: Optional[float] = None
    depth_bid_5: Optional[float] = None
    depth_ask_5: Optional[float] = None
    depth_imbalance_1: Optional[float] = None
    depth_imbalance_5: Optional[float] = None
    weighted_bid_price_5: Optional[float] = None
    weighted_ask_price_5: Optional[float] = None
    bid_ask_error: Optional[str] = None
    raw_json: Optional[str] = None


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    p2 = script_dir() / p
    return p2 if p2.exists() else p


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "." in s:
        code, market = s.split(".", 1)
        return f"{code.zfill(6)}.{market}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    if not code:
        raise ValueError(f"invalid symbol: {symbol!r}")
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def ak_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).split(".", 1)[0]


def sina_symbol(symbol: str) -> str:
    s = normalize_symbol(symbol)
    code, market = s.split(".")
    return f"{market.lower()}{code}"


def read_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if getattr(args, "symbols", None):
        symbols.extend(t.strip() for t in args.symbols.replace(";", ",").split(",") if t.strip())
    path_arg = getattr(args, "symbols_file", None)
    if not symbols and not path_arg:
        path_arg = str(DEFAULT_WATCHLIST)
    if path_arg:
        path = resolve_path(path_arg)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                symbols.append(token)
    symbols = [normalize_symbol(s) for s in symbols]
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    if getattr(args, "max_symbols", None):
        out = out[: int(args.max_symbols)]
    if not out:
        raise ValueError("provide --symbols or --symbols-file")
    return out


def to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "None", "nan", "NaN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def find_value(mapping: Dict[str, object], names: Sequence[str]) -> Optional[object]:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_akshare():
    try:
        import akshare as ak
    except Exception as exc:
        raise RuntimeError(f"akshare import failed: {type(exc).__name__}: {exc}") from exc
    return ak


def phase_of(dt: datetime, symbol: str = "") -> str:
    t = dt.time()
    market = normalize_symbol(symbol).split(".")[1] if symbol else ""
    if dtime(9, 15) <= t < dtime(9, 20):
        return "pre_open_auction_cancelable"
    if dtime(9, 20) <= t < dtime(9, 25):
        return "pre_open_auction_locked"
    if dtime(9, 25) <= t < dtime(9, 30):
        return "pre_open_auction_match_pause"
    if dtime(9, 30) <= t <= dtime(11, 30):
        return "continuous_am"
    if dtime(13, 0) <= t < dtime(14, 57):
        return "continuous_pm"
    if dtime(14, 57) <= t <= dtime(15, 0, 30):
        return "close_auction" if market == "SZ" else "close_phase"
    return "outside_trading"


def in_trading_window(now: datetime) -> bool:
    t = now.time()
    return any(start <= t <= end for start, end in TRADING_WINDOWS)


def row_mapping(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {}
    if len(df.columns) >= 2:
        cols = set(map(str, df.columns))
        if {"item", "value"}.issubset(cols):
            return dict(zip(df["item"].astype(str), df["value"]))
        if {"项目", "值"}.issubset(cols):
            return dict(zip(df["项目"].astype(str), df["值"]))
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))
    return {}


def fetch_spot_map(ak, symbols: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Fetch bulk A-share and ETF spot maps when available."""
    wanted = {ak_symbol(s) for s in symbols}
    result: Dict[str, Dict[str, object]] = {}
    funcs = ["stock_zh_a_spot_em", "fund_etf_spot_em"]
    for fname in funcs:
        fn = getattr(ak, fname, None)
        if fn is None:
            continue
        try:
            df = fn()
        except Exception:
            continue
        if df is None or df.empty:
            continue
        code_col = next((c for c in ["代码", "symbol", "code"] if c in df.columns), None)
        if code_col is None:
            continue
        tmp = df.copy()
        tmp[code_col] = tmp[code_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        tmp = tmp[tmp[code_col].isin(wanted)]
        for code, row in tmp.set_index(code_col).iterrows():
            result[normalize_symbol(code)] = row.to_dict()
    return result


def fetch_sina_spot_map(ak, symbols: Sequence[str]) -> Dict[str, Dict[str, object]]:
    wanted = {ak_symbol(s) for s in symbols}
    try:
        df = ak.stock_zh_a_spot()
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    code_col = next((c for c in ["代码", "code", "symbol"] if c in df.columns), None)
    if code_col is None:
        return {}
    tmp = df.copy()
    tmp[code_col] = tmp[code_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    tmp = tmp[tmp[code_col].isin(wanted)]
    return {normalize_symbol(code): row.to_dict() for code, row in tmp.set_index(code_col).iterrows()}


def fetch_ths_etf_spot_map(ak, symbols: Sequence[str]) -> Dict[str, Dict[str, object]]:
    wanted = {ak_symbol(s) for s in symbols}
    try:
        df = ak.fund_etf_spot_ths()
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    code_col = next((c for c in ["基金代码", "代码", "code", "symbol"] if c in df.columns), None)
    if code_col is None:
        return {}
    tmp = df.copy()
    tmp[code_col] = tmp[code_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    tmp = tmp[tmp[code_col].isin(wanted)]
    return {normalize_symbol(code): row.to_dict() for code, row in tmp.set_index(code_col).iterrows()}


def fetch_xq_spot_one(ak, symbol: str) -> Dict[str, object]:
    try:
        df = ak.stock_individual_spot_xq(symbol=sina_symbol(symbol).upper())
    except Exception:
        return {}
    return row_mapping(df)


def merge_spot_maps(ak, symbols: Sequence[str], priority: Sequence[str]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {s: {} for s in symbols}
    for source in priority:
        source = source.strip().lower()
        if not source:
            continue
        if source == "sina":
            got = fetch_sina_spot_map(ak, symbols)
        elif source == "ths":
            got = fetch_ths_etf_spot_map(ak, symbols)
        elif source == "xq":
            got = {s: fetch_xq_spot_one(ak, s) for s in symbols}
        elif source == "em":
            got = fetch_spot_map(ak, symbols)
        else:
            got = {}
        for sym, mapping in got.items():
            if mapping and not result.get(sym):
                result[sym] = mapping
    return result


def derive_snapshot(snap: Snapshot) -> None:
    bp = [getattr(snap, f"bid_price_{i}") for i in range(1, 6)]
    ap = [getattr(snap, f"ask_price_{i}") for i in range(1, 6)]
    bv = [getattr(snap, f"bid_volume_{i}") or 0.0 for i in range(1, 6)]
    av = [getattr(snap, f"ask_volume_{i}") or 0.0 for i in range(1, 6)]
    if bp[0] and ap[0]:
        snap.spread_1 = ap[0] - bp[0]
        snap.mid_price_1 = (ap[0] + bp[0]) / 2.0
    snap.depth_bid_5 = sum(bv)
    snap.depth_ask_5 = sum(av)
    if (bv[0] + av[0]) > 0:
        snap.depth_imbalance_1 = (bv[0] - av[0]) / (bv[0] + av[0])
    if (snap.depth_bid_5 + snap.depth_ask_5) > 0:
        snap.depth_imbalance_5 = (snap.depth_bid_5 - snap.depth_ask_5) / (snap.depth_bid_5 + snap.depth_ask_5)
    if sum(bv) > 0:
        snap.weighted_bid_price_5 = sum((bp[i] or 0.0) * bv[i] for i in range(5)) / sum(bv)
    if sum(av) > 0:
        snap.weighted_ask_price_5 = sum((ap[i] or 0.0) * av[i] for i in range(5)) / sum(av)


def fetch_bid_ask_map(
    ak,
    symbol: str,
    allow_em: bool,
    retries: int = 1,
    retry_sleep: float = 0.5,
) -> tuple[Dict[str, object], Optional[str], str]:
    if not allow_em:
        return {}, "five-level quote disabled: EM fallback is disabled", "none"
    attempts = max(1, int(retries))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            mapping = row_mapping(ak.stock_bid_ask_em(symbol=ak_symbol(symbol)))
            if mapping:
                return mapping, None, "eastmoney_stock_bid_ask_em"
            last_error = "empty response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts and retry_sleep > 0:
            time.sleep(retry_sleep)
    return {}, last_error, "eastmoney_stock_bid_ask_em"


def fetch_one_snapshot(
    ak,
    symbol: str,
    spot: Dict[str, object],
    include_raw: bool,
    allow_em_bid_ask: bool,
    collected_at: Optional[datetime] = None,
    bid_ask_retries: int = 1,
    retry_sleep: float = 0.5,
) -> Snapshot:
    now = collected_at or datetime.now()
    bid_ask_map, bid_ask_error, quote_source = fetch_bid_ask_map(
        ak, symbol, allow_em_bid_ask, bid_ask_retries, retry_sleep
    )

    def value(*names: str):
        v = find_value(bid_ask_map, names)
        return v if v is not None else find_value(spot, names)

    symbol_n = normalize_symbol(symbol)
    code, market = symbol_n.split(".")
    snap = Snapshot(
        vendor="akshare",
        source="preferred_spot_sources+optional_stock_bid_ask_em",
        quote_source=quote_source,
        collected_at=now.isoformat(timespec="seconds"),
        symbol=symbol_n,
        exchange=market,
        trade_date=now.strftime("%Y%m%d"),
        trade_time=now.strftime("%H%M%S"),
        phase=phase_of(now, symbol_n),
        name=value("股票简称", "名称", "name"),
        last_price=to_float(value("最新", "最新价", "现价", "最新报价")),
        open=to_float(value("今开", "开盘", "开盘价")),
        high=to_float(value("最高", "最高价")),
        low=to_float(value("最低", "最低价")),
        prev_close=to_float(value("昨收", "昨收价")),
        volume=to_float(value("成交量", "总手", "总量")),
        amount=to_float(value("成交额", "金额")),
        turnover=to_float(value("换手率")),
        pct_chg=to_float(value("涨跌幅", "涨幅")),
        bid_ask_error=bid_ask_error,
        raw_json=json.dumps({"spot": spot, "bid_ask": bid_ask_map}, ensure_ascii=False) if include_raw else None,
    )
    for i in range(1, 6):
        setattr(snap, f"bid_price_{i}", to_float(value(f"buy_{i}", f"买{i}", f"买{i}价", f"买盘{i}")))
        setattr(snap, f"ask_price_{i}", to_float(value(f"sell_{i}", f"卖{i}", f"卖{i}价", f"卖盘{i}")))
        setattr(snap, f"bid_volume_{i}", to_float(value(f"buy_{i}_vol", f"买{i}量", f"买盘{i}量")))
        setattr(snap, f"ask_volume_{i}", to_float(value(f"sell_{i}_vol", f"卖{i}量", f"卖盘{i}量")))
    # Some Eastmoney responses use "买一"/"买一量" style.
    cn_num = ["一", "二", "三", "四", "五"]
    for i, cn in enumerate(cn_num, start=1):
        if getattr(snap, f"bid_price_{i}") is None:
            setattr(snap, f"bid_price_{i}", to_float(value(f"买{cn}", f"买{cn}价")))
        if getattr(snap, f"ask_price_{i}") is None:
            setattr(snap, f"ask_price_{i}", to_float(value(f"卖{cn}", f"卖{cn}价")))
        if getattr(snap, f"bid_volume_{i}") is None:
            setattr(snap, f"bid_volume_{i}", to_float(value(f"买{cn}量")))
        if getattr(snap, f"ask_volume_{i}") is None:
            setattr(snap, f"ask_volume_{i}", to_float(value(f"卖{cn}量")))
    derive_snapshot(snap)
    return snap


def symbol_dir(out_dir: Path, date: str, symbol: str) -> Path:
    return out_dir / "pending" / date / normalize_symbol(symbol)


def append_csv(path: Path, row: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def append_snapshot(out_dir: Path, snap: Snapshot) -> Path:
    path = symbol_dir(out_dir, snap.trade_date, snap.symbol) / "snapshot_5level.csv"
    append_csv(path, pd.DataFrame([asdict(snap)]))
    return path


def has_bid_ask(snap: Snapshot) -> bool:
    return snap.bid_price_1 is not None and snap.ask_price_1 is not None


def has_l1_price(snap: Snapshot) -> bool:
    return snap.last_price is not None


def append_snapshot_error(out_dir: Path, snap: Snapshot, reason: str) -> Path:
    path = symbol_dir(out_dir, snap.trade_date, snap.symbol) / "snapshot_errors.csv"
    row = asdict(snap)
    row["reject_reason"] = reason
    append_csv(path, pd.DataFrame([row]))
    return path


def collect_once(args: argparse.Namespace) -> None:
    symbols = read_symbols(args)
    out_dir = Path(args.out_dir)
    ak = load_akshare()
    spot_priority = [x.strip() for x in args.spot_source_priority.split(",") if x.strip()]
    spot_map = {} if args.no_spot_bulk else merge_spot_maps(ak, symbols, spot_priority)
    written: List[Path] = []
    batch_time = datetime.now()
    if getattr(args, "strict_trading_hours", False) and not in_trading_window(batch_time):
        print(f"skip collect-once outside trading window: {batch_time.strftime('%H:%M:%S')}")
        return
    end_at = getattr(args, "_end_at", None)
    for idx, symbol in enumerate(symbols, start=1):
        if end_at and datetime.now().time() > end_at:
            print(f"stop symbol scan: current time {datetime.now().strftime('%H:%M:%S')} > loop end")
            break
        snap = fetch_one_snapshot(
            ak,
            symbol,
            spot_map.get(symbol, {}),
            args.include_raw,
            not args.disable_em_bid_ask,
            collected_at=batch_time,
            bid_ask_retries=args.bid_ask_retries,
            retry_sleep=args.retry_sleep,
        )
        if has_bid_ask(snap) or (args.allow_l1_only and has_l1_price(snap)):
            path = append_snapshot(out_dir, snap)
            status = "wrote"
        else:
            reason = snap.bid_ask_error or "missing bid/ask"
            path = append_snapshot_error(out_dir, snap, reason)
            status = "rejected"
        written.append(path)
        if args.per_symbol_delay > 0 and idx < len(symbols):
            time.sleep(args.per_symbol_delay)
        print(f"{status} snapshot {idx}/{len(symbols)} {symbol} {snap.trade_date} {snap.trade_time} {snap.phase}")
    write_manifest(out_dir, symbols, written, "collect-once")


def normalize_intraday_df(
    df: pd.DataFrame,
    symbol: str,
    source: str,
    trade_date: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        sc = str(c)
        sc_lower = sc.lower()
        if sc_lower in {"ticktime", "tick_time", "trade_time", "datetime", "date_time"} or sc in {"时间", "成交时间"}:
            rename[c] = "time"
            continue
        if sc_lower in {"kind", "side", "bs"} or sc in {"性质", "买卖盘性质"}:
            rename[c] = "side"
            continue
        if sc in {"时间", "成交时间", "time"}:
            rename[c] = "time"
        elif sc in {"成交价", "价格", "price"}:
            rename[c] = "price"
        elif sc in {"手数", "成交量", "volume"}:
            rename[c] = "volume"
        elif sc in {"成交额", "amount"}:
            rename[c] = "amount"
        elif sc in {"性质", "买卖盘性质", "side", "BS"}:
            rename[c] = "side"
    if source == "stock_intraday_em" and "time" not in rename.values() and len(out.columns) >= 4:
        rename[out.columns[0]] = "time"
        rename[out.columns[1]] = "price"
        rename[out.columns[2]] = "volume"
        rename[out.columns[3]] = "side"
    out = out.rename(columns=rename)
    keep = [c for c in ["time", "price", "volume", "amount", "side"] if c in out.columns]
    out = out[keep].copy() if keep else out.copy()
    now = datetime.now()
    date_text = trade_date or now.strftime("%Y%m%d")
    date_dash = f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"
    out.insert(0, "symbol", normalize_symbol(symbol))
    out.insert(1, "trade_date", date_text)
    out.insert(2, "collected_at", now.isoformat(timespec="seconds"))
    out["source"] = source
    if "time" in out.columns:
        out["time"] = out["time"].astype(str).str.strip()
        time_only = out["time"].str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False)
        parsed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        if (~time_only).any():
            parsed.loc[~time_only] = pd.to_datetime(out.loc[~time_only, "time"], errors="coerce")
        if time_only.any():
            parsed.loc[time_only] = pd.to_datetime(
                date_dash + " " + out.loc[time_only, "time"],
                errors="coerce",
            )
        out["trade_datetime"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        parsed_ok = parsed.notna()
        out.loc[parsed_ok, "trade_date"] = parsed.loc[parsed_ok].dt.strftime("%Y%m%d")
        out["trade_time"] = parsed.dt.strftime("%H%M%S")
    for c in ["price", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_trade_source_df(ak, source_key: str, symbol: str, trade_date: str) -> tuple[str, pd.DataFrame]:
    key = source_key.strip().lower()
    if key == "sina":
        return "stock_intraday_sina", ak.stock_intraday_sina(symbol=sina_symbol(symbol), date=trade_date)
    if key in {"tx", "tencent"}:
        return "stock_zh_a_tick_tx_js", ak.stock_zh_a_tick_tx_js(symbol=sina_symbol(symbol))
    if key in {"em", "eastmoney"}:
        return "stock_intraday_em", ak.stock_intraday_em(symbol=ak_symbol(symbol))
    raise ValueError(f"unknown trade source: {source_key}")


def trade_source_worker(source_key: str, symbol: str, trade_date: str, queue) -> None:
    try:
        ak = load_akshare()
        source_name, df = fetch_trade_source_df(ak, source_key, symbol, trade_date)
        queue.put({"ok": True, "source": source_name, "df": df})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def fetch_trade_source_with_timeout(
    source_key: str,
    symbol: str,
    trade_date: str,
    timeout: float,
) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    queue = mp.Queue()
    proc = mp.Process(target=trade_source_worker, args=(source_key, symbol, trade_date, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return source_key, None, f"timeout after {timeout}s"
    if queue.empty():
        return source_key, None, f"process exited with code {proc.exitcode}"
    payload = queue.get()
    if payload.get("ok"):
        return str(payload["source"]), payload.get("df"), None
    return source_key, None, str(payload.get("error"))


def fetch_intraday_trades(
    ak,
    symbol: str,
    trade_date: str,
    source_priority: Sequence[str],
    per_source_timeout: float = 45.0,
) -> pd.DataFrame:
    source_map = {
        "sina": True,
        "tx": True,
        "tencent": True,
        "em": True,
        "eastmoney": True,
    }
    attempts = [s.strip().lower() for s in source_priority if s.strip().lower() in source_map]
    errors = []
    for source_key in attempts:
        try:
            if per_source_timeout and per_source_timeout > 0:
                source, df, err = fetch_trade_source_with_timeout(source_key, symbol, trade_date, per_source_timeout)
                if err:
                    errors.append(f"{source}: {err}")
                    continue
            else:
                source, df = fetch_trade_source_df(ak, source_key, symbol, trade_date)
            if df is not None and not df.empty:
                return normalize_intraday_df(df, symbol, source, trade_date=trade_date)
            errors.append(f"{source}: empty")
        except Exception as exc:
            errors.append(f"{source_key}: {type(exc).__name__}: {exc}")
    return pd.DataFrame([{
        "symbol": normalize_symbol(symbol),
        "trade_date": trade_date,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "source": "none",
        "error": " | ".join(errors),
    }])


def collect_trades(args: argparse.Namespace) -> None:
    symbols = read_symbols(args)
    out_dir = Path(args.out_dir)
    ak = load_akshare()
    trade_date = args.trades_date or datetime.now().strftime("%Y%m%d")
    source_priority = [s.strip() for s in args.trades_source_priority.split(",") if s.strip()]
    written = []
    for idx, symbol in enumerate(symbols, start=1):
        df = fetch_intraday_trades(ak, symbol, trade_date, source_priority, args.trades_source_timeout)
        date = str(df["trade_date"].iloc[0])
        path = symbol_dir(out_dir, date, symbol) / "intraday_trades.csv"
        # Most AKShare trade endpoints return all currently available rows for the day.
        # Replace the file each fetch to avoid duplicate appends.
        path.parent.mkdir(parents=True, exist_ok=True)
        is_error_only = "error" in df.columns and len(df) == 1 and str(df.get("source", pd.Series([""])).iloc[0]) == "none"
        if is_error_only and path.exists() and not getattr(args, "overwrite_with_error", False):
            err_path = path.with_name("intraday_trades_errors.csv")
            append_csv(err_path, df)
            written.append(err_path)
            print(f"kept existing trades {idx}/{len(symbols)} {symbol}; wrote error log rows={len(df)}")
            if args.per_symbol_delay > 0 and idx < len(symbols):
                time.sleep(args.per_symbol_delay)
            continue
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
        print(f"wrote trades {idx}/{len(symbols)} {symbol} rows={len(df)}")
        if args.per_symbol_delay > 0 and idx < len(symbols):
            time.sleep(args.per_symbol_delay)
    write_manifest(out_dir, symbols, written, "collect-trades")


def write_manifest(out_dir: Path, symbols: Sequence[str], files: Sequence[Path], mode: str) -> None:
    path = out_dir / "pending" / "last_collect_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": list(symbols),
        "files": sorted({str(p) for p in files}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_loop(args: argparse.Namespace) -> None:
    interval = max(1, int(args.interval_seconds))
    end_at = None
    if args.until:
        hh, mm = args.until.split(":", 1)
        end_at = dtime(int(hh), int(mm))
    args._end_at = end_at
    last_trades_fetch = 0.0
    while True:
        now = datetime.now()
        if end_at and now.time() > end_at:
            print(f"stop: current time {now.strftime('%H:%M:%S')} > --until {args.until}")
            return
        if args.ignore_trading_hours or in_trading_window(now):
            try:
                collect_once(args)
                if end_at and datetime.now().time() > end_at:
                    print(f"stop: current time {datetime.now().strftime('%H:%M:%S')} > --until {args.until}")
                    return
                if args.with_trades and (time.time() - last_trades_fetch >= args.trades_interval_seconds):
                    collect_trades(args)
                    last_trades_fetch = time.time()
            except Exception as exc:
                print(f"collect failed at {now.isoformat(timespec='seconds')}: {type(exc).__name__}: {exc}")
        else:
            print(f"skip outside trading window: {now.strftime('%H:%M:%S')}")
        time.sleep(interval)


def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["trade_date"].astype(str) + df["trade_time"].astype(str).str.zfill(6), errors="coerce")
    return df.dropna(subset=["datetime"]).sort_values("datetime")


def build_bars(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    src = out_dir / "pending" / args.date
    if not src.exists():
        raise FileNotFoundError(src)
    freqs = [x.strip() for x in args.freqs.split(",") if x.strip()]
    for sym_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        snap_path = sym_dir / "snapshot_5level.csv"
        if not snap_path.exists():
            continue
        df = load_snapshots(snap_path)
        if df.empty:
            continue
        symbol = str(df["symbol"].iloc[0])
        px = pd.to_numeric(df["last_price"], errors="coerce").ffill()
        if px.dropna().empty:
            print(f"skip {snap_path}: no valid last_price")
            continue
        base = pd.DataFrame(index=df["datetime"])
        base["open"] = px.to_numpy()
        base["high"] = px.to_numpy()
        base["low"] = px.to_numpy()
        base["close"] = px.to_numpy()
        for c in ["volume", "amount", "depth_imbalance_1", "depth_imbalance_5", "spread_1", "mid_price_1"]:
            if c in df.columns:
                base[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        for freq in freqs:
            agg_map = {
                "open": ("open", "first"),
                "high": ("high", "max"),
                "low": ("low", "min"),
                "close": ("close", "last"),
            }
            for c in ["volume", "amount", "depth_imbalance_1", "depth_imbalance_5", "spread_1", "mid_price_1"]:
                if c in base.columns:
                    agg_map[c] = (c, "last")
            bars = base.resample(freq).agg(**agg_map).dropna(subset=["open", "high", "low", "close"], how="all")
            if "volume" in bars.columns:
                bars["bar_volume"] = bars["volume"].diff()
                bars.loc[bars["bar_volume"] < 0, "bar_volume"] = pd.NA
            if "amount" in bars.columns:
                bars["bar_amount"] = bars["amount"].diff()
                bars.loc[bars["bar_amount"] < 0, "bar_amount"] = pd.NA
            bars = bars.reset_index()
            bars.insert(0, "trade_date", args.date)
            bars.insert(0, "symbol", symbol)
            out_path = sym_dir / f"minute_bars_{freq}.csv"
            bars.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"wrote {out_path} rows={len(bars)}")
        write_daily_features(sym_dir, df, args.date, symbol)


def write_daily_features(sym_dir: Path, df: pd.DataFrame, date: str, symbol: str) -> None:
    px = pd.to_numeric(df["last_price"], errors="coerce")
    vol = pd.to_numeric(df.get("volume"), errors="coerce")
    amt = pd.to_numeric(df.get("amount"), errors="coerce")
    valid = px.dropna()
    if valid.empty:
        return
    last30 = df[df["datetime"] >= df["datetime"].max() - pd.Timedelta(minutes=30)].copy()
    row = {
        "symbol": symbol,
        "trade_date": date,
        "first_time": str(df["datetime"].min()),
        "last_time": str(df["datetime"].max()),
        "snapshots": int(len(df)),
        "open": float(valid.iloc[0]),
        "high": float(valid.max()),
        "low": float(valid.min()),
        "close": float(valid.iloc[-1]),
        "volume": float(vol.dropna().iloc[-1]) if vol.dropna().size else None,
        "amount": float(amt.dropna().iloc[-1]) if amt.dropna().size else None,
        "last_30m_ret": float(pd.to_numeric(last30["last_price"], errors="coerce").dropna().iloc[-1] / pd.to_numeric(last30["last_price"], errors="coerce").dropna().iloc[0] - 1.0)
        if pd.to_numeric(last30["last_price"], errors="coerce").dropna().size >= 2 else None,
        "last_depth_imbalance_5": float(pd.to_numeric(df.get("depth_imbalance_5"), errors="coerce").dropna().iloc[-1])
        if "depth_imbalance_5" in df.columns and pd.to_numeric(df.get("depth_imbalance_5"), errors="coerce").dropna().size else None,
    }
    if row["amount"] and row["volume"]:
        row["daily_vwap"] = row["amount"] / row["volume"]
    pd.DataFrame([row]).to_csv(sym_dir / "daily_features.csv", index=False, encoding="utf-8-sig")


def validate_day(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    day_dir = out_dir / "pending" / args.date
    if not day_dir.exists():
        raise FileNotFoundError(day_dir)
    rows = []
    for sym_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
        snap = sym_dir / "snapshot_5level.csv"
        snap_error = sym_dir / "snapshot_errors.csv"
        trade = sym_dir / "intraday_trades.csv"
        row = {
            "symbol": sym_dir.name,
            "has_snapshot": snap.exists(),
            "has_snapshot_errors": snap_error.exists(),
            "has_trades": trade.exists(),
        }
        if snap.exists():
            df = load_snapshots(snap)
            row.update({
                "snapshot_rows": int(len(df)),
                "first_time": str(df["datetime"].min()) if len(df) else None,
                "last_time": str(df["datetime"].max()) if len(df) else None,
                "last_price_missing_rate": float(pd.to_numeric(df.get("last_price"), errors="coerce").isna().mean()) if len(df) else None,
                "bid1_missing_rate": float(pd.to_numeric(df.get("bid_price_1"), errors="coerce").isna().mean()) if len(df) else None,
                "ask1_missing_rate": float(pd.to_numeric(df.get("ask_price_1"), errors="coerce").isna().mean()) if len(df) else None,
            })
            if "phase" in df.columns:
                for phase, n in df["phase"].value_counts().items():
                    row[f"phase_{phase}"] = int(n)
        if snap_error.exists():
            try:
                edf = pd.read_csv(snap_error, encoding="utf-8-sig")
                row["snapshot_error_rows"] = int(len(edf))
                if "bid_ask_error" in edf.columns:
                    row["top_snapshot_error"] = str(edf["bid_ask_error"].dropna().astype(str).value_counts().index[0]) if edf["bid_ask_error"].dropna().size else None
            except Exception as exc:
                row["snapshot_error_read_error"] = f"{type(exc).__name__}: {exc}"
        if trade.exists():
            try:
                tdf = pd.read_csv(trade, encoding="utf-8-sig")
                row["trade_rows"] = int(len(tdf))
                row["trade_error"] = str(tdf["error"].iloc[0]) if "error" in tdf.columns else None
            except Exception as exc:
                row["trade_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    report = pd.DataFrame(rows)
    out_path = day_dir / "validation_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    summary = {
        "date": args.date,
        "symbols": int(len(report)),
        "with_snapshot": int(report["has_snapshot"].sum()) if len(report) else 0,
        "with_snapshot_errors": int(report["has_snapshot_errors"].sum()) if len(report) else 0,
        "with_trades": int(report["has_trades"].sum()) if len(report) else 0,
        "out": str(out_path),
    }
    (day_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def data88_dates(data88_dir: Path) -> Dict[str, set]:
    covered: Dict[str, set] = {}
    for quote_path in data88_dir.rglob("行情.csv"):
        symbol = normalize_symbol(quote_path.parent.name)
        try:
            df = pd.read_csv(quote_path, encoding="gbk", usecols=["自然日"])
            covered.setdefault(symbol, set()).update(df["自然日"].dropna().astype(int).astype(str))
        except Exception:
            continue
    return covered


def reconcile_data88(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pending = out_dir / "pending"
    archive = out_dir / "archived_by_data88"
    data88_dir = resolve_path(args.data88_dir)
    covered = data88_dates(data88_dir)
    moved = 0
    for date_dir in sorted(p for p in pending.glob("*") if p.is_dir()):
        date = date_dir.name
        for sym_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if date in covered.get(sym_dir.name, set()):
                dst = archive / date / sym_dir.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(sym_dir), str(dst))
                moved += 1
                print(f"archived {sym_dir} -> {dst}")
        if not any(date_dir.iterdir()):
            date_dir.rmdir()
    print(f"reconcile done; archived_symbol_dirs={moved}")


def write_watchlist(args: argparse.Namespace) -> None:
    symbols = [normalize_symbol(s) for s in args.symbols.replace(";", ",").split(",") if s.strip()]
    path = resolve_path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(f"wrote {path}: {', '.join(symbols)}")


def preflight_hosts(sources: Sequence[str], timeout: float = 3.0) -> pd.DataFrame:
    hosts: List[str] = []
    for source in sources:
        hosts.extend(SOURCE_HOSTS.get(source.strip().lower(), []))
    hosts = sorted(set(hosts))
    rows = []
    for host in hosts:
        row = {"host": host, "dns_ok": False, "tcp_443_ok": False, "error": None}
        try:
            addr = socket.gethostbyname(host)
            row["dns_ok"] = True
            row["ip"] = addr
            try:
                with socket.create_connection((host, 443), timeout=timeout):
                    row["tcp_443_ok"] = True
            except Exception as exc:
                row["error"] = f"tcp: {type(exc).__name__}: {exc}"
        except Exception as exc:
            row["error"] = f"dns: {type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def preflight(args: argparse.Namespace) -> None:
    sources = []
    if getattr(args, "trades_source_priority", None):
        sources.extend(x.strip() for x in args.trades_source_priority.split(",") if x.strip())
    if getattr(args, "spot_source_priority", None):
        sources.extend(x.strip() for x in args.spot_source_priority.split(",") if x.strip())
    if not sources:
        sources = ["em", "sina", "tx"]
    report = preflight_hosts(sources, timeout=args.timeout)
    print(report.to_string(index=False))
    if not report.empty and (not report["dns_ok"].all() or not report["tcp_443_ok"].all()):
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect AKShare L1 / five-level quote and intraday trade data")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--symbols", help="Comma separated symbols, e.g. 002714,000001.SZ")
        sp.add_argument("--symbols-file", help="One symbol per line; defaults to PurchasedData/selected_watchlist.txt when --symbols is omitted")
        sp.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
        sp.add_argument("--include-raw", action="store_true")
        sp.add_argument("--no-spot-bulk", action="store_true")
        sp.add_argument("--spot-source-priority", default="sina,ths,xq,em")
        sp.add_argument("--disable-em-bid-ask", action="store_true", help="Do not use Eastmoney stock_bid_ask_em five-level quote fallback")
        sp.add_argument("--bid-ask-retries", type=int, default=2, help="Retry Eastmoney five-level quote this many times per symbol")
        sp.add_argument("--retry-sleep", type=float, default=0.5, help="Seconds between five-level quote retries")
        sp.add_argument("--allow-l1-only", action="store_true", help="Write snapshots with last_price even when bid/ask is missing")
        sp.add_argument("--strict-trading-hours", action="store_true", help="Skip snapshot writes outside configured A-share trading windows")
        sp.add_argument("--max-symbols", type=int)
        sp.add_argument("--per-symbol-delay", type=float, default=0.05)

    sp = sub.add_parser("collect-once", help="Collect one five-level quote snapshot for all symbols")
    add_common(sp)

    sp = sub.add_parser("collect-trades", help="Fetch current intraday trade detail for all symbols")
    add_common(sp)
    sp.add_argument("--trades-date", help="Trade date YYYYMMDD; defaults to today")
    sp.add_argument("--trades-source-priority", default="sina,tx,em", help="Comma separated trade sources: sina,tx,em")
    sp.add_argument("--trades-source-timeout", type=float, default=45.0, help="Hard timeout seconds for each trade source call")
    sp.add_argument("--overwrite-with-error", action="store_true", help="Overwrite intraday_trades.csv even when all trade sources fail")
    sp.add_argument("--preflight", action="store_true", help="Check source DNS/TCP connectivity before collecting trades")
    sp.add_argument("--preflight-timeout", type=float, default=3.0)

    sp = sub.add_parser("collect-loop", help="Collect snapshots repeatedly during trading windows")
    add_common(sp)
    sp.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    sp.add_argument("--until", help="Stop after HH:MM local time")
    sp.add_argument("--ignore-trading-hours", action="store_true")
    sp.add_argument("--with-trades", action="store_true")
    sp.add_argument("--trades-interval-seconds", type=int, default=300)

    sp = sub.add_parser("build-bars", help="Aggregate pending snapshots to 1m/5m bars and daily features")
    sp.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    sp.add_argument("--date", required=True, help="YYYYMMDD")
    sp.add_argument("--freqs", default="1min,5min")

    sp = sub.add_parser("validate-day", help="Write collection quality report for a date")
    sp.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    sp.add_argument("--date", required=True, help="YYYYMMDD")

    sp = sub.add_parser("reconcile-data88", help="Archive AKShare pending symbol dirs covered by data88")
    sp.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    sp.add_argument("--data88-dir", default="PurchasedData/data88")

    sp = sub.add_parser("write-watchlist", help="Write a watchlist file")
    sp.add_argument("--symbols", required=True)
    sp.add_argument("--path", default=str(DEFAULT_WATCHLIST))

    sp = sub.add_parser("preflight", help="Check source DNS/TCP connectivity")
    sp.add_argument("--trades-source-priority", default="sina,tx,em")
    sp.add_argument("--spot-source-priority", default="")
    sp.add_argument("--timeout", type=float, default=3.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "collect-once":
        collect_once(args)
    elif args.cmd == "collect-trades":
        if args.preflight:
            report = preflight_hosts([x.strip() for x in args.trades_source_priority.split(",") if x.strip()], timeout=args.preflight_timeout)
            print(report.to_string(index=False))
            if not report.empty and (not report["dns_ok"].all() or not report["tcp_443_ok"].all()):
                raise SystemExit("preflight failed; skip collect-trades")
        collect_trades(args)
    elif args.cmd == "collect-loop":
        collect_loop(args)
    elif args.cmd == "build-bars":
        build_bars(args)
    elif args.cmd == "validate-day":
        validate_day(args)
    elif args.cmd == "reconcile-data88":
        reconcile_data88(args)
    elif args.cmd == "write-watchlist":
        write_watchlist(args)
    elif args.cmd == "preflight":
        preflight(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
