#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build realistic no-EM external features for selected A-share profiles.

This builder is intentionally consistent with the existing external feature scripts:
- read an existing samples CSV with a `date` column;
- fetch daily external series from AKShare where possible;
- create lagged/as-of features to avoid look-ahead;
- write raw external features, merged samples, and validation_report.json.

The profiles are designed for the next batch of stocks:
    ai_compute                 -> 601138.SH 工业富联
    material_wind_battery      -> 002080.SZ 中材科技
    power_utility_rate         -> 601985.SH 中国核电
    fertilizer                 -> 600096.SH 云天化
    storage_power              -> 002518.SZ 科士达
    aero_nuclear_equipment     -> 603308.SH 应流股份
    optical_cable_grid         -> 600522.SH 中天科技 / 600487.SH 亨通光电

This version intentionally avoids Eastmoney/EM historical sources in the new external builder:
    - BaoStock daily bars for A-share peer baskets and ETF proxies, with local CSV cache
    - stock_board_industry_index_ths / stock_board_concept_hist_ths for board indices
    - futures_zh_daily_sina for domestic futures
    - optional yfinance for U.S. AI/semiconductor mappings, with forced T-1 as-of alignment

If a source fails, it is recorded in validation_report.json and the builder continues.
Use --strict if you want failures to stop the pipeline when no external source is available.
"""
from __future__ import annotations

import argparse
import atexit
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


@dataclass(frozen=True)
class ExternalProfile:
    key: str
    prefix: str
    description: str
    stocks: Tuple[str, ...] = ()
    etfs: Tuple[str, ...] = ()
    futures: Tuple[str, ...] = ()
    boards: Tuple[str, ...] = ()
    us_tickers: Tuple[str, ...] = ()


# Keep names short; feature groups use these prefixes.
PROFILES: Dict[str, ExternalProfile] = {
    "ai_compute": ExternalProfile(
        key="ai_compute",
        prefix="ai",
        description="AI compute/server/cloud-capex proxy for 工业富联",
        stocks=("300308", "300502", "300394", "002463", "603019", "000977", "000938"),
        etfs=("512480", "159995", "515050", "159819"),
        futures=("CU0", "AL0"),
        boards=("半导体", "通信设备", "消费电子"),
        us_tickers=("NVDA", "AMD", "AVGO", "SMCI", "DELL", "MSFT", "AMZN", "GOOGL", "META", "^IXIC", "^GSPC", "^SOX"),
    ),
    "material_wind_battery": ExternalProfile(
        key="material_wind_battery",
        prefix="mwb",
        description="Fiberglass/material + wind + battery proxy for 中材科技",
        stocks=("600176", "300196", "605006", "002202", "601615", "002812", "300568"),
        etfs=("516660", "159566", "515790", "516160"),
        futures=("SA0", "FG0", "SI0"),
        boards=("建筑材料", "玻璃玻纤", "风电设备", "电池"),
    ),
    "power_utility_rate": ExternalProfile(
        key="power_utility_rate",
        prefix="pur",
        description="Utility/power/defensive style proxy for 中国核电",
        stocks=("003816", "600900", "600025", "600011", "601991", "601611"),
        etfs=("512890", "159611", "516260"),
        futures=("ZC0",),
        boards=("电力", "公用事业"),
    ),
    "fertilizer": ExternalProfile(
        key="fertilizer",
        prefix="fert",
        description="Fertilizer/agri-chemical commodities and peer proxy for 云天化",
        stocks=("600141", "000422", "002895", "002539", "000830", "000893"),
        etfs=("159865", "516220"),
        futures=("UR0", "MA0", "SA0", "C0", "M0", "JM0", "ZC0"),
        boards=("农化制品", "化学制品", "基础化工"),
    ),
    "storage_power": ExternalProfile(
        key="storage_power",
        prefix="sp",
        description="Storage/photovoltaic inverter/data-center power proxy for 科士达",
        stocks=("300274", "300763", "688390", "605117", "300750", "002594", "300693"),
        etfs=("159566", "515790", "516160", "515030", "512480"),
        futures=("LC0", "SI0", "CU0", "AL0"),
        boards=("其他电源设备", "光伏设备", "电池", "电网设备"),
    ),
    "aero_nuclear_equipment": ExternalProfile(
        key="aero_nuclear_equipment",
        prefix="ane",
        description="Aero-engine/military/nuclear-equipment/high-alloy proxy for 应流股份",
        stocks=("600893", "600765", "000768", "003816", "601985", "601611", "300034"),
        etfs=("512660", "512670", "512710"),
        futures=("NI0", "SS0", "AL0"),
        boards=("军工装备", "军工电子", "通用设备", "专用设备"),
    ),
    "optical_cable_grid": ExternalProfile(
        key="optical_cable_grid",
        prefix="ocg",
        description="Optical communication + power cable/grid/offshore-wind proxy for 中天科技/亨通光电",
        stocks=("300308", "300502", "300394", "600522", "600487", "603606", "600973", "601869", "600498", "600406", "601179"),
        etfs=("515050", "159994", "159819", "512480", "515790"),
        futures=("CU0", "AL0"),
        boards=("通信设备", "电网设备", "风电设备", "消费电子"),
    ),
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_code(code: str) -> str:
    raw = re.sub(r"\D", "", str(code or ""))
    return raw[-6:] if len(raw) >= 6 else raw


def normalize_profile(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "ai": "ai_compute",
        "compute": "ai_compute",
        "material": "material_wind_battery",
        "wind_battery": "material_wind_battery",
        "power": "power_utility_rate",
        "utility": "power_utility_rate",
        "nuclear_power": "power_utility_rate",
        "storage": "storage_power",
        "power_storage": "storage_power",
        "aero": "aero_nuclear_equipment",
        "nuclear_equipment": "aero_nuclear_equipment",
        "optical": "optical_cable_grid",
        "cable_grid": "optical_cable_grid",
    }
    key = aliases.get(key, key)
    if key not in PROFILES:
        raise ValueError(f"unknown profile={value!r}; available={sorted(PROFILES)}")
    return key


def ymd(value: str | pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_numeric_except_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalize_ohlcv(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Normalize common AKShare OHLCV schemas into prefixed columns."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date"])
    df = raw.copy()
    date_col = first_col(df, ["日期", "date", "Date", "trade_date", "时间"])
    close_col = first_col(df, ["收盘", "收盘价", "close", "Close", "最新价"])
    open_col = first_col(df, ["开盘", "开盘价", "open", "Open"])
    high_col = first_col(df, ["最高", "最高价", "high", "High"])
    low_col = first_col(df, ["最低", "最低价", "low", "Low"])
    vol_col = first_col(df, ["成交量", "volume", "Volume", "vol"])
    amount_col = first_col(df, ["成交额", "amount", "Amount"])
    hold_col = first_col(df, ["持仓量", "hold", "open_interest"])
    settle_col = first_col(df, ["结算价", "settle", "settlement"])

    if date_col is None or close_col is None:
        return pd.DataFrame(columns=["date"])

    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce")})
    mapping = {
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
        "volume": vol_col,
        "amount": amount_col,
        "hold": hold_col,
        "settle": settle_col,
    }
    for name, src in mapping.items():
        if src is not None:
            out[f"{prefix}_{name}"] = df[src]
    out = to_numeric_except_date(out)
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if f"{prefix}_high" in out.columns and f"{prefix}_low" in out.columns:
        out[f"{prefix}_range_pct"] = out[f"{prefix}_high"] / out[f"{prefix}_low"].replace(0, np.nan) - 1.0
    return out.reset_index(drop=True)


def add_ts_features(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    for col in list(cols):
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_ret1"] = s.pct_change()
        out[f"{col}_ret3"] = s / s.shift(3) - 1.0
        out[f"{col}_ret5"] = s / s.shift(5) - 1.0
        out[f"{col}_ret20"] = s / s.shift(20) - 1.0
        out[f"{col}_ret60"] = s / s.shift(60) - 1.0
        ma20 = s.shift(1).rolling(20, min_periods=10).mean()
        sd20 = s.shift(1).rolling(20, min_periods=10).std()
        ma60 = s.shift(1).rolling(60, min_periods=20).mean()
        sd60 = s.shift(1).rolling(60, min_periods=20).std()
        out[f"{col}_ma20_gap"] = s / ma20.replace(0, np.nan) - 1.0
        out[f"{col}_z20"] = (s - ma20) / sd20.replace(0, np.nan)
        out[f"{col}_z60"] = (s - ma60) / sd60.replace(0, np.nan)
        if col.endswith("_close"):
            r1 = out[f"{col}_ret1"]
            out[f"{col}_vol20"] = r1.shift(1).rolling(20, min_periods=10).std()
    return out


def add_volume_features(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    for col in list(cols):
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        ma20 = s.shift(1).rolling(20, min_periods=10).mean()
        sd20 = s.shift(1).rolling(20, min_periods=10).std()
        out[f"{col}_shock20"] = s / ma20.replace(0, np.nan) - 1.0
        out[f"{col}_z20"] = (s - ma20) / sd20.replace(0, np.nan)
    return out


_BAOSTOCK_MODULE = None
_BAOSTOCK_LOGGED_IN = False


def baostock_symbol(code: str) -> str:
    raw = normalize_code(code)
    if raw.startswith(("6", "9", "5")):
        return f"sh.{raw}"
    return f"sz.{raw}"


def baostock_adjustflag(adjust: str) -> str:
    value = str(adjust or "").strip().lower()
    if value in {"qfq", "front", "pre"}:
        return "2"
    if value in {"hfq", "back", "post"}:
        return "1"
    return "3"


def get_baostock():
    global _BAOSTOCK_MODULE, _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK_MODULE is None:
        import baostock as bs
        _BAOSTOCK_MODULE = bs
    if not _BAOSTOCK_LOGGED_IN:
        lg = _BAOSTOCK_MODULE.login()
        if getattr(lg, "error_code", "0") != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
        _BAOSTOCK_LOGGED_IN = True
        atexit.register(lambda: _BAOSTOCK_MODULE.logout() if _BAOSTOCK_MODULE is not None else None)
    return _BAOSTOCK_MODULE


def baostock_result_to_df(rs) -> pd.DataFrame:
    rows = []
    while getattr(rs, "error_code", "0") == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def merge_daily_cache(cache_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if cache_path.exists():
        try:
            frames.append(pd.read_csv(cache_path))
        except Exception:
            pass
    if new_df is not None and not new_df.empty:
        frames.append(new_df)
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return merged


def query_baostock_daily(code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    bs = get_baostock()
    bs_code = baostock_symbol(code)
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume",
        start_date=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        end_date=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag=baostock_adjustflag(adjust),
    )
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock daily query failed for {bs_code}: {rs.error_code} {rs.error_msg}")
    df = baostock_result_to_df(rs)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_baostock_security(
    symbol: str,
    prefix: str,
    start_date: str,
    end_date: str,
    adjust: str,
    cache_dir: Path,
    source_kind: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
    raw_code = normalize_code(symbol)
    try:
        cache_path = cache_dir / source_kind / f"{raw_code}_daily_raw.csv"
        new_df = query_baostock_daily(raw_code, start_date, end_date, adjust)
        cached = merge_daily_cache(cache_path, new_df)
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        raw = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)].copy()
        out = normalize_ohlcv(raw, prefix)
        if out.empty or f"{prefix}_close" not in out.columns:
            return pd.DataFrame(columns=["date"]), "empty_or_unusable_schema"
        out = add_ts_features(out, [f"{prefix}_close", f"{prefix}_range_pct"])
        out = add_volume_features(out, [c for c in [f"{prefix}_volume", f"{prefix}_amount"] if c in out.columns])
        return out, None
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), f"{type(exc).__name__}: {exc}"


def fetch_stock(symbol: str, prefix: str, start_date: str, end_date: str, adjust: str, cache_dir: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    return fetch_baostock_security(symbol, prefix, start_date, end_date, adjust, cache_dir, "stocks")


def fetch_etf(symbol: str, prefix: str, start_date: str, end_date: str, adjust: str, cache_dir: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    return fetch_baostock_security(symbol, prefix, start_date, end_date, adjust, cache_dir, "etfs")


def fetch_future(symbol: str, prefix: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Optional[str]]:
    import akshare as ak

    try:
        raw = ak.futures_zh_daily_sina(symbol=symbol)
        out = normalize_ohlcv(raw, prefix)
        if out.empty or f"{prefix}_close" not in out.columns:
            return pd.DataFrame(columns=["date"]), "empty_or_unusable_schema"
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)].copy()
        out = add_ts_features(out, [f"{prefix}_close", f"{prefix}_range_pct"])
        out = add_volume_features(out, [c for c in [f"{prefix}_volume", f"{prefix}_hold"] if c in out.columns])
        return out, None
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), f"{type(exc).__name__}: {exc}"


def fetch_board(symbol: str, prefix: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Optional[str]]:
    import akshare as ak

    errors: List[str] = []
    for fn_name in ["stock_board_industry_index_ths", "stock_board_concept_hist_ths"]:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            errors.append(f"{fn_name}: not_available")
            continue
        try:
            raw = fn(symbol=symbol, start_date=start_date, end_date=end_date)
            out = normalize_ohlcv(raw, prefix)
            if out.empty or f"{prefix}_close" not in out.columns:
                errors.append(f"{fn_name}: empty_or_unusable_schema")
                continue
            out = add_ts_features(out, [f"{prefix}_close", f"{prefix}_range_pct"])
            out = add_volume_features(out, [c for c in [f"{prefix}_volume", f"{prefix}_amount"] if c in out.columns])
            return out, None
        except Exception as exc:
            errors.append(f"{fn_name}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(columns=["date"]), " | ".join(errors)




def safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return token or "asset"


def fetch_yfinance_symbol(ticker: str, prefix: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch one U.S./global ticker with yfinance.

    The returned rows are keyed by the U.S. trading date. Leakage is controlled later
    by source-specific as-of lag; for U.S. series we force lag >= 1 calendar day.
    """
    try:
        import yfinance as yf  # optional dependency
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), f"yfinance_not_available: {type(exc).__name__}: {exc}"

    try:
        # yfinance end is exclusive; add one day so the requested end date is included when available.
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date"]), "empty"
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance may return a single-ticker MultiIndex depending on version.
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.reset_index()
        date_col = first_col(raw, ["Date", "Datetime", "date"])
        close_col = first_col(raw, ["Adj Close", "Close", "close"])
        if date_col is None or close_col is None:
            return pd.DataFrame(columns=["date"]), "empty_or_unusable_schema"
        out = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce")})
        mapping = {
            "open": first_col(raw, ["Open", "open"]),
            "high": first_col(raw, ["High", "high"]),
            "low": first_col(raw, ["Low", "low"]),
            "close": close_col,
            "volume": first_col(raw, ["Volume", "volume"]),
        }
        for name, src in mapping.items():
            if src is not None:
                out[f"{prefix}_{name}"] = raw[src]
        out = to_numeric_except_date(out)
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        if f"{prefix}_high" in out.columns and f"{prefix}_low" in out.columns:
            out[f"{prefix}_range_pct"] = out[f"{prefix}_high"] / out[f"{prefix}_low"].replace(0, np.nan) - 1.0
        if out.empty or f"{prefix}_close" not in out.columns:
            return pd.DataFrame(columns=["date"]), "empty_or_unusable_schema"
        out = add_ts_features(out, [f"{prefix}_close", f"{prefix}_range_pct"])
        out = add_volume_features(out, [c for c in [f"{prefix}_volume"] if c in out.columns])
        return out.reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(columns=["date"]), f"{type(exc).__name__}: {exc}"

def merge_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    usable = [f for f in frames if f is not None and not f.empty and "date" in f.columns]
    if not usable:
        return pd.DataFrame(columns=["date"])
    out = usable[0].sort_values("date").copy()
    for frame in usable[1:]:
        out = out.merge(frame.sort_values("date"), on="date", how="outer")
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def add_basket_from_closes(df: pd.DataFrame, close_cols: List[str], basket_col: str) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    cols = [c for c in close_cols if c in out.columns]
    if not cols:
        return out
    returns = out[cols].apply(pd.to_numeric, errors="coerce").pct_change()
    out[f"{basket_col}_ret1"] = returns.mean(axis=1, skipna=True)
    out[basket_col] = (1.0 + out[f"{basket_col}_ret1"].fillna(0.0)).cumprod() * 100.0
    out = add_ts_features(out, [basket_col])
    return out


def build_external_features(
    profile: ExternalProfile,
    start_date: str,
    end_date: str,
    target_symbol: Optional[str] = None,
    adjust: str = "qfq",
    skip_stocks: bool = False,
    skip_etfs: bool = False,
    skip_futures: bool = False,
    skip_boards: bool = False,
    enable_us_yf: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    target_raw = normalize_code(target_symbol or "")
    frames: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}
    used: Dict[str, List[str]] = {"stocks": [], "etfs": [], "futures": [], "boards": [], "us_yf": []}

    pfx = profile.prefix
    cache_root = Path(cache_dir or (SAVED_DATA_DIR / "stock_external_baostock_cache"))

    if not skip_stocks:
        stock_frames = []
        stock_close_cols = []
        for code in profile.stocks:
            raw = normalize_code(code)
            if target_raw and raw == target_raw:
                continue
            col_prefix = f"{pfx}_stk_{raw}"
            frame, err = fetch_stock(raw, col_prefix, start_date, end_date, adjust, cache_root)
            if err:
                errors[f"stock_{raw}"] = err
                continue
            stock_frames.append(frame)
            stock_close_cols.append(f"{col_prefix}_close")
            used["stocks"].append(raw)
        stock_merged = merge_frames(stock_frames)
        if not stock_merged.empty:
            stock_merged = add_basket_from_closes(stock_merged, stock_close_cols, f"{pfx}_stock_basket_close")
            frames.append(stock_merged)

    if not skip_etfs:
        etf_frames = []
        etf_close_cols = []
        for code in profile.etfs:
            raw = normalize_code(code)
            col_prefix = f"{pfx}_etf_{raw}"
            frame, err = fetch_etf(raw, col_prefix, start_date, end_date, adjust, cache_root)
            if err:
                errors[f"etf_{raw}"] = err
                continue
            etf_frames.append(frame)
            etf_close_cols.append(f"{col_prefix}_close")
            used["etfs"].append(raw)
        etf_merged = merge_frames(etf_frames)
        if not etf_merged.empty:
            etf_merged = add_basket_from_closes(etf_merged, etf_close_cols, f"{pfx}_etf_basket_close")
            frames.append(etf_merged)

    if not skip_futures:
        fut_frames = []
        fut_close_cols = []
        for symbol in profile.futures:
            col_prefix = f"{pfx}_fut_{symbol.lower()}"
            frame, err = fetch_future(symbol, col_prefix, start_date, end_date)
            if err:
                errors[f"future_{symbol}"] = err
                continue
            fut_frames.append(frame)
            fut_close_cols.append(f"{col_prefix}_close")
            used["futures"].append(symbol)
        fut_merged = merge_frames(fut_frames)
        if not fut_merged.empty:
            fut_merged = add_basket_from_closes(fut_merged, fut_close_cols, f"{pfx}_future_basket_close")
            frames.append(fut_merged)

    if not skip_boards:
        board_frames = []
        board_close_cols = []
        used_board_tokens = set()
        for board_idx, board in enumerate(profile.boards, start=1):
            # Chinese board names such as “通信设备/电网设备/风电设备” would all
            # collapse to the same ASCII token if we simply stripped non-ASCII
            # characters.  That caused duplicate columns like
            # ocg_board_board_close and aborted merge_frames().  Keep the token
            # deterministic, human-readable enough, and unique within a profile.
            ascii_token = re.sub(r"[^0-9A-Za-z_]+", "_", str(board)).strip("_").lower()
            if ascii_token:
                safe = ascii_token
            else:
                safe = f"b{board_idx:02d}"
            while safe in used_board_tokens:
                safe = f"{safe}_{board_idx:02d}"
            used_board_tokens.add(safe)

            col_prefix = f"{pfx}_board_{safe}"
            frame, err = fetch_board(board, col_prefix, start_date, end_date)
            if err:
                errors[f"board_{board}"] = err
                continue
            board_frames.append(frame)
            board_close_cols.append(f"{col_prefix}_close")
            used["boards"].append(board)
        board_merged = merge_frames(board_frames)
        if not board_merged.empty:
            board_merged = add_basket_from_closes(board_merged, board_close_cols, f"{pfx}_board_basket_close")
            frames.append(board_merged)


    if enable_us_yf and profile.us_tickers:
        us_frames = []
        us_close_cols = []
        for ticker in profile.us_tickers:
            token = safe_token(ticker)
            col_prefix = f"{pfx}_us_{token}"
            frame, err = fetch_yfinance_symbol(ticker, col_prefix, start_date, end_date)
            if err:
                errors[f"us_yf_{ticker}"] = err
                continue
            us_frames.append(frame)
            us_close_cols.append(f"{col_prefix}_close")
            used["us_yf"].append(ticker)
        us_merged = merge_frames(us_frames)
        if not us_merged.empty:
            us_merged = add_basket_from_closes(us_merged, us_close_cols, f"{pfx}_us_basket_close")
            frames.append(us_merged)

    features = merge_frames(frames)
    # Drop fully empty feature columns but keep date.
    if not features.empty:
        non_empty = ["date"] + [c for c in features.columns if c != "date" and not features[c].isna().all()]
        features = features[non_empty].sort_values("date").reset_index(drop=True)

    source_meta = {
        "profile": asdict(profile),
        "used": used,
        "start_date": start_date,
        "end_date": end_date,
        "target_symbol_excluded": target_raw or None,
        "em_sources_used": False,
        "baostock_cache_dir": str(cache_root),
        "us_yfinance_enabled": bool(enable_us_yf),
    }
    return features, errors, source_meta




def feature_family_columns(features: pd.DataFrame, prefix: str, family: str) -> List[str]:
    """Return columns for one source family so each family can use its own lag."""
    if features.empty:
        return []
    patterns = {
        "domestic": (f"{prefix}_stk_", f"{prefix}_stock_basket", f"{prefix}_etf_", f"{prefix}_etf_basket", f"{prefix}_board_", f"{prefix}_board_basket"),
        "futures": (f"{prefix}_fut_", f"{prefix}_future_basket"),
        "us": (f"{prefix}_us_", f"{prefix}_us_basket"),
    }[family]
    return [c for c in features.columns if c != "date" and c.startswith(patterns)]


def merge_source_families_asof(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    prefix: str,
    domestic_lag_days: int,
    future_lag_days: int,
    us_lag_days: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Merge feature families using source-specific as-of lags.

    For next-day A-share decisions made after A-share close:
      - domestic A-share/ETF/THS board features can use T close, default lag 0;
      - domestic futures are conservative by default, default lag 1;
      - U.S. yfinance features must lag at least 1 calendar day because U.S. T close
        occurs after the A-share T close in Beijing time.
    """
    out = samples.sort_values("date").copy()
    policy = {
        "domestic": max(int(domestic_lag_days), 0),
        "futures": max(int(future_lag_days), 0),
        "us": max(int(us_lag_days), 1),
    }
    for family, lag in policy.items():
        cols = feature_family_columns(features, prefix, family)
        if not cols:
            continue
        fam = features[["date"] + cols].dropna(how="all", subset=cols).copy()
        out = merge_asof_lag(out, fam, lag, f"{prefix}_{family}")
    return out.reset_index(drop=True), policy

def merge_asof_lag(samples: pd.DataFrame, features: pd.DataFrame, lag_days: int, prefix: str) -> pd.DataFrame:
    s = samples.sort_values("date").copy()
    if features.empty or list(features.columns) == ["date"]:
        return s.reset_index(drop=True)
    s[f"{prefix}_asof_date"] = pd.to_datetime(s["date"]) - pd.to_timedelta(lag_days, unit="D")
    f = features.sort_values("date").copy().rename(columns={"date": f"{prefix}_feature_date"})
    merged = pd.merge_asof(
        s.sort_values(f"{prefix}_asof_date"),
        f.sort_values(f"{prefix}_feature_date"),
        left_on=f"{prefix}_asof_date",
        right_on=f"{prefix}_feature_date",
        direction="backward",
    )
    return merged.sort_values("date").drop(columns=[f"{prefix}_asof_date"]).reset_index(drop=True)


def add_relative_features(merged: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = merged.sort_values("date").copy()
    # These relative features are based on lagged external returns already merged by as-of date.
    if "close" in out.columns:
        stock_ret5 = pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["close"], errors="coerce").shift(5) - 1.0
        stock_ret20 = pd.to_numeric(out["close"], errors="coerce") / pd.to_numeric(out["close"], errors="coerce").shift(20) - 1.0
        for family in ["stock_basket", "etf_basket", "future_basket", "board_basket", "us_basket"]:
            ext5 = f"{prefix}_{family}_close_ret5"
            ext20 = f"{prefix}_{family}_close_ret20"
            if ext5 in out.columns:
                out[f"{prefix}_stock_vs_{family}_ret5"] = stock_ret5 - pd.to_numeric(out[ext5], errors="coerce")
            if ext20 in out.columns:
                out[f"{prefix}_stock_vs_{family}_ret20"] = stock_ret20 - pd.to_numeric(out[ext20], errors="coerce")
    if "bench_ret20" in out.columns:
        for family in ["stock_basket", "etf_basket", "future_basket", "board_basket", "us_basket"]:
            ext20 = f"{prefix}_{family}_close_ret20"
            if ext20 in out.columns:
                out[f"{prefix}_{family}_vs_bench_ret20"] = pd.to_numeric(out[ext20], errors="coerce") - pd.to_numeric(out["bench_ret20"], errors="coerce")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build AKShare-based external features for selected stock profiles")
    p.add_argument("--samples", required=True, help="Input samples CSV, usually sector/fundamental samples")
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "stock_external_features_out"))
    p.add_argument("--profile", required=True, help=f"One of: {','.join(sorted(PROFILES))}")
    p.add_argument("--target-symbol", default="", help="Current stock; excluded from peer baskets if present")
    p.add_argument("--lag-days", type=int, default=None, help="Legacy: use the same as-of lag for every source family")
    p.add_argument("--domestic-lag-days", type=int, default=0, help="A-share/ETF/THS board lag; default 0 for after-close next-day models")
    p.add_argument("--future-lag-days", type=int, default=1, help="Domestic futures lag; default 1 conservatively avoids night-session ambiguity")
    p.add_argument("--us-lag-days", type=int, default=1, help="U.S. yfinance lag; forced to >=1 to avoid A-share after-close leakage")
    p.add_argument("--enable-us-yf", action="store_true", help="Enable optional yfinance U.S. mappings, mainly for ai_compute")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""])
    p.add_argument("--skip-stocks", action="store_true")
    p.add_argument("--skip-etfs", action="store_true")
    p.add_argument("--skip-futures", action="store_true")
    p.add_argument("--skip-boards", action="store_true")
    p.add_argument("--baostock-cache-dir", default=None, help="Local cache dir for BaoStock stock/ETF daily external series")
    p.add_argument("--strict", action="store_true", help="Fail if no external feature source succeeds")
    args = p.parse_args()

    profile_key = normalize_profile(args.profile)
    profile = PROFILES[profile_key]
    out_dir = ensure_dir(args.out_dir)

    samples = pd.read_csv(args.samples, parse_dates=["date"]).sort_values("date")
    start_date = ymd(args.start_date or samples["date"].min())
    # Fetch to today to allow as-of merge for an unlabeled tail row; trim happens inside fetchers where needed.
    end_date = ymd(args.end_date or max(samples["date"].max(), pd.Timestamp.today()))

    features, errors, source_meta = build_external_features(
        profile=profile,
        start_date=start_date,
        end_date=end_date,
        target_symbol=args.target_symbol,
        adjust=args.adjust,
        skip_stocks=args.skip_stocks,
        skip_etfs=args.skip_etfs,
        skip_futures=args.skip_futures,
        skip_boards=args.skip_boards,
        enable_us_yf=args.enable_us_yf,
        cache_dir=Path(args.baostock_cache_dir) if args.baostock_cache_dir else (out_dir / "baostock_external_cache"),
    )
    feature_cols = [c for c in features.columns if c != "date"] if not features.empty else []
    if args.strict and not feature_cols:
        raise RuntimeError(f"no usable external features fetched for profile={profile_key}; errors={errors}")

    if args.lag_days is not None:
        domestic_lag_days = future_lag_days = int(args.lag_days)
        us_lag_days = max(int(args.lag_days), 1)
    else:
        domestic_lag_days = args.domestic_lag_days
        future_lag_days = args.future_lag_days
        us_lag_days = max(args.us_lag_days, 1)

    merged, lag_policy = merge_source_families_asof(
        samples=samples,
        features=features,
        prefix=profile.prefix,
        domestic_lag_days=domestic_lag_days,
        future_lag_days=future_lag_days,
        us_lag_days=us_lag_days,
    )
    merged = add_relative_features(merged, profile.prefix)

    raw_path = out_dir / f"stock_external_features_{profile_key}.csv"
    merged_path = out_dir / f"training_samples_with_{profile_key}_external.csv"
    report_path = out_dir / "validation_report.json"

    features.to_csv(raw_path, index=False, encoding="utf-8-sig")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    final_feature_cols = [c for c in merged.columns if c.startswith(f"{profile.prefix}_")]
    top_missing = {}
    if final_feature_cols:
        top_missing = {
            k: float(v)
            for k, v in merged[final_feature_cols].isna().mean().sort_values(ascending=False).head(50).items()
        }
    report = {
        "profile": profile_key,
        "description": profile.description,
        "sample_rows": int(len(samples)),
        "merged_rows": int(len(merged)),
        "raw_rows": int(len(features)),
        "raw_date_min": str(features["date"].min().date()) if len(features) else None,
        "raw_date_max": str(features["date"].max().date()) if len(features) else None,
        "lag_days_legacy": args.lag_days,
        "lag_policy": lag_policy,
        "enable_us_yf": bool(args.enable_us_yf),
        "feature_cols": int(len(final_feature_cols)),
        "source_meta": source_meta,
        "errors": errors,
        "top_missing": top_missing,
        "outputs": {
            "features": str(raw_path),
            "merged_samples": str(merged_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
