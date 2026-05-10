#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaoStock helper with:
1) 原始行情增量缓存；
2) 特征尾部增量重算；
3) 仍复用原 backtest/feature-engineering 模块。

说明：
- 若仅末尾追加新数据，则不全量重算历史特征；
- 日线特征默认仅重算最近 260 个交易日；
- 5分钟特征默认仅重算最近 40 个交易日；
- 若请求区间向前扩展到特征缓存起点之前，则回退为全量特征重建，以保证正确性。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd

try:
    import baostock as bs
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "未安装 baostock。请先执行: pip install -U baostock\n"
        f"原始错误: {type(e).__name__}: {e}"
    )

MINUTE_START_FLOOR = pd.Timestamp("2015-01-01 09:30:00")
BAR_5M = pd.Timedelta(minutes=5)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    digits = re.findall(r"\d+", s)
    if not digits:
        raise ValueError(f"无法从 symbol 中提取 6 位代码: {symbol}")
    return digits[-1].zfill(6)


def to_bs_stock_code(symbol: str) -> str:
    code = normalize_symbol(symbol)
    if code.startswith(("600", "601", "603", "605", "688", "900")):
        return f"sh.{code}"
    return f"sz.{code}"


def to_bs_index_code(symbol: str) -> str:
    code = normalize_symbol(symbol)
    if code.startswith("399"):
        return f"sz.{code}"
    return f"sh.{code}"


def adjust_to_baostock_flag(adjust: str) -> str:
    a = str(adjust or "").lower().strip()
    if a == "qfq":
        return "2"
    if a == "hfq":
        return "1"
    return "3"


def import_backtest_module(backtest_py: str | Path):
    backtest_py = str(backtest_py)
    module_name = "t_strategy_backtest_split_eval_mod_baostock_stateful"
    spec = importlib.util.spec_from_file_location(module_name, backtest_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载回测脚本: {backtest_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def maybe_fill_event_flags(daily_df: pd.DataFrame) -> pd.DataFrame:
    out = daily_df.copy()
    if "event_flag" not in out.columns:
        out["event_flag"] = 0
    if "no_price_limit_flag" not in out.columns:
        out["no_price_limit_flag"] = 0
    return out


def _login() -> None:
    lg = bs.login()
    if getattr(lg, "error_code", "-1") != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_code} {lg.error_msg}")


def _logout() -> None:
    try:
        bs.logout()
    except Exception:
        pass


def _rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while (rs.error_code == '0') and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query failed: {rs.error_code} {rs.error_msg}")
    if not rows:
        return pd.DataFrame(columns=getattr(rs, "fields", []))
    return pd.DataFrame(rows, columns=rs.fields)


def _parse_bs_minute_datetime(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    out = []
    for d, t in zip(date_s.astype(str), time_s.astype(str)):
        digits = ''.join(ch for ch in str(t) if ch.isdigit())
        dt = pd.NaT
        if len(digits) >= 14:
            dt = pd.to_datetime(digits[:14], format='%Y%m%d%H%M%S', errors='coerce')
        elif len(digits) >= 12:
            dt = pd.to_datetime(digits[:12], format='%Y%m%d%H%M', errors='coerce')
        elif len(digits) >= 6:
            hhmmss = digits[:6]
            d8 = str(d).replace('-', '')[:8]
            dt = pd.to_datetime(d8 + hhmmss, format='%Y%m%d%H%M%S', errors='coerce')
        out.append(dt)
    return pd.Series(out)


def _normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "event_flag", "no_price_limit_flag"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = maybe_fill_event_flags(out)
    keep = ["date", "open", "high", "low", "close", "volume", "event_flag", "no_price_limit_flag"]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out[keep]


def _normalize_intraday_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    keep = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    out = out.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
    return out[keep]


def _normalize_bench_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out[["date", "close"]]


def _normalize_daily_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    out = df.copy()
    if "date" not in out.columns:
        return pd.DataFrame(columns=["date"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def _normalize_intraday_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "date"])
    out = df.copy()
    if "datetime" not in out.columns:
        return pd.DataFrame(columns=["datetime", "date"])
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    else:
        out["date"] = out["datetime"].dt.normalize()
    out = out.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
    return out


def _read_csv_or_empty(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        if kind == "daily":
            return _normalize_daily_df(pd.DataFrame())
        if kind == "intraday":
            return _normalize_intraday_df(pd.DataFrame())
        if kind == "feat_daily":
            return _normalize_daily_feature_df(pd.DataFrame())
        if kind == "feat_intraday":
            return _normalize_intraday_feature_df(pd.DataFrame())
        return _normalize_bench_df(pd.DataFrame())
    try:
        df = pd.read_csv(path)
    except Exception:
        if kind == "daily":
            return _normalize_daily_df(pd.DataFrame())
        if kind == "intraday":
            return _normalize_intraday_df(pd.DataFrame())
        if kind == "feat_daily":
            return _normalize_daily_feature_df(pd.DataFrame())
        if kind == "feat_intraday":
            return _normalize_intraday_feature_df(pd.DataFrame())
        return _normalize_bench_df(pd.DataFrame())
    if kind == "daily":
        return _normalize_daily_df(df)
    if kind == "intraday":
        return _normalize_intraday_df(df)
    if kind == "feat_daily":
        return _normalize_daily_feature_df(df)
    if kind == "feat_intraday":
        return _normalize_intraday_feature_df(df)
    return _normalize_bench_df(df)


def _merge_and_save(df_old: pd.DataFrame, df_new: pd.DataFrame, path: Path, kind: str) -> pd.DataFrame:
    if kind == "daily":
        merged = _normalize_daily_df(pd.concat([df_old, df_new], axis=0, ignore_index=True))
    elif kind == "intraday":
        merged = _normalize_intraday_df(pd.concat([df_old, df_new], axis=0, ignore_index=True))
    elif kind == "feat_daily":
        merged = _normalize_daily_feature_df(pd.concat([df_old, df_new], axis=0, ignore_index=True))
    elif kind == "feat_intraday":
        merged = _normalize_intraday_feature_df(pd.concat([df_old, df_new], axis=0, ignore_index=True))
    else:
        merged = _normalize_bench_df(pd.concat([df_old, df_new], axis=0, ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def _date_range_log(df: pd.DataFrame, date_col: str) -> Dict[str, object]:
    if df.empty or date_col not in df.columns:
        return {"rows": 0, "min": None, "max": None}
    s = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if s.empty:
        return {"rows": int(len(df)), "min": None, "max": None}
    return {"rows": int(len(df)), "min": str(s.iloc[0]), "max": str(s.iloc[-1])}


def fetch_stock_daily_range(symbol: str, start: str, end: str, adjust: str = "qfq") -> Tuple[pd.DataFrame, Dict]:
    code = to_bs_stock_code(symbol)
    adjustflag = adjust_to_baostock_flag(adjust)
    logs: Dict[str, object] = {"provider": "baostock", "code": code, "adjustflag": adjustflag}
    _login()
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,open,high,low,close,volume",
            start_date=str(pd.to_datetime(start).date()),
            end_date=str(pd.to_datetime(end).date()),
            frequency="d",
            adjustflag=adjustflag,
        )
        df = _rs_to_df(rs)
    finally:
        _logout()
    if df.empty:
        return _normalize_daily_df(pd.DataFrame()), {"chosen_source": "baostock", "logs": {**logs, "rows": 0}}
    out = _normalize_daily_df(df)
    logs["rows"] = int(len(out))
    logs["start"] = str(pd.to_datetime(start).date())
    logs["end"] = str(pd.to_datetime(end).date())
    return out, {"chosen_source": "baostock", "logs": logs}


def fetch_stock_intraday_5m_range(symbol: str, start_dt: str, end_dt: str, adjust: str = "qfq") -> Tuple[pd.DataFrame, Dict]:
    code = to_bs_stock_code(symbol)
    adjustflag = adjust_to_baostock_flag(adjust)
    start_ts = pd.to_datetime(start_dt)
    end_ts = pd.to_datetime(end_dt)
    effective_start = max(start_ts, MINUTE_START_FLOOR)
    logs: Dict[str, object] = {
        "provider": "baostock",
        "code": code,
        "adjustflag": adjustflag,
        "requested_start": str(start_ts),
        "effective_start": str(effective_start),
        "requested_end": str(end_ts),
    }
    if effective_start > end_ts:
        return _normalize_intraday_df(pd.DataFrame()), {"chosen_source": "baostock", "logs": {**logs, "rows": 0}}
    _login()
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,time,open,high,low,close,volume,amount",
            start_date=str(effective_start.date()),
            end_date=str(end_ts.date()),
            frequency="5",
            adjustflag=adjustflag,
        )
        df = _rs_to_df(rs)
    finally:
        _logout()
    if df.empty:
        return _normalize_intraday_df(pd.DataFrame()), {"chosen_source": "baostock", "logs": {**logs, "rows": 0}}
    df["datetime"] = _parse_bs_minute_datetime(df["date"], df["time"])
    out = _normalize_intraday_df(df)
    out = out[(out["datetime"] >= effective_start) & (out["datetime"] <= end_ts)].copy().reset_index(drop=True)
    logs["rows"] = int(len(out))
    return out, {"chosen_source": "baostock", "logs": logs}


def fetch_benchmark_daily_range(index_symbol: str, start: str, end: str) -> Tuple[pd.DataFrame, Dict]:
    code = to_bs_index_code(index_symbol)
    logs: Dict[str, object] = {"provider": "baostock", "code": code}
    _login()
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,close",
            start_date=str(pd.to_datetime(start).date()),
            end_date=str(pd.to_datetime(end).date()),
            frequency="d",
            adjustflag="3",
        )
        df = _rs_to_df(rs)
    finally:
        _logout()
    out = _normalize_bench_df(df)
    logs["rows"] = int(len(out))
    logs["start"] = str(pd.to_datetime(start).date())
    logs["end"] = str(pd.to_datetime(end).date())
    return out, {"chosen_source": "baostock", "logs": logs}


def _incremental_fetch_daily(symbol: str, start: str, end: str, adjust: str, cache_path: Path, force_refresh: bool = False, cache_mode: str = "incremental") -> Tuple[pd.DataFrame, Dict]:
    req_start = pd.to_datetime(start).normalize()
    req_end = pd.to_datetime(end).normalize()
    cache_df = _read_csv_or_empty(cache_path, "daily")
    logs: Dict[str, object] = {"cache_path": str(cache_path), "requested_start": str(req_start.date()), "requested_end": str(req_end.date())}

    if force_refresh or cache_mode == "full" or cache_df.empty:
        fresh, meta = fetch_stock_daily_range(symbol, str(req_start.date()), str(req_end.date()), adjust)
        merged = _merge_and_save(_normalize_daily_df(pd.DataFrame()), fresh, cache_path, "daily")
        logs["mode"] = "full_refresh" if (force_refresh or cache_mode == "full") else "initial_fetch"
        logs["fetch_segments"] = [meta["logs"]]
        logs["cache_after"] = _date_range_log(merged, "date")
        return merged[(merged["date"] >= req_start) & (merged["date"] <= req_end)].reset_index(drop=True), logs

    fetch_segments: List[Dict[str, object]] = []
    merged = cache_df.copy()
    cmin = merged["date"].min()
    cmax = merged["date"].max()
    if req_start < cmin:
        pre_end = cmin - pd.Timedelta(days=1)
        if req_start <= pre_end:
            pre_df, pre_meta = fetch_stock_daily_range(symbol, str(req_start.date()), str(pre_end.date()), adjust)
            fetch_segments.append(pre_meta["logs"])
            merged = _merge_and_save(merged, pre_df, cache_path, "daily")
    if req_end > cmax:
        post_start = cmax + pd.Timedelta(days=1)
        if post_start <= req_end:
            post_df, post_meta = fetch_stock_daily_range(symbol, str(post_start.date()), str(req_end.date()), adjust)
            fetch_segments.append(post_meta["logs"])
            merged = _merge_and_save(merged, post_df, cache_path, "daily")
    merged = _read_csv_or_empty(cache_path, "daily")
    logs["mode"] = "incremental"
    logs["cache_before"] = _date_range_log(cache_df, "date")
    logs["fetch_segments"] = fetch_segments
    logs["cache_after"] = _date_range_log(merged, "date")
    out = merged[(merged["date"] >= req_start) & (merged["date"] <= req_end)].reset_index(drop=True)
    return out, logs


def _incremental_fetch_intraday(symbol: str, start_dt: str, end_dt: str, adjust: str, cache_path: Path, force_refresh: bool = False, cache_mode: str = "incremental") -> Tuple[pd.DataFrame, Dict]:
    req_start = max(pd.to_datetime(start_dt), MINUTE_START_FLOOR)
    req_end = pd.to_datetime(end_dt)
    cache_df = _read_csv_or_empty(cache_path, "intraday")
    logs: Dict[str, object] = {"cache_path": str(cache_path), "requested_start": str(req_start), "requested_end": str(req_end)}

    if force_refresh or cache_mode == "full" or cache_df.empty:
        fresh, meta = fetch_stock_intraday_5m_range(symbol, str(req_start), str(req_end), adjust)
        merged = _merge_and_save(_normalize_intraday_df(pd.DataFrame()), fresh, cache_path, "intraday")
        logs["mode"] = "full_refresh" if (force_refresh or cache_mode == "full") else "initial_fetch"
        logs["fetch_segments"] = [meta["logs"]]
        logs["cache_after"] = _date_range_log(merged, "datetime")
        return merged[(merged["datetime"] >= req_start) & (merged["datetime"] <= req_end)].reset_index(drop=True), logs

    fetch_segments: List[Dict[str, object]] = []
    merged = cache_df.copy()
    cmin = merged["datetime"].min()
    cmax = merged["datetime"].max()
    if req_start < cmin:
        pre_end = cmin - BAR_5M
        if req_start <= pre_end:
            pre_df, pre_meta = fetch_stock_intraday_5m_range(symbol, str(req_start), str(pre_end), adjust)
            fetch_segments.append(pre_meta["logs"])
            merged = _merge_and_save(merged, pre_df, cache_path, "intraday")
    if req_end > cmax:
        post_start = cmax + BAR_5M
        if post_start <= req_end:
            post_df, post_meta = fetch_stock_intraday_5m_range(symbol, str(post_start), str(req_end), adjust)
            fetch_segments.append(post_meta["logs"])
            merged = _merge_and_save(merged, post_df, cache_path, "intraday")
    merged = _read_csv_or_empty(cache_path, "intraday")
    logs["mode"] = "incremental"
    logs["cache_before"] = _date_range_log(cache_df, "datetime")
    logs["fetch_segments"] = fetch_segments
    logs["cache_after"] = _date_range_log(merged, "datetime")
    out = merged[(merged["datetime"] >= req_start) & (merged["datetime"] <= req_end)].reset_index(drop=True)
    return out, logs


def _incremental_fetch_benchmark(symbol: str, start: str, end: str, cache_path: Path, force_refresh: bool = False, cache_mode: str = "incremental") -> Tuple[pd.DataFrame, Dict]:
    req_start = pd.to_datetime(start).normalize()
    req_end = pd.to_datetime(end).normalize()
    cache_df = _read_csv_or_empty(cache_path, "bench")
    logs: Dict[str, object] = {"cache_path": str(cache_path), "requested_start": str(req_start.date()), "requested_end": str(req_end.date())}

    if force_refresh or cache_mode == "full" or cache_df.empty:
        fresh, meta = fetch_benchmark_daily_range(symbol, str(req_start.date()), str(req_end.date()))
        merged = _merge_and_save(_normalize_bench_df(pd.DataFrame()), fresh, cache_path, "bench")
        logs["mode"] = "full_refresh" if (force_refresh or cache_mode == "full") else "initial_fetch"
        logs["fetch_segments"] = [meta["logs"]]
        logs["cache_after"] = _date_range_log(merged, "date")
        return merged[(merged["date"] >= req_start) & (merged["date"] <= req_end)].reset_index(drop=True), logs

    fetch_segments: List[Dict[str, object]] = []
    merged = cache_df.copy()
    cmin = merged["date"].min()
    cmax = merged["date"].max()
    if req_start < cmin:
        pre_end = cmin - pd.Timedelta(days=1)
        if req_start <= pre_end:
            pre_df, pre_meta = fetch_benchmark_daily_range(symbol, str(req_start.date()), str(pre_end.date()))
            fetch_segments.append(pre_meta["logs"])
            merged = _merge_and_save(merged, pre_df, cache_path, "bench")
    if req_end > cmax:
        post_start = cmax + pd.Timedelta(days=1)
        if post_start <= req_end:
            post_df, post_meta = fetch_benchmark_daily_range(symbol, str(post_start.date()), str(req_end.date()))
            fetch_segments.append(post_meta["logs"])
            merged = _merge_and_save(merged, post_df, cache_path, "bench")
    merged = _read_csv_or_empty(cache_path, "bench")
    logs["mode"] = "incremental"
    logs["cache_before"] = _date_range_log(cache_df, "date")
    logs["fetch_segments"] = fetch_segments
    logs["cache_after"] = _date_range_log(merged, "date")
    out = merged[(merged["date"] >= req_start) & (merged["date"] <= req_end)].reset_index(drop=True)
    return out, logs


def _choose_daily_overlap_start(raw_daily: pd.DataFrame, overlap_days: int) -> pd.Timestamp:
    if raw_daily.empty:
        return pd.Timestamp.min
    idx = max(0, len(raw_daily) - max(1, int(overlap_days)))
    return pd.to_datetime(raw_daily.iloc[idx]["date"]).normalize()


def _choose_intraday_overlap_start(raw_intraday: pd.DataFrame, overlap_days: int) -> pd.Timestamp:
    if raw_intraday.empty:
        return pd.Timestamp.min
    dates = sorted(pd.to_datetime(raw_intraday["datetime"]).dt.normalize().dropna().unique())
    idx = max(0, len(dates) - max(1, int(overlap_days)))
    start_date = pd.Timestamp(dates[idx]).normalize()
    return pd.Timestamp(start_date)


def _slice_daily_feature_request(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = _normalize_daily_feature_df(df)
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    if out.empty:
        return out
    return out[(out["date"] >= s) & (out["date"] <= e)].copy().reset_index(drop=True)


def _slice_intraday_feature_request(df: pd.DataFrame, start_dt: str, end_dt: str) -> pd.DataFrame:
    out = _normalize_intraday_feature_df(df)
    s = pd.to_datetime(start_dt)
    e = pd.to_datetime(end_dt)
    if out.empty:
        return out
    return out[(out["datetime"] >= s) & (out["datetime"] <= e)].copy().reset_index(drop=True)


def _write_temp_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _build_features_full(mod, cfg, meta, daily_df: pd.DataFrame, intraday_df: pd.DataFrame, bench_df: Optional[pd.DataFrame], tmp_dir: Path):
    daily_feat = _normalize_daily_feature_df(pd.DataFrame())
    intraday_feat = _normalize_intraday_feature_df(pd.DataFrame())

    if daily_df is not None and not daily_df.empty:
        daily_path = _write_temp_csv(daily_df, tmp_dir / "full_daily_input.csv")
        bench_path = _write_temp_csv(bench_df, tmp_dir / "full_benchmark_input.csv") if bench_df is not None else None
        daily_loaded = mod.load_daily_csv(daily_path)
        bench_loaded = mod.load_benchmark_daily_csv(bench_path) if bench_path else None
        daily_feat = _normalize_daily_feature_df(mod.build_daily_features(daily_loaded, cfg, meta, bench_daily=bench_loaded))

    if intraday_df is not None and not intraday_df.empty:
        intra_path = _write_temp_csv(intraday_df, tmp_dir / "full_intraday_input.csv")
        intraday_loaded = mod.load_intraday_csv(intra_path)
        intraday_feat = _normalize_intraday_feature_df(mod.build_intraday_features(intraday_loaded, cfg))

    return daily_feat, intraday_feat


def _build_features_tail(mod, cfg, meta, daily_tail_df: pd.DataFrame, intraday_tail_df: pd.DataFrame, bench_tail_df: Optional[pd.DataFrame], tmp_dir: Path):
    daily_feat = _normalize_daily_feature_df(pd.DataFrame())
    intraday_feat = _normalize_intraday_feature_df(pd.DataFrame())

    if daily_tail_df is not None and not daily_tail_df.empty:
        daily_path = _write_temp_csv(daily_tail_df, tmp_dir / "tail_daily_input.csv")
        bench_path = _write_temp_csv(bench_tail_df, tmp_dir / "tail_benchmark_input.csv") if bench_tail_df is not None else None
        daily_loaded = mod.load_daily_csv(daily_path)
        bench_loaded = mod.load_benchmark_daily_csv(bench_path) if bench_path else None
        daily_feat = _normalize_daily_feature_df(mod.build_daily_features(daily_loaded, cfg, meta, bench_daily=bench_loaded))

    if intraday_tail_df is not None and not intraday_tail_df.empty:
        intra_path = _write_temp_csv(intraday_tail_df, tmp_dir / "tail_intraday_input.csv")
        intraday_loaded = mod.load_intraday_csv(intra_path)
        intraday_feat = _normalize_intraday_feature_df(mod.build_intraday_features(intraday_loaded, cfg))

    return daily_feat, intraday_feat




def _compute_daily_context_days(cfg, args) -> int:
    explicit = int(getattr(args, "daily_feature_context_days", 0) or 0)
    if explicit > 0:
        return explicit
    compat = int(getattr(args, "daily_feature_overlap_days", 0) or 0)
    if compat > 0:
        return compat
    candidates = [
        int(getattr(cfg, "train_window", 500) or 500),
        int(getattr(cfg, "z_window", 120) or 120),
        120, 60, 20, 14, 10, 5,
    ]
    return int(max(candidates) + 20)


def _compute_intraday_context_days(cfg, args) -> int:
    explicit = int(getattr(args, "intraday_feature_context_days", 0) or 0)
    if explicit > 0:
        return explicit
    compat = int(getattr(args, "intraday_feature_overlap_days", 0) or 0)
    if compat > 0:
        return compat
    lookback = int(getattr(cfg, "slot_lookback_days", 20) or 20)
    return int(max(lookback + 2, 22))


def _choose_daily_context_start_for_new_rows(raw_daily: pd.DataFrame, first_new_date: pd.Timestamp, context_days: int) -> pd.Timestamp:
    raw_daily = _normalize_daily_df(raw_daily)
    if raw_daily.empty:
        return pd.Timestamp.min.normalize()
    dates = pd.to_datetime(raw_daily["date"]).dt.normalize().tolist()
    idx_new = next((i for i, d in enumerate(dates) if d >= first_new_date), 0)
    idx_ctx = max(0, idx_new - max(1, int(context_days)))
    return pd.Timestamp(dates[idx_ctx]).normalize()


def _choose_intraday_context_start_for_new_rows(raw_intraday: pd.DataFrame, first_new_dt: pd.Timestamp, context_days: int) -> pd.Timestamp:
    raw_intraday = _normalize_intraday_df(raw_intraday)
    if raw_intraday.empty:
        return pd.Timestamp.min.normalize()
    dates = sorted(pd.to_datetime(raw_intraday["datetime"]).dt.normalize().dropna().unique())
    first_new_date = pd.Timestamp(first_new_dt).normalize()
    idx_new = next((i for i, d in enumerate(dates) if pd.Timestamp(d).normalize() >= first_new_date), 0)
    idx_ctx = max(0, idx_new - max(1, int(context_days)))
    return pd.Timestamp(dates[idx_ctx]).normalize()


def _save_feature_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def _build_or_update_feature_cache(
    mod,
    cfg,
    meta,
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    bench_df: Optional[pd.DataFrame],
    feature_cache_dir: Path,
    logs: Dict[str, object],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    真实增量思路：
    1) 老特征直接复用；
    2) 只对新增原始数据对应的新样本做特征计算；
    3) 为了给新增样本提供足够上下文，只带入必要的“历史原始上下文”，
       但上下文对应的旧特征不会回写覆盖，最终只 append 新日期/新分钟特征。
    """
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    daily_feat_cache_path = feature_cache_dir / "daily_features_cache.csv"
    intraday_feat_cache_path = feature_cache_dir / "intraday_features_cache.csv"
    state_path = feature_cache_dir / "feature_state.json"
    tmp_dir = feature_cache_dir / "_tmp_build"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    force_refresh = bool(getattr(args, "force_refresh", False))
    feature_cache_mode = str(getattr(args, "feature_cache_mode", "incremental") or "incremental").lower()
    daily_context_days = _compute_daily_context_days(cfg, args)
    intraday_context_days = _compute_intraday_context_days(cfg, args)

    daily_feat_cache = _read_csv_or_empty(daily_feat_cache_path, "feat_daily")
    intraday_feat_cache = _read_csv_or_empty(intraday_feat_cache_path, "feat_intraday")

    full_rebuild = False
    reason = ""
    if force_refresh or feature_cache_mode == "full":
        full_rebuild = True
        reason = "force_refresh_or_full_mode"
    elif daily_feat_cache.empty or intraday_feat_cache.empty:
        full_rebuild = True
        reason = "feature_cache_missing"
    else:
        raw_daily_min = pd.to_datetime(daily_df["date"]).min() if not daily_df.empty else pd.NaT
        raw_intra_min = pd.to_datetime(intraday_df["datetime"]).min() if not intraday_df.empty else pd.NaT
        feat_daily_min = pd.to_datetime(daily_feat_cache["date"]).min() if not daily_feat_cache.empty else pd.NaT
        feat_intra_min = pd.to_datetime(intraday_feat_cache["datetime"]).min() if not intraday_feat_cache.empty else pd.NaT
        if (pd.notna(raw_daily_min) and pd.notna(feat_daily_min) and raw_daily_min < feat_daily_min) or \
           (pd.notna(raw_intra_min) and pd.notna(feat_intra_min) and raw_intra_min < feat_intra_min):
            full_rebuild = True
            reason = "requested_range_starts_before_feature_cache"

    if full_rebuild:
        daily_feat_full, intraday_feat_full = _build_features_full(mod, cfg, meta, daily_df, intraday_df, bench_df, tmp_dir)
        daily_feat_full = _merge_and_save(_normalize_daily_feature_df(pd.DataFrame()), daily_feat_full, daily_feat_cache_path, "feat_daily")
        intraday_feat_full = _merge_and_save(_normalize_intraday_feature_df(pd.DataFrame()), intraday_feat_full, intraday_feat_cache_path, "feat_intraday")
        feature_logs = {
            "mode": "full_rebuild",
            "reason": reason,
            "daily_cache_after": _date_range_log(daily_feat_full, "date"),
            "intraday_cache_after": _date_range_log(intraday_feat_full, "datetime"),
            "daily_context_days": daily_context_days,
            "intraday_context_days": intraday_context_days,
            "daily_appended_rows": int(len(daily_feat_full)),
            "intraday_appended_rows": int(len(intraday_feat_full)),
        }
    else:
        daily_feat_full = daily_feat_cache.copy()
        intraday_feat_full = intraday_feat_cache.copy()

        raw_daily_max = pd.to_datetime(daily_df["date"]).max() if not daily_df.empty else pd.NaT
        raw_intra_max = pd.to_datetime(intraday_df["datetime"]).max() if not intraday_df.empty else pd.NaT
        feat_daily_max = pd.to_datetime(daily_feat_cache["date"]).max() if not daily_feat_cache.empty else pd.NaT
        feat_intra_max = pd.to_datetime(intraday_feat_cache["datetime"]).max() if not intraday_feat_cache.empty else pd.NaT

        daily_append_logs: Dict[str, object] = {"needed": False}
        intraday_append_logs: Dict[str, object] = {"needed": False}

        need_daily_append = pd.notna(raw_daily_max) and (pd.isna(feat_daily_max) or raw_daily_max > feat_daily_max)
        if need_daily_append:
            first_new_date = pd.to_datetime(daily_df.loc[pd.to_datetime(daily_df["date"]) > feat_daily_max, "date"]).min()
            ctx_start = _choose_daily_context_start_for_new_rows(daily_df, first_new_date, daily_context_days)
            daily_ctx_raw = daily_df[pd.to_datetime(daily_df["date"]).dt.normalize() >= ctx_start].copy().reset_index(drop=True)
            bench_ctx_raw = None
            if bench_df is not None and not bench_df.empty:
                bench_ctx_raw = bench_df[pd.to_datetime(bench_df["date"]).dt.normalize() >= ctx_start].copy().reset_index(drop=True)
            daily_ctx_feat, _ = _build_features_tail(mod, cfg, meta, daily_ctx_raw, _normalize_intraday_df(pd.DataFrame()), bench_ctx_raw, tmp_dir)
            daily_append = daily_ctx_feat[pd.to_datetime(daily_ctx_feat["date"]).dt.normalize() >= pd.Timestamp(first_new_date).normalize()].copy().reset_index(drop=True)
            daily_feat_full = _merge_and_save(daily_feat_full, daily_append, daily_feat_cache_path, "feat_daily")
            daily_append_logs = {
                "needed": True,
                "first_new_date": str(pd.Timestamp(first_new_date).normalize()),
                "context_start": str(ctx_start),
                "context_raw_rows": int(len(daily_ctx_raw)),
                "appended_rows": int(len(daily_append)),
                "cache_before": _date_range_log(daily_feat_cache, "date"),
                "cache_after": _date_range_log(daily_feat_full, "date"),
            }

        need_intraday_append = pd.notna(raw_intra_max) and (pd.isna(feat_intra_max) or raw_intra_max > feat_intra_max)
        if need_intraday_append:
            first_new_dt = pd.to_datetime(intraday_df.loc[pd.to_datetime(intraday_df["datetime"]) > feat_intra_max, "datetime"]).min()
            ctx_start_date = _choose_intraday_context_start_for_new_rows(intraday_df, first_new_dt, intraday_context_days)
            intra_ctx_raw = intraday_df[pd.to_datetime(intraday_df["datetime"]).dt.normalize() >= ctx_start_date].copy().reset_index(drop=True)
            _, intra_ctx_feat = _build_features_tail(mod, cfg, meta, _normalize_daily_df(pd.DataFrame()), intra_ctx_raw, None, tmp_dir)
            intraday_append = intra_ctx_feat[pd.to_datetime(intra_ctx_feat["datetime"]) > pd.Timestamp(feat_intra_max)].copy().reset_index(drop=True)
            intraday_feat_full = _merge_and_save(intraday_feat_full, intraday_append, intraday_feat_cache_path, "feat_intraday")
            intraday_append_logs = {
                "needed": True,
                "first_new_datetime": str(pd.Timestamp(first_new_dt)),
                "context_start_date": str(ctx_start_date),
                "context_raw_rows": int(len(intra_ctx_raw)),
                "appended_rows": int(len(intraday_append)),
                "cache_before": _date_range_log(intraday_feat_cache, "datetime"),
                "cache_after": _date_range_log(intraday_feat_full, "datetime"),
            }

        if not need_daily_append and not need_intraday_append:
            feature_logs = {
                "mode": "cache_reuse_no_rebuild",
                "daily_cache": _date_range_log(daily_feat_full, "date"),
                "intraday_cache": _date_range_log(intraday_feat_full, "datetime"),
                "daily_context_days": daily_context_days,
                "intraday_context_days": intraday_context_days,
            }
        else:
            feature_logs = {
                "mode": "stateful_incremental_append",
                "daily_context_days": daily_context_days,
                "intraday_context_days": intraday_context_days,
                "daily_append": daily_append_logs,
                "intraday_append": intraday_append_logs,
            }

    state_obj = {
        "daily_feature_max": str(pd.to_datetime(daily_feat_full["date"]).max()) if not daily_feat_full.empty else None,
        "intraday_feature_max": str(pd.to_datetime(intraday_feat_full["datetime"]).max()) if not intraday_feat_full.empty else None,
        "daily_context_days": int(daily_context_days),
        "intraday_context_days": int(intraday_context_days),
        "feature_build_mode": feature_logs.get("mode"),
    }
    _save_feature_state(state_path, state_obj)

    logs["feature_build"] = feature_logs
    logs["feature_state"] = state_obj

    daily_request_feat = _slice_daily_feature_request(daily_feat_full, args.daily_start, args.daily_end)
    intraday_request_feat = _slice_intraday_feature_request(intraday_feat_full, args.intraday_start, args.intraday_end)

    daily_out = feature_cache_dir.parent / "daily_features.csv"
    intraday_out = feature_cache_dir.parent / "intraday_features.csv"
    daily_request_feat.to_csv(daily_out, index=False, encoding="utf-8-sig")
    intraday_request_feat.to_csv(intraday_out, index=False, encoding="utf-8-sig")

    base_paths = {
        "daily_features_csv": str(daily_out),
        "intraday_features_csv": str(intraday_out),
        "daily_features_cache_csv": str(daily_feat_cache_path),
        "intraday_features_cache_csv": str(intraday_feat_cache_path),
        "feature_state_json": str(state_path),
    }
    return daily_request_feat, intraday_request_feat, base_paths

def build_feature_frames(args):
    out_dir = ensure_dir(args.output_dir)
    raw_cache_dir = Path(getattr(args, "raw_cache_dir", "") or (out_dir / "raw_cache"))
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir = Path(getattr(args, "feature_cache_dir", "") or (out_dir / "feature_cache"))
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_mode = str(getattr(args, "cache_mode", "incremental") or "incremental").lower()
    force_refresh = bool(getattr(args, "force_refresh", False))

    logs: Dict[str, object] = {
        "symbol": args.symbol,
        "benchmark_symbol": args.benchmark_symbol,
        "daily_start": args.daily_start,
        "daily_end": args.daily_end,
        "intraday_start": args.intraday_start,
        "intraday_end": args.intraday_end,
        "perf_start": (args.perf_start or str(pd.to_datetime(args.intraday_start).date())),
        "data_source": "baostock",
        "cache_mode": cache_mode,
        "force_refresh": force_refresh,
        "raw_cache_dir": str(raw_cache_dir),
        "feature_cache_dir": str(feature_cache_dir),
        "feature_cache_mode": str(getattr(args, "feature_cache_mode", "incremental") or "incremental"),
    }

    symbol_norm = normalize_symbol(args.symbol)
    daily_cache_path = raw_cache_dir / f"{symbol_norm}_daily_raw.csv"
    intraday_cache_path = raw_cache_dir / f"{symbol_norm}_5m_raw.csv"

    daily_df, daily_logs = _incremental_fetch_daily(args.symbol, args.daily_start, args.daily_end, adjust=args.adjust, cache_path=daily_cache_path, force_refresh=force_refresh, cache_mode=cache_mode)
    daily_df = maybe_fill_event_flags(daily_df)
    logs["daily"] = daily_logs

    intraday_df, intra_logs = _incremental_fetch_intraday(args.symbol, args.intraday_start, args.intraday_end, adjust=args.adjust, cache_path=intraday_cache_path, force_refresh=force_refresh, cache_mode=cache_mode)
    logs["intraday_5m"] = intra_logs

    bench_df = None
    bench_path: Optional[Path] = None
    bench_cache_path: Optional[Path] = None
    if getattr(args, "benchmark_symbol", None):
        try:
            bench_cache_path = raw_cache_dir / f"{normalize_symbol(args.benchmark_symbol)}_benchmark_daily_raw.csv"
            bench_df, bench_logs = _incremental_fetch_benchmark(args.benchmark_symbol, args.daily_start, args.daily_end, cache_path=bench_cache_path, force_refresh=force_refresh, cache_mode=cache_mode)
            logs["benchmark_daily"] = bench_logs
            bench_path = out_dir / f"{normalize_symbol(args.benchmark_symbol)}_benchmark_daily.csv"
            bench_df.to_csv(bench_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logs["benchmark_daily"] = {"status": "failed", "error_type": type(e).__name__, "error": str(e)}
            bench_df = None
            bench_path = None

    # 当前请求区间原始 CSV 仍然导出，便于排查
    daily_path = out_dir / f"{symbol_norm}_daily.csv"
    intraday_path = out_dir / f"{symbol_norm}_5m.csv"
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    intraday_df.to_csv(intraday_path, index=False, encoding="utf-8-sig")

    mod = import_backtest_module(args.backtest_py)
    meta = mod.MetaConfig(
        exchange=args.exchange,
        board=args.board,
        security_type=args.security_type,
        lot_size=args.lot_size,
        price_limit_ratio=args.price_limit_ratio,
        no_price_limit_default=False,
        t0_eligible=False,
    )
    perf_start = args.perf_start or str(pd.to_datetime(args.intraday_start).date())
    cfg = mod.StrategyConfig(
        initial_shares=args.initial_shares,
        initial_cash=args.initial_cash,
        evaluation_start_date=perf_start,
        cost_buy_rate=args.cost_buy_rate,
        cost_sell_rate=args.cost_sell_rate,
        slippage_bps=args.slippage_bps,
        force_rebuy_at_close=not args.no_force_rebuy_close,
        verbose=not args.quiet,
    )

    daily_feat, intraday_feat, feat_paths = _build_or_update_feature_cache(
        mod=mod,
        cfg=cfg,
        meta=meta,
        daily_df=daily_df,
        intraday_df=intraday_df,
        bench_df=bench_df,
        feature_cache_dir=feature_cache_dir,
        logs=logs,
        args=args,
    )

    base_paths = {
        "daily_csv": str(daily_path),
        "intraday_csv": str(intraday_path),
        "benchmark_csv": str(bench_path) if bench_path else None,
        "raw_daily_cache_csv": str(daily_cache_path),
        "raw_intraday_cache_csv": str(intraday_cache_path),
        "raw_benchmark_cache_csv": str(bench_cache_path) if bench_cache_path else None,
        **feat_paths,
    }
    return mod, cfg, meta, daily_feat, intraday_feat, logs, base_paths
