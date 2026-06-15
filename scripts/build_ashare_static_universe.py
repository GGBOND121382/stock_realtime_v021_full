#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build static A-share universes without Eastmoney market-cap sources.

Policy:
  - BaoStock is the primary source for stock basic info, industry, and history.
  - AKShare is used only for market cap, and only through non-EM sources.
  - No AKShare *_em function is called by this script.
  - If non-EM market cap is unavailable, use BaoStock approximate circulating
    market cap as a low-confidence fallback and write the missing list.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_static_universe"
DEFAULT_BAOSTOCK_CACHE = DEFAULT_OUT_DIR / "baostock_daily_cache"
DEFAULT_AK_MARKETCAP_CACHE = DEFAULT_OUT_DIR / "akshare_non_em_marketcap_cache"

HISTORY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

FINAL_COLUMNS = [
    "code",
    "baostock_code",
    "name",
    "board",
    "industry",
    "industryClassification",
    "list_date",
    "asof_date",
    "total_mv",
    "circ_mv",
    "marketcap_source",
    "marketcap_confidence",
    "rank_circ_mv",
    "rank_total_mv",
    "history_start",
    "history_end",
    "valid_days_7y",
    "valid_ratio_7y",
    "valid_days_252",
    "has_recent_st",
    "is_current_st",
    "selected_for_train",
    "selected_for_trade",
]


@dataclass(frozen=True)
class CompletenessRules:
    years: int = 7
    trading_days_per_year: int = 240
    min_valid_ratio: float = 0.90
    strict_valid_ratio: float = 0.95
    recent_window: int = 252
    min_recent_valid_days: int = 240

    @property
    def expected_days(self) -> int:
        return self.years * self.trading_days_per_year

    @property
    def min_valid_days(self) -> int:
        return int(math.ceil(self.expected_days * self.min_valid_ratio))

    @property
    def strict_min_valid_days(self) -> int:
        return int(math.ceil(self.expected_days * self.strict_valid_ratio))


_BAOSTOCK = None
_BAOSTOCK_LOGGED_IN = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def market_from_code6(code6: str) -> str:
    return "sh" if code6.startswith(("6", "9")) else "sz"


def baostock_code_from_code6(code6: str) -> str:
    return f"{market_from_code6(code6)}.{normalize_code(code6)}"


def get_baostock():
    global _BAOSTOCK, _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK is None:
        import baostock as bs

        _BAOSTOCK = bs
    if not _BAOSTOCK_LOGGED_IN:
        lg = _BAOSTOCK.login()
        if getattr(lg, "error_code", "0") != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
        _BAOSTOCK_LOGGED_IN = True
        atexit.register(lambda: _BAOSTOCK.logout() if _BAOSTOCK is not None else None)
    return _BAOSTOCK


def baostock_result_to_df(rs: Any) -> pd.DataFrame:
    rows = []
    while getattr(rs, "error_code", "0") == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        found = lower.get(str(c).strip().lower())
        if found is not None:
            return found
    return None


def to_number(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def parse_date(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        dt = pd.to_datetime(digits[:8], format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).strftime("%Y-%m-%d")


def board_from_code6(code6: str) -> str:
    c = normalize_code(code6)
    if c.startswith(("600", "601", "603", "605")):
        return "sh_mainboard"
    if c.startswith(("000", "001", "002", "003")):
        return "sz_mainboard"
    if c.startswith(("300", "301")):
        return "chinext"
    if c.startswith(("688", "689")):
        return "star"
    if c.startswith(("200", "900")):
        return "b_share"
    if c.startswith(("8", "4", "9")):
        return "bse_or_other"
    return "other"


def is_common_ashare_board(board: str) -> bool:
    return board in {"sh_mainboard", "sz_mainboard", "chinext", "star"}


def is_mainboard(board: str) -> bool:
    return board in {"sh_mainboard", "sz_mainboard"}


def query_stock_basic(bs: Any) -> pd.DataFrame:
    rs = bs.query_stock_basic()
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"query_stock_basic failed: {rs.error_code} {rs.error_msg}")
    df = baostock_result_to_df(rs)
    if df.empty:
        raise RuntimeError("query_stock_basic returned no rows")
    df["code"] = df["code"].astype(str)
    df["code6"] = df["code"].map(normalize_code)
    df["baostock_code"] = df["code"]
    df["name"] = df["code_name"].astype(str).str.strip()
    df["list_date"] = df["ipoDate"].map(parse_date)
    df["out_date"] = df["outDate"].map(parse_date)
    df["board"] = df["code6"].map(board_from_code6)
    return df


def query_stock_industry(bs: Any, date: str) -> pd.DataFrame:
    rs = bs.query_stock_industry(date=date)
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"query_stock_industry failed: {rs.error_code} {rs.error_msg}")
    df = baostock_result_to_df(rs)
    if df.empty:
        raise RuntimeError("query_stock_industry returned no rows")
    df["code"] = df["code"].astype(str)
    df["code6"] = df["code"].map(normalize_code)
    df["industry"] = df["industry"].fillna("").astype(str).str.strip()
    df["industryClassification"] = df["industryClassification"].fillna("").astype(str).str.strip()
    return df


def base_prefilter(basic: pd.DataFrame, industry: pd.DataFrame, history_start: str) -> pd.DataFrame:
    merged = basic.merge(
        industry[["code", "industry", "industryClassification"]],
        on="code",
        how="left",
    )
    merged["industry"] = merged["industry"].fillna("").astype(str).str.strip()
    merged["industryClassification"] = merged["industryClassification"].fillna("").astype(str).str.strip()
    listed_for_7y = pd.to_datetime(merged["list_date"], errors="coerce") <= pd.Timestamp(history_start)
    out = merged[
        merged["type"].astype(str).eq("1")
        & merged["status"].astype(str).eq("1")
        & merged["out_date"].isna()
        & merged["board"].map(is_common_ashare_board)
        & merged["industry"].ne("")
        & listed_for_7y
    ].copy()
    return out.sort_values("code").reset_index(drop=True)


def fetch_baostock_daily(bs: Any, bs_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        bs_code,
        HISTORY_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"history query failed for {bs_code}: {rs.error_code} {rs.error_msg}")
    return baostock_result_to_df(rs)


def merge_daily_cache(cache_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if cache_path.exists():
        try:
            frames.append(pd.read_csv(cache_path, dtype={"code": str}))
        except Exception:
            pass
    if new_df is not None and not new_df.empty:
        frames.append(new_df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = merged.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return merged


def load_or_fetch_history(bs: Any, bs_code: str, start_date: str, end_date: str, cache_dir: Path) -> pd.DataFrame:
    code6 = normalize_code(bs_code)
    cache_path = cache_dir / f"{code6}_daily_raw.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"code": str})
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
        have = cached[(cached["date"] >= pd.Timestamp(start_date)) & (cached["date"] <= pd.Timestamp(end_date))]
        if (
            not have.empty
            and have["date"].min() <= pd.Timestamp(start_date) + pd.Timedelta(days=14)
            and have["date"].max() >= pd.Timestamp(end_date) - pd.Timedelta(days=14)
        ):
            return have.copy()
    new_df = fetch_baostock_daily(bs, bs_code, start_date, end_date)
    merged = merge_daily_cache(cache_path, new_df)
    if merged.empty:
        return merged
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged[(merged["date"] >= pd.Timestamp(start_date)) & (merged["date"] <= pd.Timestamp(end_date))].copy()


def compute_history_report(row: pd.Series, hist: pd.DataFrame, start_date: str, end_date: str, rules: CompletenessRules) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "code": row["code6"],
        "baostock_code": row["baostock_code"],
        "name": row["name"],
        "board": row["board"],
        "industry": row["industry"],
        "industryClassification": row["industryClassification"],
        "list_date": row["list_date"],
        "asof_date": end_date,
        "history_start": start_date,
        "history_end": end_date,
        "history_rows": 0,
        "valid_days_7y": 0,
        "valid_ratio_7y": 0.0,
        "valid_days_252": 0,
        "has_recent_st": False,
        "is_current_st": False,
        "recent_close": pd.NA,
        "recent_turn": pd.NA,
        "recent_volume": pd.NA,
        "recent_amount": pd.NA,
        "approx_float_shares": pd.NA,
        "implied_float_shares_5d_median": pd.NA,
        "implied_float_shares_5d_min": pd.NA,
        "implied_float_shares_5d_max": pd.NA,
        "implied_float_shares_5d_obs": 0,
        "implied_float_shares_5d_range_pct": pd.NA,
        "implied_float_shares_unstable_3pct": False,
        "implied_float_shares_unstable_5pct": False,
        "approx_circ_mv": pd.NA,
        "pass_90": False,
        "pass_95": False,
        "pass_recent": False,
        "pass_current_trade": False,
        "pass_st": False,
        "error": "",
    }
    if hist is None or hist.empty:
        rec["error"] = "empty_history"
        return rec

    work = hist.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for c in [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "tradestatus",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    ]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    valid = (
        work["tradestatus"].eq(1)
        & work[["open", "high", "low", "close"]].notna().all(axis=1)
        & work["volume"].gt(0)
        & work["amount"].gt(0)
    )
    recent = work.tail(rules.recent_window)
    recent_valid = valid.loc[recent.index] if not recent.empty else pd.Series(dtype=bool)
    recent_st = recent["isST"].fillna(0).eq(1) if "isST" in recent.columns else pd.Series(dtype=bool)
    current_candidates = work[
        work["tradestatus"].eq(1)
        & work["turn"].gt(0)
        & work["volume"].gt(0)
        & work["close"].gt(0)
        & work["isST"].fillna(0).ne(1)
    ].copy()
    current = current_candidates.tail(1)
    shares_window = current_candidates.tail(5).copy()
    if not shares_window.empty:
        implied = shares_window["volume"] / (shares_window["turn"] / 100.0)
        implied = implied.replace([float("inf"), float("-inf")], pd.NA).dropna()
    else:
        implied = pd.Series(dtype="float64")

    rec["history_rows"] = int(len(work))
    rec["valid_days_7y"] = int(valid.sum())
    rec["valid_ratio_7y"] = float(rec["valid_days_7y"] / rules.expected_days)
    rec["valid_days_252"] = int(recent_valid.sum())
    rec["has_recent_st"] = bool(recent_st.any())
    rec["is_current_st"] = bool(work["isST"].fillna(0).tail(1).eq(1).any())
    rec["pass_90"] = rec["valid_days_7y"] >= rules.min_valid_days
    rec["pass_95"] = rec["valid_days_7y"] >= rules.strict_min_valid_days
    rec["pass_recent"] = rec["valid_days_252"] >= rules.min_recent_valid_days
    rec["pass_st"] = (not rec["is_current_st"]) and (not rec["has_recent_st"])
    rec["pass_current_trade"] = not current.empty
    if not current.empty:
        r = current.iloc[-1]
        rec["recent_close"] = float(r["close"])
        rec["recent_turn"] = float(r["turn"])
        rec["recent_volume"] = float(r["volume"])
        rec["recent_amount"] = float(r["amount"])
        rec["approx_float_shares"] = float(r["volume"] / (r["turn"] / 100.0))
        if not implied.empty:
            median_shares = float(implied.median())
            min_shares = float(implied.min())
            max_shares = float(implied.max())
            range_pct = float((max_shares - min_shares) / median_shares) if median_shares > 0 else pd.NA
            rec["implied_float_shares_5d_median"] = median_shares
            rec["implied_float_shares_5d_min"] = min_shares
            rec["implied_float_shares_5d_max"] = max_shares
            rec["implied_float_shares_5d_obs"] = int(len(implied))
            rec["implied_float_shares_5d_range_pct"] = range_pct
            rec["implied_float_shares_unstable_3pct"] = bool(pd.notna(range_pct) and range_pct > 0.03)
            rec["implied_float_shares_unstable_5pct"] = bool(pd.notna(range_pct) and range_pct > 0.05)
            rec["approx_circ_mv"] = float(r["close"] * median_shares)
        else:
            rec["approx_circ_mv"] = float(r["close"] * rec["approx_float_shares"])
    return rec


def build_history_report(candidates: pd.DataFrame, start_date: str, end_date: str, cache_dir: Path, rules: CompletenessRules) -> pd.DataFrame:
    bs = get_baostock()
    ensure_dir(cache_dir)
    records = []
    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        try:
            hist = load_or_fetch_history(bs, row["baostock_code"], start_date, end_date, cache_dir)
            rec = compute_history_report(row, hist, start_date, end_date, rules)
        except Exception as exc:
            rec = compute_history_report(row, pd.DataFrame(), start_date, end_date, rules)
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
        if i == 1 or i % 50 == 0 or i == total:
            print(
                f"[baostock-history] {i}/{total} {row['code6']} "
                f"valid={rec['valid_days_7y']} recent={rec['valid_days_252']} "
                f"approx={rec['approx_circ_mv']} error={rec['error']}",
                flush=True,
            )
    return pd.DataFrame(records)


def eligible_from_history(report: pd.DataFrame) -> pd.DataFrame:
    return report[
        report["pass_90"].eq(True)
        & report["pass_recent"].eq(True)
        & report["pass_st"].eq(True)
        & report["pass_current_trade"].eq(True)
        & report["industry"].astype(str).str.strip().ne("")
        & report["approx_circ_mv"].notna()
        & (pd.to_numeric(report["approx_circ_mv"], errors="coerce") > 0)
    ].copy()


def akshare_symbol_163(code6: str) -> str:
    prefix = market_from_code6(code6)
    return f"{prefix}{normalize_code(code6)}"


def normalize_marketcap_from_163(raw: pd.DataFrame, code6: str, asof_date: str) -> Dict[str, Any]:
    if raw is None or raw.empty:
        return {"code": code6, "akshare_total_mv": pd.NA, "akshare_circ_mv": pd.NA, "marketcap_error": "empty"}
    date_col = first_col(raw, ["日期", "date", "Date"])
    total_col = first_col(raw, ["总市值", "total_mv", "market_cap", "总市值(元)"])
    circ_col = first_col(raw, ["流通市值", "circ_mv", "float_market_cap", "流通市值(元)"])
    if date_col is not None:
        work = raw.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work[work[date_col] <= pd.Timestamp(asof_date)].sort_values(date_col)
    else:
        work = raw.copy()
    if work.empty:
        return {"code": code6, "akshare_total_mv": pd.NA, "akshare_circ_mv": pd.NA, "marketcap_error": "no_row_before_asof"}
    row = work.iloc[-1]
    return {
        "code": code6,
        "akshare_total_mv": pd.to_numeric(row[total_col], errors="coerce") if total_col else pd.NA,
        "akshare_circ_mv": pd.to_numeric(row[circ_col], errors="coerce") if circ_col else pd.NA,
        "marketcap_error": "" if total_col or circ_col else f"missing_marketcap_columns:{list(raw.columns)}",
    }


def fetch_akshare_163_one(ak: Any, code6: str, asof_date: str, retries: int, retry_sleep: float) -> Dict[str, Any]:
    fn = getattr(ak, "stock_zh_a_hist_163", None)
    if fn is None:
        return {
            "code": code6,
            "akshare_total_mv": pd.NA,
            "akshare_circ_mv": pd.NA,
            "marketcap_source": "akshare_stock_zh_a_hist_163",
            "marketcap_error": "akshare_function_missing",
        }
    start = (pd.Timestamp(asof_date) - pd.Timedelta(days=14)).strftime("%Y%m%d")
    end = pd.Timestamp(asof_date).strftime("%Y%m%d")
    symbols = [akshare_symbol_163(code6), normalize_code(code6)]
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        for symbol in symbols:
            try:
                raw = fn(symbol=symbol, start_date=start, end_date=end)
                rec = normalize_marketcap_from_163(raw, code6, asof_date)
                rec["marketcap_source"] = "akshare_stock_zh_a_hist_163"
                return rec
            except TypeError:
                try:
                    raw = fn(symbol=symbol)
                    rec = normalize_marketcap_from_163(raw, code6, asof_date)
                    rec["marketcap_source"] = "akshare_stock_zh_a_hist_163"
                    return rec
                except Exception as exc:
                    last_exc = exc
            except Exception as exc:
                last_exc = exc
        if attempt < retries:
            time.sleep(retry_sleep * attempt)
    return {
        "code": code6,
        "akshare_total_mv": pd.NA,
        "akshare_circ_mv": pd.NA,
        "marketcap_source": "akshare_stock_zh_a_hist_163",
        "marketcap_error": f"{type(last_exc).__name__}: {last_exc}",
    }


def fetch_marketcaps_non_em(candidates: pd.DataFrame, asof_date: str, cache_dir: Path, retries: int, retry_sleep: float, sleep_seconds: float) -> pd.DataFrame:
    import akshare as ak

    ensure_dir(cache_dir)
    records = []
    total = len(candidates)
    for i, code6 in enumerate(candidates["code"].astype(str), start=1):
        cache_path = cache_dir / f"{normalize_code(code6)}.json"
        if cache_path.exists():
            try:
                rec = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                rec = fetch_akshare_163_one(ak, code6, asof_date, retries, retry_sleep)
        else:
            rec = fetch_akshare_163_one(ak, code6, asof_date, retries, retry_sleep)
            cache_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        records.append(rec)
        if i == 1 or i % 50 == 0 or i == total:
            print(f"[akshare-non-em-marketcap] {i}/{total} {code6} error={rec.get('marketcap_error','')}", flush=True)
    out = pd.DataFrame(records)
    if out.empty:
        out = pd.DataFrame(columns=["code", "akshare_total_mv", "akshare_circ_mv", "marketcap_source", "marketcap_error"])
    return out


def rank_with_marketcap(prefilter: pd.DataFrame, marketcap: pd.DataFrame) -> pd.DataFrame:
    out = prefilter.merge(marketcap, on="code", how="left")
    out["akshare_total_mv"] = pd.to_numeric(out.get("akshare_total_mv"), errors="coerce")
    out["akshare_circ_mv"] = pd.to_numeric(out.get("akshare_circ_mv"), errors="coerce")
    out["approx_circ_mv"] = pd.to_numeric(out["approx_circ_mv"], errors="coerce")
    out["circ_mv"] = out["akshare_circ_mv"].where(out["akshare_circ_mv"].notna(), out["approx_circ_mv"])
    out["total_mv"] = out["akshare_total_mv"]
    out["marketcap_source"] = out["marketcap_source"].fillna("akshare_stock_zh_a_hist_163")
    out["marketcap_confidence"] = "akshare_non_em"
    out.loc[out["akshare_circ_mv"].isna(), "marketcap_confidence"] = "low_confidence_approx"
    out.loc[out["akshare_circ_mv"].isna(), "marketcap_source"] = "baostock_approx_circ_mv"
    out = out.sort_values(["circ_mv", "approx_circ_mv"], ascending=[False, False]).reset_index(drop=True)
    out["rank_circ_mv"] = out["circ_mv"].rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    out["rank_total_mv"] = out["total_mv"].rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    return out


def build_cross_check(ranked: pd.DataFrame) -> pd.DataFrame:
    out = ranked.copy()
    out["rank_approx"] = out["approx_circ_mv"].rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    external_rank = out["akshare_circ_mv"].rank(method="first", ascending=False, na_option="keep")
    out["rank_external"] = external_rank.astype("Int64")
    out["rank_marketcap_used"] = out["circ_mv"].rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    out["rank_diff"] = out["rank_external"] - out["rank_approx"]
    out["circ_mv_ratio"] = out["akshare_circ_mv"] / out["approx_circ_mv"].replace(0, pd.NA)
    cols = [
        "code",
        "name",
        "board",
        "industry",
        "recent_close",
        "recent_turn",
        "recent_volume",
        "approx_float_shares",
        "implied_float_shares_5d_median",
        "implied_float_shares_5d_obs",
        "implied_float_shares_5d_range_pct",
        "implied_float_shares_unstable_3pct",
        "implied_float_shares_unstable_5pct",
        "approx_circ_mv",
        "akshare_circ_mv",
        "akshare_total_mv",
        "circ_mv_ratio",
        "rank_marketcap_used",
        "rank_approx",
        "rank_external",
        "rank_diff",
        "marketcap_source",
        "marketcap_confidence",
        "marketcap_error",
    ]
    return out[cols]


def make_final_universes(ranked: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_a = ranked.head(top_n).copy()
    mainboard = ranked[ranked["board"].map(is_mainboard)].head(top_n).copy()
    all_a["selected_for_train"] = True
    all_a["selected_for_trade"] = all_a["board"].map(is_mainboard)
    mainboard["selected_for_train"] = False
    mainboard["selected_for_trade"] = True
    for df in [all_a, mainboard]:
        if "marketcap_error" not in df.columns:
            df["marketcap_error"] = ""
    return all_a[FINAL_COLUMNS], mainboard[FINAL_COLUMNS]


def default_asof_date() -> str:
    now = datetime.now()
    if now.hour < 16:
        now = now - timedelta(days=1)
    while now.weekday() >= 5:
        now = now - timedelta(days=1)
    return now.strftime("%Y-%m-%d")


def quality_summary(report: pd.DataFrame, prefilter: pd.DataFrame, ranked: pd.DataFrame, all_a: pd.DataFrame, mainboard: pd.DataFrame, rules: CompletenessRules, args: argparse.Namespace) -> Dict[str, Any]:
    def board_counts(df: pd.DataFrame) -> Dict[str, int]:
        if df.empty:
            return {}
        return {str(k): int(v) for k, v in df["board"].value_counts(dropna=False).sort_index().items()}

    top_approx = set(ranked.sort_values("approx_circ_mv", ascending=False).head(args.top_n)["code"].astype(str))
    top_final = set(ranked.head(args.top_n)["code"].astype(str))
    cross = build_cross_check(ranked)
    spearman = float(cross[["approx_circ_mv", "akshare_circ_mv"]].corr(method="spearman").iloc[0, 1]) if cross["akshare_circ_mv"].notna().sum() >= 2 else None
    return {
        "asof_date": args.asof_date,
        "history_start": args.history_start,
        "history_end": args.history_end,
        "rules": asdict(rules),
        "counts": {
            "history_report_rows": int(len(report)),
            "pass_90": int(report["pass_90"].sum()) if "pass_90" in report else 0,
            "pass_95": int(report["pass_95"].sum()) if "pass_95" in report else 0,
            "prefilter_candidates": int(len(prefilter)),
            "akshare_marketcap_present": int(ranked["akshare_circ_mv"].notna().sum()) if "akshare_circ_mv" in ranked else 0,
            "akshare_marketcap_missing": int(ranked["akshare_circ_mv"].isna().sum()) if "akshare_circ_mv" in ranked else 0,
            "final_allA": int(len(all_a)),
            "final_mainboard": int(len(mainboard)),
        },
        "quality_checks": {
            "allA_exact_top_n": bool(len(all_a) == args.top_n),
            "mainboard_exact_top_n": bool(len(mainboard) == args.top_n),
            "allA_recent_valid_lt_min": int((all_a["valid_days_252"] < rules.min_recent_valid_days).sum()) if not all_a.empty else 0,
            "mainboard_recent_valid_lt_min": int((mainboard["valid_days_252"] < rules.min_recent_valid_days).sum()) if not mainboard.empty else 0,
            "allA_recent_st": int(all_a["has_recent_st"].sum()) if not all_a.empty else 0,
            "mainboard_recent_st": int(mainboard["has_recent_st"].sum()) if not mainboard.empty else 0,
            "mainboard_growth_or_star_count": int(mainboard["board"].isin(["chinext", "star"]).sum()) if not mainboard.empty else 0,
            "forbidden_eastmoney_source_count": int(ranked["marketcap_source"].astype(str).str.contains("eastmoney|_em", case=False, regex=True).sum()) if "marketcap_source" in ranked else 0,
        },
        "marketcap_cross_check": {
            "spearman_approx_vs_akshare": spearman,
            "top1000_overlap_approx_vs_final": int(len(top_approx & top_final)),
            "circ_mv_ratio_lt_0_2_or_gt_5": int(((cross["circ_mv_ratio"] < 0.2) | (cross["circ_mv_ratio"] > 5)).sum()) if "circ_mv_ratio" in cross else 0,
        },
        "board_distribution": {
            "allA_top1000": board_counts(all_a),
            "mainboard_top1000": board_counts(mainboard),
        },
        "known_limitation": "Static current snapshot universe. Use rolling point-in-time universes for rigorous historical backtests.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build static A-share universes using BaoStock plus non-EM AKShare market cap")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--asof-date", default=default_asof_date())
    p.add_argument("--history-start", default=None, help="Default: history-end minus 7 years")
    p.add_argument("--history-end", default=None, help="Default: asof-date")
    p.add_argument("--top-n", type=int, default=1000)
    p.add_argument("--prefilter-n", type=int, default=2500, help="Candidate count after BaoStock approx-circ-mv ranking")
    p.add_argument("--min-valid-ratio", type=float, default=0.90)
    p.add_argument("--strict-valid-ratio", type=float, default=0.95)
    p.add_argument("--recent-window", type=int, default=252)
    p.add_argument("--min-recent-valid-days", type=int, default=240)
    p.add_argument("--baostock-cache-dir", default=str(DEFAULT_BAOSTOCK_CACHE))
    p.add_argument("--ak-marketcap-cache-dir", default=str(DEFAULT_AK_MARKETCAP_CACHE))
    p.add_argument("--ak-retries", type=int, default=2)
    p.add_argument("--ak-retry-sleep", type=float, default=2.0)
    p.add_argument("--ak-sleep", type=float, default=0.05)
    p.add_argument("--max-history-candidates", type=int, default=None, help="Smoke-test limit before BaoStock history checks")
    p.add_argument("--max-marketcap-candidates", type=int, default=None, help="Smoke-test limit before AKShare market cap checks")
    p.add_argument("--skip-akshare-marketcap", action="store_true", help="Use BaoStock approx_circ_mv only, marked low confidence")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    args.asof_date = pd.to_datetime(args.asof_date).strftime("%Y-%m-%d")
    args.history_end = pd.to_datetime(args.history_end or args.asof_date).strftime("%Y-%m-%d")
    args.history_start = pd.to_datetime(args.history_start or (pd.Timestamp(args.history_end) - pd.DateOffset(years=7))).strftime("%Y-%m-%d")
    if str(args.baostock_cache_dir) == str(DEFAULT_BAOSTOCK_CACHE):
        args.baostock_cache_dir = str(out_dir / "baostock_daily_cache")
    if str(args.ak_marketcap_cache_dir) == str(DEFAULT_AK_MARKETCAP_CACHE):
        args.ak_marketcap_cache_dir = str(out_dir / "akshare_non_em_marketcap_cache")
    rules = CompletenessRules(
        min_valid_ratio=args.min_valid_ratio,
        strict_valid_ratio=args.strict_valid_ratio,
        recent_window=args.recent_window,
        min_recent_valid_days=args.min_recent_valid_days,
    )

    print(f"[start] asof={args.asof_date} history={args.history_start}..{args.history_end} out={out_dir}", flush=True)
    bs = get_baostock()
    basic = query_stock_basic(bs)
    industry = query_stock_industry(bs, date=args.asof_date)
    basic.to_csv(out_dir / "01_baostock_stock_basic.csv", index=False, encoding="utf-8-sig")
    industry.to_csv(out_dir / "02_baostock_stock_industry.csv", index=False, encoding="utf-8-sig")

    base = base_prefilter(basic, industry, args.history_start)
    if args.max_history_candidates is not None:
        base = base.head(args.max_history_candidates).copy()
    print(f"[base-prefilter] rows={len(base)}", flush=True)

    report = build_history_report(base, args.history_start, args.history_end, Path(args.baostock_cache_dir), rules)
    report = report.sort_values(["approx_circ_mv", "recent_amount"], ascending=[False, False]).reset_index(drop=True)
    report.to_csv(out_dir / "03_baostock_history_completeness.csv", index=False, encoding="utf-8-sig")

    eligible = eligible_from_history(report)
    prefilter = eligible.sort_values(["approx_circ_mv", "recent_amount"], ascending=[False, False]).head(args.prefilter_n).copy()
    prefilter.to_csv(out_dir / "04_baostock_prefilter_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"[history-prefilter] eligible={len(eligible)} candidates={len(prefilter)}", flush=True)

    if args.max_marketcap_candidates is not None:
        marketcap_scope = prefilter.head(args.max_marketcap_candidates).copy()
    else:
        marketcap_scope = prefilter.copy()

    if args.skip_akshare_marketcap:
        marketcap = pd.DataFrame(
            {
                "code": marketcap_scope["code"].astype(str),
                "akshare_total_mv": pd.NA,
                "akshare_circ_mv": pd.NA,
                "marketcap_source": "akshare_stock_zh_a_hist_163",
                "marketcap_error": "skipped_by_user",
            }
        )
    else:
        marketcap = fetch_marketcaps_non_em(
            marketcap_scope,
            args.asof_date,
            Path(args.ak_marketcap_cache_dir),
            retries=args.ak_retries,
            retry_sleep=args.ak_retry_sleep,
            sleep_seconds=args.ak_sleep,
        )
    marketcap.to_csv(out_dir / "05_akshare_non_em_marketcap.csv", index=False, encoding="utf-8-sig")

    ranked_scope = prefilter[prefilter["code"].isin(marketcap_scope["code"])].copy()
    ranked = rank_with_marketcap(ranked_scope, marketcap)
    cross = build_cross_check(ranked)
    cross.to_csv(out_dir / "06_marketcap_cross_check.csv", index=False, encoding="utf-8-sig")
    missing = ranked[ranked["akshare_circ_mv"].isna()].copy()
    missing.to_csv(out_dir / "05_akshare_non_em_marketcap_missing.csv", index=False, encoding="utf-8-sig")

    all_a, mainboard = make_final_universes(ranked, args.top_n)
    all_a.to_csv(out_dir / "07_universe_allA_top1000_static.csv", index=False, encoding="utf-8-sig")
    mainboard.to_csv(out_dir / "08_universe_mainboard_top1000_static.csv", index=False, encoding="utf-8-sig")

    summary = quality_summary(report, prefilter, ranked, all_a, mainboard, rules, args)
    (out_dir / "universe_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["quality_checks"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["marketcap_cross_check"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["board_distribution"], ensure_ascii=False, indent=2), flush=True)
    if len(all_a) != args.top_n or len(mainboard) != args.top_n:
        print("[warn] final universe count is below top-n; increase smoke limits/prefilter-n for full run", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
