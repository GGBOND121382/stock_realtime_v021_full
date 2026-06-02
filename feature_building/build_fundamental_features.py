#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build point-in-time fundamental features for the current stock dataset.

Data sources:
  - BaoStock: daily valuation fields and quarterly financial indicators.
  - AKShare: optional individual fund-flow fields when available.

The output is date-level and can be merged into the existing daily samples.
Quarterly financial records are aligned by publication date when BaoStock
provides pubDate.  When pubDate is missing, the script falls back to a
conservative report-date lag and records that fallback in validation output.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def normalize_symbol(symbol: str) -> Tuple[str, str, str]:
    clean = "".join(ch for ch in str(symbol) if ch.isdigit()).zfill(6)
    exchange = "sh" if clean.startswith(("6", "9")) else "sz"
    return clean, f"{exchange}.{clean}", exchange


def result_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def to_numeric_except_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.lower().endswith("date") or c in {"date", "code", "statDate", "pubDate"}:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def baostock_login():
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
    return bs


def fetch_baostock_daily_valuation(bs, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"query_history_k_data_plus failed: {rs.error_code} {rs.error_msg}")
    df = result_to_df(rs)
    if df.empty:
        return df
    df = to_numeric_except_date(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def fetch_baostock_quarter_table(bs, func_name: str, code: str, years: Iterable[int], quarters: Iterable[int]) -> pd.DataFrame:
    func = getattr(bs, func_name)
    parts = []
    for year in years:
        for quarter in quarters:
            rs = func(code=code, year=int(year), quarter=int(quarter))
            if rs.error_code != "0":
                continue
            df = result_to_df(rs)
            if not df.empty:
                df["source_table"] = func_name
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return to_numeric_except_date(out)


def infer_effective_date(fin: pd.DataFrame, fallback_lag_days: int) -> pd.DataFrame:
    out = fin.copy()
    if "pubDate" in out.columns:
        out["pubDate_dt"] = pd.to_datetime(out["pubDate"], errors="coerce")
    else:
        out["pubDate_dt"] = pd.NaT
    out["statDate_dt"] = pd.to_datetime(out.get("statDate"), errors="coerce")
    fallback = out["statDate_dt"] + pd.to_timedelta(fallback_lag_days, unit="D")
    out["used_pubDate"] = out["pubDate_dt"].notna()
    out["effective_date"] = out["pubDate_dt"].fillna(fallback) + pd.Timedelta(days=1)
    return out.dropna(subset=["effective_date"]).sort_values("effective_date")


def prefixed(df: pd.DataFrame, prefix: str, key_cols: Optional[set] = None) -> pd.DataFrame:
    if key_cols is None:
        key_cols = {"code", "pubDate", "statDate", "pubDate_dt", "statDate_dt", "used_pubDate", "effective_date", "source_table"}
    out = df.copy()
    rename = {c: f"{prefix}_{c}" for c in out.columns if c not in key_cols}
    return out.rename(columns=rename)


def build_quarterly_fundamentals(bs, code: str, start_year: int, end_year: int, fallback_lag_days: int) -> pd.DataFrame:
    specs = [
        ("query_profit_data", "profit"),
        ("query_operation_data", "operation"),
        ("query_growth_data", "growth"),
        ("query_balance_data", "solvency"),
        ("query_cash_flow_data", "cashflow"),
        ("query_dupont_data", "dupont"),
    ]
    years = range(start_year, end_year + 1)
    quarters = range(1, 5)
    merged: Optional[pd.DataFrame] = None
    raw_parts = []
    for func_name, prefix in specs:
        if not hasattr(bs, func_name):
            continue
        table = fetch_baostock_quarter_table(bs, func_name, code, years, quarters)
        if table.empty:
            continue
        raw_parts.append(table)
        table = infer_effective_date(table, fallback_lag_days)
        table = prefixed(table, prefix)
        keys = ["code", "statDate", "effective_date", "pubDate", "used_pubDate"]
        keys = [c for c in keys if c in table.columns]
        value_cols = [c for c in table.columns if c not in {"pubDate_dt", "statDate_dt", "source_table"}]
        table = table[value_cols].drop_duplicates(subset=[c for c in keys if c in table.columns])
        if merged is None:
            merged = table
        else:
            join_keys = [c for c in ["code", "statDate", "effective_date", "pubDate", "used_pubDate"] if c in merged.columns and c in table.columns]
            merged = merged.merge(table, on=join_keys, how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("effective_date").reset_index(drop=True)


def try_fetch_akshare_fund_flow(symbol: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        import akshare as ak
    except Exception as e:
        return pd.DataFrame(), f"akshare import failed: {type(e).__name__}: {e}"
    try:
        market = "sh" if symbol.startswith("6") else "sz"
        df = ak.stock_individual_fund_flow(stock=symbol, market=market)
    except Exception as e:
        return pd.DataFrame(), f"stock_individual_fund_flow failed: {type(e).__name__}: {e}"
    if df is None or df.empty:
        return pd.DataFrame(), "stock_individual_fund_flow returned empty"
    out = df.copy()
    date_col = next((c for c in out.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    if date_col is None:
        return pd.DataFrame(), f"fund flow date column not found: {list(out.columns)}"
    out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in out.columns:
        if c == "date":
            continue
        out[c] = pd.to_numeric(out[c].astype(str).str.replace("%", "", regex=False), errors="coerce")
    rename = {c: f"ak_fund_{c}" for c in out.columns if c != "date"}
    out = out.rename(columns=rename).dropna(subset=["date"]).sort_values("date")
    return out, None


def merge_features(
    sample_dates: pd.DataFrame,
    valuation: pd.DataFrame,
    quarterly: pd.DataFrame,
    fund_flow: pd.DataFrame,
) -> pd.DataFrame:
    base = sample_dates[["date"]].drop_duplicates().sort_values("date").copy()
    out = base.merge(valuation.drop(columns=["code"], errors="ignore"), on="date", how="left")

    if not quarterly.empty:
        q = quarterly.copy()
        q["effective_date"] = pd.to_datetime(q["effective_date"], errors="coerce")
        q = q.dropna(subset=["effective_date"]).sort_values("effective_date")
        out = pd.merge_asof(
            out.sort_values("date"),
            q.drop(columns=["code"], errors="ignore").sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            direction="backward",
            allow_exact_matches=True,
        )
        out["fund_days_since_effective"] = (out["date"] - out["effective_date"]).dt.days
    if not fund_flow.empty:
        out = out.merge(fund_flow, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def apply_asof1455_valuation_policy(features: pd.DataFrame, sample_dates: pd.DataFrame) -> pd.DataFrame:
    """Convert daily valuation fields to a strict 14:55-known scale.

    BaoStock daily valuation fields are date-level and may only be stable after
    the close.  For asof1455 samples, use the previous trading day's known
    valuation and scale it by today's 14:55 price over previous EOD close:

        valuation_asof_T = valuation_eod_T-1 * close_asof1455_T / close_eod_T-1

    The public feature names are intentionally kept as peTTM/pbMRQ/psTTM/
    pcfNcfTTM so downstream model definitions do not need a new feature set.
    Raw BaoStock same-day values are preserved as *_eod for audit only.
    """
    if "close_asof1455" not in sample_dates.columns or "close" not in sample_dates.columns:
        return features
    out = features.sort_values("date").reset_index(drop=True).copy()
    sample = sample_dates[["date", "close", "close_asof1455"]].drop_duplicates("date").sort_values("date").copy()
    sample["date"] = pd.to_datetime(sample["date"], errors="coerce")
    out = out.merge(sample.rename(columns={"close": "_sample_close_eod", "close_asof1455": "_sample_close_asof1455"}), on="date", how="left")
    prev_close = pd.to_numeric(out["_sample_close_eod"], errors="coerce").shift(1)
    close_asof = pd.to_numeric(out["_sample_close_asof1455"], errors="coerce")
    ratio = close_asof / prev_close.replace(0, np.nan)
    valuation_cols = ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    applied_cols = []
    for col in valuation_cols:
        if col not in out.columns:
            continue
        out[f"{col}_eod"] = out[col]
        prev_val = pd.to_numeric(out[col], errors="coerce").shift(1)
        asof_val = prev_val * ratio
        out[col] = asof_val.where(asof_val.notna(), out[col])
        applied_cols.append(col)
    out["valuation_time_mode"] = "asof1455_from_lagged_eod" if applied_cols else ""
    out["valuation_reference_lag_days"] = 1 if applied_cols else np.nan
    return out.drop(columns=["_sample_close_eod", "_sample_close_asof1455"], errors="ignore")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        if col in out.columns:
            out[f"{col}_rank252"] = out[col].rolling(252, min_periods=60).rank(pct=True)
    numeric_cols = [c for c in out.columns if c not in {"date", "pubDate", "statDate"} and pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_cols:
        if col.startswith(("profit_", "growth_", "solvency_", "cashflow_", "dupont_")):
            out[f"{col}_chg4q"] = out[col] - out[col].shift(4)
    fund_cols = [c for c in out.columns if c.startswith("ak_fund_") and pd.api.types.is_numeric_dtype(out[c])]
    for col in fund_cols[:12]:
        out[f"{col}_sum5"] = out[col].rolling(5, min_periods=3).sum()
        out[f"{col}_sum10"] = out[col].rolling(10, min_periods=5).sum()
    return out


def validate_features(features: pd.DataFrame, sample_dates: pd.DataFrame, quarterly: pd.DataFrame, errors: List[str]) -> Dict:
    report: Dict[str, object] = {
        "rows": int(len(features)),
        "date_min": str(features["date"].min().date()) if len(features) else None,
        "date_max": str(features["date"].max().date()) if len(features) else None,
        "source_errors": errors,
    }
    if len(features):
        miss = features.isna().mean().sort_values(ascending=False)
        report["top_missing_rates"] = {k: float(v) for k, v in miss.head(30).items()}
        report["valuation_non_null"] = {c: int(features[c].notna().sum()) for c in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"] if c in features.columns}
        if "effective_date" in features.columns:
            future_mask = pd.to_datetime(features["effective_date"], errors="coerce") > features["date"]
            report["future_effective_date_violations"] = int(future_mask.sum())
            report["rows_with_quarterly"] = int(features["effective_date"].notna().sum())
    report["sample_rows"] = int(len(sample_dates))
    report["quarterly_raw_rows"] = int(len(quarterly))
    if not quarterly.empty and "used_pubDate" in quarterly.columns:
        report["quarterly_used_pubDate_rows"] = int(quarterly["used_pubDate"].sum())
        report["quarterly_fallback_rows"] = int((~quarterly["used_pubDate"]).sum())
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build fundamental features and merge them into daily samples")
    p.add_argument("--symbol", default="002714")
    p.add_argument("--daily-samples", default=str(SAVED_DATA_DIR / "603308_pipeline_out" / "01_samples" / "training_samples.csv"))
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "603308_pipeline_out" / "02_fundamental"))
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--fallback-lag-days", type=int, default=120)
    p.add_argument("--skip-akshare", action="store_true")
    args = p.parse_args()

    start = time.time()
    out_dir = ensure_dir(args.out_dir)
    symbol, bs_code, _ = normalize_symbol(args.symbol)
    samples = pd.read_csv(args.daily_samples, parse_dates=["date"]).sort_values("date")
    start_date = args.start_date or str(samples["date"].min().date())
    end_date = args.end_date or str(samples["date"].max().date())
    errors: List[str] = []

    bs = baostock_login()
    try:
        valuation = fetch_baostock_daily_valuation(bs, bs_code, start_date, end_date)
        quarterly = build_quarterly_fundamentals(
            bs,
            bs_code,
            pd.to_datetime(start_date).year - 1,
            pd.to_datetime(end_date).year,
            args.fallback_lag_days,
        )
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    fund_flow = pd.DataFrame()
    if not args.skip_akshare:
        fund_flow, err = try_fetch_akshare_fund_flow(symbol)
        if err:
            errors.append(err)

    features = merge_features(samples, valuation, quarterly, fund_flow)
    features = apply_asof1455_valuation_policy(features, samples)
    features = add_derived_features(features)
    merged = samples.merge(features, on="date", how="left", suffixes=("", "_fund"))

    valuation.to_csv(out_dir / "baostock_daily_valuation.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(out_dir / "baostock_quarterly_fundamentals_pit.csv", index=False, encoding="utf-8-sig")
    if not fund_flow.empty:
        fund_flow.to_csv(out_dir / "akshare_fund_flow.csv", index=False, encoding="utf-8-sig")
    features.to_csv(out_dir / "fundamental_features.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(out_dir / "training_samples_with_fundamentals.csv", index=False, encoding="utf-8-sig")

    report = validate_features(features, samples, quarterly, errors)
    report.update({
        "elapsed_seconds": round(time.time() - start, 3),
        "symbol": symbol,
        "baostock_code": bs_code,
        "valuation_rows": int(len(valuation)),
        "fund_flow_rows": int(len(fund_flow)),
        "feature_columns": int(len(features.columns)),
        "merged_columns": int(len(merged.columns)),
        "outputs": {
            "daily_valuation": str(out_dir / "baostock_daily_valuation.csv"),
            "quarterly_fundamentals": str(out_dir / "baostock_quarterly_fundamentals_pit.csv"),
            "fundamental_features": str(out_dir / "fundamental_features.csv"),
            "merged_samples": str(out_dir / "training_samples_with_fundamentals.csv"),
        },
    })
    save_json(report, out_dir / "validation_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
