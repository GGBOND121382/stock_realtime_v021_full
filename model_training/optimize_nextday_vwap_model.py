#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare model designs for next-day VWAP rebound filtering.

Designs:
  - return regression: predict next_day_vwap_ret_vwap on all rows
  - trade regression: predict close-only net trade return on entry rows
  - trade classification: predict whether the close-only trade is profitable on entry rows

Feature groups:
  - base: original technical/daily features
  - reversal: overnight/daytime/VWAP/intraday reversal features
  - base_fundamental: base + BaoStock valuation and PIT quarterly fundamentals
  - fundamental_only: valuation and PIT quarterly fundamentals
  - reversal_fundamental: reversal + BaoStock valuation and PIT quarterly fundamentals

Evaluation:
  - chronological train/valid/test split inherited from sample rows
  - valid chooses top-quantile thresholds
  - test reports return-filtered strategy results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"
EPS = 1e-12


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_corr(y, p, method: str) -> float:
    s1 = pd.Series(y)
    s2 = pd.Series(p)
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return float(s1.corr(s2, method=method))


def regression_metrics(y, p) -> Dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "rank_ic": safe_corr(y, p, "spearman"),
        "pearson_ic": safe_corr(y, p, "pearson"),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "target_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "target_std": float(np.std(y)),
        "pred_std": float(np.std(p)),
    }


def trade_metrics(ret: pd.Series) -> Dict:
    r = ret.dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return {"trades": 0}
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    return {
        "trades": int(len(r)),
        "win_rate": float(np.mean(r > 0)),
        "avg_return": float(np.mean(r)),
        "median_return": float(np.median(r)),
        "compound_return": float(eq[-1] - 1),
        "max_drawdown": float(np.min(dd)),
        "profit_factor": float(r[r > 0].sum() / abs(r[r < 0].sum())) if np.any(r < 0) else np.inf,
    }


def split_chrono(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * 0.60)
    n_valid = int(n * 0.20)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_valid].copy(), df.iloc[n_train + n_valid:].copy()


def add_ratio(out: pd.DataFrame, name: str, numerator: str, denominator: str) -> None:
    if numerator in out.columns and denominator in out.columns:
        out[name] = out[numerator] / out[denominator].replace(0, np.nan) - 1.0


def add_reversal_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    if "close" in out.columns:
        out["prev_close"] = out["close"].shift(1)
    add_ratio(out, "overnight_ret", "open", "prev_close")
    add_ratio(out, "intraday_ret", "close", "open")
    add_ratio(out, "close_to_vwap", "close", "daily_vwap")
    add_ratio(out, "open_to_vwap", "open", "daily_vwap")
    add_ratio(out, "high_to_close", "high", "close")
    add_ratio(out, "close_to_low", "close", "low")
    add_ratio(out, "high_to_open", "high", "open")
    add_ratio(out, "low_to_open", "low", "open")
    if {"high", "low", "close"}.issubset(out.columns):
        out["range_pct"] = out["high"] / out["low"].replace(0, np.nan) - 1.0
        out["close_range_pos"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
        out["upper_shadow_pct"] = out["high"] / out[["open", "close"]].max(axis=1).replace(0, np.nan) - 1.0
        out["lower_shadow_pct"] = out[["open", "close"]].min(axis=1) / out["low"].replace(0, np.nan) - 1.0
    for col in ["volume", "daily_vwap_volume", "daily_vwap_pv", "range_pct", "intraday_ret"]:
        if col in out.columns:
            prev_mean = out[col].shift(1).rolling(20, min_periods=10).mean()
            prev_std = out[col].shift(1).rolling(20, min_periods=10).std()
            out[f"{col}_shock20"] = out[col] / prev_mean.replace(0, np.nan) - 1.0
            out[f"{col}_z20"] = (out[col] - prev_mean) / prev_std.replace(0, np.nan)
    if {"overnight_ret", "intraday_ret"}.issubset(out.columns):
        out["overnight_intraday_reversal"] = -out["overnight_ret"] * out["intraday_ret"]
        out["overnight_plus_intraday"] = out["overnight_ret"] + out["intraday_ret"]
    if {"intraday_ret", "close_to_vwap"}.issubset(out.columns):
        out["down_day_below_vwap"] = ((out["intraday_ret"] < 0) & (out["close_to_vwap"] < 0)).astype(float)
    return out


def segment_ret(g: pd.DataFrame, start: str, end: str) -> float:
    part = g[(g["time_str"] >= start) & (g["time_str"] <= end)].sort_values("datetime")
    if part.empty:
        return np.nan
    first_open = part["open"].iloc[0]
    if not np.isfinite(first_open) or abs(first_open) < EPS:
        return np.nan
    return float(part["close"].iloc[-1] / first_open - 1.0)


def segment_vwap(g: pd.DataFrame, start: str, end: str) -> float:
    part = g[(g["time_str"] >= start) & (g["time_str"] <= end)]
    if part.empty or "amount" not in part.columns or "volume" not in part.columns:
        return np.nan
    vol = part["volume"].sum()
    return float(part["amount"].sum() / vol) if vol > 0 else np.nan


def build_intraday_reversal_features(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    bars = pd.read_csv(p, parse_dates=["datetime"])
    bars = bars.replace([np.inf, -np.inf], np.nan).dropna(subset=["datetime", "open", "high", "low", "close"])
    bars["date"] = bars["datetime"].dt.normalize()
    bars["time_str"] = bars["datetime"].dt.strftime("%H:%M:%S")
    rows = []
    for date, g in bars.groupby("date", sort=True):
        g = g.sort_values("datetime")
        row = {
            "date": date,
            "bar_count": int(len(g)),
            "first_30m_ret": segment_ret(g, "09:35:00", "10:00:00"),
            "first_60m_ret": segment_ret(g, "09:35:00", "10:30:00"),
            "morning_ret": segment_ret(g, "09:35:00", "11:30:00"),
            "afternoon_ret": segment_ret(g, "13:05:00", "15:00:00"),
            "last_30m_ret": segment_ret(g, "14:35:00", "15:00:00"),
            "last_60m_ret": segment_ret(g, "14:05:00", "15:00:00"),
            "morning_vwap": segment_vwap(g, "09:35:00", "11:30:00"),
            "afternoon_vwap": segment_vwap(g, "13:05:00", "15:00:00"),
            "last_30m_vwap": segment_vwap(g, "14:35:00", "15:00:00"),
        }
        if "volume" in g.columns:
            row["first_60m_volume_share"] = g[(g["time_str"] >= "09:35:00") & (g["time_str"] <= "10:30:00")]["volume"].sum() / max(g["volume"].sum(), EPS)
            row["last_30m_volume_share"] = g[(g["time_str"] >= "14:35:00") & (g["time_str"] <= "15:00:00")]["volume"].sum() / max(g["volume"].sum(), EPS)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date")
    if out.empty:
        return out
    for col in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
        out[f"{col}_to_close"] = np.nan
    return out


def add_reversal_features(df: pd.DataFrame, intraday_bars: str | Path | None) -> pd.DataFrame:
    out = add_reversal_daily_features(df)
    if intraday_bars:
        intra = build_intraday_reversal_features(intraday_bars)
        if not intra.empty:
            out = out.merge(intra, on="date", how="left")
            for col in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
                if col in out.columns and "close" in out.columns:
                    out[f"{col}_to_close"] = out["close"] / out[col].replace(0, np.nan) - 1.0
            if {"morning_ret", "afternoon_ret"}.issubset(out.columns):
                out["morning_afternoon_reversal"] = -out["morning_ret"] * out["afternoon_ret"]
            if {"first_60m_ret", "last_30m_ret"}.issubset(out.columns):
                out["first60_last30_reversal"] = -out["first_60m_ret"] * out["last_30m_ret"]
    return out


def add_market_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    if "bench_close" in out.columns:
        out["bench_ret20"] = out["bench_close"] / out["bench_close"].shift(20) - 1.0
        out["bench_ret60"] = out["bench_close"] / out["bench_close"].shift(60) - 1.0
        out["bench_ma20_gap"] = out["bench_close"] / out["bench_close"].shift(1).rolling(20, min_periods=10).mean() - 1.0
        out["bench_ma60_gap"] = out["bench_close"] / out["bench_close"].shift(1).rolling(60, min_periods=30).mean() - 1.0
        out["bench_vol20"] = out["bench_close"].pct_change().shift(1).rolling(20, min_periods=10).std()
    if "close" in out.columns:
        out["stock_ret20"] = out["close"] / out["close"].shift(20) - 1.0
        out["stock_ret60"] = out["close"] / out["close"].shift(60) - 1.0
        out["stock_ma20_gap"] = out["close"] / out["close"].shift(1).rolling(20, min_periods=10).mean() - 1.0
        out["stock_ma60_gap"] = out["close"] / out["close"].shift(1).rolling(60, min_periods=30).mean() - 1.0
        out["stock_vol20"] = out["close"].pct_change().shift(1).rolling(20, min_periods=10).std()
    if {"bench_ret20", "bench_ma20_gap", "stock_ret20", "stock_ma20_gap"}.issubset(out.columns):
        out["regime_market_ok"] = (
            (out["bench_ret20"] > -0.08)
            & (out["bench_ma20_gap"] > -0.05)
            & (out["stock_ret20"] > -0.18)
            & (out["stock_ma20_gap"] > -0.10)
        ).astype(float)
        out["regime_market_strict"] = (
            (out["bench_ret20"] > -0.03)
            & (out["bench_ma20_gap"] > -0.02)
            & (out["stock_ret20"] > -0.10)
            & (out["stock_ma20_gap"] > -0.05)
        ).astype(float)
    return out


def is_first_asof_drop_feature(col: str) -> bool:
    text = str(col)
    return bool(
        "range_pct" in text
        or "_range_pct_ret" in text
        or text.endswith("_range_pct_z20")
        or text.endswith("_range_pct_z60")
        or text.endswith("_range_pct_ma20_gap")
        or text.endswith("_amount_shock20")
        or text.endswith("_amount_z20")
        or text.endswith("_volume_shock20")
        or text.endswith("_volume_z20")
        or text in {"range_pct_shock20", "range_pct_z20", "range_pct_z60"}
        or text.startswith(("last_30m_", "last_60m_", "afternoon_"))
    )


def is_lagged_daily_external_feature_name(col: str) -> bool:
    text = str(col).lower()
    return (
        "_fut_" in text
        or "_future_basket_" in text
        or "_us_" in text
        or "_us_basket_" in text
        or "_stock_vs_future_basket_ret" in text
        or "_stock_vs_us_basket_ret" in text
    )


def is_asof_allowed_feature(col: str) -> bool:
    text = str(col)
    if text.endswith("_eod"):
        return False
    fund_prefixes = (
        "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        "profit_", "operation_", "growth_", "solvency_", "cashflow_", "dupont_",
        "fund_days_since_effective",
    )
    return (
        "_asof" in text
        or text.endswith("_asof1455")
        or text.startswith(fund_prefixes)
        or text in {"peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"}
        or is_lagged_daily_external_feature_name(text)
    )


def feature_groups(df: pd.DataFrame, max_missing: float, feature_time_mode: str | None = "eod") -> Dict[str, List[str]]:
    leak = {
        "date", "next_date", "next_day_vwap", "next_day_close",
        "next_day_low",
        "next_day_high", "next_day_vwap_ret_close", "next_day_vwap_ret_vwap", "next_day_close_ret_close",
        "target", "pred", "error", "abs_error", "pred_direction", "target_direction",
        "label_rev", "daily_vwap", "daily_vwap_pv", "daily_vwap_volume", "n_intraday_bars",
        "pubDate", "statDate", "effective_date", "used_pubDate",
        "entry_signal", "trade_net_close_return", "trade_net_high_return", "trade_target_or_close_return",
        "trade_label_profit", "trade_hit_label", "trade_close_profit_label",
        "entry_price", "entry_price_source",
        "feature_time_mode", "feature_cutoff_time", "asof_last_bar_time",
        "selected", "selected_return", "selected_eval_return", "eval_label",
        "score", "hit_score", "threshold", "chosen_threshold", "chosen_quantile",
        "signal", "signal_raw_score_pass",
    }
    numeric = [c for c in df.columns if c not in leak and pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in numeric if df[c].isna().mean() <= max_missing]
    if str(feature_time_mode or "eod").strip().lower() in {"asof", "asof1455"}:
        numeric = [c for c in numeric if is_asof_allowed_feature(c) and not is_first_asof_drop_feature(c)]
    fund_prefixes = (
        "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        "profit_", "operation_", "growth_", "solvency_", "cashflow_", "dupont_",
        "fund_days_since_effective",
    )
    fund = [c for c in numeric if c.startswith(fund_prefixes) or c in {"peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"}]
    reversal_prefixes = (
        "prev_close", "overnight_", "intraday_", "close_to_", "open_to_", "high_to_", "low_to_",
        "range_", "volume_", "daily_vwap_", "upper_shadow_", "lower_shadow_", "down_day_",
        "first_", "morning_", "afternoon_", "last_", "bar_count",
    )
    regime_prefixes = ("bench_ret", "bench_ma", "bench_vol", "stock_ret", "stock_ma", "stock_vol", "regime_")
    sector_prefixes = ("sector_", "stock_vs_sector_")
    hog_prefixes = ("hog_",)
    feed_prefixes = ("feed_",)
    zijin_external_prefixes = (
        "gold_", "copper_", "silver_", "zijin_hk_", "zijin_a_h_",
        "precious_", "industrial_metal_", "minor_metal_",
        "stock_vs_gold_", "stock_vs_copper_", "stock_vs_precious_", "stock_vs_industrial_",
    )
    # Generic AKShare-driven external profiles added by
    # feature_building/build_stock_external_features.py.
    stock_external_prefixes = (
        "ai_",    # ai_compute: 工业富联 AI 算力/服务器链
        "mwb_",   # material_wind_battery: 中材科技 玻纤/风电/锂电链
        "pur_",   # power_utility_rate: 中国核电 电力/公用事业/利率代理
        "fert_",  # fertilizer: 云天化 化肥/农化/商品链
        "sp_",    # storage_power: 科士达 储能/光伏/电源设备链
        "ane_",   # aero_nuclear_equipment: 应流股份 航发/军工/核电设备链
        "ocg_",   # optical_cable_grid: 中天科技/亨通光电 光通信/电缆/电网链
    )
    reversal_exact = {
        "close_range_pos", "morning_afternoon_reversal", "first60_last30_reversal",
    }
    reversal = [c for c in numeric if c.startswith(reversal_prefixes) or c in reversal_exact]
    regime = [c for c in numeric if c.startswith(regime_prefixes)]
    sector = [c for c in numeric if c.startswith(sector_prefixes)]
    hog = [c for c in numeric if c.startswith(hog_prefixes)]
    feed = [c for c in numeric if c.startswith(feed_prefixes)]
    zijin_external = [c for c in numeric if c.startswith(zijin_external_prefixes)]
    stock_external = [c for c in numeric if c.startswith(stock_external_prefixes)]
    external_all = list(dict.fromkeys(hog + feed + zijin_external + stock_external))
    base = [
        c for c in numeric
        if c not in fund and c not in reversal and c not in regime
        and c not in sector and c not in external_all
        and not c.startswith("ak_fund_") and not c.endswith("_fund")
    ]
    return {
        "base": base,
        "reversal": reversal,
        "regime": regime,
        "sector": sector,
        "hog": hog,
        "feed": feed,
        "zijin_external": zijin_external,
        "stock_external": stock_external,
        "external": external_all,
        "base_reversal": list(dict.fromkeys(base + reversal)),
        "reversal_regime": list(dict.fromkeys(reversal + regime)),
        "reversal_sector": list(dict.fromkeys(reversal + sector)),
        "base_fundamental": list(dict.fromkeys(base + fund)),
        "fundamental_only": fund,
        "reversal_fundamental": list(dict.fromkeys(reversal + fund)),
        "reversal_fundamental_regime": list(dict.fromkeys(reversal + fund + regime)),
        "reversal_fundamental_regime_sector": list(dict.fromkeys(reversal + fund + regime + sector)),
        "reversal_fundamental_regime_external": list(dict.fromkeys(reversal + fund + regime + external_all)),
        "reversal_fundamental_regime_sector_external": list(dict.fromkeys(reversal + fund + regime + sector + external_all)),
        "base_reversal_regime_external": list(dict.fromkeys(base + reversal + regime + external_all)),
        "reversal_fundamental_regime_sector_hog": list(dict.fromkeys(reversal + fund + regime + sector + hog)),
        "reversal_fundamental_regime_feed": list(dict.fromkeys(reversal + fund + regime + feed)),
        "base_reversal_regime_feed": list(dict.fromkeys(base + reversal + regime + feed)),
        "reversal_regime_zijin_external": list(dict.fromkeys(reversal + regime + zijin_external)),
        "base_reversal_regime_zijin_external": list(dict.fromkeys(base + reversal + regime + zijin_external)),
        "all_no_ak": list(dict.fromkeys(base + reversal + fund + regime + sector + external_all)),
    }


def prepare_xy(train, valid, test, cols, target_col):
    med = train[cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    X_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    X_valid = valid[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    X_test = test[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    return X_train, X_valid, X_test, train[target_col].to_numpy(float), valid[target_col].to_numpy(float), test[target_col].to_numpy(float)


def fit_regressor(X, y):
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=4,
        max_depth=2,
        learning_rate=0.03,
        n_estimators=200,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    model.fit(X, y)
    return model


def fit_classifier(X, y):
    pos = max(float(np.sum(y == 1)), 1.0)
    neg = max(float(np.sum(y == 0)), 1.0)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=4,
        max_depth=2,
        learning_rate=0.03,
        n_estimators=200,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
    )
    model.fit(X, y)
    return model


ENTRY_POLICIES = {"vwap_low", "all_days"}


def normalize_entry_policy(policy: str | None) -> str:
    """Normalize entry policy names used by search/save/predict scripts."""
    value = str(policy or "vwap_low").strip().lower().replace("-", "_")
    aliases = {
        "default": "vwap_low",
        "low_vwap": "vwap_low",
        "below_vwap": "vwap_low",
        "candidate": "vwap_low",
        "all": "all_days",
        "all_day": "all_days",
        "all_dates": "all_days",
        "full": "all_days",
    }
    value = aliases.get(value, value)
    if value not in ENTRY_POLICIES:
        raise ValueError(f"unknown entry_policy={policy}; available={sorted(ENTRY_POLICIES)}")
    return value


def compute_entry_signal(
    df: pd.DataFrame,
    entry_policy: str | None = "vwap_low",
    entry_vwap_premium_bps: float = 50.0,
    feature_time_mode: str | None = "eod",
) -> pd.Series:
    """Return tradable-row mask for a given entry policy.

    vwap_low: only allow rows where close <= daily_vwap * (1 + premium).
    all_days: allow all rows with valid close.
    """
    policy = normalize_entry_policy(entry_policy)
    mode = str(feature_time_mode or "eod").strip().lower()
    close_col = "close_asof1455" if mode in {"asof", "asof1455"} and "close_asof1455" in df.columns else "close"
    vwap_col = "vwap_asof1455" if mode in {"asof", "asof1455"} and "vwap_asof1455" in df.columns else "daily_vwap"

    if policy == "all_days":
        if close_col in df.columns:
            return pd.to_numeric(df[close_col], errors="coerce").notna()
        return pd.Series(True, index=df.index)

    if close_col not in df.columns or vwap_col not in df.columns:
        return pd.Series(False, index=df.index)
    close = pd.to_numeric(df[close_col], errors="coerce")
    vwap = pd.to_numeric(df[vwap_col], errors="coerce")
    premium = float(entry_vwap_premium_bps) / 10000.0
    return (close <= vwap * (1.0 + premium)).fillna(False)


def add_trade_returns(
    df: pd.DataFrame,
    cost_bps: float,
    target_bps: float,
    entry_policy: str | None = "vwap_low",
    entry_vwap_premium_bps: float = 50.0,
    feature_time_mode: str | None = "eod",
) -> pd.DataFrame:
    out = df.copy()
    cost = cost_bps / 10000.0
    target_ret = target_bps / 10000.0
    mode = str(feature_time_mode or "eod").strip().lower()
    entry_col = "close_asof1455" if mode in {"asof", "asof1455"} and "close_asof1455" in out.columns else "close"
    entry_price = pd.to_numeric(out[entry_col], errors="coerce")
    out["entry_price"] = entry_price
    out["entry_price_source"] = entry_col
    out["entry_signal"] = compute_entry_signal(out, entry_policy, entry_vwap_premium_bps, feature_time_mode).astype(bool)
    if "next_day_high" not in out.columns and "high" in out.columns:
        out["next_day_high"] = out["high"].shift(-1)
    out["trade_net_close_return"] = out["next_day_close"] / entry_price - 1.0 - cost
    out["trade_net_high_return"] = out["next_day_high"] / entry_price - 1.0 - cost
    out["trade_hit_label"] = (out["trade_net_high_return"] >= target_ret).astype(int)
    out["trade_target_or_close_return"] = np.where(
        out["trade_hit_label"] == 1,
        target_ret,
        out["trade_net_close_return"],
    )
    out["trade_label_profit"] = (out["trade_net_close_return"] > 0).astype(int)
    out["trade_close_profit_label"] = out["trade_label_profit"]
    return out


def eval_thresholds(valid: pd.DataFrame, test: pd.DataFrame, score_col: str, quantiles: List[float]) -> List[Dict]:
    rows = []
    valid_entries = valid[valid["entry_signal"]].copy()
    test_entries = test[test["entry_signal"]].copy()
    for q in quantiles:
        thr = float(valid_entries[score_col].quantile(q))
        for split, part in [("valid", valid_entries), ("test", test_entries)]:
            chosen = part[part[score_col] >= thr].copy()
            row = {
                "score_col": score_col,
                "quantile": q,
                "threshold": thr,
                "split": split,
                "score_mean": float(chosen[score_col].mean()) if len(chosen) else np.nan,
                "target_mean": float(chosen["next_day_vwap_ret_vwap"].mean()) if len(chosen) else np.nan,
            }
            row.update(trade_metrics(chosen["trade_net_close_return"]))
            rows.append(row)
    return rows


def eval_thresholds_by_year(valid: pd.DataFrame, test: pd.DataFrame, score_col: str, quantiles: List[float]) -> List[Dict]:
    rows = []
    valid_entries = valid[valid["entry_signal"]].copy()
    test_entries = test[test["entry_signal"]].copy()
    for q in quantiles:
        thr = float(valid_entries[score_col].quantile(q))
        for split, part in [("valid", valid_entries), ("test", test_entries)]:
            chosen = part[part[score_col] >= thr].copy()
            if chosen.empty:
                continue
            chosen["year"] = pd.to_datetime(chosen["date"]).dt.year
            for year, year_part in chosen.groupby("year"):
                row = {
                    "score_col": score_col,
                    "quantile": q,
                    "threshold": thr,
                    "split": split,
                    "year": int(year),
                    "score_mean": float(year_part[score_col].mean()),
                    "target_mean": float(year_part["next_day_vwap_ret_vwap"].mean()),
                }
                row.update(trade_metrics(year_part["trade_net_close_return"]))
                rows.append(row)
    return rows


def make_pred_frame(data: pd.DataFrame, target_col: str, group_name: str, model_type: str, split: str, score) -> pd.DataFrame:
    frame = data[["date", target_col, "entry_signal", "trade_net_close_return"]].copy()
    frame["feature_group"] = group_name
    frame["model_type"] = model_type
    frame["split"] = split
    frame["score"] = score
    return frame


def prepare_x_by_median(train: pd.DataFrame, apply: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = train[cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    x_apply = apply[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    return x_train, x_apply


def regime_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "none":
        return pd.Series(True, index=df.index)
    col = {"market_ok": "regime_market_ok", "strict": "regime_market_strict"}.get(name)
    if col and col in df.columns:
        return df[col].fillna(0).astype(float) > 0
    return pd.Series(True, index=df.index)


def iter_walk_forward_windows(df: pd.DataFrame, train_rows: int, valid_rows: int, test_rows: int) -> List[Tuple[int, int, int, int]]:
    windows = []
    start = 0
    n = len(df)
    while start + train_rows + valid_rows + test_rows <= n:
        train_end = start + train_rows
        valid_end = train_end + valid_rows
        test_end = valid_end + test_rows
        windows.append((start, train_end, valid_end, test_end))
        start += test_rows
    return windows


def choose_valid_threshold(
    valid: pd.DataFrame,
    score_col: str,
    quantiles: List[float],
    min_valid_trades: int,
    return_col: str,
    regime: str,
) -> Optional[Dict]:
    candidates = []
    valid_entries = valid[valid["entry_signal"] & regime_mask(valid, regime)].copy()
    if valid_entries.empty:
        return None
    for q in quantiles:
        threshold = float(valid_entries[score_col].quantile(q))
        chosen = valid_entries[valid_entries[score_col] >= threshold]
        metrics = trade_metrics(chosen[return_col])
        if metrics.get("trades", 0) >= min_valid_trades:
            candidates.append({"quantile": q, "threshold": threshold, **metrics})
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: (r.get("avg_return", -np.inf), r.get("profit_factor", -np.inf)), reverse=True)[0]


def eval_walk_forward_design(
    df: pd.DataFrame,
    cols: List[str],
    group_name: str,
    model_type: str,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    quantiles: List[float],
    min_valid_trades: int,
    regime: str,
    target_col: str,
) -> Tuple[List[Dict], List[pd.DataFrame], List[pd.DataFrame]]:
    rows = []
    pred_parts = []
    importance_parts = []
    windows = iter_walk_forward_windows(df, train_rows, valid_rows, test_rows)
    for window_id, (start, train_end, valid_end, test_end) in enumerate(windows, start=1):
        train = df.iloc[start:train_end].copy()
        valid = df.iloc[train_end:valid_end].copy()
        test = df.iloc[valid_end:test_end].copy()
        entry_train = train["entry_signal"].to_numpy(bool)
        if entry_train.sum() < 80:
            continue
        fit_train = train.loc[entry_train].copy()
        apply = pd.concat([valid, test], ignore_index=False)
        x_train, x_apply = prepare_x_by_median(fit_train, apply, cols)

        if model_type == "hit_classifier":
            y = fit_train["trade_hit_label"].to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            model = fit_classifier(x_train, y)
            score = model.predict_proba(x_apply)[:, 1]
            score_col = "hit_score"
            return_col = "trade_target_or_close_return"
        elif model_type == "high_regressor":
            y = fit_train["trade_net_high_return"].to_numpy(float)
            model = fit_regressor(x_train, y)
            score = model.predict(x_apply)
            score_col = "high_reg_score"
            return_col = "trade_target_or_close_return"
        else:
            y = fit_train[target_col].to_numpy(float)
            model = fit_regressor(x_train, y)
            score = model.predict(x_apply)
            score_col = "close_reg_score"
            return_col = "trade_net_close_return"

        scored = apply[["date", "entry_signal", "trade_net_close_return", "trade_net_high_return", "trade_target_or_close_return", "trade_hit_label"]].copy()
        scored["feature_group"] = group_name
        scored["model_type"] = model_type
        scored["window_id"] = window_id
        scored["score_col"] = score_col
        scored[score_col] = score
        scored["split"] = np.where(scored.index < valid_end, "valid", "test")
        valid_scored = scored[scored["split"] == "valid"].copy()
        test_scored = scored[scored["split"] == "test"].copy()
        chosen = choose_valid_threshold(valid_scored, score_col, quantiles, min_valid_trades, return_col, regime)
        if chosen is None:
            continue
        threshold = chosen["threshold"]
        for split, part in [("valid", valid_scored), ("test", test_scored)]:
            trade_part = part[part["entry_signal"] & regime_mask(part, regime) & (part[score_col] >= threshold)].copy()
            row = {
                "feature_group": group_name,
                "model_type": model_type,
                "window_id": window_id,
                "split": split,
                "regime": regime,
                "score_col": score_col,
                "quantile": chosen["quantile"],
                "threshold": threshold,
                "return_col": return_col,
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "valid_start": valid["date"].min(),
                "valid_end": valid["date"].max(),
                "test_start": test["date"].min(),
                "test_end": test["date"].max(),
            }
            row.update(trade_metrics(trade_part[return_col]))
            rows.append(row)
        scored["chosen_threshold"] = threshold
        scored["chosen_quantile"] = chosen["quantile"]
        scored["regime"] = regime
        scored["return_col"] = return_col
        selected_mask = scored["entry_signal"] & regime_mask(scored, regime) & (scored[score_col] >= threshold)
        scored["selected"] = selected_mask.astype(int)
        scored["selected_return"] = np.where(selected_mask, scored[return_col], np.nan)
        pred_parts.append(scored)
        importance_parts.append(pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_,
            "feature_group": group_name,
            "model_type": model_type,
            "window_id": window_id,
        }))
    return rows, pred_parts, importance_parts


def run_walk_forward(df: pd.DataFrame, groups: Dict[str, List[str]], out_dir: Path, args) -> None:
    selected_groups = [g.strip() for g in args.walk_forward_groups.split(",") if g.strip()]
    selected_models = [m.strip() for m in args.walk_forward_models.split(",") if m.strip()]
    selected_regimes = [r.strip() for r in args.walk_forward_regimes.split(",") if r.strip()]
    quantiles = [float(x) for x in args.walk_forward_quantiles.split(",") if x.strip()]
    metric_rows = []
    pred_parts = []
    importance_parts = []
    for group_name in selected_groups:
        cols = groups.get(group_name, [])
        if not cols:
            continue
        for model_type in selected_models:
            for regime in selected_regimes:
                rows, preds, imps = eval_walk_forward_design(
                    df=df,
                    cols=cols,
                    group_name=group_name,
                    model_type=model_type,
                    train_rows=args.walk_forward_train_rows,
                    valid_rows=args.walk_forward_valid_rows,
                    test_rows=args.walk_forward_test_rows,
                    quantiles=quantiles,
                    min_valid_trades=args.walk_forward_min_valid_trades,
                    regime=regime,
                    target_col=args.target,
                )
                metric_rows.extend(rows)
                pred_parts.extend(preds)
                importance_parts.extend(imps)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    if pred_parts:
        pd.concat(pred_parts, ignore_index=True).to_csv(out_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    if importance_parts:
        pd.concat(importance_parts, ignore_index=True).to_csv(out_dir / "walk_forward_feature_importance.csv", index=False, encoding="utf-8-sig")
    if metrics.empty:
        summary = {"walk_forward_rows": 0}
    else:
        pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
        trade_summary_rows = []
        if not pred_df.empty:
            for key_vals, part in pred_df[pred_df["split"] == "test"].groupby(["feature_group", "model_type", "regime", "return_col"], dropna=False):
                ret = part.loc[part["selected"] == 1, "selected_return"].dropna()
                row = dict(zip(["feature_group", "model_type", "regime", "return_col"], key_vals))
                row.update(trade_metrics(ret))
                row["windows"] = int(part["window_id"].nunique())
                trade_summary_rows.append(row)
        trade_summary_df = pd.DataFrame(trade_summary_rows)
        if not trade_summary_df.empty:
            trade_summary_df.to_csv(out_dir / "walk_forward_trade_summary.csv", index=False, encoding="utf-8-sig")
        key = ["feature_group", "model_type", "regime", "split", "return_col"]
        summary_df = metrics.groupby(key).agg(
            windows=("window_id", "nunique"),
            trades=("trades", "sum"),
            avg_return=("avg_return", "mean"),
            median_window_return=("avg_return", "median"),
            win_rate=("win_rate", "mean"),
            compound_windows=("compound_return", "sum"),
            worst_window=("avg_return", "min"),
            max_drawdown=("max_drawdown", "min"),
        ).reset_index()
        summary_df.to_csv(out_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
        summary = {
            "walk_forward_rows": int(len(metrics)),
            "top_test_windows": summary_df[summary_df["split"] == "test"].sort_values("avg_return", ascending=False).head(20).to_dict(orient="records"),
            "top_test_trades": trade_summary_df.sort_values("avg_return", ascending=False).head(20).to_dict(orient="records") if not trade_summary_df.empty else [],
        }
    with open(out_dir / "walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description="Optimize next-day VWAP model design with fundamentals")
    p.add_argument("--samples", default=str(SAVED_DATA_DIR / "fundamental_features_out" / "training_samples_with_fundamentals.csv"))
    p.add_argument("--out-dir", default=str(SAVED_DATA_DIR / "optimized_nextday_vwap_model_out"))
    p.add_argument("--target", default="next_day_vwap_ret_vwap")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--target-hit-bps", type=float, default=80.0)
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--intraday-bars", default=str(SAVED_DATA_DIR / "dual_opp_out_002714_v12" / "raw_cache" / "002714_5m_raw.csv"))
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--walk-forward-groups", default="reversal,reversal_fundamental,reversal_fundamental_regime,all_no_ak")
    p.add_argument("--walk-forward-models", default="hit_classifier,high_regressor,close_regressor")
    p.add_argument("--walk-forward-regimes", default="none,market_ok,strict")
    p.add_argument("--walk-forward-quantiles", default="0.5,0.6,0.7,0.8")
    p.add_argument("--walk-forward-train-rows", type=int, default=756)
    p.add_argument("--walk-forward-valid-rows", type=int, default=126)
    p.add_argument("--walk-forward-test-rows", type=int, default=63)
    p.add_argument("--walk-forward-min-valid-trades", type=int, default=12)
    p.add_argument("--entry-policy", choices=["vwap_low", "all_days"], default="vwap_low")
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)
    p.add_argument("--feature-time-mode", choices=["eod", "asof", "asof1455"], default="eod")
    p.add_argument("--feature-cutoff-time", default="")
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = pd.read_csv(args.samples, parse_dates=["date"])
    feature_time_mode = getattr(args, "feature_time_mode", "eod")
    if str(feature_time_mode).lower() not in {"asof", "asof1455"}:
        df = add_reversal_features(df, args.intraday_bars)
    df = add_market_state_features(df)
    entry_col = "close_asof1455" if str(feature_time_mode).lower() in {"asof", "asof1455"} and "close_asof1455" in df.columns else "close"
    vwap_col = "vwap_asof1455" if str(feature_time_mode).lower() in {"asof", "asof1455"} and "vwap_asof1455" in df.columns else "daily_vwap"
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[args.target, "next_day_close", entry_col, vwap_col]).reset_index(drop=True)
    df = add_trade_returns(
        df,
        args.round_trip_cost_bps,
        args.target_hit_bps,
        getattr(args, "entry_policy", "vwap_low"),
        getattr(args, "entry_vwap_premium_bps", 50.0),
        feature_time_mode,
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["trade_net_close_return", "trade_net_high_return", "trade_target_or_close_return"]).reset_index(drop=True)
    train, valid, test = split_chrono(df)
    groups = feature_groups(df, args.max_missing, feature_time_mode)
    if args.walk_forward:
        run_walk_forward(df, groups, out_dir, args)
        return
    quantiles = [0.50, 0.60, 0.70, 0.80]

    model_rows = []
    threshold_rows = []
    yearly_rows = []
    pred_frames = []
    importance_frames = []
    for group_name, cols in groups.items():
        if not cols:
            continue
        X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_xy(train, valid, test, cols, args.target)

        reg = fit_regressor(X_train, y_train)
        for split, data, X, y in [("train", train, X_train, y_train), ("valid", valid, X_valid, y_valid), ("test", test, X_test, y_test)]:
            pred = reg.predict(X)
            row = {"feature_group": group_name, "model_type": "regressor", "split": split, "n_features": len(cols)}
            row.update(regression_metrics(y, pred))
            model_rows.append(row)
            pred_frames.append(make_pred_frame(data, args.target, group_name, "regressor", split, pred))
        reg_valid = pred_frames[-2].rename(columns={"score": "reg_score"})
        reg_test = pred_frames[-1].rename(columns={"score": "reg_score"})
        threshold_rows.extend([
            {**r, "feature_group": group_name, "model_type": "regressor"}
            for r in eval_thresholds(reg_valid, reg_test, "reg_score", quantiles)
        ])
        yearly_rows.extend([
            {**r, "feature_group": group_name, "model_type": "regressor"}
            for r in eval_thresholds_by_year(reg_valid, reg_test, "reg_score", quantiles)
        ])
        importance_frames.append(pd.DataFrame({"feature": cols, "importance": reg.feature_importances_, "feature_group": group_name, "model_type": "regressor"}))

        entry_train = train["entry_signal"].to_numpy(bool)
        trade_reg = fit_regressor(X_train.loc[entry_train], train.loc[entry_train, "trade_net_close_return"].to_numpy(float))
        for split, data, X, y_trade in [
            ("train", train, X_train, train["trade_net_close_return"].to_numpy(float)),
            ("valid", valid, X_valid, valid["trade_net_close_return"].to_numpy(float)),
            ("test", test, X_test, test["trade_net_close_return"].to_numpy(float)),
        ]:
            pred = trade_reg.predict(X)
            row = {"feature_group": group_name, "model_type": "entry_trade_regressor", "split": split, "n_features": len(cols)}
            row.update(regression_metrics(y_trade, pred))
            model_rows.append(row)
            pred_frames.append(make_pred_frame(data, args.target, group_name, "entry_trade_regressor", split, pred))
        trade_reg_valid = pred_frames[-2].rename(columns={"score": "trade_reg_score"})
        trade_reg_test = pred_frames[-1].rename(columns={"score": "trade_reg_score"})
        threshold_rows.extend([
            {**r, "feature_group": group_name, "model_type": "entry_trade_regressor"}
            for r in eval_thresholds(trade_reg_valid, trade_reg_test, "trade_reg_score", quantiles)
        ])
        yearly_rows.extend([
            {**r, "feature_group": group_name, "model_type": "entry_trade_regressor"}
            for r in eval_thresholds_by_year(trade_reg_valid, trade_reg_test, "trade_reg_score", quantiles)
        ])
        importance_frames.append(pd.DataFrame({"feature": cols, "importance": trade_reg.feature_importances_, "feature_group": group_name, "model_type": "entry_trade_regressor"}))

        y_cls_fit = train.loc[entry_train, "trade_label_profit"].to_numpy(int)
        y_cls_train = train["trade_label_profit"].to_numpy(int)
        y_cls_valid = valid["trade_label_profit"].to_numpy(int)
        y_cls_test = test["trade_label_profit"].to_numpy(int)
        clf = fit_classifier(X_train.loc[entry_train], y_cls_fit)
        for split, data, X, y_cls in [("train", train, X_train, y_cls_train), ("valid", valid, X_valid, y_cls_valid), ("test", test, X_test, y_cls_test)]:
            proba = clf.predict_proba(X)[:, 1]
            row = {
                "feature_group": group_name,
                "model_type": "entry_classifier",
                "split": split,
                "n_features": len(cols),
                "auc": float(roc_auc_score(y_cls, proba)) if len(np.unique(y_cls)) > 1 else np.nan,
                "label_mean": float(np.mean(y_cls)),
                "pred_mean": float(np.mean(proba)),
            }
            model_rows.append(row)
            pred_frames.append(make_pred_frame(data, args.target, group_name, "entry_classifier", split, proba))
        clf_valid = pred_frames[-2].rename(columns={"score": "clf_score"})
        clf_test = pred_frames[-1].rename(columns={"score": "clf_score"})
        threshold_rows.extend([
            {**r, "feature_group": group_name, "model_type": "entry_classifier"}
            for r in eval_thresholds(clf_valid, clf_test, "clf_score", quantiles)
        ])
        yearly_rows.extend([
            {**r, "feature_group": group_name, "model_type": "entry_classifier"}
            for r in eval_thresholds_by_year(clf_valid, clf_test, "clf_score", quantiles)
        ])
        importance_frames.append(pd.DataFrame({"feature": cols, "importance": clf.feature_importances_, "feature_group": group_name, "model_type": "entry_classifier"}))

    model_df = pd.DataFrame(model_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    importance_df = pd.concat(importance_frames, ignore_index=True).sort_values(["feature_group", "model_type", "importance"], ascending=[True, True, False])

    model_df.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(out_dir / "threshold_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(out_dir / "yearly_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(df)),
        "splits": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "feature_time_mode": feature_time_mode,
        "feature_cutoff_time": getattr(args, "feature_cutoff_time", ""),
        "feature_groups": {k: len(v) for k, v in groups.items()},
        "top_test_by_avg_return": threshold_df[threshold_df["split"] == "test"].sort_values("avg_return", ascending=False).head(15).to_dict(orient="records"),
        "outputs": {
            "model_metrics": str(out_dir / "model_metrics.csv"),
            "threshold_strategy_metrics": str(out_dir / "threshold_strategy_metrics.csv"),
            "yearly_strategy_metrics": str(out_dir / "yearly_strategy_metrics.csv"),
            "feature_importance": str(out_dir / "feature_importance.csv"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
