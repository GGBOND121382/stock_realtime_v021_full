#!/usr/bin/env python3
"""Per-symbol single-target asof1455 regression model search.

This is intentionally separate from the pooled cross-sectional ML4T search:
each stock_code gets its own time-series CV and its own model fit. The target is
always next-day close return from the 14:55 entry price, in bps.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
except Exception as exc:  # pragma: no cover
    RandomForestRegressor = DecisionTreeRegressor = None
    SimpleImputer = LinearRegression = Ridge = Lasso = ElasticNet = None
    make_pipeline = StandardScaler = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover
    LGBMRegressor = None
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None

try:
    from catboost import CatBoostRegressor
except Exception as exc:  # pragma: no cover
    CatBoostRegressor = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None


RANDOM_STATE = 42
REQUIRED_SAMPLE_COLS = [
    "date",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "open_asof1455",
    "high_asof1455",
    "low_asof1455",
    "close_asof1455",
    "vwap_asof1455",
    "volume_asof1455",
    "amount_asof1455",
    "next_day_close",
]
DEFAULT_SAMPLE_GLOBS = [
    "saved_data/*_pipeline_out/04_external/*/training_samples_with_*external*.csv",
    "saved_data/*_pipeline_out/03_sector/training_samples_with_sector.csv",
    "saved_data/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv",
    "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv",
]
REQUIRED_FEATURE_SET_A_COLS = {
    "date",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "close_asof1455",
    "high_asof1455",
    "low_asof1455",
    "volume_asof1455",
    "amount_asof1455",
    "next_day_close",
}
LEAK_COLS = {
    "date", "next_date", "next_day_vwap", "next_day_close", "next_day_low", "next_day_high",
    "next_day_vwap_ret_close", "next_day_vwap_ret_vwap", "next_day_close_ret_close",
    "target", "target_next_close_bps", "pred", "error", "abs_error", "pred_direction",
    "target_direction", "label_rev", "daily_vwap", "daily_vwap_pv", "daily_vwap_volume",
    "n_intraday_bars", "pubDate", "statDate", "effective_date", "used_pubDate",
    "entry_signal", "trade_net_close_return", "trade_net_high_return",
    "trade_target_or_close_return", "trade_label_profit", "trade_hit_label",
    "trade_close_profit_label", "entry_price", "entry_price_source", "feature_time_mode",
    "feature_cutoff_time", "asof_last_bar_time", "selected", "selected_return",
    "selected_eval_return", "eval_label", "score", "hit_score", "threshold",
    "chosen_threshold", "chosen_quantile", "signal", "signal_raw_score_pass",
}
FUND_PREFIXES = (
    "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
    "profit_", "operation_", "growth_", "solvency_", "cashflow_", "dupont_",
    "fund_days_since_effective",
)
REGIME_PREFIXES = ("bench_ret", "bench_ma", "bench_vol", "stock_ret", "stock_ma", "stock_vol", "regime_")
SECTOR_PREFIXES = ("sector_", "stock_vs_sector_")
EXTERNAL_PREFIXES = (
    "hog_", "feed_", "gold_", "copper_", "silver_", "zijin_hk_", "zijin_a_h_",
    "precious_", "industrial_metal_", "minor_metal_", "stock_vs_gold_",
    "stock_vs_copper_", "stock_vs_precious_", "stock_vs_industrial_",
    "ai_", "mwb_", "pur_", "fert_", "sp_", "ane_", "ocg_",
)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    name: str
    params: Dict


def parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("input sample is missing required column: date")
    raw = df["date"]
    if pd.api.types.is_integer_dtype(raw) or pd.api.types.is_float_dtype(raw):
        as_str = raw.dropna().astype("Int64").astype(str)
        if len(as_str) and as_str.str.fullmatch(r"\d{8}").mean() > 0.8:
            df["date"] = pd.to_datetime(raw.astype("Int64").astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
        else:
            df["date"] = pd.to_datetime(raw, errors="coerce").dt.normalize()
    else:
        as_str = raw.astype(str).str.strip()
        if len(as_str.dropna()) and as_str.dropna().str.fullmatch(r"\d{8}").mean() > 0.8:
            df["date"] = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce").dt.normalize()
        else:
            df["date"] = pd.to_datetime(raw, errors="coerce").dt.normalize()
    bad = int(df["date"].isna().sum())
    if bad:
        raise ValueError(f"failed to parse {bad} date values")
    return df


def infer_stock_code(path: Path, df: pd.DataFrame) -> str:
    for col in ["stock_code", "symbol", "code", "ts_code"]:
        if col in df.columns and df[col].notna().any():
            return str(df[col].dropna().iloc[0]).strip()
    text = str(path)
    import re

    m = re.search(r"(\d{6})(?:[_-]pipeline|[_-]asof|[_-]5m|\.csv)", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6})", path.name)
    return m.group(1) if m else path.parent.name


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    m = xx.notna() & yy.notna()
    if int(m.sum()) < 3 or xx[m].nunique() < 2 or yy[m].nunique() < 2:
        return np.nan
    return float(xx[m].corr(yy[m], method="spearman"))


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    m = xx.notna() & yy.notna()
    if int(m.sum()) < 3 or xx[m].nunique() < 2 or yy[m].nunique() < 2:
        return np.nan
    return float(xx[m].corr(yy[m], method="pearson"))


def date_windows(dates: Sequence[pd.Timestamp], train_days: int, test_days: int, embargo_days: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    need = int(train_days + embargo_days + test_days)
    start = 0
    while start + need <= len(unique_dates):
        train = unique_dates[start : start + train_days]
        test_start = start + train_days + embargo_days
        test = unique_dates[test_start : test_start + test_days]
        windows.append((train, test))
        start += int(test_days)
    return windows


def pipeline_root(path: Path) -> str:
    for parent in [path] + list(path.parents):
        if parent.name.endswith("_pipeline_out"):
            return str(parent)
    return str(path.parent)


def pipeline_stock_code(path: Path) -> str:
    # The symbol filter must be based on the canonical pipeline root only.
    # Do not match by substring in the full path: recycle/backup paths can also
    # contain the same code and must never be selected for a symbol run.
    for parent in [path] + list(path.parents):
        if parent.name.endswith("_pipeline_out"):
            return parent.name[: -len("_pipeline_out")]
    return ""


def sample_priority(path: Path) -> int:
    text = str(path).replace("\\", "/")
    if "/04_external/" in text:
        return 4
    if "/03_sector/" in text:
        return 3
    if "/02_fundamental/" in text:
        return 2
    if "/01_samples_asof1455/" in text:
        return 1
    return 0


def sample_has_required_columns(path: Path) -> bool:
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return REQUIRED_FEATURE_SET_A_COLS.issubset(cols)


def is_recycle_path(path: Path) -> bool:
    # Recycle directories are archival output from cleanup. They are not valid
    # training/search inputs, even if their nested pipeline directory name looks
    # like the requested symbol.
    return any(part.startswith("_recycle_data_cleanup_") for part in path.parts)


def expand_samples(sample_paths: Sequence[str], sample_globs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in sample_paths:
        path = Path(p)
        if is_recycle_path(path):
            continue
        if path.exists() and path.is_file():
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    for pat in sample_globs:
        for name in glob.glob(pat, recursive=True):
            path = Path(name)
            if is_recycle_path(path):
                continue
            if path.exists() and path.is_file():
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(path)
    explicit = [Path(p) for p in sample_paths if Path(p).exists() and Path(p).is_file() and not is_recycle_path(Path(p))]
    explicit_keys = {str(p.resolve()) for p in explicit}
    chosen: Dict[str, List[Path]] = {}
    for path in out:
        if not sample_has_required_columns(path):
            continue
        if str(path.resolve()) in explicit_keys:
            chosen.setdefault(str(path.resolve()), [path])
            continue
        root = pipeline_root(path)
        current = chosen.get(root, [])
        if not current:
            chosen[root] = [path]
            continue
        old_pri = sample_priority(current[0])
        new_pri = sample_priority(path)
        if new_pri > old_pri:
            chosen[root] = [path]
        elif new_pri == old_pri:
            current.append(path)
    flattened = [p for paths in chosen.values() for p in paths]
    return sorted(flattened, key=lambda p: (pipeline_root(p), -sample_priority(p), str(p)))


def load_one_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_date_col(df)
    if "stock_code" not in df.columns:
        df["stock_code"] = infer_stock_code(path, df)
    return df


def build_feature_set_a(df: pd.DataFrame, round_trip_cost_bps: float) -> Tuple[pd.DataFrame, List[str]]:
    g = df.sort_values("date").copy()
    required = ["close", "open", "high", "low", "volume", "close_asof1455", "high_asof1455", "low_asof1455", "volume_asof1455", "amount_asof1455", "next_day_close"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"missing required columns for Feature Set A: {missing}")

    close = to_num(g["close"])
    prev_close = close.shift(1)
    openp = to_num(g["open"])
    high = to_num(g["high"])
    low = to_num(g["low"])
    volume = to_num(g["volume"])
    amount = to_num(g["amount"]) if "amount" in g.columns else close * volume
    close1455 = to_num(g["close_asof1455"])
    high1455 = to_num(g["high_asof1455"])
    low1455 = to_num(g["low_asof1455"])
    volume1455 = to_num(g["volume_asof1455"])
    amount1455 = to_num(g["amount_asof1455"])

    g["target_next_close_bps"] = 10000.0 * (to_num(g["next_day_close"]) / close1455.replace(0, np.nan) - 1.0) - float(round_trip_cost_bps)
    g["ret_prevclose_to_1455"] = close1455 / prev_close.replace(0, np.nan) - 1.0
    g["ret_open_to_1455"] = close1455 / openp.replace(0, np.nan) - 1.0
    g["gap_open"] = openp / prev_close.replace(0, np.nan) - 1.0
    if "vwap_asof1455" in g.columns:
        g["vwap_dev_1455"] = close1455 / to_num(g["vwap_asof1455"]).replace(0, np.nan) - 1.0
    g["range_1455"] = high1455 / low1455.replace(0, np.nan) - 1.0
    g["pos_in_range_1455"] = (close1455 - low1455) / (high1455 - low1455).replace(0, np.nan)
    g["high_ret_1455"] = high1455 / prev_close.replace(0, np.nan) - 1.0
    g["low_ret_1455"] = low1455 / prev_close.replace(0, np.nan) - 1.0
    g["amount_ratio_1455"] = amount1455 / amount.shift(1).rolling(20, min_periods=10).mean().replace(0, np.nan)
    g["volume_ratio_1455"] = volume1455 / volume.shift(1).rolling(20, min_periods=10).mean().replace(0, np.nan)

    for h in [1, 2, 3, 5, 10, 20]:
        g[f"ret_{h}d"] = prev_close / close.shift(h + 1).replace(0, np.nan) - 1.0

    daily_ret = close / close.shift(1).replace(0, np.nan) - 1.0
    known_ret = daily_ret.shift(1)
    for w in [5, 10, 20]:
        mean = known_ret.rolling(w, min_periods=max(3, w // 2)).mean()
        std = known_ret.rolling(w, min_periods=max(3, w // 2)).std()
        g[f"rolling_ret_mean_{w}"] = mean
        g[f"rolling_ret_std_{w}"] = std
        if w == 20:
            g["rolling_ret_z_20"] = (known_ret - mean) / std.replace(0, np.nan)

    eod_range = (high / low.replace(0, np.nan) - 1.0).shift(1)
    g["rolling_range_mean_10"] = eod_range.rolling(10, min_periods=5).mean()
    vol_mean = volume.shift(1).rolling(20, min_periods=10).mean()
    vol_std = volume.shift(1).rolling(20, min_periods=10).std()
    amt_mean = amount.shift(1).rolling(20, min_periods=10).mean()
    amt_std = amount.shift(1).rolling(20, min_periods=10).std()
    g["rolling_volume_z_20"] = (volume1455 - vol_mean) / vol_std.replace(0, np.nan)
    g["rolling_amount_z_20"] = (amount1455 - amt_mean) / amt_std.replace(0, np.nan)

    features = [
        "ret_prevclose_to_1455",
        "ret_open_to_1455",
        "gap_open",
        "vwap_dev_1455",
        "range_1455",
        "pos_in_range_1455",
        "high_ret_1455",
        "low_ret_1455",
        "amount_ratio_1455",
        "volume_ratio_1455",
        "ret_1d",
        "ret_2d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "rolling_ret_mean_5",
        "rolling_ret_mean_10",
        "rolling_ret_mean_20",
        "rolling_ret_std_5",
        "rolling_ret_std_10",
        "rolling_ret_std_20",
        "rolling_ret_z_20",
        "rolling_range_mean_10",
        "rolling_volume_z_20",
        "rolling_amount_z_20",
    ]
    features = [f for f in features if f in g.columns]
    g = g.replace([np.inf, -np.inf], np.nan)
    return g, features


def build_technical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    g = df.copy()
    close = to_num(g["close"])
    high = to_num(g["high"])
    low = to_num(g["low"])
    openp = to_num(g["open"])
    volume = to_num(g["volume"])
    amount = to_num(g["amount"]) if "amount" in g.columns else close * volume
    close1455 = to_num(g["close_asof1455"])
    high1455 = to_num(g["high_asof1455"])
    low1455 = to_num(g["low_asof1455"])
    prev_close = close.shift(1)
    known_close = prev_close
    known_ret = close.pct_change().shift(1)

    features: List[str] = []
    for w in [3, 5, 10, 20, 40, 60, 120]:
        ma = known_close.rolling(w, min_periods=max(3, w // 2)).mean()
        std = known_close.rolling(w, min_periods=max(3, w // 2)).std()
        g[f"tech_close_ma_dev_{w}"] = close1455 / ma.replace(0, np.nan) - 1.0
        g[f"tech_close_z_{w}"] = (close1455 - ma) / std.replace(0, np.nan)
        g[f"tech_ret_sum_{w}"] = known_ret.rolling(w, min_periods=max(3, w // 2)).sum()
        g[f"tech_ret_vol_{w}"] = known_ret.rolling(w, min_periods=max(3, w // 2)).std()
        features.extend([f"tech_close_ma_dev_{w}", f"tech_close_z_{w}", f"tech_ret_sum_{w}", f"tech_ret_vol_{w}"])

    delta = close.diff().shift(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for w in [6, 12, 14, 24]:
        avg_gain = gain.rolling(w, min_periods=max(3, w // 2)).mean()
        avg_loss = loss.rolling(w, min_periods=max(3, w // 2)).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g[f"tech_rsi_{w}"] = 100.0 - (100.0 / (1.0 + rs))
        features.append(f"tech_rsi_{w}")

    ema12 = known_close.ewm(span=12, adjust=False, min_periods=6).mean()
    ema26 = known_close.ewm(span=26, adjust=False, min_periods=13).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=5).mean()
    g["tech_macd"] = macd / known_close.replace(0, np.nan)
    g["tech_macd_signal"] = signal / known_close.replace(0, np.nan)
    g["tech_macd_hist"] = (macd - signal) / known_close.replace(0, np.nan)
    features.extend(["tech_macd", "tech_macd_signal", "tech_macd_hist"])

    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1).shift(1)
    for w in [10, 14, 20]:
        g[f"tech_atr_pct_{w}"] = tr.rolling(w, min_periods=max(3, w // 2)).mean() / known_close.replace(0, np.nan)
        features.append(f"tech_atr_pct_{w}")

    daily_range = (high / low.replace(0, np.nan) - 1.0).shift(1)
    asof_range = high1455 / low1455.replace(0, np.nan) - 1.0
    g["tech_asof_vs_range20"] = asof_range / daily_range.rolling(20, min_periods=10).mean().replace(0, np.nan)
    g["tech_open_gap_z20"] = ((openp / prev_close.replace(0, np.nan) - 1.0) - known_ret.rolling(20, min_periods=10).mean()) / known_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    g["tech_amount_turnover_z20"] = (amount - amount.shift(1).rolling(20, min_periods=10).mean()) / amount.shift(1).rolling(20, min_periods=10).std().replace(0, np.nan)
    features.extend(["tech_asof_vs_range20", "tech_open_gap_z20", "tech_amount_turnover_z20"])
    return g.replace([np.inf, -np.inf], np.nan), features


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


def is_context_col(col: str) -> bool:
    text = str(col)
    lower = text.lower()
    return (
        text.startswith(FUND_PREFIXES)
        or text in {"peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"}
        or text.startswith(REGIME_PREFIXES)
        or text.startswith(SECTOR_PREFIXES)
        or text.startswith(EXTERNAL_PREFIXES)
        or is_lagged_daily_external_feature_name(text)
        or lower.endswith("_rank252")
        or lower.endswith("_rank60")
        or lower.endswith("_rank20")
        or "_rank" in lower
    )


def native_context_features(df: pd.DataFrame, max_missing: float) -> List[str]:
    features: List[str] = []
    for col in df.columns:
        if col in LEAK_COLS or col in REQUIRED_SAMPLE_COLS or col == "stock_code":
            continue
        if col.endswith("_eod"):
            continue
        if not is_context_col(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 3 or s.isna().mean() > max_missing:
            continue
        features.append(col)
    return features


def build_feature_sets(raw: pd.DataFrame, round_trip_cost_bps: float, max_missing: float, selected: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df, a_features = build_feature_set_a(raw, round_trip_cost_bps=round_trip_cost_bps)
    df, tech_features = build_technical_features(df)
    context_features = native_context_features(df, max_missing=max_missing)
    selected_set = {x.strip().lower() for x in selected if x.strip()}
    if "all" in selected_set:
        selected_set.update(["a", "b", "c", "full"])

    feature_sets: Dict[str, List[str]] = {}
    if "a" in selected_set or "base" in selected_set:
        feature_sets["A"] = a_features
    if "b" in selected_set or "technical" in selected_set:
        feature_sets["B_technical"] = list(dict.fromkeys(a_features + tech_features))
    if "c" in selected_set or "context" in selected_set:
        feature_sets["C_context"] = list(dict.fromkeys(a_features + context_features))
    if "full" in selected_set or "abc" in selected_set:
        feature_sets["ABC_full"] = list(dict.fromkeys(a_features + tech_features + context_features))

    cleaned: Dict[str, List[str]] = {}
    for name, cols in feature_sets.items():
        usable = []
        for col in cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() >= 3 and s.isna().mean() <= max_missing:
                usable.append(col)
        if usable:
            cleaned[name] = usable
    return df.replace([np.inf, -np.inf], np.nan), cleaned


def make_model_specs(model_families: Sequence[str], n_jobs: int) -> List[ModelSpec]:
    families = {m.strip().lower() for m in model_families if m.strip()}
    specs: List[ModelSpec] = []
    if "constant" in families:
        specs.append(ModelSpec("constant", "constant_mean", {}))
    if "ewma" in families:
        for h in [5, 10, 20]:
            specs.append(ModelSpec("ewma", f"last_ewma_h{h}", {"halflife": h}))
    if "ols" in families:
        specs.append(ModelSpec("ols", "ols", {}))
    if "ridge" in families:
        for a in [0.1, 1, 3, 10, 30, 100]:
            specs.append(ModelSpec("ridge", f"ridge_alpha{a:g}", {"alpha": a}))
    if "lasso" in families:
        for a in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
            specs.append(ModelSpec("lasso", f"lasso_alpha{a:g}", {"alpha": a}))
    if "elasticnet" in families:
        for a in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            for r in [0.05, 0.15, 0.5]:
                specs.append(ModelSpec("elasticnet", f"elasticnet_alpha{a:g}_l1{r:g}", {"alpha": a, "l1_ratio": r}))
    if "tree" in families:
        for d in [2, 3, 4, 6]:
            for leaf in [30, 50, 100]:
                for mf in ["sqrt", 0.7, 1.0]:
                    specs.append(ModelSpec("tree", f"tree_d{d}_leaf{leaf}_mf{mf}", {"max_depth": d, "min_samples_leaf": leaf, "max_features": mf}))
    if "randomforest" in families:
        for d in [3, 5, 8]:
            for leaf in [20, 50]:
                for mf in ["sqrt", 0.5, 0.8]:
                    specs.append(ModelSpec("randomforest", f"rf_d{d}_leaf{leaf}_mf{mf}", {"max_depth": d, "min_samples_leaf": leaf, "max_features": mf, "n_jobs": n_jobs}))
    if "lgbm_l2" in families:
        specs.extend(make_lgbm_specs("lgbm_l2"))
    if "lgbm_l1" in families:
        specs.extend(make_lgbm_specs("lgbm_l1"))
    if "lgbm_huber" in families:
        specs.extend(make_lgbm_specs("lgbm_huber"))
    if "catboost_rmse" in families:
        specs.extend(make_catboost_specs("catboost_rmse"))
    if "catboost_mae" in families:
        specs.extend(make_catboost_specs("catboost_mae"))
    if "catboost_huber" in families:
        specs.extend(make_catboost_specs("catboost_huber"))
    return specs


def make_lgbm_specs(family: str) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for leaves in [4, 8]:
        for depth in [2, 3]:
            for min_leaf in [30, 50, 80]:
                for ff in [0.7, 0.9]:
                    for l1 in [0, 1]:
                        for l2 in [1, 10]:
                            name = f"{family}_leaves{leaves}_d{depth}_leaf{min_leaf}_ff{ff:g}_l1{l1:g}_l2{l2:g}"
                            out.append(ModelSpec(family, name, {
                                "num_leaves": leaves,
                                "max_depth": depth,
                                "min_data_in_leaf": min_leaf,
                                "feature_fraction": ff,
                                "lambda_l1": l1,
                                "lambda_l2": l2,
                            }))
    return out


def make_catboost_specs(family: str) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for depth in [2, 3, 4]:
        for l2 in [3, 10, 30]:
            for lr in [0.02, 0.03]:
                name = f"{family}_d{depth}_lr{lr:g}_l2{l2:g}"
                out.append(ModelSpec(family, name, {"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2}))
    return out


def instantiate(spec: ModelSpec, n_jobs: int):
    if spec.family in {"constant", "ewma"}:
        return None
    if spec.family in {"ols", "ridge", "lasso", "elasticnet", "tree", "randomforest"} and LinearRegression is None:
        raise ImportError(f"scikit-learn is required: {_SKLEARN_IMPORT_ERROR}")
    if spec.family == "ols":
        base = LinearRegression(fit_intercept=True)
    elif spec.family == "ridge":
        base = Ridge(alpha=float(spec.params["alpha"]), solver="lsqr")
    elif spec.family == "lasso":
        base = Lasso(alpha=float(spec.params["alpha"]), max_iter=20000, random_state=RANDOM_STATE)
    elif spec.family == "elasticnet":
        base = ElasticNet(alpha=float(spec.params["alpha"]), l1_ratio=float(spec.params["l1_ratio"]), max_iter=20000, random_state=RANDOM_STATE)
    elif spec.family == "tree":
        base = DecisionTreeRegressor(random_state=RANDOM_STATE, **spec.params)
    elif spec.family == "randomforest":
        p = dict(spec.params)
        p["n_estimators"] = 300
        p["bootstrap"] = True
        p["random_state"] = RANDOM_STATE
        p["n_jobs"] = n_jobs
        base = RandomForestRegressor(**p)
    elif spec.family in {"lgbm_l2", "lgbm_l1", "lgbm_huber"}:
        if LGBMRegressor is None:
            raise ImportError(f"lightgbm is required: {_LIGHTGBM_IMPORT_ERROR}")
        objective = {
            "lgbm_l2": "regression",
            "lgbm_l1": "regression_l1",
            "lgbm_huber": "huber",
        }[spec.family]
        base = LGBMRegressor(
            objective=objective,
            n_estimators=300 if spec.family == "lgbm_huber" else 200,
            learning_rate=0.03,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=-1,
            **spec.params,
        )
    elif spec.family in {"catboost_rmse", "catboost_mae", "catboost_huber"}:
        if CatBoostRegressor is None:
            raise ImportError(f"catboost is required: {_CATBOOST_IMPORT_ERROR}")
        loss_function = {
            "catboost_rmse": "RMSE",
            "catboost_mae": "MAE",
            "catboost_huber": "Huber:delta=1.0",
        }[spec.family]
        base = CatBoostRegressor(
            loss_function=loss_function,
            iterations=350,
            random_seed=RANDOM_STATE,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
            **spec.params,
        )
    else:
        raise ValueError(f"unknown model family={spec.family}")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), base)


def predict_window(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], spec: ModelSpec, n_jobs: int) -> Tuple[np.ndarray, np.ndarray]:
    y = to_num(train["target_next_close_bps"])
    if spec.family == "constant":
        pred = float(y.mean())
        return np.full(len(train), pred), np.full(len(test), pred)
    if spec.family == "ewma":
        h = float(spec.params["halflife"])
        pred = float(y.ewm(halflife=h, min_periods=1).mean().iloc[-1])
        train_pred = y.ewm(halflife=h, min_periods=1).mean().to_numpy(dtype=float)
        return train_pred, np.full(len(test), pred)

    x_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    x_test = test.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if spec.family in {"ols", "ridge", "lasso", "elasticnet"} and math.isfinite(y_std) and y_std > 1e-12:
        fit_y = (y - y_mean) / y_std
        model = instantiate(spec, n_jobs=n_jobs)
        model.fit(x_train, fit_y)
        train_pred = np.asarray(model.predict(x_train), dtype=float) * y_std + y_mean
        test_pred = np.asarray(model.predict(x_test), dtype=float) * y_std + y_mean
        return train_pred, test_pred
    model = instantiate(spec, n_jobs=n_jobs)
    model.fit(x_train, y)
    return np.asarray(model.predict(x_train), dtype=float), np.asarray(model.predict(x_test), dtype=float)


def eval_pred(y: pd.Series, pred: np.ndarray) -> Dict[str, float | int]:
    yy = to_num(y)
    pp = pd.Series(pred, index=yy.index)
    m = yy.notna() & pp.notna()
    yy = yy[m]
    pp = pp[m]
    if len(yy) == 0:
        return {"n": 0, "spearman": np.nan, "pearson": np.nan, "rmse": np.nan, "target_std": np.nan, "rmse_norm": np.nan, "r2": np.nan, "pred_std": np.nan, "pred_std_ratio": np.nan}
    err = pp - yy
    rmse = float(np.sqrt(np.mean(np.square(err))))
    target_std = float(yy.std(ddof=0))
    pred_std = float(pp.std(ddof=0))
    sse = float(np.sum(np.square(err)))
    sst = float(np.sum(np.square(yy - yy.mean())))
    return {
        "n": int(len(yy)),
        "spearman": safe_spearman(pp, yy),
        "pearson": safe_pearson(pp, yy),
        "rmse": rmse,
        "target_std": target_std,
        "rmse_norm": rmse / target_std if target_std > 1e-12 else np.nan,
        "r2": 1.0 - sse / sst if sst > 1e-12 else np.nan,
        "pred_std": pred_std,
        "pred_std_ratio": pred_std / target_std if target_std > 1e-12 else np.nan,
    }


def flush_progress(out_dir: Path, stock_code: str, current_rows: Sequence[Dict], prior_rows: Sequence[Dict]) -> None:
    partial = pd.DataFrame(current_rows)
    partial.to_csv(out_dir / f"partial_window_metrics_{stock_code}.csv", index=False, encoding="utf-8-sig")
    summarize(partial).to_csv(out_dir / f"partial_search_summary_{stock_code}.csv", index=False, encoding="utf-8-sig")

    combined = pd.DataFrame([*prior_rows, *current_rows])
    combined.to_csv(out_dir / "window_metrics.csv", index=False, encoding="utf-8-sig")
    summarize(combined).to_csv(out_dir / "search_summary.csv", index=False, encoding="utf-8-sig")


def run_symbol(
    path: Path,
    args,
    specs: Sequence[ModelSpec],
    out_dir: Optional[Path] = None,
    prior_rows: Optional[Sequence[Dict]] = None,
) -> Tuple[List[Dict], Dict]:
    raw = load_one_sample(path)
    stock_code = infer_stock_code(path, raw)
    selected_feature_sets = [x for x in str(args.feature_sets).split(",") if x.strip()]
    df, feature_sets = build_feature_sets(
        raw,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        max_missing=float(args.max_missing),
        selected=selected_feature_sets,
    )
    del raw
    df = df.dropna(subset=["date", "target_next_close_bps"]).sort_values("date").reset_index(drop=True)
    feature_sets = {
        name: [f for f in features if f in df.columns and pd.to_numeric(df[f], errors="coerce").isna().mean() <= float(args.max_missing)]
        for name, features in feature_sets.items()
    }
    feature_sets = {name: features for name, features in feature_sets.items() if features}
    if len(df) < int(args.min_rows) or not feature_sets:
        return [], {
            "stock_code": stock_code,
            "sample_path": str(path),
            "rows": int(len(df)),
            "features": 0,
            "feature_sets": "",
            "status": "skipped",
            "reason": "insufficient_rows_or_features",
        }

    rows: List[Dict] = []
    train_windows = [int(x) for x in str(args.train_windows).split(",") if str(x).strip()]
    symbol_started = time.time()
    print(
        f"[INFO] symbol={stock_code} rows={len(df)} feature_sets={','.join(feature_sets.keys())} "
        f"specs={len(specs)} train_windows={','.join(map(str, train_windows))}",
        flush=True,
    )
    for train_days in train_windows:
        windows = date_windows(df["date"].unique(), train_days=train_days, test_days=int(args.test_days), embargo_days=int(args.embargo_days))
        if not windows:
            continue
        for feature_set_name, features in feature_sets.items():
            for spec_idx, spec in enumerate(specs, start=1):
                spec_started = time.time()
                last_progress = spec_started
                before_rows = len(rows)
                print(
                    f"[RUN] symbol={stock_code} train_days={train_days} feature_set={feature_set_name} "
                    f"model={spec.name} spec={spec_idx}/{len(specs)} windows={len(windows)}",
                    flush=True,
                )
                for window_id, (train_dates, test_dates) in enumerate(windows, start=1):
                    train = df[df["date"].isin(train_dates)].copy()
                    test = df[df["date"].isin(test_dates)].copy()
                    train_fit = train.dropna(subset=["target_next_close_bps"]).copy()
                    if len(train_fit) < int(args.min_train_rows) or test.empty:
                        continue
                    try:
                        train_pred, pred = predict_window(train_fit, test, features, spec, n_jobs=int(args.n_jobs))
                        train_metrics = {
                            f"train_fit_{k}": v
                            for k, v in eval_pred(train_fit["target_next_close_bps"], train_pred).items()
                        }
                        metrics = eval_pred(test["target_next_close_bps"], pred)
                        rows.append({
                            "stock_code": stock_code,
                            "sample_path": str(path),
                            "feature_set": feature_set_name,
                            "n_features": int(len(features)),
                            "train_days": train_days,
                            "test_days": int(args.test_days),
                            "embargo_days": int(args.embargo_days),
                            "window_id": window_id,
                            "train_start": str(pd.Timestamp(train_dates[0]).date()),
                            "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                            "test_start": str(pd.Timestamp(test_dates[0]).date()),
                            "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                            "model_family": spec.family,
                            "model_name": spec.name,
                            "params": json.dumps(spec.params, ensure_ascii=False, sort_keys=True),
                            **train_metrics,
                            **metrics,
                        })
                        now = time.time()
                        if window_id == len(windows) or now - last_progress >= 60:
                            print(
                                f"[PROGRESS] symbol={stock_code} train_days={train_days} feature_set={feature_set_name} "
                                f"model={spec.name} window={window_id}/{len(windows)} elapsed={now - spec_started:.1f}s",
                                flush=True,
                            )
                            last_progress = now
                    except Exception as exc:
                        rows.append({
                            "stock_code": stock_code,
                            "sample_path": str(path),
                            "feature_set": feature_set_name,
                            "n_features": int(len(features)),
                            "train_days": train_days,
                            "test_days": int(args.test_days),
                            "embargo_days": int(args.embargo_days),
                            "window_id": window_id,
                            "model_family": spec.family,
                            "model_name": spec.name,
                            "params": json.dumps(spec.params, ensure_ascii=False, sort_keys=True),
                            "n": 0,
                            "spearman": np.nan,
                            "pearson": np.nan,
                            "rmse": np.nan,
                            "target_std": np.nan,
                            "rmse_norm": np.nan,
                            "r2": np.nan,
                            "pred_std": np.nan,
                            "pred_std_ratio": np.nan,
                            "error": str(exc)[:500],
                        })
                added = len(rows) - before_rows
                print(
                    f"[DONE] symbol={stock_code} train_days={train_days} feature_set={feature_set_name} "
                    f"model={spec.name} rows_added={added} elapsed={time.time() - spec_started:.1f}s",
                    flush=True,
                )
                if out_dir is not None:
                    flush_progress(out_dir, stock_code, rows, prior_rows or [])
    meta = {
        "stock_code": stock_code,
        "sample_path": str(path),
        "rows": int(len(df)),
        "features": int(max(len(v) for v in feature_sets.values())),
        "feature_sets": ",".join(feature_sets.keys()),
        "target": "target_next_close_bps",
        "status": "ok",
        "feature_columns_by_set": json.dumps(feature_sets, ensure_ascii=False),
        "elapsed_seconds": float(time.time() - symbol_started),
    }
    return rows, meta


def summarize(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()
    ok = window_df[pd.to_numeric(window_df["n"], errors="coerce").fillna(0) > 0].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["stock_code", "sample_path", "feature_set", "train_days", "model_family", "model_name", "params"]
    rows = []
    for key, g in ok.groupby(group_cols, dropna=False):
        target_std = pd.to_numeric(g["target_std"], errors="coerce")
        rows.append({
            **dict(zip(group_cols, key)),
            "n_features": int(pd.to_numeric(g.get("n_features", pd.Series(dtype=float)), errors="coerce").max()) if "n_features" in g.columns else 0,
            "n_windows": int(len(g)),
            "n_test_rows": int(pd.to_numeric(g["n"], errors="coerce").fillna(0).sum()),
            "median_spearman": float(pd.to_numeric(g["spearman"], errors="coerce").median()),
            "mean_spearman": float(pd.to_numeric(g["spearman"], errors="coerce").mean()),
            "mean_pearson": float(pd.to_numeric(g["pearson"], errors="coerce").mean()),
            "median_rmse_norm": float(pd.to_numeric(g["rmse_norm"], errors="coerce").median()),
            "mean_rmse_norm": float(pd.to_numeric(g["rmse_norm"], errors="coerce").mean()),
            "median_r2": float(pd.to_numeric(g["r2"], errors="coerce").median()),
            "mean_r2": float(pd.to_numeric(g["r2"], errors="coerce").mean()),
            "median_pred_std_ratio": float(pd.to_numeric(g["pred_std_ratio"], errors="coerce").median()),
            "median_train_fit_spearman": float(pd.to_numeric(g.get("train_fit_spearman", pd.Series(dtype=float)), errors="coerce").median()),
            "mean_train_fit_spearman": float(pd.to_numeric(g.get("train_fit_spearman", pd.Series(dtype=float)), errors="coerce").mean()),
            "median_train_fit_rmse_norm": float(pd.to_numeric(g.get("train_fit_rmse_norm", pd.Series(dtype=float)), errors="coerce").median()),
            "mean_train_fit_rmse_norm": float(pd.to_numeric(g.get("train_fit_rmse_norm", pd.Series(dtype=float)), errors="coerce").mean()),
            "median_train_fit_r2": float(pd.to_numeric(g.get("train_fit_r2", pd.Series(dtype=float)), errors="coerce").median()),
            "mean_train_fit_r2": float(pd.to_numeric(g.get("train_fit_r2", pd.Series(dtype=float)), errors="coerce").mean()),
            "median_train_fit_pred_std_ratio": float(pd.to_numeric(g.get("train_fit_pred_std_ratio", pd.Series(dtype=float)), errors="coerce").median()),
            "mean_target_std": float(target_std.mean()),
        })
    out = pd.DataFrame(rows)
    out["passes_min_signal"] = (
        (out["median_spearman"] > 0.04)
        & (out["mean_spearman"] > 0)
        & (out["median_rmse_norm"] < 1.0)
        & (out["median_pred_std_ratio"] >= 0.20)
    )
    out = out.sort_values(
        ["stock_code", "feature_set", "train_days", "passes_min_signal", "median_spearman", "median_rmse_norm"],
        ascending=[True, True, True, False, False, True],
    )
    return out


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Single-symbol single-target asof1455 regression search")
    p.add_argument("--samples", nargs="*", default=[])
    p.add_argument("--sample-glob", action="append", default=DEFAULT_SAMPLE_GLOBS)
    p.add_argument("--symbols", default="", help="Comma-separated symbols/codes to include, e.g. 603308,600312")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--out-dir", default="")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--train-windows", default="756,504")
    p.add_argument("--test-days", type=int, default=21)
    p.add_argument("--embargo-days", type=int, default=1)
    p.add_argument("--max-missing", type=float, default=0.70)
    p.add_argument("--min-rows", type=int, default=600)
    p.add_argument("--min-train-rows", type=int, default=252)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--models", default="constant,ewma,ols,ridge,lasso,elasticnet,tree,randomforest,lgbm_l2,lgbm_l1,lgbm_huber,catboost_rmse,catboost_mae,catboost_huber")
    p.add_argument("--feature-sets", default="all", help="Comma-separated: a,b,c,full,all")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"saved_data/single_target_asof1455_model_search_out/search_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_globs = [g for item in args.sample_glob for g in str(item).split(";") if g.strip()]
    paths = expand_samples(args.samples, sample_globs)
    include = {s.strip().split(".")[0] for s in str(args.symbols).split(",") if s.strip()}
    if include:
        # Hard rule: --symbols 603308 means saved_data/603308_pipeline_out only.
        # Never use `code in str(path)` here; that can pull old recycle data.
        paths = [p for p in paths if pipeline_stock_code(p) in include]
    if int(args.max_symbols) > 0:
        paths = paths[: int(args.max_symbols)]
    if not paths:
        raise FileNotFoundError("no sample files found")

    specs = make_model_specs(str(args.models).split(","), n_jobs=int(args.n_jobs))
    if not specs:
        raise ValueError("no model specs selected")

    all_rows: List[Dict] = []
    meta_rows: List[Dict] = []
    print(f"[INFO] samples={len(paths)} models={len(specs)} out_dir={out_dir}")
    for i, path in enumerate(paths, start=1):
        print(f"[INFO] symbol_job {i}/{len(paths)} sample={path}")
        rows, meta = run_symbol(path, args, specs, out_dir=out_dir, prior_rows=all_rows)
        all_rows.extend(rows)
        meta_rows.append(meta)
        pd.DataFrame(all_rows).to_csv(out_dir / "window_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(meta_rows).to_csv(out_dir / "symbol_manifest.csv", index=False, encoding="utf-8-sig")
        summarize(pd.DataFrame(all_rows)).to_csv(out_dir / "search_summary.csv", index=False, encoding="utf-8-sig")

    window_df = pd.DataFrame(all_rows)
    summary = summarize(window_df)
    summary.to_csv(out_dir / "search_summary.csv", index=False, encoding="utf-8-sig")
    if not summary.empty:
        best = summary.sort_values(["stock_code", "passes_min_signal", "median_spearman", "median_rmse_norm"], ascending=[True, False, False, True]).groupby("stock_code", as_index=False).head(1)
        best.to_csv(out_dir / "best_by_symbol.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[DONE] out_dir={out_dir}")


if __name__ == "__main__":
    main()
