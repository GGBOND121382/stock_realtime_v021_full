#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML4T asof-14:55 LightGBM forward-return backtest.

Purpose
-------
Replace the old close_profit / hit classifier backtest with a book-aligned
workflow:

1) Build point-in-time 14:55 features only.
2) Predict 1-day forward return from the 14:55 entry price.
3) Train a pooled cross-sectional LightGBM regressor.
4) Evaluate the signal using IC and prediction quantiles/deciles.
5) Backtest fixed top-ranked selections without treating a classifier score as EV.

Expected input
--------------
One or more per-stock CSV files produced by, or compatible with,
feature_building/build_asof1455_training_samples.py.
Required columns:
    date, close_asof1455, vwap_asof1455(optional), open_asof1455,
    high_asof1455, low_asof1455, volume_asof1455, amount_asof1455,
    close, high, low, volume, next_day_close and/or next_day_vwap

Typical command
---------------
python3 model_training/backtest_ml4t_asof1455_lgbm.py \
  --sample-glob "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv" \
  --bars-glob "saved_data/*_pipeline_out/00_base/*_5m.csv" \
  --out-root saved_data/ml4t_asof1455_lgbm_pipeline_out \
  --entry-price-col close_asof1455 \
  --exit-price-col next_day_close \
  --train-days 756 \
  --test-days 21 \
  --embargo-days 1 \
  --selection-rule strict_top_decile_positive \
  --min-pred-return-bps 0.0 \
  --min-daily-candidates 10 \
  --max-positions 3

Notes
-----
- This script intentionally does NOT create hit50/hit80 labels.
- The only supervised label is a realized forward return from the 14:55 entry.
- All day-t feature values that use the current day are computed from asof1455
  fields. Prior days use completed EOD daily fields.
"""
from __future__ import annotations

import argparse
import copy
import gc
import glob
import json
import math
import re
import sys
import subprocess
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover
    LGBMRegressor = None
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    ExtraTreesRegressor = RandomForestRegressor = None
    SimpleImputer = ElasticNet = Ridge = make_pipeline = StandardScaler = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:
    from catboost import CatBoostRegressor
except Exception as exc:  # pragma: no cover
    CatBoostRegressor = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None

RANDOM_STATE = 42
EPS = 1e-12

# Conservative leakage blocklist. Anything containing these substrings is not used
# even if it is numeric.
LEAK_SUBSTRINGS = (
    "next_day",
    "target",
    "label",
    "pred",
    "score",
    "selected",
    "trade_",
    "future",
    "tomorrow",
)

# Strict v1 feature whitelist created by this script. Existing columns outside this
# prefix set are not automatically pulled in, to avoid accidental leakage.
ML4T_FEATURE_PREFIXES = (
    "ml4t_ret_",
    "rank_ml4t_ret_",
    "ml4t_ppo_",
    "ml4t_natr",
    "ml4t_rsi",
    "ml4t_bb_",
    "ml4t_dollar_volume_rank_pct",
    "ml4t_liquidity_ok",
    "intraday_ret_lag_",
    "intraday_bop_",
    "intraday_cci",
    "intraday_stochrsi",
    "intraday_mfi",
)
ML4T_FEATURE_EXACT = {"year", "month", "weekday", "ml4t_dollar_volume_prev"}

FUNDAMENTAL_FEATURE_PREFIXES = (
    "profit_",
    "operation_",
    "growth_",
    "solvency_",
    "cashflow_",
    "dupont_",
    "fund_",
)
FUNDAMENTAL_FEATURE_EXACT = {
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "peTTM_rank252",
    "pbMRQ_rank252",
    "psTTM_rank252",
    "pcfNcfTTM_rank252",
}
SECTOR_FEATURE_PREFIXES = (
    "sector_",
    "industry_",
    "ths_",
    "sw_",
    "rank_sector_",
    "stock_vs_sector_",
)
EXTERNAL_FEATURE_PREFIXES = (
    "external_",
    "ext_",
    "feed_",
    "hog_",
    "fert_",
    "zijin_",
    "ane_",
    "optical_",
    "storage_",
    "power_",
    "ai_",
    "material_",
    "muyuan_",
)
EXTERNAL_FEATURE_SUBSTRINGS = (
    "_fut_",
    "_hk_",
    "_etf_",
    "_board_",
    "_external",
)

BASE_NON_FEATURE_COLUMNS = {
    "date",
    "stock_code",
    "symbol",
    "code",
    "ts_code",
    "sample_path",
    "entry_signal",
    "entry_price",
    "exit_price",
    "feature_time_mode",
    "feature_cutoff_time",
    "valuation_time_mode",
    "asof_last_bar_time",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bool_mask(s: pd.Series) -> pd.Series:
    """Convert a possibly object/nullable boolean series to a strict bool mask without pandas downcast warnings."""
    return s.astype("boolean").fillna(False).astype(bool)


def safe_div(a, b):
    return a / pd.Series(b).replace(0, np.nan)


def parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("input sample is missing required column: date")
    raw = df["date"]
    # Avoid pandas treating integer YYYYMMDD values as nanosecond timestamps.
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
    m = re.search(r"(\d{6})(?:[_-]pipeline|[_-]asof|[_-]5m|\.csv)", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6})", path.name)
    if m:
        return m.group(1)
    return path.parent.name


def profit_factor(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna().to_numpy(float)
    if len(r) == 0:
        return np.nan
    gain = r[r > 0].sum()
    loss = -r[r < 0].sum()
    if loss <= 0:
        return float("inf") if gain > 0 else np.nan
    return float(gain / loss)


def return_metrics(ret: pd.Series) -> Dict[str, float | int]:
    r = pd.to_numeric(ret, errors="coerce").dropna().to_numpy(float)
    if len(r) == 0:
        return {
            "n": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "sum_return": np.nan,
            "compound_return": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, EPS) - 1.0
    return {
        "n": int(len(r)),
        "win_rate": float(np.mean(r > 0)),
        "avg_return": float(np.mean(r)),
        "median_return": float(np.median(r)),
        "sum_return": float(np.sum(r)),
        "compound_return": float(eq[-1] - 1.0),
        "profit_factor": profit_factor(pd.Series(r)),
        "max_drawdown": float(np.min(dd)),
    }


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return np.nan
    if x[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan
    return float(spearmanr(x[m], y[m]).correlation)


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return np.nan
    if x[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan
    return float(x[m].corr(y[m], method="pearson"))



def safe_series_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if len(v) else np.nan


def safe_series_median(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.median()) if len(v) else np.nan

def date_windows(
    dates: Sequence[pd.Timestamp],
    train_days: int,
    valid_days: int,
    test_days: int,
    embargo_days: int,
    step_days: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rolling date windows; split by dates, not rows.

    If valid_days == 0, the layout is train -> embargo -> test.
    If valid_days > 0, the layout is train -> embargo -> valid -> embargo -> test.
    """
    unique_dates = np.array(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    step = int(step_days or test_days)
    n = len(unique_dates)
    windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    need = train_days + embargo_days + valid_days + (embargo_days if valid_days > 0 else 0) + test_days
    while start + need <= n:
        train_start = start
        train_end = train_start + train_days
        if valid_days > 0:
            valid_start = train_end + embargo_days
            valid_end = valid_start + valid_days
            test_start = valid_end + embargo_days
        else:
            valid_start = valid_end = train_end + embargo_days
            test_start = train_end + embargo_days
        test_end = test_start + test_days
        train_dates = unique_dates[train_start:train_end]
        valid_dates = unique_dates[valid_start:valid_end] if valid_days > 0 else np.array([], dtype="datetime64[ns]")
        test_dates = unique_dates[test_start:test_end]
        windows.append((train_dates, valid_dates, test_dates))
        start += step
    return windows


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def expand_inputs(sample_paths: Sequence[str], sample_globs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for p in sample_paths:
        paths.append(Path(p))
    for pattern in sample_globs:
        paths.extend(Path(x) for x in glob.glob(pattern, recursive=True))
    seen = set()
    out = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            out.append(p)
    return sorted(out)


def load_one_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_date_col(df)
    if "stock_code" not in df.columns:
        df["stock_code"] = infer_stock_code(path, df)
    df["sample_path"] = str(path)
    float_cols = df.select_dtypes(include=["float"]).columns
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    int_cols = df.select_dtypes(include=["integer"]).columns
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
    return df


def load_samples(sample_paths: Sequence[Path]) -> pd.DataFrame:
    if not sample_paths:
        raise FileNotFoundError("no sample files found; pass --samples or --sample-glob")
    parts = []
    for p in sample_paths:
        try:
            part = load_one_sample(p)
            parts.append(part)
        except Exception as exc:
            print(f"[WARN] skip sample {p}: {exc}", file=sys.stderr)
    if not parts:
        raise RuntimeError("all sample files failed to load")
    df = pd.concat(parts, ignore_index=True, sort=False)
    del parts
    gc.collect()
    df["stock_code"] = df["stock_code"].astype(str)
    # Keep latest row if duplicate stock/date appears.
    df = df.sort_values(["stock_code", "date", "sample_path"]).drop_duplicates(["stock_code", "date"], keep="last")
    df = df.reset_index(drop=True)
    gc.collect()
    return df


def load_sample_dates(sample_paths: Sequence[Path], chunksize: int = 200_000) -> np.ndarray:
    dates: List[pd.Timestamp] = []
    for p in sample_paths:
        try:
            for chunk in pd.read_csv(p, usecols=["date"], chunksize=chunksize):
                chunk = parse_date_col(chunk)
                dates.extend(chunk["date"].dropna().tolist())
        except Exception as exc:
            print(f"[WARN] skip date scan {p}: {exc}", file=sys.stderr)
    if not dates:
        raise RuntimeError("no dates found in sample files")
    return np.array(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))


def raw_column_needed_for_lazy(c: str, feature_group: str, entry_price_col: str, exit_price_col: str) -> bool:
    required = {
        "date", "stock_code", "symbol", "code", "ts_code",
        "close", "high", "low", "volume",
        "open_asof1455", "high_asof1455", "low_asof1455", "close_asof1455",
        "volume_asof1455", "amount_asof1455", "vwap_asof1455",
        entry_price_col, exit_price_col,
    }
    if c in required:
        return True
    if c in BASE_NON_FEATURE_COLUMNS:
        return False
    if any(s in c.lower() for s in LEAK_SUBSTRINGS):
        return False
    return feature_allowed_by_group(c, feature_group)


def lazy_usecols(args):
    if not bool(getattr(args, "lazy_usecols", True)):
        return None

    def _usecols(c: str) -> bool:
        return raw_column_needed_for_lazy(
            c,
            feature_group=str(args.feature_group),
            entry_price_col=str(args.entry_price_col),
            exit_price_col=str(args.exit_price_col),
        )

    return _usecols


def load_one_sample_between(
    path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunksize: int = 100_000,
    usecols=None,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        chunk = parse_date_col(chunk)
        mask = (chunk["date"] >= start_date) & (chunk["date"] <= end_date)
        if not mask.any():
            continue
        part = chunk.loc[mask].copy()
        if "stock_code" not in part.columns:
            part["stock_code"] = infer_stock_code(path, part)
        part["sample_path"] = str(path)
        float_cols = part.select_dtypes(include=["float"]).columns
        for c in float_cols:
            part[c] = pd.to_numeric(part[c], errors="coerce", downcast="float")
        int_cols = part.select_dtypes(include=["integer"]).columns
        for c in int_cols:
            part[c] = pd.to_numeric(part[c], errors="coerce", downcast="integer")
        parts.append(part)
        del chunk, part
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    del parts
    gc.collect()
    return df


def load_samples_between(
    sample_paths: Sequence[Path],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunksize: int = 100_000,
    usecols=None,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for p in sample_paths:
        try:
            part = load_one_sample_between(p, start_date=start_date, end_date=end_date, chunksize=chunksize, usecols=usecols)
            if not part.empty:
                parts.append(part)
        except Exception as exc:
            print(f"[WARN] skip sample window {p}: {exc}", file=sys.stderr)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    del parts
    gc.collect()
    df["stock_code"] = df["stock_code"].astype(str)
    df = df.sort_values(["stock_code", "date", "sample_path"]).drop_duplicates(["stock_code", "date"], keep="last")
    return df.reset_index(drop=True)


def build_bars_map(bars_globs: Sequence[str]) -> Dict[str, Path]:
    files: List[Path] = []
    for pattern in bars_globs:
        files.extend(Path(x) for x in glob.glob(pattern, recursive=True))
    mp: Dict[str, Path] = {}
    for p in files:
        if not p.exists() or not p.is_file():
            continue
        m = re.search(r"(\d{6}).*5m.*\.csv$", p.name)
        if not m:
            m = re.search(r"(\d{6})", str(p))
        if m:
            mp.setdefault(m.group(1), p)
    return mp


# ---------------------------------------------------------------------------
# Optional project-native data preparation
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    # Normal placement is <root>/scripts/backtest_ml4t_asof1455_lgbm.py.
    if here.parent.name in {"scripts", "model_training"}:
        return here.parent.parent
    # Fallback for direct testing from project root or /mnt/data.
    for parent in [here.parent, *here.parents]:
        if (parent / "pipelines" / "run_nextday_pipeline.py").exists():
            return parent
    return Path.cwd().resolve()


def today_iso() -> str:
    return dt.date.today().isoformat()


def normalize_stock_code(value: str) -> str:
    s = str(value or "").strip().upper().replace("_", ".")
    if not s:
        return s
    m = re.match(r"^(\d{6})\.(SH|SZ)$", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.match(r"^(SH|SZ)[.:-]?(\d{6})$", s)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    code = re.sub(r"\D", "", s)
    if len(code) >= 6:
        code = code[-6:]
        suffix = "SH" if code.startswith(("6", "9")) else "SZ"
        return f"{code}.{suffix}"
    return s


def stock_code6(value: str) -> str:
    m = re.search(r"(\d{6})", str(value or ""))
    if not m:
        raise ValueError(f"cannot infer 6-digit stock code from {value!r}")
    return m.group(1)


def stock_pipeline_dir(symbol: str, root: Optional[Path] = None) -> Path:
    root = root or resolve_project_root()
    return root / "saved_data" / f"{stock_code6(symbol)}_pipeline_out"


def canonical_asof_stage_dir(symbol: str, stage_name: str = "01_samples_asof1455", root: Optional[Path] = None) -> Path:
    return stock_pipeline_dir(symbol, root=root) / stage_name


def find_project_asof_sample(symbol: str, stage_name: str = "01_samples_asof1455", root: Optional[Path] = None) -> Optional[Path]:
    """Find the best project-generated per-stock asof1455 sample.

    Prefer source stages that are rebuilt by run_nextday_pipeline. The ML4T
    stage is a stable copy target and may be stale if a newer 01 sample exists.
    """
    root = root or resolve_project_root()
    pipe = stock_pipeline_dir(symbol, root=root)
    candidates = [
        pipe / "01_samples_asof1455" / "training_samples_asof1455.csv",
    ]
    if stage_name != "01_samples_asof1455":
        candidates.insert(0, pipe / stage_name / "training_samples_asof1455.csv")
    candidates += sorted(
        pipe.glob("**/training_samples_asof1455.csv"),
        key=lambda x: x.stat().st_mtime if x.exists() else 0,
        reverse=True,
    )
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists() and c.is_file():
            return c
    return None


def copy_asof_sample_to_ml4t_stage(symbol: str, stage_name: str = "01_samples_asof1455", root: Optional[Path] = None) -> Optional[Path]:
    """Ensure the canonical per-stock asof1455 sample exists."""
    root = root or resolve_project_root()
    src = find_project_asof_sample(symbol, stage_name=stage_name, root=root)
    if src is None:
        return None
    dst_dir = canonical_asof_stage_dir(symbol, stage_name=stage_name, root=root)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "training_samples_asof1455.csv"
    if src.resolve() != dst.resolve():
        import shutil
        shutil.copy2(src, dst)
        vr = src.parent / "validation_report.json"
        if vr.exists():
            shutil.copy2(vr, dst_dir / "validation_report.json")
    manifest = {
        "stock_code": stock_code6(symbol),
        "source_sample": str(src),
        "canonical_asof_sample": str(dst),
        "stage_name": stage_name,
        "created_by": Path(__file__).name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(dst_dir / "ml4t_asof_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    return dst


def split_symbols_arg(value: str) -> List[str]:
    out: List[str] = []
    for x in str(value or "").split(","):
        x = x.strip()
        if x:
            out.append(normalize_stock_code(x))
    return out


def infer_symbols_from_paths(paths: Sequence[Path]) -> List[str]:
    symbols: List[str] = []
    for p in paths:
        m = re.search(r"(\d{6})", str(p))
        if m:
            symbols.append(normalize_stock_code(m.group(1)))
    return sorted(set(symbols))


def infer_symbols_from_pipeline_dirs(project_root: Path) -> List[str]:
    """Infer the trading universe from saved_data/<code>_pipeline_out directories.

    This is the default universe for this project. It intentionally only accepts
    directory names like 601899_pipeline_out or 603308_pipeline_out, so summary
    folders such as ml4t_asof1455_lgbm_pipeline_out are ignored.
    """
    saved_data = project_root / "saved_data"
    if not saved_data.exists() or not saved_data.is_dir():
        return []
    symbols: List[str] = []
    for d in sorted(saved_data.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(\d{6})_pipeline_out", d.name)
        if not m:
            continue
        symbols.append(normalize_stock_code(m.group(1)))
    return sorted(set(symbols))


def pipeline_universe_dirs(project_root: Path) -> Dict[str, Path]:
    """Return the allowed per-stock data roots under saved_data.

    Only exact saved_data/<6-digit-code>_pipeline_out directories are allowed.
    This deliberately excludes lookalike experiment folders such as
    000657_pipeline_out_v2_new27_full and non-stock summary/cache folders.
    """
    saved_data = project_root / "saved_data"
    if not saved_data.exists() or not saved_data.is_dir():
        return {}
    out: Dict[str, Path] = {}
    for d in sorted(saved_data.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(\d{6})_pipeline_out", d.name)
        if not m:
            continue
        out[m.group(1)] = d
    return out


def is_relative_to_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def filter_paths_to_pipeline_universe(paths: Sequence[Path], project_root: Path, label: str) -> List[Path]:
    """Keep only files that live inside exact saved_data/<code>_pipeline_out roots."""
    allowed_dirs = pipeline_universe_dirs(project_root)
    if not allowed_dirs:
        if paths:
            print(f"[INFO] filtered {len(paths)} {label}: no saved_data/<code>_pipeline_out directories found")
        return []
    out: List[Path] = []
    dropped: List[Path] = []
    for p in paths:
        if any(is_relative_to_path(p, d) for d in allowed_dirs.values()):
            out.append(p)
        else:
            dropped.append(p)
    if dropped:
        print(
            f"[INFO] filtered {len(dropped)} {label} outside saved_data/<code>_pipeline_out "
            f"(kept {len(out)}, allowed_symbols={len(allowed_dirs)})"
        )
        for p in dropped[:20]:
            print(f"  - filtered {p}")
        if len(dropped) > 20:
            print(f"  ... ({len(dropped) - 20} more filtered)")
    return out


def pipeline_code_for_path(path: Path, project_root: Path) -> Optional[str]:
    for code, root in pipeline_universe_dirs(project_root).items():
        if is_relative_to_path(path, root):
            return code
    return None


def sample_stage_rank(path: Path, stage_name: str, feature_group: str) -> int:
    parts = path.parts
    group = str(feature_group or "ml4t_intraday").strip().lower()
    if group in {"external", "ml4t_external", "core_external", "core_sector_external_lagged", "all"}:
        order = ["04_external", "03_sector", "02_fundamental", "01_samples_asof1455"]
    elif group in {"sector", "ml4t_sector", "core_sector"}:
        order = ["03_sector", "02_fundamental", "01_samples_asof1455"]
    elif group in {"fundamental", "ml4t_fundamental", "core_fundamental"}:
        order = ["02_fundamental", "01_samples_asof1455"]
    else:
        order = [stage_name, "01_samples_asof1455"]
    for i, name in enumerate(order):
        if name in parts:
            return i
    return len(order)


def select_one_sample_per_symbol(paths: Sequence[Path], project_root: Path, stage_name: str, feature_group: str) -> List[Path]:
    """Avoid mixing sample stages for the same stock.

    The preferred stage depends on --feature-group, with deeper asof-derived
    feature files preferred and shallower stages used only as fallback.
    """
    by_code: Dict[str, List[Path]] = {}
    no_code: List[Path] = []
    for p in paths:
        code = pipeline_code_for_path(p, project_root)
        if code:
            by_code.setdefault(code, []).append(p)
        else:
            no_code.append(p)

    selected: List[Path] = []
    dropped: List[Path] = []
    for code, code_paths in sorted(by_code.items()):
        ranked = sorted(
            code_paths,
            key=lambda p: (
                sample_stage_rank(p, stage_name, feature_group),
                -(p.stat().st_mtime if p.exists() else 0),
                str(p),
            ),
        )
        selected.append(ranked[0])
        dropped.extend(ranked[1:])
    selected.extend(no_code)
    if dropped:
        print(
            f"[INFO] dropped {len(dropped)} duplicate sample files to avoid cross-stage mixing "
            f"(kept {len(selected)})"
        )
        for p in dropped[:20]:
            print(f"  - dropped {p}")
        if len(dropped) > 20:
            print(f"  ... ({len(dropped) - 20} more dropped)")
    return sorted(selected)


def filter_bars_map_to_pipeline_universe(bars_map: Dict[str, Path], project_root: Path) -> Dict[str, Path]:
    allowed_dirs = pipeline_universe_dirs(project_root)
    if not allowed_dirs:
        if bars_map:
            print(f"[INFO] filtered {len(bars_map)} 5m bar mappings: no saved_data/<code>_pipeline_out directories found")
        return {}
    out: Dict[str, Path] = {}
    dropped: Dict[str, Path] = {}
    for code, path in bars_map.items():
        allowed_dir = allowed_dirs.get(stock_code6(code))
        if allowed_dir is not None and is_relative_to_path(path, allowed_dir):
            out[code] = path
        else:
            dropped[code] = path
    if dropped:
        print(f"[INFO] filtered {len(dropped)} 5m bar mappings outside saved_data/<code>_pipeline_out (kept {len(out)})")
    return out


def read_symbols_file(path: Path) -> List[str]:
    """Read stock symbols from a simple watchlist file.

    Supported formats per line:
      600312.SH
      600312
      600312.SH,002311.SZ

    Lines after # are ignored.
    """
    if not path.exists() or not path.is_file():
        return []
    symbols: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for token in re.split(r"[,\s]+", line):
            token = token.strip()
            if token:
                symbols.append(normalize_stock_code(token))
    return sorted(set([s for s in symbols if re.search(r"\d{6}", s)]))


def discover_symbol_paths_for_auto_update(args, project_root: Path) -> List[Path]:
    """Find existing project artifacts that identify which stocks should be auto-updated.

    This is used when no asof sample exists yet. In that case relying only on
    --sample-glob cannot work, so we infer symbols from already-existing
    pipeline dirs and 5m bar files.
    """
    paths: List[Path] = []

    # 1) User-supplied bar patterns, e.g. saved_data/*_pipeline_out/00_base/*_5m.csv
    for pat in getattr(args, "bars_glob", []) or []:
        paths.extend(Path(x) for x in glob.glob(pat, recursive=True))
        paths.extend(project_root.glob(pat))

    # 2) Standard per-stock pipeline folders and 5m bars under saved_data.
    paths.extend(project_root.glob("saved_data/*_pipeline_out"))
    paths.extend(project_root.glob("saved_data/*_pipeline_out/00_base/*_5m.csv"))
    paths.extend(project_root.glob("saved_data/*_pipeline_out/**/training_samples*.csv"))

    # 3) Saved model metadata/paths often include the 6-digit stock code.
    paths.extend(project_root.glob("saved_models/**/*.json"))

    # Deduplicate existing paths; keep non-existing glob-string paths out.
    seen = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(p.resolve()) if p.exists() else str(p)
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            out.append(p)
    return out


def infer_symbols_for_auto_update(sample_paths: Sequence[Path], raw: Optional[pd.DataFrame], args, project_root: Path) -> List[str]:
    """Infer auto-update symbols.

    Default project rule: use all folders under saved_data whose names match
    <6-digit-code>_pipeline_out, e.g. 601899_pipeline_out or 603308_pipeline_out.
    Explicit --auto-symbols still overrides this default.
    """
    symbols = split_symbols_arg(args.auto_symbols)
    if symbols:
        print(f"[INFO] using symbols from --auto-symbols: {len(symbols)}")
        return symbols

    # Project default: the universe is exactly saved_data/<code>_pipeline_out.
    symbols = infer_symbols_from_pipeline_dirs(project_root)
    if symbols:
        print(f"[INFO] inferred auto-update symbols from saved_data/*_pipeline_out dirs: {len(symbols)}")
        return symbols

    print("[INFO] no saved_data/<code>_pipeline_out dirs found for auto-update symbol inference")
    return []


def discover_asof_sample_paths(project_root: Path, stage_name: str = "01_samples_asof1455") -> List[Path]:
    # Prefer the organized ML4T stage under each stock pipeline_out.
    patterns = [
        "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv",
    ]
    if stage_name != "01_samples_asof1455":
        patterns.insert(0, f"saved_data/*_pipeline_out/{stage_name}/training_samples_asof1455.csv")
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(project_root.glob(pat))
    seen = set()
    out: List[Path] = []
    for p in paths:
        if p.exists() and p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    out = filter_paths_to_pipeline_universe(out, project_root, "auto-discovered sample files")
    return select_one_sample_per_symbol(out, project_root, stage_name, "ml4t_intraday")


def required_window_dates(args) -> int:
    return int(args.train_days + args.embargo_days + args.valid_days + (args.embargo_days if args.valid_days > 0 else 0) + args.test_days)


def run_project_data_update(symbols: Sequence[str], args, out_dir: Path) -> None:
    root = resolve_project_root()
    pipeline = root / "pipelines" / "run_nextday_pipeline.py"
    if not pipeline.exists():
        raise FileNotFoundError(f"cannot auto-update: missing {pipeline}")
    if not symbols:
        raise RuntimeError("cannot auto-update: no symbols inferred; pass --auto-symbols 600312.SH,...")

    py = args.auto_update_python or sys.executable
    end_date = today_iso() if str(args.auto_update_end_date).lower() == "today" else str(args.auto_update_end_date)
    log_path = Path(args.out_root) / "logs" / "auto_update_data.log"
    if not log_path.is_absolute():
        log_path = root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    max_symbols = int(args.auto_max_symbols or 0)
    todo = list(symbols)[:max_symbols] if max_symbols > 0 else list(symbols)
    print(f"[INFO] auto-update symbols={len(todo)} start={args.auto_update_start_date} end={end_date}")

    failures: List[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== auto update start symbols={len(todo)} start={args.auto_update_start_date} end={end_date} ===\n")
        for i, sym in enumerate(todo, start=1):
            cmd = [
                py, str(pipeline),
                "--symbol", sym,
                "--start-date", str(args.auto_update_start_date),
                "--end-date", end_date,
                "--feature-time-mode", "asof1455",
                "--feature-cutoff-time", str(args.cutoff_time),
                "--feature-pipeline", "",
                "--no-fundamental",
                "--only-stages", "update_data,samples,asof_samples",
                "--cache-mode", str(args.auto_cache_mode),
                "--feature-cache-mode", str(args.auto_feature_cache_mode),
                "--min-bars", str(args.auto_min_bars),
            ]
            if args.auto_force_refresh:
                cmd.append("--force-refresh")
            print(f"[INFO] auto-update {i}/{len(todo)} {sym}")
            log.write("\n[CMD] " + " ".join(cmd) + "\n")
            if args.auto_update_dry_run:
                continue
            proc = subprocess.run(cmd, cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(proc.stdout or "")
            log.write(f"[RET] {proc.returncode}\n")
            if proc.returncode != 0:
                failures.append(sym)
                print(f"[WARN] auto-update failed for {sym}; see {log_path}", file=sys.stderr)
                if not args.auto_keep_going:
                    raise RuntimeError(f"auto-update failed for {sym}; see {log_path}")
            else:
                copied = copy_asof_sample_to_ml4t_stage(sym, stage_name=args.ml4t_sample_stage, root=root)
                log.write(f"[ML4T_SAMPLE] {sym} -> {copied}\n")
    if failures:
        print(f"[WARN] auto-update failures: {','.join(failures)}; see {log_path}", file=sys.stderr)


def ensure_data_if_needed(sample_paths: List[Path], raw: Optional[pd.DataFrame], args, out_dir: Path) -> Tuple[List[Path], Optional[pd.DataFrame]]:
    need = required_window_dates(args)
    have = 0 if raw is None or raw.empty else int(pd.to_datetime(raw["date"], errors="coerce").nunique())
    if sample_paths and have >= need:
        return sample_paths, raw
    if not args.auto_update_data:
        return sample_paths, raw

    root = resolve_project_root()
    symbols = infer_symbols_for_auto_update(sample_paths, raw, args, root)
    if not symbols:
        searched = {
            "sample_paths": len(sample_paths),
            "bars_glob": list(getattr(args, "bars_glob", []) or []),
            "pipeline_dirs": len(list(root.glob("saved_data/*_pipeline_out"))),
            "bar_files": len(list(root.glob("saved_data/*_pipeline_out/00_base/*_5m.csv"))),
        }
        raise RuntimeError(
            "cannot auto-update: no symbols inferred. "
            "Pass --auto-symbols 600312.SH,002311.SZ or --auto-symbols-file selected_watchlist.txt. "
            f"Searched: {searched}"
        )
    run_project_data_update(symbols, args, out_dir)

    # Re-scan broadly because run_nextday_pipeline writes to 01_samples_asof1455 by default.
    new_paths = discover_asof_sample_paths(resolve_project_root(), stage_name=args.ml4t_sample_stage)
    if not new_paths:
        raise FileNotFoundError("auto-update finished, but no training_samples_asof1455.csv was found under saved_data")
    new_raw = load_samples(new_paths)
    return new_paths, new_raw


# ---------------------------------------------------------------------------
# As-of feature engineering
# ---------------------------------------------------------------------------

def add_asof_return_features(g: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    out = g.sort_values("date").copy()
    close_eod = to_num(out["close"])
    p_asof = to_num(out["close_asof1455"])
    for h in horizons:
        out[f"ml4t_ret_{h}d_asof"] = p_asof / close_eod.shift(h).replace(0, np.nan) - 1.0
    return out


def ema_asof(close_eod: pd.Series, p_asof: pd.Series, span: int) -> pd.Series:
    alpha = 2.0 / (span + 1.0)
    ema_prev = close_eod.shift(1).ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()
    return alpha * p_asof + (1.0 - alpha) * ema_prev


def rolling_with_current(prev_series: pd.Series, current_value: pd.Series, window: int, func: str) -> pd.Series:
    """Rolling statistic over previous window-1 EOD values plus current asof value."""
    prev = prev_series.shift(1)
    if func == "mean":
        s = prev.rolling(window - 1, min_periods=window - 1).sum() + current_value
        return s / float(window)
    if func == "std":
        # Simpler and safer than trying to update variance by hand.
        values = []
        arr_prev = prev_series.to_numpy(float)
        arr_cur = current_value.to_numpy(float)
        for i in range(len(prev_series)):
            start = i - (window - 1)
            if start < 0:
                values.append(np.nan)
                continue
            hist = arr_prev[start:i]
            vals = np.r_[hist, arr_cur[i]]
            if np.isfinite(vals).sum() < window:
                values.append(np.nan)
            else:
                values.append(float(np.nanstd(vals, ddof=0)))
        return pd.Series(values, index=prev_series.index)
    raise ValueError(func)


def add_asof_technical_features(g: pd.DataFrame) -> pd.DataFrame:
    out = g.sort_values("date").copy()
    close_eod = to_num(out["close"])
    high_eod = to_num(out["high"])
    low_eod = to_num(out["low"])
    vol_eod = to_num(out["volume"]) if "volume" in out.columns else pd.Series(np.nan, index=out.index)
    p_asof = to_num(out["close_asof1455"])
    high_asof = to_num(out["high_asof1455"])
    low_asof = to_num(out["low_asof1455"])
    vol_asof = to_num(out["volume_asof1455"]) if "volume_asof1455" in out.columns else pd.Series(np.nan, index=out.index)

    ema14 = ema_asof(close_eod, p_asof, 14)
    ema26 = ema_asof(close_eod, p_asof, 26)
    out["ml4t_ppo_14_26_asof"] = (ema14 - ema26) / ema26.replace(0, np.nan)

    # RSI14 as-of approximation: previous EOD Wilder averages + current asof delta.
    delta_eod = close_eod.diff()
    gain_eod = delta_eod.clip(lower=0.0)
    loss_eod = (-delta_eod).clip(lower=0.0)
    n = 14
    avg_gain_prev = gain_eod.shift(1).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss_prev = loss_eod.shift(1).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    delta_cur = p_asof - close_eod.shift(1)
    gain_cur = delta_cur.clip(lower=0.0)
    loss_cur = (-delta_cur).clip(lower=0.0)
    avg_gain_asof = (avg_gain_prev * (n - 1.0) + gain_cur) / n
    avg_loss_asof = (avg_loss_prev * (n - 1.0) + loss_cur) / n
    rs = avg_gain_asof / avg_loss_asof.replace(0, np.nan)
    out["ml4t_rsi14_asof"] = 100.0 - 100.0 / (1.0 + rs)

    # NATR14 as-of approximation.
    prev_close = close_eod.shift(1)
    tr_eod = pd.concat([
        high_eod - low_eod,
        (high_eod - prev_close).abs(),
        (low_eod - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_prev = tr_eod.shift(1).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    tr_asof = pd.concat([
        high_asof - low_asof,
        (high_asof - prev_close).abs(),
        (low_asof - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_asof = (atr_prev * (n - 1.0) + tr_asof) / n
    out["ml4t_natr14_asof"] = 100.0 * atr_asof / p_asof.replace(0, np.nan)

    # Bollinger ratios over previous 19 completed closes + current asof price.
    mid = rolling_with_current(close_eod, p_asof, 20, "mean")
    std = rolling_with_current(close_eod, p_asof, 20, "std")
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    out["ml4t_bb_upper_ratio_asof"] = p_asof / upper.replace(0, np.nan) - 1.0
    out["ml4t_bb_middle_ratio_asof"] = p_asof / mid.replace(0, np.nan) - 1.0
    out["ml4t_bb_lower_ratio_asof"] = p_asof / lower.replace(0, np.nan) - 1.0

    # Daily-asof OHLCV technical indicators that can be approximated from 14:55 fields.
    denom = (high_asof - low_asof).replace(0, np.nan)
    out["intraday_bop_asof"] = (p_asof - to_num(out["open_asof1455"])) / denom

    typical_eod = (high_eod + low_eod + close_eod) / 3.0
    typical_asof = (high_asof + low_asof + p_asof) / 3.0
    tp_ma20 = rolling_with_current(typical_eod, typical_asof, 20, "mean")
    # Mean absolute deviation over previous 19 + current.
    mad_vals = []
    arr_tp = typical_eod.to_numpy(float)
    arr_cur = typical_asof.to_numpy(float)
    arr_mid = tp_ma20.to_numpy(float)
    for i in range(len(out)):
        start = i - 19
        if start < 0 or not np.isfinite(arr_mid[i]):
            mad_vals.append(np.nan)
            continue
        vals = np.r_[arr_tp[start:i], arr_cur[i]]
        if np.isfinite(vals).sum() < 20:
            mad_vals.append(np.nan)
        else:
            mad_vals.append(float(np.nanmean(np.abs(vals - arr_mid[i]))))
    mad = pd.Series(mad_vals, index=out.index)
    out["intraday_cci20_asof"] = (typical_asof - tp_ma20) / (0.015 * mad.replace(0, np.nan))

    # StochRSI from current RSI and previous EOD RSI values approximated by existing rsi14 if present,
    # otherwise the same asof RSI series shifted is used only for historical values.
    if "rsi14" in out.columns:
        rsi_eod = to_num(out["rsi14"])
    else:
        rsi_eod = out["ml4t_rsi14_asof"].shift(1)
    rsi_cur = out["ml4t_rsi14_asof"]
    rsi_min = pd.concat([rsi_eod.shift(1).rolling(13, min_periods=13).min(), rsi_cur], axis=1).min(axis=1)
    rsi_max = pd.concat([rsi_eod.shift(1).rolling(13, min_periods=13).max(), rsi_cur], axis=1).max(axis=1)
    out["intraday_stochrsi14_asof"] = (rsi_cur - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)

    # MFI14 approximation: previous 13 EOD money-flow signs + current asof money-flow sign.
    raw_eod = typical_eod * vol_eod
    raw_cur = typical_asof * vol_asof
    sign_eod = typical_eod.diff()
    pos_eod = raw_eod.where(sign_eod > 0, 0.0)
    neg_eod = raw_eod.where(sign_eod < 0, 0.0)
    sign_cur = typical_asof - typical_eod.shift(1)
    pos_cur = raw_cur.where(sign_cur > 0, 0.0)
    neg_cur = raw_cur.where(sign_cur < 0, 0.0)
    pos_sum = pos_eod.shift(1).rolling(13, min_periods=13).sum() + pos_cur
    neg_sum = neg_eod.shift(1).rolling(13, min_periods=13).sum() + neg_cur
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    out["intraday_mfi14_asof"] = 100.0 - 100.0 / (1.0 + mfr)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    out["year"] = dt.dt.year.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["weekday"] = dt.dt.weekday.astype(int)
    return out


def add_liquidity_and_rank_features(df: pd.DataFrame, horizons: Sequence[int], min_liquidity_rank_pct: float) -> pd.DataFrame:
    out = df.copy()
    # Liquidity filter: rank yesterday's completed close*volume across the current daily universe.
    out = out.sort_values(["stock_code", "date"])
    dollar_eod = to_num(out["close"]) * to_num(out["volume"])
    out["ml4t_dollar_volume_prev"] = dollar_eod.groupby(out["stock_code"]).shift(1)
    out["ml4t_dollar_volume_rank_pct"] = out.groupby("date")["ml4t_dollar_volume_prev"].rank(pct=True, method="average")
    out["ml4t_liquidity_ok"] = (out["ml4t_dollar_volume_rank_pct"] >= float(min_liquidity_rank_pct)).astype(float)
    for h in horizons:
        col = f"ml4t_ret_{h}d_asof"
        if col in out.columns:
            out[f"rank_{col}"] = out.groupby("date")[col].rank(pct=True, method="average")
    return out


def parse_cutoff_minutes(cutoff_time: str) -> int:
    hh, mm = str(cutoff_time).split(":", 1)
    return int(hh) * 60 + int(mm)


def build_intraday_lag_features_from_bars(path: Path, cutoff_time: str, max_lag: int = 10) -> pd.DataFrame:
    bars = pd.read_csv(path)
    if "datetime" not in bars.columns:
        return pd.DataFrame()
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    bars = bars.dropna(subset=["datetime"]).copy()
    if bars.empty or "close" not in bars.columns:
        return pd.DataFrame()
    bars["date"] = bars["datetime"].dt.normalize()
    minute = bars["datetime"].dt.hour * 60 + bars["datetime"].dt.minute
    bars = bars[minute <= parse_cutoff_minutes(cutoff_time)].copy()
    bars["close"] = to_num(bars["close"])
    rows = []
    for d, g in bars.groupby("date", sort=True):
        g = g.sort_values("datetime")
        close = g["close"].to_numpy(float)
        row = {"date": d}
        # ret_lag_1bar = last completed bar return. ret_lag_kbar walks backward.
        for k in range(1, max_lag + 1):
            if len(close) >= k + 1 and np.isfinite(close[-k]) and np.isfinite(close[-k - 1]) and abs(close[-k - 1]) > EPS:
                row[f"intraday_ret_lag_{k}bar"] = float(close[-k] / close[-k - 1] - 1.0)
            else:
                row[f"intraday_ret_lag_{k}bar"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_bar_lag_features(df: pd.DataFrame, bars_map: Dict[str, Path], cutoff_time: str, max_lag: int) -> pd.DataFrame:
    if not bars_map:
        return df
    parts = []
    for stock_code, g in df.groupby("stock_code", sort=False):
        code6 = re.search(r"\d{6}", str(stock_code))
        key = code6.group(0) if code6 else str(stock_code)
        path = bars_map.get(key)
        gg = g.copy()
        if path is not None:
            try:
                lag_df = build_intraday_lag_features_from_bars(path, cutoff_time, max_lag=max_lag)
                if not lag_df.empty:
                    gg = gg.merge(lag_df, on="date", how="left")
            except Exception as exc:
                print(f"[WARN] failed to build intraday bar features for {stock_code} from {path}: {exc}", file=sys.stderr)
        parts.append(gg)
    return pd.concat(parts, ignore_index=True, sort=False)


def build_ml4t_features(
    df: pd.DataFrame,
    horizons: Sequence[int],
    min_liquidity_rank_pct: float,
    bars_map: Optional[Dict[str, Path]],
    cutoff_time: str,
    max_intraday_lag: int,
) -> pd.DataFrame:
    required = [
        "date", "stock_code", "close", "high", "low", "volume",
        "open_asof1455", "high_asof1455", "low_asof1455", "close_asof1455",
        "volume_asof1455", "amount_asof1455",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sample is missing required asof columns: {missing}")
    out = df.copy()
    parts = []
    for _, g in out.groupby("stock_code", sort=False):
        gg = add_asof_return_features(g, horizons)
        gg = add_asof_technical_features(gg)
        parts.append(gg)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = add_time_features(out)
    out = add_liquidity_and_rank_features(out, horizons, min_liquidity_rank_pct)
    if bars_map:
        out = add_bar_lag_features(out, bars_map, cutoff_time=cutoff_time, max_lag=max_intraday_lag)
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Target and entry rules
# ---------------------------------------------------------------------------

def add_target_and_entry(
    df: pd.DataFrame,
    entry_price_col: str,
    exit_price_col: str,
    round_trip_cost_bps: float,
    entry_policy: str,
    entry_vwap_premium_bps: float,
) -> pd.DataFrame:
    out = df.copy()
    if entry_price_col not in out.columns:
        raise ValueError(f"missing entry_price_col={entry_price_col}")
    if exit_price_col not in out.columns:
        raise ValueError(f"missing exit_price_col={exit_price_col}; available columns include {list(out.columns)[:30]}...")
    entry = to_num(out[entry_price_col])
    exitp = to_num(out[exit_price_col])
    out["entry_price"] = entry
    out["exit_price"] = exitp
    out["target_1d_forward_return_bps"] = 10000.0 * (exitp / entry.replace(0, np.nan) - 1.0) - float(round_trip_cost_bps)

    policy = str(entry_policy).strip().lower()
    if policy in {"all", "all_days", "all-day"}:
        out["entry_signal"] = entry.notna() & exitp.notna()
    elif policy in {"vwap_low", "below_vwap", "low_vwap"}:
        if "vwap_asof1455" not in out.columns:
            raise ValueError("entry_policy=vwap_low requires vwap_asof1455")
        premium = float(entry_vwap_premium_bps) / 10000.0
        out["entry_signal"] = entry.notna() & exitp.notna() & (entry <= to_num(out["vwap_asof1455"]) * (1.0 + premium))
    else:
        raise ValueError(f"unknown entry_policy={entry_policy}")
    return out


# ---------------------------------------------------------------------------
# Modeling and evaluation
# ---------------------------------------------------------------------------

def is_ml4t_feature(c: str) -> bool:
    return c in ML4T_FEATURE_EXACT or c.startswith(ML4T_FEATURE_PREFIXES)


def is_fundamental_feature(c: str) -> bool:
    return c in FUNDAMENTAL_FEATURE_EXACT or c.startswith(FUNDAMENTAL_FEATURE_PREFIXES)


def is_sector_feature(c: str) -> bool:
    return c.startswith(SECTOR_FEATURE_PREFIXES)


def is_external_feature(c: str) -> bool:
    text = c.lower()
    return c.startswith(EXTERNAL_FEATURE_PREFIXES) or any(s in text for s in EXTERNAL_FEATURE_SUBSTRINGS)


def feature_allowed_by_group(c: str, feature_group: str) -> bool:
    group = str(feature_group or "ml4t_intraday").strip().lower()
    if group in {"ml4t", "core", "ml4t_core", "ml4t_intraday", "core_intraday"}:
        return is_ml4t_feature(c)
    if group in {"fundamental", "ml4t_fundamental", "core_fundamental"}:
        return is_ml4t_feature(c) or is_fundamental_feature(c)
    if group in {"sector", "ml4t_sector", "core_sector"}:
        return is_ml4t_feature(c) or is_fundamental_feature(c) or is_sector_feature(c)
    if group in {"external", "ml4t_external", "core_external", "core_sector_external_lagged", "all"}:
        return is_ml4t_feature(c) or is_fundamental_feature(c) or is_sector_feature(c) or is_external_feature(c)
    if group in {"all_numeric_asof", "all_numeric"}:
        return True
    raise ValueError(f"unknown --feature-group={feature_group}")


def feature_time_policy(c: str) -> str:
    if is_ml4t_feature(c):
        return "asof1455_or_lagged_price_volume"
    if is_fundamental_feature(c):
        return "fundamental_point_in_time_or_asof1455_valuation"
    if is_sector_feature(c):
        return "sector_lagged_or_asof_source_required"
    if is_external_feature(c):
        return "external_lagged_or_asof_source_required"
    return "other_numeric_allowed_by_group"


def candidate_features(df: pd.DataFrame, max_missing: float, feature_group: str = "ml4t_intraday") -> List[str]:
    feats: List[str] = []
    for c in df.columns:
        if c in BASE_NON_FEATURE_COLUMNS:
            continue
        if any(s in c.lower() for s in LEAK_SUBSTRINGS):
            continue
        if not feature_allowed_by_group(c, feature_group):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().mean() <= float(max_missing):
            feats.append(c)
    return sorted(dict.fromkeys(feats))


def make_lgbm(args) -> LGBMRegressor:
    if LGBMRegressor is None:
        raise ImportError(f"lightgbm is required for this script: {_LIGHTGBM_IMPORT_ERROR}")
    family = str(args.model_family).strip().lower()
    objective = {
        "lgbm_l2": "regression",
        "lgbm": "regression",
        "lgbm_l1": "regression_l1",
        "lgbm_huber": "huber",
        "lgbm_quantile": "quantile",
    }.get(family, "regression")
    return LGBMRegressor(
        objective=objective,
        alpha=float(args.objective_alpha),
        random_state=RANDOM_STATE,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_data_in_leaf),
        subsample=float(args.bagging_fraction),
        subsample_freq=1,
        colsample_bytree=float(args.feature_fraction),
        reg_lambda=float(args.reg_lambda),
        n_jobs=int(args.n_jobs),
        verbose=-1,
    )


def make_model(args):
    family = str(args.model_family).strip().lower()
    if family in {"lgbm", "lgbm_l2", "lgbm_l1", "lgbm_huber", "lgbm_quantile"}:
        return make_lgbm(args), "lgbm"
    if family == "ridge":
        if Ridge is None:
            raise ImportError(f"scikit-learn is required for ridge: {_SKLEARN_IMPORT_ERROR}")
        return make_pipeline(
            SimpleImputer(strategy="median", copy=False),
            StandardScaler(copy=False),
            Ridge(alpha=float(args.ridge_alpha), solver=str(args.ridge_solver)),
        ), "sklearn"
    if family == "elasticnet":
        if ElasticNet is None:
            raise ImportError(f"scikit-learn is required for elasticnet: {_SKLEARN_IMPORT_ERROR}")
        return make_pipeline(
            SimpleImputer(strategy="median", copy=False),
            StandardScaler(copy=False),
            ElasticNet(
                alpha=float(args.elasticnet_alpha),
                l1_ratio=float(args.elasticnet_l1_ratio),
                max_iter=int(args.elasticnet_max_iter),
                random_state=RANDOM_STATE,
            ),
        ), "sklearn"
    if family == "extratrees":
        if ExtraTreesRegressor is None:
            raise ImportError(f"scikit-learn is required for extratrees: {_SKLEARN_IMPORT_ERROR}")
        return ExtraTreesRegressor(
            n_estimators=int(args.tree_n_estimators),
            max_depth=None if int(args.tree_max_depth) <= 0 else int(args.tree_max_depth),
            min_samples_leaf=int(args.tree_min_samples_leaf),
            min_samples_split=int(args.tree_min_samples_split),
            max_features=float(args.tree_max_features),
            bootstrap=False,
            random_state=RANDOM_STATE,
            n_jobs=int(args.n_jobs),
        ), "sklearn_tree"
    if family == "randomforest":
        if RandomForestRegressor is None:
            raise ImportError(f"scikit-learn is required for randomforest: {_SKLEARN_IMPORT_ERROR}")
        return RandomForestRegressor(
            n_estimators=int(args.tree_n_estimators),
            max_depth=None if int(args.tree_max_depth) <= 0 else int(args.tree_max_depth),
            min_samples_leaf=int(args.tree_min_samples_leaf),
            min_samples_split=int(args.tree_min_samples_split),
            max_features=float(args.tree_max_features),
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=int(args.n_jobs),
        ), "sklearn_tree"
    if family in {"catboost_rmse", "catboost_huber", "catboost_quantile"}:
        if CatBoostRegressor is None:
            raise ImportError(f"catboost is required for {family}: {_CATBOOST_IMPORT_ERROR}")
        loss = {
            "catboost_rmse": "RMSE",
            "catboost_huber": f"Huber:delta={float(args.catboost_huber_delta):g}",
            "catboost_quantile": f"Quantile:alpha={float(args.objective_alpha):g}",
        }[family]
        return CatBoostRegressor(
            loss_function=loss,
            iterations=int(args.catboost_iterations),
            learning_rate=float(args.catboost_learning_rate),
            depth=int(args.catboost_depth),
            l2_leaf_reg=float(args.catboost_l2_leaf_reg),
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
        ), "catboost"
    raise ValueError(f"unknown --model-family={args.model_family}")


def prepare_x(train: pd.DataFrame, apply: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x_train = train.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    x_apply = apply.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    with np.errstate(all="ignore"):
        med = np.nanmedian(x_train, axis=0).astype(np.float32, copy=False)
    med = np.where(np.isfinite(med), med, np.float32(0.0)).astype(np.float32, copy=False)
    train_nan = ~np.isfinite(x_train)
    if train_nan.any():
        x_train[train_nan] = np.take(med, np.where(train_nan)[1])
    apply_nan = ~np.isfinite(x_apply)
    if apply_nan.any():
        x_apply[apply_nan] = np.take(med, np.where(apply_nan)[1])
    return x_train, x_apply


def training_target(train_fit: pd.DataFrame, args) -> Tuple[pd.Series, Dict[str, float]]:
    y = to_num(train_fit["target_1d_forward_return_bps"])
    meta = {"target_winsor_q_low": np.nan, "target_winsor_q_high": np.nan}
    if not bool(getattr(args, "winsorize_target", False)):
        return y, meta
    q_low = float(getattr(args, "winsor_q_low", 0.01))
    q_high = float(getattr(args, "winsor_q_high", 0.99))
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("--winsor-q-low/--winsor-q-high must satisfy 0 <= low < high <= 1")
    lo = float(y.quantile(q_low))
    hi = float(y.quantile(q_high))
    meta["target_winsor_q_low"] = lo
    meta["target_winsor_q_high"] = hi
    return y.clip(lo, hi), meta


def add_rank_candidates(scored: pd.DataFrame, require_liquidity_ok: bool) -> pd.DataFrame:
    out = scored.copy()
    base = bool_mask(out["entry_signal"])
    if require_liquidity_ok and "ml4t_liquidity_ok" in out.columns:
        base &= pd.to_numeric(out["ml4t_liquidity_ok"], errors="coerce").fillna(0) > 0
    out["rank_candidate"] = base
    out["daily_candidate_count"] = out.groupby("date")["rank_candidate"].transform("sum").astype("Int64")
    return out


def assign_daily_ranks(scored: pd.DataFrame, require_liquidity_ok: bool) -> pd.DataFrame:
    out = add_rank_candidates(scored, require_liquidity_ok=require_liquidity_ok)
    out["pred_rank_pct"] = np.nan
    out["pred_decile"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
    cand = bool_mask(out["rank_candidate"])
    if cand.any():
        # pct=True ascending=True means high predictions have high rank_pct.
        ranks = out.loc[cand].groupby("date")["pred_return_bps"].rank(pct=True, method="first")
        out.loc[cand, "pred_rank_pct"] = ranks
        dec = np.ceil(out.loc[cand, "pred_rank_pct"].astype(float) * 10.0).clip(1, 10)
        out.loc[cand, "pred_decile"] = dec.astype("Int64")
    return out


def select_trades(scored: pd.DataFrame, args) -> pd.DataFrame:
    out = scored.copy()
    base = bool_mask(out["rank_candidate"])
    base &= pd.to_numeric(out["daily_candidate_count"], errors="coerce").fillna(0) >= int(args.min_daily_candidates)
    # Evaluation and baseline pools should match the universe where a cross-sectional
    # decile signal is meaningful. Days with too few candidates are excluded from
    # IC/decile-lift/baseline metrics as well as from selection.
    out["eval_candidate"] = base

    if args.selection_rule == "strict_top_decile_positive":
        base &= out["pred_decile"].astype("Int64") == 10
        base &= pd.to_numeric(out["pred_return_bps"], errors="coerce") > float(args.min_pred_return_bps)
    elif args.selection_rule == "top_quantile":
        base &= pd.to_numeric(out["pred_rank_pct"], errors="coerce") >= float(args.top_quantile)
        base &= pd.to_numeric(out["pred_return_bps"], errors="coerce") > float(args.min_pred_return_bps)
    else:
        raise ValueError(f"unknown --selection-rule: {args.selection_rule}")

    out["selection_candidate"] = base
    out["selected"] = False
    group_cols = ["split", "date"] if "split" in out.columns else ["date"]
    for _, idx in out[base].groupby(group_cols).groups.items():
        part = out.loc[list(idx)].sort_values("pred_return_bps", ascending=False)
        chosen_idx = part.head(int(args.max_positions)).index
        out.loc[chosen_idx, "selected"] = True
    return out


def _eval_pool(scored: pd.DataFrame) -> pd.DataFrame:
    if "eval_candidate" in scored.columns:
        return scored[bool_mask(scored["eval_candidate"])].copy()
    if "rank_candidate" in scored.columns:
        return scored[bool_mask(scored["rank_candidate"])].copy()
    return scored[bool_mask(scored["entry_signal"])].copy()


def daily_ic_frame(scored: pd.DataFrame, window_id: int, split: str) -> pd.DataFrame:
    rows = []
    pool = _eval_pool(scored)
    for d, g in pool.groupby("date", sort=True):
        rows.append({
            "window_id": window_id,
            "split": split,
            "date": d,
            "n": int(len(g)),
            "daily_ic": safe_spearman(g["pred_return_bps"], g["target_1d_forward_return_bps"]),
            "daily_pearson": safe_pearson(g["pred_return_bps"], g["target_1d_forward_return_bps"]),
        })
    return pd.DataFrame(rows)


def quantile_lift_frame(scored: pd.DataFrame, window_id: int, split: str) -> pd.DataFrame:
    rows = []
    pool = _eval_pool(scored)
    if pool.empty:
        return pd.DataFrame(rows)
    ret = pool["target_1d_forward_return_bps"] / 10000.0
    pool = pool.assign(_ret=ret)
    for decile, g in pool.groupby("pred_decile", dropna=True):
        m = return_metrics(g["_ret"])
        rows.append({
            "window_id": window_id,
            "split": split,
            "pred_decile": int(decile),
            "rows": int(len(g)),
            "avg_target_bps": float(pd.to_numeric(g["target_1d_forward_return_bps"], errors="coerce").mean()),
            "median_target_bps": float(pd.to_numeric(g["target_1d_forward_return_bps"], errors="coerce").median()),
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
        })
    return pd.DataFrame(rows)


def regression_prediction_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float | int]:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_pred, errors="coerce")
    m = y.notna() & p.notna()
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return {
            "n": 0,
            "spearman": np.nan,
            "pearson": np.nan,
            "rmse": np.nan,
            "target_std": np.nan,
            "rmse_norm": np.nan,
            "r2": np.nan,
            "pred_std": np.nan,
            "pred_std_ratio": np.nan,
        }
    err = p - y
    rmse = float(np.sqrt(np.mean(np.square(err))))
    target_std = float(y.std(ddof=0))
    pred_std = float(p.std(ddof=0))
    sse = float(np.sum(np.square(err)))
    sst = float(np.sum(np.square(y - y.mean())))
    return {
        "n": int(len(y)),
        "spearman": safe_spearman(p, y),
        "pearson": safe_pearson(p, y),
        "rmse": rmse,
        "target_std": target_std,
        "rmse_norm": rmse / target_std if target_std > 1e-12 else np.nan,
        "r2": 1.0 - sse / sst if sst > 1e-12 else np.nan,
        "pred_std": pred_std,
        "pred_std_ratio": pred_std / target_std if target_std > 1e-12 else np.nan,
    }


def evaluate_selected(scored: pd.DataFrame, window_id: int, split: str) -> Tuple[Dict, pd.DataFrame]:
    out = scored.copy()
    out["realized_return"] = out["target_1d_forward_return_bps"] / 10000.0
    selected = out[bool_mask(out["selected"])].copy()
    daily = selected.groupby("date", as_index=False).agg(
        n_positions=("stock_code", "count"),
        daily_return=("realized_return", "mean"),
        avg_pred_bps=("pred_return_bps", "mean"),
        avg_realized_bps=("target_1d_forward_return_bps", "mean"),
    )
    daily["window_id"] = window_id
    daily["split"] = split

    pool = _eval_pool(out)
    row = {
        "window_id": window_id,
        "split": split,
        "start_date": str(out["date"].min().date()) if len(out) else None,
        "end_date": str(out["date"].max().date()) if len(out) else None,
        "n_rows": int(len(out)),
        "n_eval_rows": int(len(pool)),
        "n_dates": int(out["date"].nunique()),
        "n_stocks": int(out["stock_code"].nunique()),
        "overall_ic": safe_spearman(pool["pred_return_bps"], pool["target_1d_forward_return_bps"]) if not pool.empty else np.nan,
        "overall_pearson": safe_pearson(pool["pred_return_bps"], pool["target_1d_forward_return_bps"]) if not pool.empty else np.nan,
        "selected_trades": int(len(selected)),
        "selected_dates": int(daily["date"].nunique()) if not daily.empty else 0,
    }
    trade_m = return_metrics(selected["realized_return"])
    row.update({f"trade_{k}": v for k, v in trade_m.items()})
    port_m = return_metrics(daily["daily_return"] if not daily.empty else pd.Series(dtype=float))
    row.update({f"daily_portfolio_{k}": v for k, v in port_m.items()})

    baseline_trade_m = return_metrics(pool["target_1d_forward_return_bps"] / 10000.0 if not pool.empty else pd.Series(dtype=float))
    row.update({f"baseline_trade_{k}": v for k, v in baseline_trade_m.items()})
    if not pool.empty:
        baseline_daily = pool.assign(_ret=pool["target_1d_forward_return_bps"] / 10000.0).groupby("date")['_ret'].mean()
    else:
        baseline_daily = pd.Series(dtype=float)
    baseline_daily_m = return_metrics(baseline_daily)
    row.update({f"baseline_daily_{k}": v for k, v in baseline_daily_m.items()})
    return row, daily

def fit_predict_window(
    df: pd.DataFrame,
    train_dates: np.ndarray,
    valid_dates: np.ndarray,
    test_dates: np.ndarray,
    features: Sequence[str],
    args,
    window_id: int,
) -> Tuple[List[Dict], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], pd.DataFrame]:
    train_mask = df["date"].isin(train_dates)
    valid_mask = df["date"].isin(valid_dates) if len(valid_dates) else pd.Series(False, index=df.index)
    test_mask = df["date"].isin(test_dates)
    base_cols = [
        "date", "stock_code", "entry_signal", "entry_price", "exit_price",
        "target_1d_forward_return_bps", "ml4t_dollar_volume_rank_pct", "ml4t_liquidity_ok",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    work_cols = list(dict.fromkeys(base_cols + list(features)))
    # Train only rows that would have been tradable and have known label.
    train_fit_mask = (
        train_mask
        & bool_mask(df["entry_signal"])
        & df["target_1d_forward_return_bps"].notna()
    )
    train_fit = df.loc[train_fit_mask, work_cols].copy()
    min_dates = int(args.min_train_dates) if int(args.min_train_dates) > 0 else max(20, int(args.train_days * 0.2))
    if len(train_fit) < int(args.min_train_rows) or train_fit["date"].nunique() < min_dates:
        print(f"[WARN] skip window {window_id}: insufficient train rows={len(train_fit)} dates={train_fit['date'].nunique() if len(train_fit) else 0}; need rows>={args.min_train_rows}, dates>={min_dates}", file=sys.stderr)
        del train_fit, train_mask, valid_mask, test_mask
        gc.collect()
        return [], [], [], [], pd.DataFrame(), pd.DataFrame()
    apply_mask = test_mask if not len(valid_dates) else (valid_mask | test_mask)
    apply = df.loc[apply_mask, work_cols].copy()
    if apply.empty:
        del train_fit, apply, train_mask, valid_mask, test_mask, apply_mask
        gc.collect()
        return [], [], [], [], pd.DataFrame(), pd.DataFrame()
    X_train, X_apply = prepare_x(train_fit, apply, features)
    y_train, target_meta = training_target(train_fit, args)
    del train_fit, train_mask, valid_mask, test_mask, apply_mask
    gc.collect()
    model, model_kind = make_model(args)
    if model_kind == "lgbm":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_fit_metrics = {
        f"train_fit_{k}": v
        for k, v in regression_prediction_metrics(
            pd.Series(y_train).reset_index(drop=True),
            pd.Series(train_pred).reset_index(drop=True),
        ).items()
    }
    pred = model.predict(X_apply)
    del X_train, X_apply, y_train, train_pred
    gc.collect()
    scored = apply[[
        "date", "stock_code", "entry_signal", "entry_price", "exit_price",
        "target_1d_forward_return_bps", "ml4t_dollar_volume_rank_pct", "ml4t_liquidity_ok",
    ]].copy()
    del apply
    scored["window_id"] = window_id
    scored["pred_return_bps"] = pred
    scored["split"] = np.where(scored["date"].isin(test_dates), "test", "valid")
    scored = assign_daily_ranks(scored, require_liquidity_ok=args.require_liquidity_ok)
    scored = select_trades(scored, args)

    metrics_rows: List[Dict] = []
    daily_port_parts: List[pd.DataFrame] = []
    daily_ic_parts: List[pd.DataFrame] = []
    quantile_parts: List[pd.DataFrame] = []
    for split in ["valid", "test"]:
        part = scored[scored["split"] == split].copy()
        if part.empty:
            continue
        row, daily_port = evaluate_selected(part, window_id, split)
        row.update(target_meta)
        row.update(train_fit_metrics)
        row["model_family"] = str(args.model_family)
        row["feature_group"] = str(args.feature_group)
        metrics_rows.append(row)
        daily_port_parts.append(daily_port)
        daily_ic_parts.append(daily_ic_frame(part, window_id, split))
        quantile_parts.append(quantile_lift_frame(part, window_id, split))

    # Feature importance.
    importance = pd.DataFrame({
        "window_id": window_id,
        "feature": list(features),
        "importance": getattr(model, "feature_importances_", np.zeros(len(features))),
        "model_family": str(args.model_family),
        "feature_group": str(args.feature_group),
    })
    daily_port = pd.concat(daily_port_parts, ignore_index=True) if daily_port_parts else pd.DataFrame()
    del model, pred, daily_port_parts
    gc.collect()
    return metrics_rows, [scored], daily_ic_parts, quantile_parts, daily_port, importance



def finite_json(obj):
    """Recursively convert NaN/Inf to None for strict JSON output."""
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [finite_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def expected_prediction_columns() -> List[str]:
    return [
        "date", "stock_code", "entry_signal", "entry_price", "exit_price",
        "target_1d_forward_return_bps", "ml4t_dollar_volume_rank_pct", "ml4t_liquidity_ok",
        "window_id", "pred_return_bps", "split", "rank_candidate", "daily_candidate_count",
        "eval_candidate", "pred_rank_pct", "pred_decile", "selection_candidate", "selected",
    ]


def ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(cols))
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out

def run_backtest(df: pd.DataFrame, features: Sequence[str], args, out_dir: Path) -> Dict:
    windows = date_windows(
        dates=df["date"].unique(),
        train_days=args.train_days,
        valid_days=args.valid_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        step_days=args.step_days,
    )
    if not windows:
        need = required_window_dates(args)
        raise RuntimeError(
            f"no rolling windows produced; unique_dates={df['date'].nunique()} need>={need} "
            f"train={args.train_days} valid={args.valid_days} embargo={args.embargo_days} test={args.test_days}. "
            f"Either reduce --train-days or run with --auto-update-data --auto-update-start-date 2018-01-01."
        )
    print(f"[INFO] rolling windows: {len(windows)}")
    metrics_rows: List[Dict] = []
    pred_parts: List[pd.DataFrame] = []
    daily_ic_parts: List[pd.DataFrame] = []
    quantile_parts: List[pd.DataFrame] = []
    daily_port_parts: List[pd.DataFrame] = []
    importance_parts: List[pd.DataFrame] = []

    for i, (train_dates, valid_dates, test_dates) in enumerate(windows, start=1):
        print(
            f"[INFO] window={i} "
            f"train={pd.Timestamp(train_dates[0]).date()}..{pd.Timestamp(train_dates[-1]).date()} "
            + (f"valid={pd.Timestamp(valid_dates[0]).date()}..{pd.Timestamp(valid_dates[-1]).date()} " if len(valid_dates) else "")
            + f"test={pd.Timestamp(test_dates[0]).date()}..{pd.Timestamp(test_dates[-1]).date()}"
        )
        rows, preds, ics, qs, daily_port, imp = fit_predict_window(df, train_dates, valid_dates, test_dates, features, args, i)
        metrics_rows.extend(rows)
        pred_parts.extend(preds)
        daily_ic_parts.extend(ics)
        quantile_parts.extend(qs)
        if not daily_port.empty:
            daily_port_parts.append(daily_port)
        if not imp.empty:
            importance_parts.append(imp)
        del rows, preds, ics, qs, daily_port, imp
        gc.collect()

    metrics_cols = ["window_id", "split", "model_family", "feature_group", "start_date", "end_date", "n_rows", "n_eval_rows", "n_dates", "n_stocks", "overall_ic", "overall_pearson", "selected_trades", "selected_dates", "target_winsor_q_low", "target_winsor_q_high"]
    metrics = ensure_columns(pd.DataFrame(metrics_rows), metrics_cols)
    preds = ensure_columns(pd.concat(pred_parts, ignore_index=True, sort=False) if pred_parts else pd.DataFrame(), expected_prediction_columns())
    daily_ic = ensure_columns(pd.concat(daily_ic_parts, ignore_index=True, sort=False) if daily_ic_parts else pd.DataFrame(), ["window_id", "split", "date", "n", "daily_ic", "daily_pearson"])
    qlift = ensure_columns(pd.concat(quantile_parts, ignore_index=True, sort=False) if quantile_parts else pd.DataFrame(), ["window_id", "split", "pred_decile", "rows", "avg_target_bps", "median_target_bps", "win_rate", "profit_factor"])
    daily_port = ensure_columns(pd.concat(daily_port_parts, ignore_index=True, sort=False) if daily_port_parts else pd.DataFrame(), ["date", "n_positions", "daily_return", "avg_pred_bps", "avg_realized_bps", "window_id", "split"])
    importance = pd.concat(importance_parts, ignore_index=True, sort=False) if importance_parts else pd.DataFrame()

    metrics.to_csv(out_dir / "window_metrics.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    selected_all = preds[bool_mask(preds["selected"])].copy() if "selected" in preds.columns else pd.DataFrame(columns=preds.columns)
    selected_all.to_csv(out_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    selected_test = selected_all[selected_all["split"] == "test"].copy() if not selected_all.empty and "split" in selected_all.columns else pd.DataFrame(columns=preds.columns)
    selected_test.to_csv(out_dir / "selected_trades_test.csv", index=False, encoding="utf-8-sig")
    daily_ic.to_csv(out_dir / "daily_ic.csv", index=False, encoding="utf-8-sig")
    qlift.to_csv(out_dir / "quantile_lift.csv", index=False, encoding="utf-8-sig")
    daily_port.to_csv(out_dir / "daily_portfolio_returns.csv", index=False, encoding="utf-8-sig")
    if not importance.empty:
        imp_sum = importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        importance.to_csv(out_dir / "feature_importance_by_window.csv", index=False, encoding="utf-8-sig")
        imp_sum.to_csv(out_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")
        del imp_sum

    feature_policy = pd.DataFrame({
        "feature": list(features),
        "policy": [feature_time_policy(f) for f in features],
        "missing_rate": [float(df[f].isna().mean()) if f in df.columns else np.nan for f in features],
        "feature_group": str(args.feature_group),
    })
    feature_policy.to_csv(out_dir / "feature_time_policy_report.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics[metrics["split"] == "test"].copy() if not metrics.empty else pd.DataFrame()
    selected = selected_test.copy()
    test_daily_port = daily_port[daily_port["split"] == "test"].copy() if not daily_port.empty else pd.DataFrame()
    test_ic = daily_ic[daily_ic["split"] == "test"].copy() if not daily_ic.empty else pd.DataFrame()

    summary = {
        "n_input_rows": int(len(df)),
        "n_dates": int(df["date"].nunique()),
        "n_stocks": int(df["stock_code"].nunique()),
        "n_features": int(len(features)),
        "features": list(features),
        "feature_group": str(args.feature_group),
        "model_family": str(args.model_family),
        "n_windows": int(len(windows)),
        "test_selected_trades": int(len(selected)),
        "test_selected_dates": int(test_daily_port["date"].nunique()) if not test_daily_port.empty else 0,
        "test_mean_daily_ic": safe_series_mean(test_ic["daily_ic"]) if not test_ic.empty else np.nan,
        "test_median_daily_ic": safe_series_median(test_ic["daily_ic"]) if not test_ic.empty else np.nan,
        "test_trade_metrics": return_metrics(selected["target_1d_forward_return_bps"] / 10000.0) if not selected.empty else return_metrics(pd.Series(dtype=float)),
        "test_daily_portfolio_metrics": return_metrics(test_daily_port["daily_return"]) if not test_daily_port.empty else return_metrics(pd.Series(dtype=float)),
        "config": vars(args),
    }
    summary = finite_json(summary)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return summary


def prepare_window_dataset(
    sample_paths: Sequence[Path],
    bars_map: Dict[str, Path],
    args,
    train_dates: np.ndarray,
    valid_dates: np.ndarray,
    test_dates: np.ndarray,
) -> Tuple[pd.DataFrame, List[str]]:
    horizons = [1, 5, 10, 21, 63]
    window_start = pd.Timestamp(train_dates[0]) - pd.Timedelta(days=int(args.lazy_window_lookback_days))
    window_end = pd.Timestamp(test_dates[-1])
    raw = load_samples_between(
        sample_paths,
        start_date=window_start.normalize(),
        end_date=window_end.normalize(),
        chunksize=int(args.lazy_chunk_rows),
        usecols=lazy_usecols(args),
    )
    if raw.empty:
        return pd.DataFrame(), []
    df = build_ml4t_features(
        raw,
        horizons=horizons,
        min_liquidity_rank_pct=args.min_liquidity_rank_pct,
        bars_map=bars_map,
        cutoff_time=args.cutoff_time,
        max_intraday_lag=args.max_intraday_lag,
    )
    del raw
    gc.collect()
    df = add_target_and_entry(
        df,
        entry_price_col=args.entry_price_col,
        exit_price_col=args.exit_price_col,
        round_trip_cost_bps=args.round_trip_cost_bps,
        entry_policy=args.entry_policy,
        entry_vwap_premium_bps=args.entry_vwap_premium_bps,
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "stock_code", "entry_price", "exit_price", "target_1d_forward_return_bps"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    features = candidate_features(df, max_missing=args.max_missing, feature_group=args.feature_group)
    if not features:
        return pd.DataFrame(), []
    keep_cols = [
        "date", "stock_code", "entry_signal", "entry_price", "exit_price",
        "target_1d_forward_return_bps", "ml4t_dollar_volume_rank_pct", "ml4t_liquidity_ok",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.loc[:, list(dict.fromkeys(keep_cols + list(features)))].copy()
    gc.collect()
    return df, features


def run_backtest_lazy(
    sample_paths: Sequence[Path],
    bars_map: Dict[str, Path],
    args,
    out_dir: Path,
) -> Dict:
    dates = load_sample_dates(sample_paths, chunksize=int(args.lazy_chunk_rows))
    windows = date_windows(
        dates=dates,
        train_days=args.train_days,
        valid_days=args.valid_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        step_days=args.step_days,
    )
    if not windows:
        need = required_window_dates(args)
        raise RuntimeError(f"no rolling windows produced; unique_dates={len(dates)} need>={need}")
    print(f"[INFO] rolling windows: {len(windows)} lazy_window_load=1")

    metrics_rows: List[Dict] = []
    pred_parts: List[pd.DataFrame] = []
    daily_ic_parts: List[pd.DataFrame] = []
    quantile_parts: List[pd.DataFrame] = []
    daily_port_parts: List[pd.DataFrame] = []
    importance_parts: List[pd.DataFrame] = []
    feature_union: List[str] = []
    feature_policy_rows: List[Dict] = []
    total_rows = 0
    all_dates: set = set()
    all_stocks: set = set()

    for i, (train_dates, valid_dates, test_dates) in enumerate(windows, start=1):
        print(
            f"[INFO] window={i} "
            f"train={pd.Timestamp(train_dates[0]).date()}..{pd.Timestamp(train_dates[-1]).date()} "
            + (f"valid={pd.Timestamp(valid_dates[0]).date()}..{pd.Timestamp(valid_dates[-1]).date()} " if len(valid_dates) else "")
            + f"test={pd.Timestamp(test_dates[0]).date()}..{pd.Timestamp(test_dates[-1]).date()}"
        )
        df, features = prepare_window_dataset(sample_paths, bars_map, args, train_dates, valid_dates, test_dates)
        if df.empty or not features:
            print(f"[WARN] skip window {i}: no lazy-loaded features/data survived", file=sys.stderr)
            continue
        total_rows += int(len(df))
        all_dates.update(pd.to_datetime(df["date"]).dropna().dt.strftime("%Y-%m-%d").tolist())
        all_stocks.update(df["stock_code"].dropna().astype(str).tolist())
        feature_union = sorted(dict.fromkeys(feature_union + list(features)))
        feature_policy_rows.extend({
            "window_id": i,
            "feature": f,
            "policy": feature_time_policy(f),
            "missing_rate": float(df[f].isna().mean()) if f in df.columns else np.nan,
            "feature_group": str(args.feature_group),
        } for f in features)
        rows, preds, ics, qs, daily_port, imp = fit_predict_window(df, train_dates, valid_dates, test_dates, features, args, i)
        metrics_rows.extend(rows)
        pred_parts.extend(preds)
        daily_ic_parts.extend(ics)
        quantile_parts.extend(qs)
        if not daily_port.empty:
            daily_port_parts.append(daily_port)
        if not imp.empty:
            importance_parts.append(imp)
        del df, features, rows, preds, ics, qs, daily_port, imp
        gc.collect()

    metrics_cols = ["window_id", "split", "model_family", "feature_group", "start_date", "end_date", "n_rows", "n_eval_rows", "n_dates", "n_stocks", "overall_ic", "overall_pearson", "selected_trades", "selected_dates", "target_winsor_q_low", "target_winsor_q_high"]
    metrics = ensure_columns(pd.DataFrame(metrics_rows), metrics_cols)
    preds = ensure_columns(pd.concat(pred_parts, ignore_index=True, sort=False) if pred_parts else pd.DataFrame(), expected_prediction_columns())
    daily_ic = ensure_columns(pd.concat(daily_ic_parts, ignore_index=True, sort=False) if daily_ic_parts else pd.DataFrame(), ["window_id", "split", "date", "n", "daily_ic", "daily_pearson"])
    qlift = ensure_columns(pd.concat(quantile_parts, ignore_index=True, sort=False) if quantile_parts else pd.DataFrame(), ["window_id", "split", "pred_decile", "rows", "avg_target_bps", "median_target_bps", "win_rate", "profit_factor"])
    daily_port = ensure_columns(pd.concat(daily_port_parts, ignore_index=True, sort=False) if daily_port_parts else pd.DataFrame(), ["date", "n_positions", "daily_return", "avg_pred_bps", "avg_realized_bps", "window_id", "split"])
    importance = pd.concat(importance_parts, ignore_index=True, sort=False) if importance_parts else pd.DataFrame()

    metrics.to_csv(out_dir / "window_metrics.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    selected_all = preds[bool_mask(preds["selected"])].copy() if "selected" in preds.columns else pd.DataFrame(columns=preds.columns)
    selected_all.to_csv(out_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    selected_test = selected_all[selected_all["split"] == "test"].copy() if not selected_all.empty and "split" in selected_all.columns else pd.DataFrame(columns=preds.columns)
    selected_test.to_csv(out_dir / "selected_trades_test.csv", index=False, encoding="utf-8-sig")
    daily_ic.to_csv(out_dir / "daily_ic.csv", index=False, encoding="utf-8-sig")
    qlift.to_csv(out_dir / "quantile_lift.csv", index=False, encoding="utf-8-sig")
    daily_port.to_csv(out_dir / "daily_portfolio_returns.csv", index=False, encoding="utf-8-sig")
    if not importance.empty:
        imp_sum = importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        importance.to_csv(out_dir / "feature_importance_by_window.csv", index=False, encoding="utf-8-sig")
        imp_sum.to_csv(out_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_policy_rows).to_csv(out_dir / "feature_time_policy_report.csv", index=False, encoding="utf-8-sig")
    pd.Series(feature_union, name="feature").to_csv(out_dir / "feature_columns.txt", index=False, header=False, encoding="utf-8")

    test_metrics = metrics[metrics["split"] == "test"].copy() if not metrics.empty else pd.DataFrame()
    selected = selected_test.copy()
    test_daily_port = daily_port[daily_port["split"] == "test"].copy() if not daily_port.empty else pd.DataFrame()
    test_ic = daily_ic[daily_ic["split"] == "test"].copy() if not daily_ic.empty else pd.DataFrame()
    summary = finite_json({
        "n_input_rows": int(total_rows),
        "n_dates": int(len(all_dates)),
        "n_stocks": int(len(all_stocks)),
        "n_features": int(len(feature_union)),
        "features": list(feature_union),
        "feature_group": str(args.feature_group),
        "model_family": str(args.model_family),
        "n_windows": int(len(windows)),
        "lazy_window_load": True,
        "test_selected_trades": int(len(selected)),
        "test_selected_dates": int(test_daily_port["date"].nunique()) if not test_daily_port.empty else 0,
        "test_mean_daily_ic": safe_series_mean(test_ic["daily_ic"]) if not test_ic.empty else np.nan,
        "test_median_daily_ic": safe_series_median(test_ic["daily_ic"]) if not test_ic.empty else np.nan,
        "test_trade_metrics": return_metrics(selected["target_1d_forward_return_bps"] / 10000.0) if not selected.empty else return_metrics(pd.Series(dtype=float)),
        "test_daily_portfolio_metrics": return_metrics(test_daily_port["daily_return"]) if not test_daily_port.empty else return_metrics(pd.Series(dtype=float)),
        "config": vars(args),
    })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    return summary




def default_run_name(args) -> str:
    return (
        f"backtest_lgbm_asof1455_"
        f"{args.selection_rule}_"
        f"train{args.train_days}_test{args.test_days}_embargo{args.embargo_days}_"
        f"pos{args.max_positions}"
    )


def resolve_output_dirs(args) -> Tuple[Path, Path]:
    root = resolve_project_root()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        run_name = args.run_name or default_run_name(args)
        out_dir = out_root / "99_summary" / run_name
    ensure_dir(out_root)
    ensure_dir(out_root / "logs")
    ensure_dir(out_dir)
    return out_root, out_dir


def write_pipeline_root_summary(out_root: Path, out_dir: Path, args, summary: Dict) -> None:
    payload = {
        "pipeline_name": "ml4t_asof1455_lgbm_pipeline_out",
        "latest_run_dir": str(out_dir),
        "latest_summary_file": str(out_dir / "summary.json"),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name or default_run_name(args),
        "selection_rule": args.selection_rule,
        "train_days": int(args.train_days),
        "test_days": int(args.test_days),
        "embargo_days": int(args.embargo_days),
        "max_positions": int(args.max_positions),
        "summary": finite_json(summary),
    }
    with open(out_root / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Backtest ML4T asof1455 LightGBM forward-return model")
    p.add_argument("--samples", nargs="*", default=[], help="explicit training_samples_asof1455.csv files")
    p.add_argument("--sample-glob", action="append", default=[], help="glob for asof1455 sample files; may be repeated")
    p.add_argument("--bars-glob", action="append", default=[], help="optional glob for per-stock 5m bar CSVs, e.g. saved_data/*_pipeline_out/00_base/*_5m.csv")
    p.add_argument("--out-dir", default="", help="Exact run output directory. If omitted, writes under --out-root/99_summary/<run-name>.")
    p.add_argument("--out-root", default="saved_data/ml4t_asof1455_lgbm_pipeline_out", help="Organized ML4T pipeline root, similar to saved_data/<code>_pipeline_out.")
    p.add_argument("--run-name", default="", help="Optional subfolder name under --out-root/99_summary.")
    p.add_argument("--ml4t-sample-stage", default="01_samples_asof1455", help="Canonical per-stock pipeline_out stage for asof1455 samples.")

    p.add_argument("--auto-update-data", action="store_true", help="If sample files are missing or too short for the requested rolling window, run the project-native pipelines/run_nextday_pipeline.py to download/build longer asof1455 samples, then reload.")
    p.add_argument("--auto-symbols", default="", help="Comma-separated symbols for --auto-update-data; if empty, infer from loaded samples, watchlist, bars, or pipeline dirs.")
    p.add_argument("--auto-symbols-file", default="", help="Optional watchlist file used by --auto-update-data when no asof samples exist yet.")
    p.add_argument("--auto-max-symbols", type=int, default=0, help="Limit auto-updated symbols for testing; 0 means no limit.")
    p.add_argument("--auto-update-start-date", default="2018-01-01")
    p.add_argument("--auto-update-end-date", default="today")
    p.add_argument("--auto-update-python", default=None, help="Python executable for project pipeline; default current interpreter.")
    p.add_argument("--auto-cache-mode", default="incremental", choices=["incremental", "full"])
    p.add_argument("--auto-feature-cache-mode", default="incremental", choices=["incremental", "full"])
    p.add_argument("--auto-min-bars", type=int, default=40)
    p.add_argument("--auto-force-refresh", action="store_true")
    p.add_argument("--auto-keep-going", action="store_true", help="Continue auto-updating other symbols when one symbol fails.")
    p.add_argument("--auto-update-dry-run", action="store_true", help="Print/write auto-update commands without executing them.")

    p.add_argument("--cutoff-time", default="14:55")
    p.add_argument("--entry-price-col", default="close_asof1455", choices=["close_asof1455", "vwap_asof1455"])
    p.add_argument("--exit-price-col", default="next_day_close", help="next_day_close or next_day_vwap")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--entry-policy", default="all_days", choices=["all_days", "vwap_low"])
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)

    p.add_argument("--train-days", type=int, default=756)
    p.add_argument("--valid-days", type=int, default=0, help="optional validation days; default 0 keeps a direct train->embargo->test backtest")
    p.add_argument("--test-days", type=int, default=21)
    p.add_argument("--embargo-days", type=int, default=1)
    p.add_argument("--step-days", type=int, default=0, help="default=test-days")
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--min-train-dates", type=int, default=0, help="minimum unique training dates; 0 uses max(20, train_days*0.2)")

    p.add_argument("--selection-rule", default="strict_top_decile_positive", choices=["strict_top_decile_positive", "top_quantile"], help="strict default: pred_decile==10 and pred_return_bps > min_pred_return_bps")
    p.add_argument("--top-quantile", type=float, default=0.90, help="used only when --selection-rule top_quantile")
    p.add_argument("--min-pred-return-bps", type=float, default=0.0, help="minimum predicted net return in bps for selected trades")
    p.add_argument("--min-daily-candidates", type=int, default=10, help="minimum rank candidates per day before decile-based selection is allowed")
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--min-liquidity-rank-pct", type=float, default=0.0, help="0 disables liquidity filtering; 0.5 keeps top half by yesterday dollar volume")
    p.add_argument("--require-liquidity-ok", action="store_true", help="require ml4t_liquidity_ok for selected trades")
    p.add_argument("--max-missing", type=float, default=0.70)
    p.add_argument("--save-prepared", action="store_true")
    p.add_argument("--lazy-window-load", action="store_true", help="Load/build only the rows needed for each rolling window, then release them.")
    p.add_argument("--lazy-window-lookback-days", type=int, default=180, help="Calendar-day lookback loaded before each train window for lagged/asof features.")
    p.add_argument("--lazy-chunk-rows", type=int, default=100000, help="CSV chunksize used by --lazy-window-load.")
    p.add_argument("--lazy-usecols", dest="lazy_usecols", action="store_true", default=True, help="In lazy mode, only read raw columns needed by the selected feature group.")
    p.add_argument("--no-lazy-usecols", dest="lazy_usecols", action="store_false", help="In lazy mode, read all columns for debugging/compatibility.")
    p.add_argument("--feature-group", default="ml4t_intraday", choices=["ml4t", "core", "ml4t_core", "ml4t_intraday", "core_intraday", "fundamental", "ml4t_fundamental", "core_fundamental", "sector", "ml4t_sector", "core_sector", "external", "ml4t_external", "core_external", "core_sector_external_lagged", "all", "all_numeric_asof", "all_numeric"])
    p.add_argument("--model-family", default="lgbm_l2", choices=["ridge", "elasticnet", "extratrees", "randomforest", "lgbm", "lgbm_l2", "lgbm_l1", "lgbm_huber", "lgbm_quantile", "catboost_rmse", "catboost_huber", "catboost_quantile"])
    p.add_argument("--winsorize-target", action="store_true", help="Clip the training target within each rolling train window only.")
    p.add_argument("--winsor-q-low", type=float, default=0.01)
    p.add_argument("--winsor-q-high", type=float, default=0.99)

    # LightGBM defaults: deliberately small/regularized. The book-style knobs are exposed.
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--min-data-in-leaf", type=int, default=250)
    p.add_argument("--bagging-fraction", type=float, default=0.75)
    p.add_argument("--feature-fraction", type=float, default=0.75)
    p.add_argument("--reg-lambda", type=float, default=2.0)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--max-intraday-lag", type=int, default=10)
    p.add_argument("--objective-alpha", type=float, default=0.90, help="Huber/quantile alpha where supported.")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--ridge-solver", default="auto", choices=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
    p.add_argument("--elasticnet-alpha", type=float, default=0.01)
    p.add_argument("--elasticnet-l1-ratio", type=float, default=0.15)
    p.add_argument("--elasticnet-max-iter", type=int, default=20000)
    p.add_argument("--tree-n-estimators", type=int, default=600)
    p.add_argument("--tree-max-depth", type=int, default=0, help="<=0 means unlimited")
    p.add_argument("--tree-min-samples-leaf", type=int, default=20)
    p.add_argument("--tree-min-samples-split", type=int, default=40)
    p.add_argument("--tree-max-features", type=float, default=0.6)
    p.add_argument("--catboost-iterations", type=int, default=1000)
    p.add_argument("--catboost-learning-rate", type=float, default=0.03)
    p.add_argument("--catboost-depth", type=int, default=6)
    p.add_argument("--catboost-l2-leaf-reg", type=float, default=10.0)
    p.add_argument("--catboost-huber-delta", type=float, default=100.0)
    args = p.parse_args(argv)
    if args.step_days <= 0:
        args.step_days = args.test_days
    if not 0 < args.top_quantile <= 1:
        raise ValueError("--top-quantile must be in (0,1]")
    if args.min_daily_candidates < 1:
        raise ValueError("--min-daily-candidates must be >= 1")
    if args.max_positions < 1:
        raise ValueError("--max-positions must be >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = resolve_project_root()
    out_root, out_dir = resolve_output_dirs(args)

    sample_paths = expand_inputs(args.samples, args.sample_glob)
    sample_paths = filter_paths_to_pipeline_universe(sample_paths, root, "sample files")
    sample_paths = select_one_sample_per_symbol(sample_paths, root, args.ml4t_sample_stage, args.feature_group)
    print(f"[INFO] sample files: {len(sample_paths)}")
    for p in sample_paths[:20]:
        print(f"  - {p}")
    if len(sample_paths) > 20:
        print(f"  ... ({len(sample_paths) - 20} more)")

    bars_map = filter_bars_map_to_pipeline_universe(build_bars_map(args.bars_glob), root)
    if bars_map:
        print(f"[INFO] 5m bar files mapped: {len(bars_map)}")

    if args.lazy_window_load:
        if args.auto_update_data:
            raise ValueError("--lazy-window-load does not support --auto-update-data in the same run")
        manifest = {
            "sample_files": [str(p) for p in sample_paths],
            "bars_files": {k: str(v) for k, v in sorted(bars_map.items())},
            "feature_group": args.feature_group,
            "model_family": args.model_family,
            "target": "target_1d_forward_return_bps",
            "entry_price_col": args.entry_price_col,
            "exit_price_col": args.exit_price_col,
            "feature_time_mode": "asof1455",
            "out_root": str(out_root),
            "out_dir": str(out_dir),
            "run_name": args.run_name or default_run_name(args),
            "ml4t_sample_stage": args.ml4t_sample_stage,
            "lazy_window_load": True,
        }
        manifest = finite_json(manifest)
        with open(out_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        summary = run_backtest_lazy(sample_paths, bars_map, args, out_dir)
        write_pipeline_root_summary(out_root, out_dir, args, summary)
        print(f"[INFO] ML4T pipeline root: {out_root}")
        print(f"[INFO] ML4T run output: {out_dir}")
        return

    raw: Optional[pd.DataFrame] = None
    if sample_paths:
        raw = load_samples(sample_paths)
    elif args.auto_update_data:
        raw = pd.DataFrame(columns=["date", "stock_code"])
    else:
        raw = load_samples(sample_paths)
    sample_paths, raw = ensure_data_if_needed(sample_paths, raw, args, out_dir)
    if raw is None or raw.empty:
        raw = load_samples(sample_paths)
    if args.auto_update_data:
        # Bars may have been created/extended by the project pipeline.
        auto_bar_patterns = list(args.bars_glob) + ["saved_data/*_pipeline_out/00_base/*_5m.csv"]
        bars_map = filter_bars_map_to_pipeline_universe(build_bars_map(auto_bar_patterns), root)
        if bars_map:
            print(f"[INFO] 5m bar files mapped after auto-update: {len(bars_map)}")

    horizons = [1, 5, 10, 21, 63]
    df = build_ml4t_features(
        raw,
        horizons=horizons,
        min_liquidity_rank_pct=args.min_liquidity_rank_pct,
        bars_map=bars_map,
        cutoff_time=args.cutoff_time,
        max_intraday_lag=args.max_intraday_lag,
    )
    del raw
    gc.collect()
    df = add_target_and_entry(
        df,
        entry_price_col=args.entry_price_col,
        exit_price_col=args.exit_price_col,
        round_trip_cost_bps=args.round_trip_cost_bps,
        entry_policy=args.entry_policy,
        entry_vwap_premium_bps=args.entry_vwap_premium_bps,
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "stock_code", "entry_price", "exit_price", "target_1d_forward_return_bps"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)

    features = candidate_features(df, max_missing=args.max_missing, feature_group=args.feature_group)
    if not features:
        raise RuntimeError("no ML4T feature columns survived; check input asof columns and --max-missing")
    keep_cols = [
        "date", "stock_code", "entry_signal", "entry_price", "exit_price",
        "target_1d_forward_return_bps", "ml4t_dollar_volume_rank_pct", "ml4t_liquidity_ok",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.loc[:, list(dict.fromkeys(keep_cols + list(features)))].copy()
    gc.collect()

    manifest = {
        "sample_files": [str(p) for p in sample_paths],
        "bars_files": {k: str(v) for k, v in sorted(bars_map.items())},
        "rows_after_prepare": int(len(df)),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "n_stocks": int(df["stock_code"].nunique()),
        "features": features,
        "feature_group": args.feature_group,
        "model_family": args.model_family,
        "target": "target_1d_forward_return_bps",
        "entry_price_col": args.entry_price_col,
        "exit_price_col": args.exit_price_col,
        "feature_time_mode": "asof1455",
        "out_root": str(out_root),
        "out_dir": str(out_dir),
        "run_name": args.run_name or default_run_name(args),
        "ml4t_sample_stage": args.ml4t_sample_stage,
    }
    manifest = finite_json(manifest)
    with open(out_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
    pd.Series(features, name="feature").to_csv(out_dir / "feature_columns.txt", index=False, header=False, encoding="utf-8")
    if args.save_prepared:
        df.to_csv(out_dir / "prepared_ml4t_samples.csv", index=False, encoding="utf-8-sig")

    print(f"[INFO] prepared rows={len(df)} dates={df['date'].nunique()} stocks={df['stock_code'].nunique()} features={len(features)}")
    summary = run_backtest(df, features, args, out_dir)
    write_pipeline_root_summary(out_root, out_dir, args, summary)
    print(f"[INFO] ML4T pipeline root: {out_root}")
    print(f"[INFO] ML4T run output: {out_dir}")


if __name__ == "__main__":
    main()
