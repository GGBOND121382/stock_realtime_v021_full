#!/usr/bin/env python3
"""Ridge-only single-symbol asof1455 regression search.

Target:
    target_next_close_bps = 10000 * (next_day_close / close_asof1455 - 1) - cost_bps

This script intentionally avoids tree models, portfolio metrics, selected
trades, top-decile diagnostics, and test-based model selection.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except Exception as exc:  # pragma: no cover
    Parallel = delayed = None
    _JOBLIB_IMPORT_ERROR = exc
else:
    _JOBLIB_IMPORT_ERROR = None

try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.kernel_approximation import Nystroem
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except Exception as exc:  # pragma: no cover
    PCA = SimpleImputer = Nystroem = KernelRidge = Ridge = None
    PolynomialFeatures = StandardScaler = rbf_kernel = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


DEFAULT_SAMPLE_GLOBS = [
    "saved_data/*_pipeline_out/04_external/*/training_samples_with_*external*.csv",
    "saved_data/*_pipeline_out/03_sector/training_samples_with_sector.csv",
    "saved_data/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv",
    "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv",
]
DEFAULT_ALPHA_GRID = "logspace:-6:8:29"
DEFAULT_KERNEL_ALPHA_GRID = "logspace:-4:10:29"
DEFAULT_N_COMPONENTS_GRID = "64,128,256,512,1024"
DEFAULT_SVD_COMPONENTS_GRID = "16,32,64,128,256"
GAMMA_MULTIPLIERS = [1 / 300, 1 / 100, 1 / 30, 1 / 10, 1 / 3, 1, 3, 10, 30]
EXACT_GAMMA_MULTIPLIERS = [1 / 300, 1 / 100, 1 / 30, 1 / 10, 1 / 3, 1, 3, 10]
MODEL_COMPLEXITY = {
    "ridge_linear": 0,
    "ridge_svd": 1,
    "ridge_poly_svd": 2,
    "ridge_rbf_nystroem": 3,
    "ridge_svd_rbf_nystroem": 4,
    "kernel_ridge_rbf": 5,
    "kernel_ridge_svd_rbf": 6,
}
EXCLUDE_EXACT = {
    "date",
    "next_date",
    "stock_code",
    "symbol",
    "code",
    "ts_code",
    "entry_price",
    "entry_price_source",
    "feature_time_mode",
    "feature_cutoff_time",
    "asof_last_bar_time",
}
EXCLUDE_SUBSTRINGS = [
    "target",
    "label",
    "next_day_",
    "future_return",
    "forward_return",
    "pred",
    "score",
    "signal",
    "selected",
    "threshold",
    "portfolio",
    "compound",
    "trade_",
    "eval_",
]


@dataclass(frozen=True)
class Config:
    model_type: str
    alpha: float
    gamma: Optional[float] = None
    n_components: Optional[int] = None
    svd_components: Optional[int] = None
    degree: Optional[int] = None
    interaction_only: Optional[bool] = None

    def key(self) -> Tuple:
        return (
            self.model_type,
            round(float(self.alpha), 14),
            None if self.gamma is None else round(float(self.gamma), 14),
            self.n_components,
            self.svd_components,
            self.degree,
            self.interaction_only,
        )

    def params_json(self) -> str:
        return json.dumps(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "n_components": self.n_components,
                "svd_components": self.svd_components,
                "degree": self.degree,
                "interaction_only": self.interaction_only,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


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


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if math.isfinite(v) else np.nan


def infer_stock_code(path: Path, df: pd.DataFrame) -> str:
    for col in ["stock_code", "symbol", "code", "ts_code"]:
        if col in df.columns and df[col].notna().any():
            return str(df[col].dropna().iloc[0]).strip()
    for parent in [path] + list(path.parents):
        if parent.name.endswith("_pipeline_out"):
            return parent.name[: -len("_pipeline_out")]
    m = re.search(r"(\d{6})", str(path))
    return m.group(1) if m else path.parent.name


def pipeline_stock_code(path: Path) -> str:
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


def sample_has_required_columns(path: Path, entry_col: str, exit_col: str) -> bool:
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return {"date", entry_col, exit_col}.issubset(cols)


def is_recycle_path(path: Path) -> bool:
    return any(part.startswith("_recycle_data_cleanup_") for part in path.parts)


def expand_samples(sample_paths: Sequence[str], sample_globs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for item in sample_paths:
        path = Path(item)
        if path.exists() and path.is_file() and not is_recycle_path(path):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    for pat in sample_globs:
        for name in glob.glob(pat, recursive=True):
            path = Path(name)
            if is_recycle_path(path):
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    return sorted(out, key=lambda p: (pipeline_stock_code(p), -sample_priority(p), str(p)))


def choose_best_sample_per_symbol(paths: Sequence[Path], symbols: Optional[Sequence[str]], max_symbols: int) -> List[Path]:
    symbol_set = {s.strip()[:6] for s in symbols or [] if s.strip()}
    best: Dict[str, Path] = {}
    for path in paths:
        code = pipeline_stock_code(path)
        if not code:
            try:
                code = infer_stock_code(path, pd.read_csv(path, nrows=5))
            except Exception:
                continue
        raw = code[:6]
        if symbol_set and raw not in symbol_set and code not in symbol_set:
            continue
        current = best.get(raw)
        if current is None or sample_priority(path) > sample_priority(current):
            best[raw] = path
    selected = [best[k] for k in sorted(best)]
    if max_symbols > 0:
        selected = selected[:max_symbols]
    return selected


def parse_number_grid(text: str) -> List[float]:
    text = str(text).strip()
    if text.startswith("logspace:"):
        _, a, b, n = text.split(":")
        return [float(x) for x in np.logspace(float(a), float(b), int(n))]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_grid(text: str) -> List[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def gamma_grid(d: int, exact: bool = False) -> List[float]:
    if d <= 0:
        return []
    multipliers = EXACT_GAMMA_MULTIPLIERS if exact else GAMMA_MULTIPLIERS
    return [float(m / d) for m in multipliers]


def is_excluded_feature(col: str) -> bool:
    lower = str(col).lower()
    if lower in EXCLUDE_EXACT:
        return True
    return any(token in lower for token in EXCLUDE_SUBSTRINGS)


def build_all_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    rows: List[Dict] = []
    converted: Dict[str, pd.Series] = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = float(df[col].isna().mean()) if len(df) else np.nan
        non_null = int(df[col].notna().sum())
        used = False
        reason = ""
        if is_excluded_feature(col):
            reason = "excluded_future_or_result_or_id"
        else:
            s = pd.to_numeric(df[col], errors="coerce")
            if int(s.notna().sum()) == 0:
                reason = "all_null_or_non_numeric"
            else:
                used = True
                converted[col] = s.replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature_name": col,
                "dtype": dtype,
                "missing_rate_full": missing_rate,
                "non_null_count_full": non_null,
                "used": used,
                "drop_reason": reason if not used else "",
            }
        )
    features = [r["feature_name"] for r in rows if r["used"]]
    out = df.copy()
    for col, s in converted.items():
        out[col] = s
    manifest = pd.DataFrame(rows)
    return out.replace([np.inf, -np.inf], np.nan), features, manifest


def add_target(df: pd.DataFrame, entry_col: str, exit_col: str, cost_bps: float) -> pd.DataFrame:
    if entry_col not in df.columns:
        raise ValueError(f"missing entry price column: {entry_col}")
    if exit_col not in df.columns:
        raise ValueError(f"missing exit price column: {exit_col}")
    out = df.copy()
    entry = to_num(out[entry_col])
    exitp = to_num(out[exit_col])
    out["target_next_close_bps"] = 10000.0 * (exitp / entry.replace(0, np.nan) - 1.0) - float(cost_bps)
    return out


def date_windows(
    dates: Sequence[pd.Timestamp],
    train_window: int,
    valid_window: int,
    test_window: int,
    embargo: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    need = int(train_window + embargo + valid_window + embargo + test_window)
    windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    while start + need <= len(unique_dates):
        train = unique_dates[start : start + train_window]
        valid_start = start + train_window + embargo
        valid = unique_dates[valid_start : valid_start + valid_window]
        test_start = valid_start + valid_window + embargo
        test = unique_dates[test_start : test_start + test_window]
        windows.append((train, valid, test))
        start += int(test_window)
    return windows


def safe_corr(y_true: pd.Series, y_pred: pd.Series, method: str) -> float:
    yy = pd.to_numeric(y_true, errors="coerce")
    pp = pd.to_numeric(y_pred, errors="coerce")
    m = yy.notna() & pp.notna()
    if int(m.sum()) < 3 or yy[m].nunique() < 2 or pp[m].nunique() < 2:
        return np.nan
    return safe_float(yy[m].corr(pp[m], method=method))


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float], prefix: str = "") -> Dict[str, float | int]:
    yy = pd.Series(y_true, dtype="float64")
    pp = pd.Series(y_pred, dtype="float64")
    m = yy.notna() & pp.notna()
    yy = yy[m]
    pp = pp[m]
    if len(yy) == 0:
        names = ["rmse", "rmse_norm", "mae", "r2", "pearson", "spearman", "target_mean", "target_std", "pred_mean", "pred_std", "pred_std_ratio", "pred_min", "pred_max"]
        return {f"{prefix}{name}": np.nan for name in names} | {f"{prefix}n": 0}
    err = pp - yy
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    target_std = float(yy.std(ddof=0))
    pred_std = float(pp.std(ddof=0))
    sse = float(np.sum(np.square(err)))
    sst = float(np.sum(np.square(yy - yy.mean())))
    return {
        f"{prefix}n": int(len(yy)),
        f"{prefix}rmse": rmse,
        f"{prefix}rmse_norm": rmse / target_std if target_std > 1e-12 else np.nan,
        f"{prefix}mae": mae,
        f"{prefix}r2": 1.0 - sse / sst if sst > 1e-12 else np.nan,
        f"{prefix}pearson": safe_corr(yy, pp, "pearson"),
        f"{prefix}spearman": safe_corr(yy, pp, "spearman"),
        f"{prefix}target_mean": safe_float(yy.mean()),
        f"{prefix}target_std": target_std,
        f"{prefix}pred_mean": safe_float(pp.mean()),
        f"{prefix}pred_std": pred_std,
        f"{prefix}pred_std_ratio": pred_std / target_std if target_std > 1e-12 else np.nan,
        f"{prefix}pred_min": safe_float(pp.min()),
        f"{prefix}pred_max": safe_float(pp.max()),
    }


def bps_limit_diagnostics(y_true: Sequence[float], y_pred: Sequence[float], low: float = -2700, high: float = 3500) -> Dict[str, float]:
    yy = np.asarray(y_true, dtype=float)
    pp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(pp)
    if not mask.any():
        return {"pred_out_of_limit_rate": np.nan, "pred_clip_rmse": np.nan}
    pp = pp[mask]
    yy = yy[mask]
    clipped = np.clip(pp, low, high)
    return {
        "pred_out_of_limit_rate": float(((pp < low) | (pp > high)).mean()),
        "pred_clip_rmse": float(np.sqrt(np.mean(np.square(clipped - yy)))),
    }


def finite_rank_singulars(x: np.ndarray) -> Tuple[int, np.ndarray]:
    if x.size == 0:
        return 0, np.array([], dtype=float)
    try:
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    except Exception:
        return 0, np.array([], dtype=float)
    tol = np.finfo(float).eps * max(x.shape) * (float(s[0]) if len(s) else 0.0)
    return int((s > tol).sum()), s


def ridge_effective_df_from_singulars(s: np.ndarray, alpha: float) -> float:
    if len(s) == 0:
        return np.nan
    ss = np.square(s.astype(float))
    return float(np.sum(ss / (ss + float(alpha))))


def kernel_capacity(x: np.ndarray, alpha: float, gamma: float) -> Dict[str, float]:
    if rbf_kernel is None:
        raise ImportError(f"scikit-learn is required: {_SKLEARN_IMPORT_ERROR}")
    if x.size == 0:
        return {"kernel_effective_df": np.nan, "kernel_effective_df_over_n": np.nan, "kernel_condition_number": np.nan}
    k = rbf_kernel(x, x, gamma=float(gamma))
    vals = np.linalg.eigvalsh(k)
    vals = np.clip(vals, 0.0, None)
    eff = float(np.sum(vals / (vals + float(alpha))))
    positive = vals[vals > 1e-12]
    cond = float((positive.max() + alpha) / (positive.min() + alpha)) if len(positive) else np.nan
    return {
        "kernel_effective_df": eff,
        "kernel_effective_df_over_n": eff / x.shape[0] if x.shape[0] else np.nan,
        "kernel_condition_number": cond,
    }


def fit_preprocessor(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]):
    if SimpleImputer is None:
        raise ImportError(f"scikit-learn is required: {_SKLEARN_IMPORT_ERROR}")
    x_train_raw = train.loc[:, features].to_numpy(dtype=float, copy=True)
    x_valid_raw = valid.loc[:, features].to_numpy(dtype=float, copy=True)
    x_test_raw = test.loc[:, features].to_numpy(dtype=float, copy=True)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    x_train_imp = imputer.fit_transform(x_train_raw)
    x_valid_imp = imputer.transform(x_valid_raw)
    x_test_imp = imputer.transform(x_test_raw)
    x_train = scaler.fit_transform(x_train_imp)
    x_valid = scaler.transform(x_valid_imp)
    x_test = scaler.transform(x_test_imp)
    return x_train, x_valid, x_test, int(x_train_imp.shape[1])


def fit_pca(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, components: int):
    n_comp = int(min(components, x_train.shape[0], x_train.shape[1]))
    if n_comp < 1:
        return None, x_train[:, :0], x_valid[:, :0], x_test[:, :0]
    pca = PCA(n_components=n_comp, random_state=0)
    z_train = pca.fit_transform(x_train)
    return pca, z_train, pca.transform(x_valid), pca.transform(x_test)


def fit_ridge_predictions(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, alpha: float):
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if not math.isfinite(y_std) or y_std < 1e-12:
        y_std = 1.0
    y_scaled = (y_train - y_mean) / y_std
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(x_train, y_scaled)
    return (
        np.asarray(model.predict(x_train), dtype=float) * y_std + y_mean,
        np.asarray(model.predict(x_valid), dtype=float) * y_std + y_mean,
        np.asarray(model.predict(x_test), dtype=float) * y_std + y_mean,
    )


def fit_kernel_predictions(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, alpha: float, gamma: float):
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if not math.isfinite(y_std) or y_std < 1e-12:
        y_std = 1.0
    y_scaled = (y_train - y_mean) / y_std
    model = KernelRidge(alpha=float(alpha), kernel="rbf", gamma=float(gamma))
    model.fit(x_train, y_scaled)
    return (
        np.asarray(model.predict(x_train), dtype=float) * y_std + y_mean,
        np.asarray(model.predict(x_valid), dtype=float) * y_std + y_mean,
        np.asarray(model.predict(x_test), dtype=float) * y_std + y_mean,
    )


def base_metric_row(cfg: Config, window_id: str, train_window: int, valid_window: int, test_window: int) -> Dict:
    return {
        "window_id": window_id,
        "train_window": int(train_window),
        "valid_window": int(valid_window),
        "test_window": int(test_window),
        "model_type": cfg.model_type,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "n_components": cfg.n_components,
        "svd_components": cfg.svd_components,
        "degree": cfg.degree,
        "interaction_only": cfg.interaction_only,
    }


def baseline_metrics(y_train: np.ndarray, y_valid: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    train_mean = float(np.mean(y_train))
    zero_train = np.zeros_like(y_train, dtype=float)
    zero_valid = np.zeros_like(y_valid, dtype=float)
    zero_test = np.zeros_like(y_test, dtype=float)
    mean_train = np.full_like(y_train, train_mean, dtype=float)
    mean_valid = np.full_like(y_valid, train_mean, dtype=float)
    mean_test = np.full_like(y_test, train_mean, dtype=float)
    return {
        "zero_train_rmse": regression_metrics(y_train, zero_train)["rmse"],
        "zero_valid_rmse": regression_metrics(y_valid, zero_valid)["rmse"],
        "zero_test_rmse": regression_metrics(y_test, zero_test)["rmse"],
        "train_mean_train_rmse": regression_metrics(y_train, mean_train)["rmse"],
        "train_mean_valid_rmse": regression_metrics(y_valid, mean_valid)["rmse"],
        "train_mean_test_rmse": regression_metrics(y_test, mean_test)["rmse"],
    }


def risk_flags(row: Dict) -> Tuple[str, bool, bool]:
    capacity: List[str] = []
    if safe_float(row.get("effective_df_over_n")) > 0.8:
        capacity.append("high_effective_df")
    if safe_float(row.get("kernel_effective_df_over_n")) > 0.8:
        capacity.append("high_kernel_capacity")
    if safe_float(row.get("poly_features_over_n")) > 2.0:
        capacity.append("high_poly_capacity")
    overfit = False
    if math.isfinite(safe_float(row.get("train_rmse_norm"))) and math.isfinite(safe_float(row.get("valid_rmse_norm"))):
        overfit = safe_float(row.get("train_rmse_norm")) + 0.15 < safe_float(row.get("valid_rmse_norm"))
    collapse = math.isfinite(safe_float(row.get("valid_pred_std_ratio"))) and safe_float(row.get("valid_pred_std_ratio")) < 0.05
    return ";".join(capacity), bool(overfit), bool(collapse)


def add_metrics_to_row(row: Dict, y_train, pred_train, y_valid, pred_valid, y_test, pred_test, base: Dict) -> Dict:
    row.update(regression_metrics(y_train, pred_train, "train_"))
    row.update(regression_metrics(y_valid, pred_valid, "valid_"))
    row.update(regression_metrics(y_test, pred_test, "test_"))
    row.update(bps_limit_diagnostics(y_test, pred_test))
    row.update(
        {
            "zero_valid_rmse": base["zero_valid_rmse"],
            "train_mean_valid_rmse": base["train_mean_valid_rmse"],
            "zero_test_rmse": base["zero_test_rmse"],
            "train_mean_test_rmse": base["train_mean_test_rmse"],
            "valid_rmse_minus_zero": row["valid_rmse"] - base["zero_valid_rmse"],
            "valid_rmse_minus_train_mean": row["valid_rmse"] - base["train_mean_valid_rmse"],
            "test_rmse_minus_zero": row["test_rmse"] - base["zero_test_rmse"],
            "test_rmse_minus_train_mean": row["test_rmse"] - base["train_mean_test_rmse"],
            "train_valid_rmse_gap": row["valid_rmse_norm"] - row["train_rmse_norm"],
            "valid_test_rmse_gap": row["test_rmse_norm"] - row["valid_rmse_norm"],
        }
    )
    capacity, overfit, collapse = risk_flags(row)
    row["capacity_risk"] = capacity
    row["overfit_risk"] = overfit
    row["collapse_risk"] = collapse
    return row


def prediction_rows(frame: pd.DataFrame, y_pred: np.ndarray, split: str, window_id: str, stock_code: str, train_mean: float) -> List[Dict]:
    rows: List[Dict] = []
    for (_, r), pred in zip(frame.iterrows(), y_pred):
        rows.append(
            {
                "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                "stock_code": stock_code,
                "window_id": window_id,
                "split": split,
                "y_true_bps": safe_float(r["target_next_close_bps"]),
                "y_pred_bps": safe_float(pred),
                "zero_pred_bps": 0.0,
                "train_mean_pred_bps": train_mean,
            }
        )
    return rows


def run_config(
    cfg: Config,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    base_row: Dict,
    base: Dict,
    features_after_impute: int,
    max_poly_features: int,
    max_exact_kernel_train_rows: int,
    random_state: int,
    store_predictions: bool = False,
) -> Tuple[Dict, List[Dict]]:
    started = time.time()
    y_train = train["target_next_close_bps"].to_numpy(dtype=float)
    y_valid = valid["target_next_close_bps"].to_numpy(dtype=float)
    y_test = test["target_next_close_bps"].to_numpy(dtype=float)
    row = dict(base_row)
    row.update(
        {
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "n_test": int(len(test)),
            "n_features_after_impute": int(features_after_impute),
            "x_rank": np.nan,
            "x_rank_over_n": np.nan,
            "effective_df": np.nan,
            "effective_df_over_n": np.nan,
            "kernel_effective_df": np.nan,
            "kernel_effective_df_over_n": np.nan,
            "kernel_condition_number": np.nan,
            "poly_features_count": np.nan,
            "poly_features_over_n": np.nan,
            "components_over_n": np.nan,
            "n_components_over_n": np.nan,
            "dim_reduction_type": "",
            "pre_reduction_dim": np.nan,
            "post_reduction_dim": np.nan,
            "explained_variance_ratio": np.nan,
            "explained_variance_cum": np.nan,
            "skip_reason": "",
        }
    )
    preds: List[Dict] = []
    try:
        z_train, z_valid, z_test = x_train, x_valid, x_test
        if cfg.model_type in {"ridge_svd", "ridge_poly_svd", "ridge_svd_rbf_nystroem", "kernel_ridge_svd_rbf"}:
            pca, z_train, z_valid, z_test = fit_pca(x_train, x_valid, x_test, int(cfg.svd_components or 0))
            row.update(
                {
                    "dim_reduction_type": "pca",
                    "pre_reduction_dim": int(x_train.shape[1]),
                    "post_reduction_dim": int(z_train.shape[1]),
                    "explained_variance_ratio": safe_float(np.mean(pca.explained_variance_ratio_)) if pca is not None else np.nan,
                    "explained_variance_cum": safe_float(np.sum(pca.explained_variance_ratio_)) if pca is not None else np.nan,
                    "components_over_n": z_train.shape[1] / len(train) if len(train) else np.nan,
                }
            )

        if cfg.model_type == "ridge_poly_svd":
            poly = PolynomialFeatures(degree=int(cfg.degree or 2), interaction_only=bool(cfg.interaction_only), include_bias=False)
            poly.fit(z_train[: min(len(z_train), 3)])
            poly_count = int(poly.n_output_features_)
            row["poly_features_count"] = poly_count
            row["poly_features_over_n"] = poly_count / len(train) if len(train) else np.nan
            if poly_count > int(max_poly_features):
                row["skip_reason"] = "poly_feature_explosion"
                row["fit_seconds"] = time.time() - started
                return row, []
            z_train = poly.fit_transform(z_train)
            z_valid = poly.transform(z_valid)
            z_test = poly.transform(z_test)

        if cfg.model_type in {"ridge_rbf_nystroem", "ridge_svd_rbf_nystroem"}:
            n_comp = int(min(int(cfg.n_components or 0), len(train)))
            if n_comp < 1:
                row["skip_reason"] = "invalid_n_components"
                row["fit_seconds"] = time.time() - started
                return row, []
            mapper = Nystroem(kernel="rbf", gamma=float(cfg.gamma), n_components=n_comp, random_state=int(random_state))
            z_train = mapper.fit_transform(z_train)
            z_valid = mapper.transform(z_valid)
            z_test = mapper.transform(z_test)
            row["n_components"] = n_comp
            row["n_components_over_n"] = n_comp / len(train) if len(train) else np.nan

        if cfg.model_type in {"kernel_ridge_rbf", "kernel_ridge_svd_rbf"}:
            if len(train) > int(max_exact_kernel_train_rows):
                row["skip_reason"] = "exact_kernel_train_rows_exceeded"
                row["fit_seconds"] = time.time() - started
                return row, []
            row.update(kernel_capacity(z_train, float(cfg.alpha), float(cfg.gamma)))
            fit_start = time.time()
            pred_train, pred_valid, pred_test = fit_kernel_predictions(z_train, z_valid, z_test, y_train, float(cfg.alpha), float(cfg.gamma))
        else:
            rank, s = finite_rank_singulars(z_train)
            eff = ridge_effective_df_from_singulars(s, float(cfg.alpha))
            row.update(
                {
                    "x_rank": rank,
                    "x_rank_over_n": rank / len(train) if len(train) else np.nan,
                    "effective_df": eff,
                    "effective_df_over_n": eff / len(train) if len(train) else np.nan,
                }
            )
            fit_start = time.time()
            pred_train, pred_valid, pred_test = fit_ridge_predictions(z_train, z_valid, z_test, y_train, float(cfg.alpha))
        row["n_features_after_transform"] = int(z_train.shape[1])
        row["fit_seconds"] = time.time() - fit_start
        predict_start = time.time()
        row = add_metrics_to_row(row, y_train, pred_train, y_valid, pred_valid, y_test, pred_test, base)
        row["predict_seconds"] = time.time() - predict_start
        if store_predictions:
            train_mean = float(np.mean(y_train))
            preds.extend(prediction_rows(train, pred_train, "train", str(row["window_id"]), str(row["stock_code"]), train_mean))
            preds.extend(prediction_rows(valid, pred_valid, "valid", str(row["window_id"]), str(row["stock_code"]), train_mean))
            preds.extend(prediction_rows(test, pred_test, "test", str(row["window_id"]), str(row["stock_code"]), train_mean))
    except Exception as exc:
        row["skip_reason"] = f"error:{type(exc).__name__}:{str(exc)[:120]}"
        row["fit_seconds"] = time.time() - started
    return row, preds


def make_configs(args, feature_dim: int) -> List[Config]:
    linear_alphas = parse_number_grid(args.alpha_grid)
    kernel_alphas = parse_number_grid(args.kernel_alpha_grid or args.alpha_grid)
    n_components_grid = parse_int_grid(args.n_components_grid)
    svd_grid = parse_int_grid(args.svd_components_grid)
    poly_svd_grid = [x for x in svd_grid if x in {16, 32, 64}] or svd_grid[:3]
    pre_kernel_grid = [x for x in svd_grid if x in {32, 64, 128, 256}] or svd_grid
    out: List[Config] = []
    models = {m.strip() for m in str(args.model_types).split(",") if m.strip()}
    if "all" in models:
        models = set(MODEL_COMPLEXITY)
    if "ridge_linear" in models:
        out.extend(Config("ridge_linear", alpha=a) for a in linear_alphas)
    if "ridge_svd" in models:
        for k in svd_grid:
            out.extend(Config("ridge_svd", alpha=a, svd_components=k) for a in linear_alphas)
    if "ridge_poly_svd" in models:
        for k in poly_svd_grid:
            for interaction_only in [True, False]:
                out.extend(Config("ridge_poly_svd", alpha=a, svd_components=k, degree=int(args.poly_degree), interaction_only=interaction_only) for a in kernel_alphas)
    if "ridge_rbf_nystroem" in models:
        for n in n_components_grid:
            for g in gamma_grid(feature_dim, exact=False):
                out.extend(Config("ridge_rbf_nystroem", alpha=a, gamma=g, n_components=n) for a in kernel_alphas)
    if "ridge_svd_rbf_nystroem" in models:
        for k in pre_kernel_grid:
            for n in [x for x in n_components_grid if x <= 512]:
                for g in gamma_grid(k, exact=False):
                    out.extend(Config("ridge_svd_rbf_nystroem", alpha=a, gamma=g, n_components=n, svd_components=k) for a in kernel_alphas)
    if "kernel_ridge_rbf" in models:
        for g in gamma_grid(feature_dim, exact=True):
            out.extend(Config("kernel_ridge_rbf", alpha=a, gamma=g) for a in kernel_alphas)
    if "kernel_ridge_svd_rbf" in models:
        for k in pre_kernel_grid:
            for g in gamma_grid(k, exact=True):
                out.extend(Config("kernel_ridge_svd_rbf", alpha=a, gamma=g, svd_components=k) for a in kernel_alphas)
    return out


def run_symbol(path: Path, args, out_dir: Path) -> Tuple[List[Dict], Dict]:
    raw = pd.read_csv(path)
    raw = parse_date_col(raw)
    stock_code = str(args.stock_code or infer_stock_code(path, raw))[:6]
    df = add_target(raw, args.entry_price_col, args.exit_price_col, float(args.cost_bps))
    df, features, manifest = build_all_features(df)
    df = df.dropna(subset=["date", "target_next_close_bps"]).sort_values("date").reset_index(drop=True)
    manifest.insert(0, "stock_code", stock_code)
    manifest.insert(1, "sample_path", str(path))
    manifest.to_csv(out_dir / f"feature_manifest_{stock_code}.csv", index=False, encoding="utf-8-sig")
    if not features or len(df) < int(args.min_rows):
        return [], {"stock_code": stock_code, "sample_path": str(path), "status": "skipped", "reason": "insufficient_rows_or_features"}

    all_rows: List[Dict] = []
    print(f"[INFO] symbol={stock_code} rows={len(df)} all_features={len(features)} sample={path}", flush=True)
    for train_window in parse_int_grid(args.train_window):
        windows = date_windows(df["date"], train_window, int(args.valid_window), int(args.test_window), int(args.embargo))
        print(f"[INFO] symbol={stock_code} train_window={train_window} windows={len(windows)}", flush=True)
        for window_idx, (train_dates, valid_dates, test_dates) in enumerate(windows, start=1):
            train = df[df["date"].isin(train_dates)].copy()
            valid = df[df["date"].isin(valid_dates)].copy()
            test = df[df["date"].isin(test_dates)].copy()
            train = train.dropna(subset=["target_next_close_bps"])
            valid = valid.dropna(subset=["target_next_close_bps"])
            test = test.dropna(subset=["target_next_close_bps"])
            if len(train) < int(args.min_train_rows) or len(valid) == 0 or len(test) == 0:
                continue
            train_features = []
            for col in features:
                s = pd.to_numeric(train[col], errors="coerce")
                if s.notna().sum() == 0:
                    continue
                if s.nunique(dropna=True) <= 1:
                    continue
                train_features.append(col)
            if not train_features:
                continue
            x_train, x_valid, x_test, features_after_impute = fit_preprocessor(train, valid, test, train_features)
            configs = make_configs(args, features_after_impute)
            y_train = train["target_next_close_bps"].to_numpy(dtype=float)
            y_valid = valid["target_next_close_bps"].to_numpy(dtype=float)
            y_test = test["target_next_close_bps"].to_numpy(dtype=float)
            base = baseline_metrics(y_train, y_valid, y_test)
            window_id = f"{stock_code}_tw{train_window}_w{window_idx:03d}"
            base_common = {
                "stock_code": stock_code,
                "sample_path": str(path),
                "n_features_raw": int(len(train_features)),
                "zero_train_rmse": base["zero_train_rmse"],
                "train_mean_train_rmse": base["train_mean_train_rmse"],
            }
            print(f"[RUN] symbol={stock_code} window={window_id} configs={len(configs)} features={len(train_features)}", flush=True)
            def evaluate_one(cfg: Config) -> Dict:
                row = base_metric_row(cfg, window_id, train_window, int(args.valid_window), int(args.test_window))
                row.update(base_common)
                metric_row, _ = run_config(
                    cfg,
                    x_train,
                    x_valid,
                    x_test,
                    train,
                    valid,
                    test,
                    row,
                    base,
                    features_after_impute,
                    int(args.max_poly_features),
                    int(args.max_exact_kernel_train_rows),
                    int(args.random_state),
                    False,
                )
                return metric_row

            n_jobs = int(args.n_jobs)
            if n_jobs == 1 or len(configs) <= 1:
                metric_rows = [evaluate_one(cfg) for cfg in configs]
            else:
                if Parallel is None:
                    raise ImportError(f"joblib is required for --n-jobs > 1: {_JOBLIB_IMPORT_ERROR}")
                metric_rows = Parallel(n_jobs=min(n_jobs, len(configs)), prefer="threads")(
                    delayed(evaluate_one)(cfg) for cfg in configs
                )
            all_rows.extend(metric_rows)
            pd.DataFrame(all_rows).to_csv(out_dir / "ridge_grid_window_metrics.csv", index=False, encoding="utf-8-sig")
    return all_rows, {"stock_code": stock_code, "sample_path": str(path), "status": "ok", "rows": len(df), "features": len(features)}


def num_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def summarize(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()
    ok = window_df[window_df["skip_reason"].fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["stock_code", "model_type", "alpha", "gamma", "n_components", "svd_components", "degree", "interaction_only"]
    rows: List[Dict] = []
    for keys, g in ok.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n_windows"] = int(len(g))
        for src, funcs in {
            "valid_rmse_norm": ["median", "mean"],
            "valid_rmse": ["median"],
            "valid_spearman": ["median", "mean"],
            "valid_pred_std_ratio": ["median"],
            "test_rmse_norm": ["median", "mean"],
            "test_spearman": ["median", "mean"],
            "effective_df_over_n": ["median"],
            "kernel_effective_df_over_n": ["median"],
            "train_valid_rmse_gap": ["median"],
            "valid_test_rmse_gap": ["median"],
            "zero_valid_rmse": ["median"],
            "train_mean_valid_rmse": ["median"],
        }.items():
            s = num_series(g, src)
            for fn in funcs:
                row[f"{fn}_{src}"] = safe_float(getattr(s, fn)())
        row["passes_valid_gate"] = bool(
            safe_float(row.get("median_valid_rmse_norm")) < 1.0
            and safe_float(row.get("median_valid_rmse")) < safe_float(row.get("median_zero_valid_rmse"))
            and safe_float(row.get("median_valid_rmse")) < safe_float(row.get("median_train_mean_valid_rmse"))
            and safe_float(row.get("mean_valid_spearman")) > 0
            and safe_float(row.get("median_valid_pred_std_ratio")) >= 0.05
        )
        row["capacity_risk_rate"] = float((g["capacity_risk"].fillna("") != "").mean())
        row["overfit_risk_rate"] = float(pd.Series(g["overfit_risk"]).astype(bool).mean())
        row["collapse_risk_rate"] = float(pd.Series(g["collapse_risk"]).astype(bool).mean())
        row["model_complexity"] = MODEL_COMPLEXITY.get(str(row["model_type"]), 999)
        row["params"] = Config(
            str(row["model_type"]),
            float(row["alpha"]),
            None if pd.isna(row["gamma"]) else float(row["gamma"]),
            None if pd.isna(row["n_components"]) else int(row["n_components"]),
            None if pd.isna(row["svd_components"]) else int(row["svd_components"]),
            None if pd.isna(row["degree"]) else int(row["degree"]),
            None if pd.isna(row["interaction_only"]) else bool(row["interaction_only"]),
        ).params_json()
        rows.append(row)
    return pd.DataFrame(rows)


def selection_by_valid(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    sort_cols = [
        "stock_code",
        "passes_valid_gate",
        "median_valid_rmse_norm",
        "median_valid_spearman",
        "median_train_valid_rmse_gap",
        "median_effective_df_over_n",
        "model_complexity",
    ]
    ascending = [True, False, True, False, True, True, True]
    ranked = summary.sort_values(sort_cols, ascending=ascending, na_position="last").copy()
    ranked["rank_by_valid"] = ranked.groupby("stock_code").cumcount() + 1
    ranked["risk_flags"] = ranked.apply(
        lambda r: ";".join(
            x
            for x in [
                "capacity" if safe_float(r.get("capacity_risk_rate")) > 0 else "",
                "overfit" if safe_float(r.get("overfit_risk_rate")) > 0 else "",
                "collapse" if safe_float(r.get("collapse_risk_rate")) > 0 else "",
            ]
            if x
        ),
        axis=1,
    )
    cols = [
        "rank_by_valid",
        "stock_code",
        "model_type",
        "params",
        "median_valid_rmse_norm",
        "median_valid_spearman",
        "median_test_rmse_norm",
        "median_test_spearman",
        "median_effective_df_over_n",
        "median_kernel_effective_df_over_n",
        "risk_flags",
        "passes_valid_gate",
    ]
    return ranked[[c for c in cols if c in ranked.columns]]


def best_configs_by_stock(selection: pd.DataFrame) -> Dict[str, List[Config]]:
    out: Dict[str, List[Config]] = {}
    if selection.empty:
        return out
    best = selection[selection["rank_by_valid"] == 1].copy()
    for r in best.itertuples():
        params = json.loads(r.params)
        cfg = Config(
            str(r.model_type),
            float(params["alpha"]),
            params.get("gamma"),
            params.get("n_components"),
            params.get("svd_components"),
            params.get("degree"),
            params.get("interaction_only"),
        )
        out.setdefault(str(r.stock_code)[:6], []).append(cfg)
    return out


def generate_best_predictions(paths: Sequence[Path], args, selection: pd.DataFrame) -> pd.DataFrame:
    configs_by_stock = best_configs_by_stock(selection)
    if not configs_by_stock:
        return pd.DataFrame(columns=["date", "stock_code", "window_id", "split", "y_true_bps", "y_pred_bps", "zero_pred_bps", "train_mean_pred_bps"])
    pred_rows: List[Dict] = []
    for path in paths:
        raw = parse_date_col(pd.read_csv(path))
        stock_code = str(args.stock_code or infer_stock_code(path, raw))[:6]
        wanted = configs_by_stock.get(stock_code)
        if not wanted:
            continue
        df = add_target(raw, args.entry_price_col, args.exit_price_col, float(args.cost_bps))
        df, features, _ = build_all_features(df)
        df = df.dropna(subset=["date", "target_next_close_bps"]).sort_values("date").reset_index(drop=True)
        for train_window in parse_int_grid(args.train_window):
            windows = date_windows(df["date"], train_window, int(args.valid_window), int(args.test_window), int(args.embargo))
            for window_idx, (train_dates, valid_dates, test_dates) in enumerate(windows, start=1):
                train = df[df["date"].isin(train_dates)].copy().dropna(subset=["target_next_close_bps"])
                valid = df[df["date"].isin(valid_dates)].copy().dropna(subset=["target_next_close_bps"])
                test = df[df["date"].isin(test_dates)].copy().dropna(subset=["target_next_close_bps"])
                if len(train) < int(args.min_train_rows) or len(valid) == 0 or len(test) == 0:
                    continue
                train_features = []
                for col in features:
                    s = pd.to_numeric(train[col], errors="coerce")
                    if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
                        continue
                    train_features.append(col)
                if not train_features:
                    continue
                x_train, x_valid, x_test, features_after_impute = fit_preprocessor(train, valid, test, train_features)
                y_train = train["target_next_close_bps"].to_numpy(dtype=float)
                y_valid = valid["target_next_close_bps"].to_numpy(dtype=float)
                y_test = test["target_next_close_bps"].to_numpy(dtype=float)
                base = baseline_metrics(y_train, y_valid, y_test)
                window_id = f"{stock_code}_tw{train_window}_w{window_idx:03d}"
                base_common = {
                    "stock_code": stock_code,
                    "sample_path": str(path),
                    "n_features_raw": int(len(train_features)),
                    "zero_train_rmse": base["zero_train_rmse"],
                    "train_mean_train_rmse": base["train_mean_train_rmse"],
                }
                for cfg in wanted:
                    row = base_metric_row(cfg, window_id, train_window, int(args.valid_window), int(args.test_window))
                    row.update(base_common)
                    _, preds = run_config(
                        cfg,
                        x_train,
                        x_valid,
                        x_test,
                        train,
                        valid,
                        test,
                        row,
                        base,
                        features_after_impute,
                        int(args.max_poly_features),
                        int(args.max_exact_kernel_train_rows),
                        int(args.random_state),
                        True,
                    )
                    pred_rows.extend(preds)
    return pd.DataFrame(pred_rows)


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Ridge-only single-symbol asof1455 regression search")
    p.add_argument("--samples", nargs="*", default=[])
    p.add_argument("--sample-glob", action="append", default=[])
    p.add_argument("--stock-code", default="")
    p.add_argument("--symbols", default="")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--entry-price-col", default="close_asof1455")
    p.add_argument("--exit-price-col", default="next_day_close")
    p.add_argument("--cost-bps", type=float, default=1.7)
    p.add_argument("--train-window", default="756")
    p.add_argument("--valid-window", type=int, default=63)
    p.add_argument("--test-window", type=int, default=21)
    p.add_argument("--embargo", type=int, default=1)
    p.add_argument("--alpha-grid", default=DEFAULT_ALPHA_GRID)
    p.add_argument("--kernel-alpha-grid", default=DEFAULT_KERNEL_ALPHA_GRID)
    p.add_argument("--gamma-grid-mode", default="scaled_by_d", choices=["scaled_by_d"])
    p.add_argument("--n-components-grid", default=DEFAULT_N_COMPONENTS_GRID)
    p.add_argument("--svd-components-grid", default=DEFAULT_SVD_COMPONENTS_GRID)
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--max-poly-features", type=int, default=20000)
    p.add_argument("--max-exact-kernel-train-rows", type=int, default=1200)
    p.add_argument("--model-types", default="all")
    p.add_argument("--min-rows", type=int, default=300)
    p.add_argument("--min-train-rows", type=int, default=120)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(f"scikit-learn is required: {_SKLEARN_IMPORT_ERROR}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_globs = args.sample_glob or DEFAULT_SAMPLE_GLOBS
    symbols = [x.strip() for x in str(args.symbols or args.stock_code).split(",") if x.strip()]
    candidate_paths = [
        p
        for p in expand_samples(args.samples, sample_globs)
        if sample_has_required_columns(p, str(args.entry_price_col), str(args.exit_price_col))
    ]
    paths = choose_best_sample_per_symbol(candidate_paths, symbols or None, int(args.max_symbols))
    if not paths:
        raise SystemExit("no sample files matched")
    print(f"[INFO] samples={len(paths)} out_dir={out_dir} n_jobs={int(args.n_jobs)}", flush=True)
    all_rows: List[Dict] = []
    statuses: List[Dict] = []
    for path in paths:
        rows, status = run_symbol(path, args, out_dir)
        all_rows.extend(rows)
        statuses.append(status)
        pd.DataFrame(all_rows).to_csv(out_dir / "ridge_grid_window_metrics.csv", index=False, encoding="utf-8-sig")
    manifest_parts = [pd.read_csv(p) for p in sorted(out_dir.glob("feature_manifest_*.csv"))]
    if manifest_parts:
        pd.concat(manifest_parts, ignore_index=True).to_csv(out_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig")
    window_df = pd.DataFrame(all_rows)
    window_df.to_csv(out_dir / "ridge_grid_window_metrics.csv", index=False, encoding="utf-8-sig")
    summary = summarize(window_df)
    summary.to_csv(out_dir / "ridge_grid_summary.csv", index=False, encoding="utf-8-sig")
    selection = selection_by_valid(summary)
    selection.to_csv(out_dir / "ridge_model_selection_by_valid.csv", index=False, encoding="utf-8-sig")
    best_preds = generate_best_predictions(paths, args, selection)
    best_preds.to_csv(out_dir / "ridge_best_config_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(statuses).to_csv(out_dir / "ridge_run_status.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] rows={len(window_df)} summary_rows={len(summary)} selection_rows={len(selection)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
