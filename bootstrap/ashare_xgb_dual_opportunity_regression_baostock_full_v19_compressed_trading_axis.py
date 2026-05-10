#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaoStock 5分钟数据 + 双机会 XGBoost 回归模型（v13 plot checked）

目标不再是“当前到收盘终点收益”，而是两条路径机会。
训练标签按“当前 bar 收盘产生信号、下一根 bar 开盘成交”的假设，
用下一根 bar 的 open 作为可执行基准价 P_exec：
  sell_opportunity = (P_exec - future_min_price) / P_exec - round_trip_cost
  buy_opportunity  = (future_max_price - P_exec) / P_exec - round_trip_cost

数据链路：
  本脚本不直接访问 AKShare；训练数据通过 helper_py 指定的 BaoStock helper 获取：
  ashare_fetch_and_train_xgb_sell_signal_baostock_state_cache_helper_fix2.py

输出：
  - signal_samples.csv
  - valid_predictions.csv / test_predictions.csv / all_signal_scores.csv
  - artifacts/xgb_sell_opportunity_model.json
  - artifacts/xgb_buy_opportunity_model.json
  - artifacts/feature_schema.json
  - artifacts/model_artifacts.json
  - plots/*.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False

try:
    from xgboost import XGBRegressor
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "未安装 xgboost。请先执行: pip install -U xgboost\n"
        f"原始错误: {type(e).__name__}: {e}"
    )

EPS = 1e-12
RANDOM_STATE = 42
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class XGBRegModelConfig:
    train_ratio: float = 0.60
    valid_ratio: float = 0.20
    test_ratio: float = 0.20
    n_splits: int = 5
    xgb_n_jobs: int = 4

    min_total_samples: int = 300
    min_train_samples: int = 120
    min_valid_samples: int = 50
    min_test_samples: int = 50
    min_cv_train_samples: int = 120
    min_cv_val_samples: int = 40
    max_cv_param_combos: int = 24

    max_depth_grid: Tuple[int, ...] = (2, 3, 4)
    learning_rate_grid: Tuple[float, ...] = (0.03, 0.05)
    n_estimators_grid: Tuple[int, ...] = (200, 500)
    min_child_weight_grid: Tuple[int, ...] = (5, 20)
    subsample_grid: Tuple[float, ...] = (0.8,)
    colsample_bytree_grid: Tuple[float, ...] = (0.8,)
    reg_lambda_grid: Tuple[float, ...] = (1.0, 5.0, 10.0)


@dataclass
class TargetConfig:
    # opportunity: 从当前 bar 后一根到当日最后一根 bar 的路径高低点机会
    # fixed_horizon_opportunity: 从当前 bar 后一根到 i+horizon_bars 的路径高低点机会
    target_mode: str = "opportunity"
    horizon_bars: int = 12
    price_field: str = "bar_vwap"  # bar_vwap | close | session_vwap | vwap(alias=bar_vwap)
    future_extreme_mode: str = "high_low"  # high_low | price_field
    min_future_bars: int = 2
    min_complete_day_bars: int = 40
    round_trip_cost: Optional[float] = None


@dataclass
class SignalRuleConfig:
    q_edges: Tuple[float, float, float, float] = (0.2, 0.4, 0.6, 0.8)
    min_action_edge: float = 0.0010
    min_action_opportunity: float = 0.0015


def import_module_from_path(path: str | Path, module_name: str):
    path = str(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(np.asarray(y_true, dtype=float))
    s2 = pd.Series(np.asarray(y_pred, dtype=float))
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return float(s1.corr(s2, method="spearman"))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(np.asarray(y_true, dtype=float))
    s2 = pd.Series(np.asarray(y_pred, dtype=float))
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return float(s1.corr(s2, method="pearson"))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def quantile_spread(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.2) -> Dict[str, float]:
    df = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 20:
        return {"top_mean": np.nan, "bottom_mean": np.nan, "spread": np.nan, "q": q}
    n = max(1, int(np.floor(len(df) * q)))
    top = df.nlargest(n, "pred")
    bot = df.nsmallest(n, "pred")
    return {
        "top_mean": float(top["y"].mean()),
        "bottom_mean": float(bot["y"].mean()),
        "spread": float(top["y"].mean() - bot["y"].mean()),
        "q": q,
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": (lambda yt, yp: float(np.mean(np.abs(yt - yp))) if len(yt) else np.nan)(
            np.asarray(y_true, dtype=float)[np.isfinite(np.asarray(y_true, dtype=float)) & np.isfinite(np.asarray(y_pred, dtype=float))],
            np.asarray(y_pred, dtype=float)[np.isfinite(np.asarray(y_true, dtype=float)) & np.isfinite(np.asarray(y_pred, dtype=float))],
        ),
        "rank_ic": safe_spearman(y_true, y_pred),
        "pearson_ic": safe_pearson(y_true, y_pred),
        "target_mean": float(np.nanmean(y_true)),
        "pred_mean": float(np.nanmean(y_pred)),
        "target_std": float(np.nanstd(y_true)),
        "pred_std": float(np.nanstd(y_pred)),
        "top_bottom_spread_q20": quantile_spread(y_true, y_pred, q=0.2),
    }


def calc_round_trip_cost(args: argparse.Namespace, cfg, target_cfg: TargetConfig) -> float:
    if target_cfg.round_trip_cost is not None and np.isfinite(float(target_cfg.round_trip_cost)):
        return float(target_cfg.round_trip_cost)
    buy_rate = float(getattr(args, "cost_buy_rate", getattr(cfg, "cost_buy_rate", 0.0)) or 0.0)
    sell_rate = float(getattr(args, "cost_sell_rate", getattr(cfg, "cost_sell_rate", 0.0)) or 0.0)
    slippage_bps = float(getattr(args, "slippage_bps", getattr(cfg, "slippage_bps", 0.0)) or 0.0)
    return float(buy_rate + sell_rate + 2.0 * slippage_bps / 10000.0)


def _finite_positive(x) -> bool:
    try:
        v = float(x)
        return np.isfinite(v) and v > 0
    except Exception:
        return False


def add_intraday_price_fields(intraday: pd.DataFrame) -> pd.DataFrame:
    out = intraday.copy()
    if out.empty:
        return out
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    else:
        out["date"] = out["datetime"].dt.normalize()

    # session_vwap：优先使用已有 vwap；否则用 cum_pv/cum_volume；再否则退回 close。
    if "vwap" in out.columns:
        out["session_vwap"] = pd.to_numeric(out["vwap"], errors="coerce")
    else:
        out["session_vwap"] = np.nan

    if "cum_pv" in out.columns and "cum_volume" in out.columns:
        out["cum_pv"] = pd.to_numeric(out["cum_pv"], errors="coerce")
        out["cum_volume"] = pd.to_numeric(out["cum_volume"], errors="coerce")
        parts = []
        for _, g in out.groupby("date", sort=True):
            g = g.sort_values("datetime").copy()
            g["bar_pv"] = g["cum_pv"].diff()
            g["bar_volume"] = g["cum_volume"].diff()
            if len(g):
                g.loc[g.index[0], "bar_pv"] = g.iloc[0]["cum_pv"]
                g.loc[g.index[0], "bar_volume"] = g.iloc[0]["cum_volume"]
            g["bar_vwap"] = g["bar_pv"] / g["bar_volume"].replace(0, np.nan)
            parts.append(g)
        out = pd.concat(parts, axis=0).sort_values("datetime").reset_index(drop=True)
    elif "amount" in out.columns and "volume" in out.columns:
        # BaoStock 5m amount/volume 可用于近似 bar_vwap；若口径异常，会在下面回退 close。
        out["bar_pv"] = pd.to_numeric(out["amount"], errors="coerce")
        out["bar_volume"] = pd.to_numeric(out["volume"], errors="coerce")
        out["bar_vwap"] = out["bar_pv"] / out["bar_volume"].replace(0, np.nan)
    else:
        out["bar_pv"] = np.nan
        out["bar_volume"] = np.nan
        out["bar_vwap"] = np.nan

    close_num = pd.to_numeric(out.get("close", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["bar_vwap"] = pd.to_numeric(out.get("bar_vwap", np.nan), errors="coerce")
    out["session_vwap"] = pd.to_numeric(out.get("session_vwap", np.nan), errors="coerce")
    out["bar_vwap"] = out["bar_vwap"].where(np.isfinite(out["bar_vwap"]) & (out["bar_vwap"] > 0), close_num)
    out["session_vwap"] = out["session_vwap"].where(np.isfinite(out["session_vwap"]) & (out["session_vwap"] > 0), close_num)
    return out


def get_ref_price(bar: pd.Series, price_field: str) -> float:
    field = str(price_field or "bar_vwap").lower()
    candidates = []
    if field in {"bar_vwap", "vwap"}:
        candidates = ["bar_vwap", "close", "session_vwap", "vwap"]
    elif field == "session_vwap":
        candidates = ["session_vwap", "vwap", "close", "bar_vwap"]
    elif field == "close":
        candidates = ["close", "bar_vwap", "session_vwap", "vwap"]
    else:
        candidates = [field, "bar_vwap", "close", "session_vwap", "vwap"]
    for c in candidates:
        if c in bar.index and _finite_positive(bar[c]):
            return float(bar[c])
    return np.nan


def future_extreme_prices(future_bars: pd.DataFrame, target_cfg: TargetConfig) -> Tuple[float, float]:
    mode = str(target_cfg.future_extreme_mode or "high_low").lower()
    if future_bars.empty:
        return np.nan, np.nan
    if mode == "high_low" and "low" in future_bars.columns and "high" in future_bars.columns:
        lows = pd.to_numeric(future_bars["low"], errors="coerce")
        highs = pd.to_numeric(future_bars["high"], errors="coerce")
        min_px = float(lows.replace([np.inf, -np.inf], np.nan).min())
        max_px = float(highs.replace([np.inf, -np.inf], np.nan).max())
        if _finite_positive(min_px) and _finite_positive(max_px):
            return min_px, max_px
    # 退回到 price_field 序列。
    pxs = []
    for _, b in future_bars.iterrows():
        px = get_ref_price(b, target_cfg.price_field)
        if _finite_positive(px):
            pxs.append(float(px))
    if not pxs:
        return np.nan, np.nan
    return float(np.min(pxs)), float(np.max(pxs))


def in_market_hours(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return ((pd.Timestamp("09:30").time() <= t <= pd.Timestamp("11:30").time()) or
            (pd.Timestamp("13:00").time() <= t <= pd.Timestamp("15:00").time()))


def _clock_slot_id_from_minute(minute_of_day: pd.Series) -> pd.Series:
    """
    BaoStock 5分钟 bar 的时间戳通常是 bar 结束时刻：
    09:35, 09:40, ..., 11:30, 13:05, ..., 15:00，共 48 根。
    若遇到 09:30 / 13:00 这类起点时间戳，也尽量映射到对应首根 bar。
    """
    m = pd.to_numeric(minute_of_day, errors="coerce").to_numpy(dtype=float)
    slot = np.full(len(m), np.nan)

    # 上午：09:35-11:30 -> 0..23；若时间戳是 09:30，也归为 0。
    morning_first_end = 9 * 60 + 35
    morning_last_end = 11 * 60 + 30
    afternoon_first_end = 13 * 60 + 5
    afternoon_last_end = 15 * 60

    mask_m = (m >= 9 * 60 + 30) & (m <= morning_last_end)
    slot[mask_m] = np.floor((np.maximum(m[mask_m], morning_first_end) - morning_first_end) / 5.0)

    # 下午：13:05-15:00 -> 24..47；若时间戳是 13:00，也归为 24。
    mask_a = (m >= 13 * 60) & (m <= afternoon_last_end)
    slot[mask_a] = 24.0 + np.floor((np.maximum(m[mask_a], afternoon_first_end) - afternoon_first_end) / 5.0)

    slot = np.clip(slot, 0, 47)
    return pd.Series(slot)


def _select_day_slot_ids(day_bars: pd.DataFrame, expected_bars_per_day: int = 48) -> Tuple[pd.Series, str]:
    """
    返回 0-based 日内 slot。

    优先使用时钟映射，因为 BaoStock 5分钟通常是 bar 结束时刻；
    但有些数据源/缓存可能使用 bar 起点时间戳（09:30...14:55），
    这时最后一根会被时钟映射成 46 而不是 47，导致完整交易日被误判为不完整。
    因此若存在 bar_no 且 max(bar_no) >= expected_bars_per_day，说明一整天 48 根已经齐全，
    此时用 bar_no-1 作为 slot，更稳。
    """
    if day_bars.empty:
        return pd.Series(dtype=float), "empty"

    dt = pd.to_datetime(day_bars["datetime"], errors="coerce")
    minutes = dt.dt.hour * 60 + dt.dt.minute
    clock_slots = _clock_slot_id_from_minute(minutes).reset_index(drop=True)

    if "bar_no" in day_bars.columns:
        bar_no = pd.to_numeric(day_bars["bar_no"], errors="coerce").reset_index(drop=True)
        if bar_no.notna().any():
            max_bar_no = float(bar_no.max())
            max_clock_slot = float(pd.to_numeric(clock_slots, errors="coerce").max()) if clock_slots.notna().any() else np.nan
            min_minute = float(pd.to_numeric(minutes, errors="coerce").min()) if pd.to_numeric(minutes, errors="coerce").notna().any() else np.nan
            # 若第一根是 09:30，通常说明 timestamp 是 bar 起点口径；实时半日数据也应直接用 bar_no-1。
            looks_like_start_timestamp = np.isfinite(min_minute) and min_minute <= 9 * 60 + 30
            # 完整日但时钟未映射到 47，也通常意味着 timestamp 是 bar 起点口径。
            full_day_but_clock_not_close = np.isfinite(max_bar_no) and max_bar_no >= expected_bars_per_day and (not np.isfinite(max_clock_slot) or max_clock_slot < expected_bars_per_day - 1)
            if looks_like_start_timestamp or full_day_but_clock_not_close:
                return (bar_no - 1).clip(lower=0, upper=expected_bars_per_day - 1), "bar_no"

    return clock_slots, "clock"


def add_time_features(df: pd.DataFrame, time_col: str = "signal_time") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    out["signal_date"] = out[time_col].dt.date.astype(str)
    out["hour"] = out[time_col].dt.hour
    out["minute"] = out[time_col].dt.minute
    out["weekday"] = out[time_col].dt.weekday
    out["month"] = out[time_col].dt.month
    minute_of_day = out["hour"] * 60 + out["minute"]
    out["minute_of_day"] = minute_of_day.astype(int)

    # 优先使用构造样本时的日内顺序，避免 BaoStock bar 结束时间戳导致 slot off-by-one。
    if "sample_bar_index_in_day" in out.columns:
        slot = pd.to_numeric(out["sample_bar_index_in_day"], errors="coerce")
    else:
        slot = _clock_slot_id_from_minute(minute_of_day)
    out["slot_id"] = slot.fillna(-1).astype(int)

    n_slots = 48.0
    slot_clip = out["slot_id"].clip(lower=0, upper=47)
    out["slot_sin"] = np.sin(2.0 * np.pi * slot_clip / n_slots)
    out["slot_cos"] = np.cos(2.0 * np.pi * slot_clip / n_slots)
    out["tod_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)

    # 下面这些交易阶段特征必须基于 slot_id，而不是直接基于 hour/minute。
    # 原因：不同数据源可能把 5分钟 bar 标成起点时间戳(09:30...14:55)或终点时间戳(09:35...15:00)。
    # 若直接用钟点，会让同一根交易 bar 在不同口径下落入不同阶段；slot_id 已在上游统一为 0..47。
    slot_int = out["slot_id"].astype(int)
    out["is_morning_session"] = ((slot_int >= 0) & (slot_int <= 23)).astype(int)
    out["is_afternoon_session"] = ((slot_int >= 24) & (slot_int <= 47)).astype(int)
    out["is_opening_30m"] = ((slot_int >= 0) & (slot_int <= 5)).astype(int)
    out["is_late_morning"] = ((slot_int >= 18) & (slot_int <= 23)).astype(int)
    out["is_afternoon_open"] = ((slot_int >= 24) & (slot_int <= 29)).astype(int)
    out["is_last_60m"] = ((slot_int >= 36) & (slot_int <= 47)).astype(int)
    out["is_last_30m"] = ((slot_int >= 42) & (slot_int <= 47)).astype(int)

    def stage_from_slot(k: int) -> str:
        if 0 <= k <= 5:
            return "open_30m"
        if 6 <= k <= 17:
            return "mid_morning"
        if 18 <= k <= 23:
            return "late_morning"
        if 24 <= k <= 29:
            return "afternoon_open"
        if 30 <= k <= 35:
            return "mid_afternoon"
        if 36 <= k <= 47:
            return "last_60m"
        return "out_of_session"

    out["session_stage"] = [stage_from_slot(int(k)) for k in slot_int]
    if "remaining_bars_to_close" in out.columns:
        out["remaining_minutes_to_close_sched"] = pd.to_numeric(out["remaining_bars_to_close"], errors="coerce") * 5.0
    return out


def add_slot_deviation_features(samples: pd.DataFrame) -> pd.DataFrame:
    """按历史同 slot 中位数做简单去季节化。只用 expanding median 的 shift，避免用到当前/未来。"""
    out = samples.copy().sort_values("signal_time").reset_index(drop=True)
    if "slot_id" not in out.columns:
        return out
    candidates = {
        "bar_volume": "bar_volume_slot_med_exp",
        "bar_atr5m12": "bar_atr5m12_slot_med_exp",
        "bar_bar_volume": "bar_bar_volume_slot_med_exp",
    }
    for col, med_col in candidates.items():
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        out[med_col] = vals.groupby(out["slot_id"]).transform(lambda s: s.expanding(min_periods=20).median().shift(1))
        out[f"{col}_slot_ratio"] = vals / (pd.to_numeric(out[med_col], errors="coerce") + EPS)
    return out


def build_opportunity_samples(
    daily_feat: pd.DataFrame,
    intraday_feat: pd.DataFrame,
    cfg,
    meta,
    mod,
    symbol: str,
    target_cfg: TargetConfig,
    round_trip_cost: float,
    include_targets: bool = True,
    require_complete_day_for_targets: bool = True,
) -> pd.DataFrame:
    if intraday_feat.empty:
        raise ValueError("intraday_features 为空，无法构造样本。")

    daily = daily_feat.copy().sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["date"]).reset_index(drop=True)
    intraday = add_intraday_price_fields(intraday_feat)
    intraday["date"] = pd.to_datetime(intraday["date"], errors="coerce").dt.normalize()

    daily_dates = list(daily["date"])
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(daily_dates)}

    rows: List[Dict[str, object]] = []
    dropped_invalid = 0
    total_candidates = 0
    skipped_incomplete_days = 0

    prev_daily_cols = [
        "open", "high", "low", "close", "volume",
        "ema20", "ema60", "atr14", "atrp14", "rsi14", "rvol20", "close_pos",
        "b10_prev", "r5", "resistance", "obv", "obv_ma20", "obv_std20",
        "bench_close", "bench_ret5", "ret5", "rs5",
        "x1_price_vs_ema20", "x2_ema20_vs_ema60", "x3_break_b10", "x4_close_pos",
        "x5_log_rvol20", "x6_obv_z", "x7_rs5", "x8_atrp_rank",
        "z_x1_price_vs_ema20", "z_x2_ema20_vs_ema60", "z_x3_break_b10", "z_x4_close_pos",
        "z_x5_log_rvol20", "z_x6_obv_z", "z_x7_rs5", "z_x8_atrp_rank",
        "cv_best_l2", "cv_logloss", "cv_folds_used", "train_n", "coef_intercept",
        "coef_z_x1_price_vs_ema20", "coef_z_x2_ema20_vs_ema60", "coef_z_x3_break_b10",
        "coef_z_x4_close_pos", "coef_z_x5_log_rvol20", "coef_z_x6_obv_z",
        "coef_z_x7_rs5", "coef_z_x8_atrp_rank", "atrp14_med60",
    ]
    intra_cols = [
        "open", "high", "low", "close", "volume", "amount", "vwap", "session_vwap", "bar_vwap",
        "bar_pv", "bar_volume", "atr5m12", "rsi5m6", "rsi_pos", "cum_ret_from_open",
        "cum_volume", "slot_vol_ratio", "cum_vol_ratio", "avg_slot_volume", "avg_slot_cum_volume",
        "upper_shadow_ratio", "bar_close_pos", "dev_vwap", "pull_vwap", "dev_vwap_q70", "pull_vwap_q60",
    ]

    for trade_date, day_bars in intraday.groupby("date", sort=True):
        trade_date = pd.Timestamp(trade_date).normalize()
        day_bars = day_bars.sort_values("datetime").reset_index(drop=True)
        # 训练标签需要至少“当前 bar + 下一根 bar”；实时预测不构造标签，
        # 09:35 第一根 5分钟 bar 出来后就应允许生成“下一根开盘执行”的信号。
        if include_targets and len(day_bars) < 2:
            continue
        if (not include_targets) and len(day_bars) < 1:
            continue

        expected_bars_per_day = 48
        day_slots, day_slot_source = _select_day_slot_ids(day_bars, expected_bars_per_day=expected_bars_per_day)
        day_slots_num = pd.to_numeric(day_slots, errors="coerce").dropna()
        day_last_slot = int(day_slots_num.max()) if len(day_slots_num) else -1
        day_unique_slots = int(day_slots_num.astype(int).nunique()) if len(day_slots_num) else 0
        min_complete_day_bars = max(2, int(getattr(target_cfg, "min_complete_day_bars", 40) or 40))
        day_has_close_bar = bool(day_last_slot >= expected_bars_per_day - 1 and day_unique_slots >= min_complete_day_bars)
        if include_targets and require_complete_day_for_targets and not day_has_close_bar:
            # 训练标签必须基于完整交易日。否则盘中最新日会被误当作“到收盘”，严重污染路径机会标签。
            skipped_incomplete_days += 1
            continue

        # 训练时通常有当日 daily row；盘中实时预测时，BaoStock 日线可能还没有当天记录。
        # 因此：若当日 daily row 存在，用“昨日 daily 特征 + 当日 open/限制参数”；
        # 若不存在，用最近一个已完成交易日作为 prev_day，并用当日第一根 5m open 合成 curr_day。
        di = date_to_idx.get(trade_date)
        if di is not None:
            if di <= 0:
                continue
            prev_day = daily.iloc[di - 1]
            curr_day = daily.iloc[di].copy()
            current_daily_available = True
        else:
            prior_daily = daily[daily["date"] < trade_date]
            if prior_daily.empty:
                continue
            prev_day = prior_daily.iloc[-1]
            curr_day = prev_day.copy()
            curr_day["date"] = trade_date
            if "open" in day_bars.columns and _finite_positive(day_bars.iloc[0].get("open", np.nan)):
                curr_day["open"] = float(day_bars.iloc[0]["open"])
            current_daily_available = False

        p_rev = float(prev_day.get("p_rev", np.nan)) if pd.notna(prev_day.get("p_rev", np.nan)) else np.nan
        try:
            regime = mod.classify_regime(p_rev, cfg)
        except Exception:
            regime = "UNKNOWN"

        prev_close_px = float(prev_day.get("close", np.nan)) if _finite_positive(prev_day.get("close", np.nan)) else np.nan
        curr_open = float(curr_day.get("open", np.nan)) if _finite_positive(curr_day.get("open", np.nan)) else np.nan
        gap_abs = abs(curr_open / (prev_close_px + EPS) - 1.0) if _finite_positive(prev_close_px) and _finite_positive(curr_open) else np.nan

        for i in range(len(day_bars)):
            bar = day_bars.iloc[i]
            bar_dt = pd.Timestamp(bar["datetime"])
            if not in_market_hours(bar_dt):
                continue
            cur_px = get_ref_price(bar, target_cfg.price_field)
            if not _finite_positive(cur_px):
                dropped_invalid += 1
                continue

            # 训练时用真实可见的未来 bars 构造标签；实时预测时不能因为没有未来 bars 就丢掉当前样本。
            if str(target_cfg.target_mode).lower() == "fixed_horizon_opportunity":
                end_idx = min(len(day_bars) - 1, i + int(target_cfg.horizon_bars))
            else:
                end_idx = len(day_bars) - 1

            # slot 已在 day 级别按 clock/bar_no 口径统一确定，避免 start/end timestamp 混用导致 off-by-one。
            if i < len(day_slots) and pd.notna(day_slots.iloc[i]):
                bar_slot = int(day_slots.iloc[i])
            else:
                bar_slot = int(i)
            scheduled_remaining_bars = max(0, expected_bars_per_day - 1 - bar_slot)
            available_future_bars = max(0, end_idx - i)

            if include_targets:
                if end_idx <= i:
                    continue
                future_bars = day_bars.iloc[i + 1:end_idx + 1].copy()
                if len(future_bars) < int(target_cfg.min_future_bars):
                    continue

                # 标签必须尽量对齐“信号在当前 bar 收盘产生，下一根 bar 开盘成交”的执行假设。
                # 因此机会收益用下一根 bar 的 open（或退回下一根 bar 的参考价）作为可执行基准价，
                # 而不是直接用当前 bar 的 close/vwap，否则快速拉升/跳水时会系统性偏乐观。
                next_bar = day_bars.iloc[i + 1]
                if "open" in next_bar.index and _finite_positive(next_bar.get("open", np.nan)):
                    label_ref_px = float(next_bar["open"])
                else:
                    label_ref_px = get_ref_price(next_bar, target_cfg.price_field)
                if not _finite_positive(label_ref_px):
                    dropped_invalid += 1
                    continue

                target_end_time = pd.Timestamp(day_bars.iloc[end_idx]["datetime"])
                # 这两个字段仅作诊断：训练时是实际可用未来 bars，实时时是日程估计，不能作为特征。
                target_horizon_bars = int(end_idx - i)
                target_horizon_minutes = float((target_end_time - bar_dt).total_seconds() / 60.0)
            else:
                # 实时预测只需要当前状态特征；目标终点按交易日收盘时间写入诊断字段。
                future_bars = pd.DataFrame()
                label_ref_px = np.nan
                target_end_time = pd.Timestamp(trade_date) + pd.Timedelta(hours=15)
                target_horizon_bars = int(scheduled_remaining_bars)
                target_horizon_minutes = float(scheduled_remaining_bars * 5.0)

            total_candidates += 1

            row: Dict[str, object] = {
                "symbol": symbol,
                "signal_time": bar_dt,
                "trade_date": str(trade_date.date()),
                "prev_trade_date": str(pd.Timestamp(prev_day["date"]).date()) if "date" in prev_day.index else None,
                "sample_bar_index_in_day": int(bar_slot),
                "raw_bar_index_in_day": int(i),
                "target_end_time": target_end_time,
                "target_horizon_bars": target_horizon_bars,
                "target_horizon_minutes": target_horizon_minutes,
                "available_future_bars_to_close": int(available_future_bars),
                "remaining_bars_to_close": int(scheduled_remaining_bars),
                "current_ref_price": float(cur_px),
                "label_ref_price": float(label_ref_px) if include_targets and np.isfinite(label_ref_px) else np.nan,
                "execution_ref_price": float(label_ref_px) if include_targets and np.isfinite(label_ref_px) else np.nan,
                "target_kind": f"{target_cfg.target_mode}:{target_cfg.price_field}:{target_cfg.future_extreme_mode}:next_open_exec",
                "round_trip_cost": float(round_trip_cost),
                "p_rev": p_rev,
                "regime": regime,
                "gap_abs": float(gap_abs) if np.isfinite(gap_abs) else np.nan,
                "price_limit_ratio": float(curr_day.get("price_limit_ratio", getattr(meta, "price_limit_ratio", np.nan))),
                "current_daily_available": int(bool(current_daily_available)),
                "is_complete_intraday_day": int(bool(day_has_close_bar)),
                "day_unique_slots": int(day_unique_slots),
                "day_slot_source": str(day_slot_source),
            }

            if include_targets:
                min_future_px, max_future_px = future_extreme_prices(future_bars, target_cfg)
                if not (_finite_positive(min_future_px) and _finite_positive(max_future_px)):
                    dropped_invalid += 1
                    continue
                row.update({
                    "future_min_price": float(min_future_px),
                    "future_max_price": float(max_future_px),
                    "sell_opportunity": float((label_ref_px - min_future_px) / (label_ref_px + EPS) - round_trip_cost),
                    "buy_opportunity": float((max_future_px - label_ref_px) / (label_ref_px + EPS) - round_trip_cost),
                })
                row["opportunity_edge"] = float(row["sell_opportunity"] - row["buy_opportunity"])

            for c in prev_daily_cols:
                if c in prev_day.index:
                    row[f"prev_{c}"] = prev_day[c]
            for c in intra_cols:
                if c in bar.index:
                    row[f"bar_{c}"] = bar[c]

            rows.append(row)

    if not rows:
        raise ValueError(
            "没有构造出任何样本；请检查 BaoStock 数据区间、intraday_start/intraday_end、股票代码。"
            f" candidates={total_candidates}, dropped_invalid_target={dropped_invalid}, "
            f"skipped_incomplete_days={skipped_incomplete_days}. "
            "如果训练区间包含未收盘交易日，默认会跳过；若你确认要用部分交易日，才使用 --allow_partial_train_day。"
        )

    samples = pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)
    if samples.empty:
        raise ValueError("没有构造出任何样本；请检查 BaoStock 数据区间、intraday_start/intraday_end、股票代码。")
    samples.attrs["candidate_events_before_drop"] = total_candidates
    samples.attrs["dropped_invalid_target"] = dropped_invalid
    samples.attrs["skipped_incomplete_days"] = skipped_incomplete_days
    samples = add_time_features(samples, time_col="signal_time")
    samples = add_slot_deviation_features(samples)
    samples["sample_idx"] = np.arange(len(samples), dtype=int)
    return samples


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    drop = {
        "sell_opportunity", "buy_opportunity", "opportunity_edge",
        "future_min_price", "future_max_price",
        "available_future_bars_to_close", "raw_bar_index_in_day",
        "signal_time", "sample_idx", "signal_date", "trade_date", "prev_trade_date",
        "target_end_time", "target_kind", "symbol",
        "target_horizon_bars", "target_horizon_minutes",  # 训练来自实际未来 bars，实时来自日程估计；剔除避免错配/泄漏
        "label_ref_price", "execution_ref_price",  # 标签计算诊断字段，不作为特征
        "current_daily_available",  # 实时诊断字段；不要让模型把“当天日线是否已出”当成信号
        "is_complete_intraday_day",  # 训练筛选/诊断字段，不作为交易信号
        "day_unique_slots",  # 数据完整性诊断字段，不作为交易信号
        "day_slot_source",  # timestamp/bar_no 口径诊断字段，不作为交易信号
        "current_ref_price",  # 当前价格可视为状态也可作为尺度；默认剔除，避免模型主要学价格水平
        # 以下钟点字段受 5分钟 bar 起点/终点时间戳口径影响。模型只用 slot_id/slot_sin/slot_cos/阶段特征。
        "hour", "minute", "minute_of_day", "tod_sin", "tod_cos",
    }
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        elif pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s):
            cat_cols.append(c)
    return num_cols, cat_cols


def build_design_matrix(df: pd.DataFrame, num_cols: Sequence[str], cat_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    out = df.copy()
    fill_values: Dict[str, float] = {}
    for c in num_cols:
        s = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = s.median()
        fill = float(med) if pd.notna(med) else 0.0
        fill_values[c] = fill
        out[c] = s.fillna(fill)
    X_num = out[list(num_cols)].copy() if num_cols else pd.DataFrame(index=out.index)
    if cat_cols:
        X_cat = pd.get_dummies(out[list(cat_cols)].astype("string"), dummy_na=True, drop_first=False)
    else:
        X_cat = pd.DataFrame(index=out.index)
    X = pd.concat([X_num, X_cat], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, list(X.columns), fill_values


def make_feature_schema(num_cols: Sequence[str], cat_cols: Sequence[str], feature_names: Sequence[str], fill_values: Dict[str, float]) -> Dict[str, object]:
    return {
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "feature_names": list(feature_names),
        "num_fill_values": {k: float(v) for k, v in fill_values.items()},
    }


def build_design_matrix_from_schema(df: pd.DataFrame, schema: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    num_cols = list(schema.get("num_cols", []))
    cat_cols = list(schema.get("cat_cols", []))
    feature_names = list(schema.get("feature_names", []))
    fill_map = dict(schema.get("num_fill_values", {}))
    for c in num_cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(fill_map.get(c, 0.0)))
    X_num = out[num_cols].copy() if num_cols else pd.DataFrame(index=out.index)
    if cat_cols:
        for c in cat_cols:
            if c not in out.columns:
                out[c] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
        X_cat = pd.get_dummies(out[cat_cols].astype("string"), dummy_na=True, drop_first=False)
    else:
        X_cat = pd.DataFrame(index=out.index)
    X = pd.concat([X_num, X_cat], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_names:
        X = X.reindex(columns=feature_names, fill_value=0.0)
    return X


def chronological_split(samples: pd.DataFrame, cfg: XGBRegModelConfig, perf_start: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = samples.sort_values("signal_time").reset_index(drop=True).copy()
    n = len(df)
    if n < cfg.min_total_samples:
        raise ValueError(f"样本太少：{n} < {cfg.min_total_samples}。请拉长 intraday_start/intraday_end，或者降低 min_total_samples。")

    if perf_start:
        perf_ts = pd.to_datetime(perf_start)
        pre = df[df["signal_time"] < perf_ts].copy()
        test = df[df["signal_time"] >= perf_ts].copy()
        if len(pre) >= cfg.min_train_samples + cfg.min_valid_samples and len(test) >= cfg.min_test_samples:
            valid_len = max(cfg.min_valid_samples, int(len(pre) * cfg.valid_ratio / max(cfg.train_ratio + cfg.valid_ratio, EPS)))
            train = pre.iloc[:-valid_len].copy()
            valid = pre.iloc[-valid_len:].copy()
            meta = {
                "split_mode": "perf_start",
                "perf_start": str(perf_ts),
                "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test)),
            }
            return train, valid, test, meta

    train_end = max(cfg.min_train_samples, int(np.floor(n * cfg.train_ratio)))
    valid_end = max(train_end + cfg.min_valid_samples, int(np.floor(n * (cfg.train_ratio + cfg.valid_ratio))))
    valid_end = min(valid_end, n - cfg.min_test_samples)
    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()
    if len(train) < cfg.min_train_samples or len(valid) < cfg.min_valid_samples or len(test) < cfg.min_test_samples:
        raise ValueError(f"切分后样本不足：train={len(train)}, valid={len(valid)}, test={len(test)}")
    meta = {"split_mode": "ratio", "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test))}
    return train, valid, test, meta


def make_expanding_time_series_splits(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """轻量版 TimeSeriesSplit，避免为这一个功能强依赖 scikit-learn。"""
    n_splits = int(max(1, n_splits))
    if n_samples <= n_splits:
        return []
    test_size = n_samples // (n_splits + 1)
    if test_size <= 0:
        return []
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_end = n_samples - (n_splits - k) * test_size
        val_start = train_end
        val_end = val_start + test_size
        if train_end <= 0 or val_end > n_samples:
            continue
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return splits


def fallback_params(cfg: XGBRegModelConfig) -> Dict[str, object]:
    return {
        "max_depth": int(cfg.max_depth_grid[0]),
        "learning_rate": float(cfg.learning_rate_grid[0]),
        "n_estimators": int(cfg.n_estimators_grid[0]),
        "min_child_weight": int(cfg.min_child_weight_grid[0]),
        "subsample": float(cfg.subsample_grid[0]),
        "colsample_bytree": float(cfg.colsample_bytree_grid[0]),
        "reg_lambda": float(cfg.reg_lambda_grid[0]),
        "n_jobs": int(getattr(cfg, "xgb_n_jobs", 4)),
    }


def fit_one_xgb(X: pd.DataFrame, y: np.ndarray, params: Dict[str, object]) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=int(params.get("n_jobs", 4)),
        tree_method="hist",
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
    )
    model.fit(X, y)
    return model


def cv_search_params(X: pd.DataFrame, y: np.ndarray, cfg: XGBRegModelConfig, target_name: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    if len(X) < cfg.min_cv_train_samples + cfg.min_cv_val_samples:
        params = fallback_params(cfg)
        return params, pd.DataFrame([{**params, "target": target_name, "cv_rmse": np.nan, "cv_rank_ic": np.nan, "folds_used": 0, "note": "insufficient_samples"}])

    combos = []
    for md in cfg.max_depth_grid:
        for lr in cfg.learning_rate_grid:
            for ne in cfg.n_estimators_grid:
                for mcw in cfg.min_child_weight_grid:
                    for ss in cfg.subsample_grid:
                        for cs in cfg.colsample_bytree_grid:
                            for rl in cfg.reg_lambda_grid:
                                combos.append({
                                    "max_depth": md, "learning_rate": lr, "n_estimators": ne,
                                    "min_child_weight": mcw, "subsample": ss,
                                    "colsample_bytree": cs, "reg_lambda": rl,
                                    "n_jobs": int(getattr(cfg, "xgb_n_jobs", 4)),
                                })
    max_combos = int(getattr(cfg, "max_cv_param_combos", 0) or 0)
    if max_combos > 0 and len(combos) > max_combos:
        # 全网格最多 72 组 * 2目标 * 5折，训练会非常慢；默认按网格顺序均匀抽取 24 组。
        keep_idx = np.unique(np.linspace(0, len(combos) - 1, max_combos).round().astype(int))
        combos = [combos[int(i)] for i in keep_idx]

    n_splits = min(cfg.n_splits, max(2, len(X) // max(cfg.min_cv_val_samples, 1)))
    splits = make_expanding_time_series_splits(len(X), n_splits=n_splits)
    rows = []
    for params in combos:
        rmses, rics = [], []
        folds_used = 0
        for tr_idx, va_idx in splits:
            if len(tr_idx) < cfg.min_cv_train_samples or len(va_idx) < cfg.min_cv_val_samples:
                continue
            model = fit_one_xgb(X.iloc[tr_idx], y[tr_idx], params)
            pred = model.predict(X.iloc[va_idx])
            rmses.append(rmse(y[va_idx], pred))
            rics.append(safe_spearman(y[va_idx], pred))
            folds_used += 1
        if folds_used:
            rows.append({**params, "target": target_name, "cv_rmse": float(np.mean(rmses)), "cv_rank_ic": float(np.nanmean(rics)), "folds_used": folds_used})
    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        params = fallback_params(cfg)
        return params, pd.DataFrame([{**params, "target": target_name, "cv_rmse": np.nan, "cv_rank_ic": np.nan, "folds_used": 0, "note": "no_valid_fold"}])
    cv_df = cv_df.sort_values(["cv_rmse", "cv_rank_ic"], ascending=[True, False]).reset_index(drop=True)
    best = cv_df.iloc[0][["max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"]].to_dict()
    best["n_jobs"] = int(getattr(cfg, "xgb_n_jobs", 4))
    return best, cv_df


def add_prediction_quantiles(df: pd.DataFrame, valid_df: pd.DataFrame, pred_col: str, prefix: str, q_edges: Sequence[float]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = df.copy()
    v = pd.to_numeric(valid_df[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    thresholds: Dict[str, float] = {}
    out[f"{prefix}_pred_quantile"] = pd.NA
    if len(v) >= 10:
        qs = np.quantile(v.to_numpy(), q_edges)
        thresholds = {f"q{int(q * 100)}": float(x) for q, x in zip(q_edges, qs)}
        # 不用 pd.cut，避免预测值低方差时重复 bin edge 直接报错。
        x = pd.to_numeric(out[pred_col], errors="coerce")
        labels = np.full(len(out), pd.NA, dtype=object)
        finite = np.isfinite(x.to_numpy(dtype=float))
        xv = x.to_numpy(dtype=float)
        labels[finite & (xv <= qs[0])] = "Q1"
        labels[finite & (xv > qs[0]) & (xv <= qs[1])] = "Q2"
        labels[finite & (xv > qs[1]) & (xv <= qs[2])] = "Q3"
        labels[finite & (xv > qs[2]) & (xv <= qs[3])] = "Q4"
        labels[finite & (xv > qs[3])] = "Q5"
        out[f"{prefix}_pred_quantile"] = labels
    return out, thresholds


def decide_action(row: pd.Series, rule_cfg: SignalRuleConfig) -> str:
    ps = float(row.get("pred_sell_opportunity", np.nan))
    pb = float(row.get("pred_buy_opportunity", np.nan))
    if not np.isfinite(ps) or not np.isfinite(pb):
        return "WAIT"
    edge = ps - pb
    if ps >= rule_cfg.min_action_opportunity and edge >= rule_cfg.min_action_edge:
        return "SELL_T"
    if pb >= rule_cfg.min_action_opportunity and -edge >= rule_cfg.min_action_edge:
        return "BUY_T"
    return "WAIT"


def data_diagnostics(daily_feat: pd.DataFrame, intraday_feat: pd.DataFrame, samples: pd.DataFrame, fetch_logs: Dict) -> Dict[str, object]:
    def range_info(df: pd.DataFrame, col: str) -> Dict[str, object]:
        if df is None or df.empty or col not in df.columns:
            return {"rows": 0, "min": None, "max": None, "n_dates": 0}
        s = pd.to_datetime(df[col], errors="coerce").dropna()
        if s.empty:
            return {"rows": int(len(df)), "min": None, "max": None, "n_dates": 0}
        return {
            "rows": int(len(df)),
            "min": str(s.min()),
            "max": str(s.max()),
            "n_dates": int(s.dt.normalize().nunique()),
        }
    return {
        "data_source": fetch_logs.get("data_source", "unknown") if isinstance(fetch_logs, dict) else "unknown",
        "daily_features": range_info(daily_feat, "date"),
        "intraday_features": range_info(intraday_feat, "datetime"),
        "samples": range_info(samples, "signal_time"),
        "candidate_events_before_drop": int(samples.attrs.get("candidate_events_before_drop", len(samples))),
        "dropped_invalid_target": int(samples.attrs.get("dropped_invalid_target", 0)),
        "skipped_incomplete_days": int(samples.attrs.get("skipped_incomplete_days", 0)),
        "fetch_logs": fetch_logs,
    }


def run_update_data(args: argparse.Namespace) -> Dict:
    helper = import_module_from_path(args.helper_py, "helper_mod_dual_update")
    out_dir = ensure_dir(args.output_dir)
    mod, cfg, meta, daily_feat, intraday_feat, fetch_logs, base_paths = helper.build_feature_frames(args)
    summary = {
        "mode": "update_data",
        "symbol": args.symbol,
        "data_source": fetch_logs.get("data_source", "unknown") if isinstance(fetch_logs, dict) else "unknown",
        "daily_rows": int(len(daily_feat)),
        "intraday_rows": int(len(intraday_feat)),
        "daily_min": str(pd.to_datetime(daily_feat["date"], errors="coerce").min()) if not daily_feat.empty and "date" in daily_feat.columns else None,
        "daily_max": str(pd.to_datetime(daily_feat["date"], errors="coerce").max()) if not daily_feat.empty and "date" in daily_feat.columns else None,
        "intraday_min": str(pd.to_datetime(intraday_feat["datetime"], errors="coerce").min()) if not intraday_feat.empty and "datetime" in intraday_feat.columns else None,
        "intraday_max": str(pd.to_datetime(intraday_feat["datetime"], errors="coerce").max()) if not intraday_feat.empty and "datetime" in intraday_feat.columns else None,
        "base_paths": base_paths,
        "fetch_logs": fetch_logs,
    }
    save_json(summary, out_dir / "update_data_summary.json")
    return summary




def _series_or_nan(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _prepare_plot_df(df: pd.DataFrame, max_points: int = 1200) -> pd.DataFrame:
    out = df.copy()
    if "signal_time" not in out.columns:
        return pd.DataFrame()
    out["signal_time"] = pd.to_datetime(out["signal_time"], errors="coerce")
    out = out.dropna(subset=["signal_time"]).sort_values("signal_time").reset_index(drop=True)
    if max_points and max_points > 0 and len(out) > max_points:
        out = out.tail(max_points).reset_index(drop=True)
    return out


def _macd_from_price(price: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = pd.to_numeric(price, errors="coerce").astype(float)
    dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    macd = 2.0 * (dif - dea)
    return dif, dea, macd


def add_prediction_quantiles_from_thresholds(df: pd.DataFrame, pred_col: str, prefix: str, thresholds: Optional[Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    qcol = f"{prefix}_pred_quantile"
    out[qcol] = pd.NA
    if not thresholds:
        return out
    try:
        q20 = float(thresholds.get("q20"))
        q40 = float(thresholds.get("q40"))
        q60 = float(thresholds.get("q60"))
        q80 = float(thresholds.get("q80"))
    except Exception:
        return out
    x = pd.to_numeric(out.get(pred_col), errors="coerce")
    xv = x.to_numpy(dtype=float)
    finite = np.isfinite(xv)
    labels = np.full(len(out), pd.NA, dtype=object)
    labels[finite & (xv <= q20)] = "Q1"
    labels[finite & (xv > q20) & (xv <= q40)] = "Q2"
    labels[finite & (xv > q40) & (xv <= q60)] = "Q3"
    labels[finite & (xv > q60) & (xv <= q80)] = "Q4"
    labels[finite & (xv > q80)] = "Q5"
    out[qcol] = labels
    return out


def _local_peak_mask(series: pd.Series, radius: int = 1) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    arr = s.to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return pd.Series([], dtype=bool, index=s.index)
    mask = np.isfinite(arr)
    radius = max(1, int(radius or 1))
    for k in range(1, radius + 1):
        prev = np.full(n, np.nan, dtype=float)
        nxt = np.full(n, np.nan, dtype=float)
        prev[k:] = arr[:-k]
        nxt[:-k] = arr[k:]
        mask &= np.isnan(prev) | (arr >= prev)
        mask &= np.isnan(nxt) | (arr > nxt)
    return pd.Series(mask, index=s.index)


def _sparsify_peak_mask(series: pd.Series, mask: pd.Series, min_gap: int = 3) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    values = s.to_numpy(dtype=float)
    base_mask = np.asarray(mask, dtype=bool)
    idx = np.where(base_mask)[0]
    keep = np.zeros(len(values), dtype=bool)
    if len(idx) == 0:
        return pd.Series(keep, index=s.index)
    min_gap = max(0, int(min_gap or 0))
    order = idx[np.argsort(values[idx])[::-1]]
    chosen: List[int] = []
    for i in order:
        if not np.isfinite(values[i]):
            continue
        if all(abs(int(i) - j) > min_gap for j in chosen):
            keep[int(i)] = True
            chosen.append(int(i))
    return pd.Series(keep, index=s.index)


def annotate_high_opportunity_points(
    df: pd.DataFrame,
    sell_thresholds: Optional[Dict[str, float]] = None,
    buy_thresholds: Optional[Dict[str, float]] = None,
    high_quantile: str = "Q5",
    local_peak_radius: int = 1,
    min_marker_gap: int = 3,
) -> pd.DataFrame:
    out = df.copy()
    if "sell_pred_quantile" not in out.columns:
        out = add_prediction_quantiles_from_thresholds(out, "pred_sell_opportunity", "sell", sell_thresholds)
    if "buy_pred_quantile" not in out.columns:
        out = add_prediction_quantiles_from_thresholds(out, "pred_buy_opportunity", "buy", buy_thresholds)

    pred_sell = pd.to_numeric(out.get("pred_sell_opportunity"), errors="coerce")
    pred_buy = pd.to_numeric(out.get("pred_buy_opportunity"), errors="coerce")

    if "sell_pred_quantile" in out.columns and out["sell_pred_quantile"].notna().any():
        sell_base = out["sell_pred_quantile"].astype(str).eq(str(high_quantile))
    else:
        qv = float(pred_sell.quantile(0.8)) if pred_sell.notna().any() else np.nan
        sell_base = pd.Series(np.isfinite(pred_sell.to_numpy(dtype=float)) & (pred_sell.to_numpy(dtype=float) >= qv), index=out.index)

    if "buy_pred_quantile" in out.columns and out["buy_pred_quantile"].notna().any():
        buy_base = out["buy_pred_quantile"].astype(str).eq(str(high_quantile))
    else:
        qv = float(pred_buy.quantile(0.8)) if pred_buy.notna().any() else np.nan
        buy_base = pd.Series(np.isfinite(pred_buy.to_numpy(dtype=float)) & (pred_buy.to_numpy(dtype=float) >= qv), index=out.index)

    sell_peak = _local_peak_mask(pred_sell, radius=local_peak_radius)
    buy_peak = _local_peak_mask(pred_buy, radius=local_peak_radius)
    out["high_sell_opp_point"] = _sparsify_peak_mask(pred_sell, sell_base & sell_peak, min_gap=min_marker_gap).astype(int)
    out["high_buy_opp_point"] = _sparsify_peak_mask(pred_buy, buy_base & buy_peak, min_gap=min_marker_gap).astype(int)
    return out


def add_quantiles_from_self(df: pd.DataFrame, value_col: str, prefix: str, q_edges: Sequence[float] = (0.2, 0.4, 0.6, 0.8)) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = df.copy()
    qcol = f"{prefix}_quantile"
    out[qcol] = pd.NA
    x = pd.to_numeric(out.get(value_col), errors="coerce")
    if x.notna().sum() == 0:
        return out, {}
    q20, q40, q60, q80 = [float(x.quantile(float(q))) for q in q_edges]
    xv = x.to_numpy(dtype=float)
    finite = np.isfinite(xv)
    labels = np.full(len(out), pd.NA, dtype=object)
    labels[finite & (xv <= q20)] = "Q1"
    labels[finite & (xv > q20) & (xv <= q40)] = "Q2"
    labels[finite & (xv > q40) & (xv <= q60)] = "Q3"
    labels[finite & (xv > q60) & (xv <= q80)] = "Q4"
    labels[finite & (xv > q80)] = "Q5"
    out[qcol] = labels
    thresholds = {"q20": q20, "q40": q40, "q60": q60, "q80": q80}
    return out, thresholds


def annotate_realized_opportunity_points(
    df: pd.DataFrame,
    high_quantile: str = "Q5",
    local_peak_radius: int = 1,
    min_marker_gap: int = 3,
) -> pd.DataFrame:
    out = df.copy()
    if "sell_opportunity" not in out.columns and "buy_opportunity" not in out.columns:
        out["high_true_sell_opp_point"] = 0
        out["high_true_buy_opp_point"] = 0
        return out
    if "true_sell_quantile" not in out.columns and "sell_opportunity" in out.columns:
        out, _ = add_quantiles_from_self(out, "sell_opportunity", "true_sell")
    if "true_buy_quantile" not in out.columns and "buy_opportunity" in out.columns:
        out, _ = add_quantiles_from_self(out, "buy_opportunity", "true_buy")

    true_sell = pd.to_numeric(out.get("sell_opportunity"), errors="coerce")
    true_buy = pd.to_numeric(out.get("buy_opportunity"), errors="coerce")

    if "true_sell_quantile" in out.columns and out["true_sell_quantile"].notna().any():
        sell_base = out["true_sell_quantile"].astype(str).eq(str(high_quantile))
    else:
        qv = float(true_sell.quantile(0.8)) if true_sell.notna().any() else np.nan
        sell_base = pd.Series(np.isfinite(true_sell.to_numpy(dtype=float)) & (true_sell.to_numpy(dtype=float) >= qv), index=out.index)

    if "true_buy_quantile" in out.columns and out["true_buy_quantile"].notna().any():
        buy_base = out["true_buy_quantile"].astype(str).eq(str(high_quantile))
    else:
        qv = float(true_buy.quantile(0.8)) if true_buy.notna().any() else np.nan
        buy_base = pd.Series(np.isfinite(true_buy.to_numpy(dtype=float)) & (true_buy.to_numpy(dtype=float) >= qv), index=out.index)

    sell_peak = _local_peak_mask(true_sell, radius=local_peak_radius)
    buy_peak = _local_peak_mask(true_buy, radius=local_peak_radius)
    out["high_true_sell_opp_point"] = _sparsify_peak_mask(true_sell, sell_base & sell_peak, min_gap=min_marker_gap).astype(int)
    out["high_true_buy_opp_point"] = _sparsify_peak_mask(true_buy, buy_base & buy_peak, min_gap=min_marker_gap).astype(int)
    return out


def plot_dual_opportunity_predictions(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    max_points: int = 1200,
    include_true_targets: bool = True,
    sell_thresholds: Optional[Dict[str, float]] = None,
    buy_thresholds: Optional[Dict[str, float]] = None,
    high_quantile: str = "Q5",
    local_peak_radius: int = 1,
    min_marker_gap: int = 3,
    show_pred_markers: bool = True,
    show_true_markers: bool = False,
) -> Optional[str]:
    """保存价格 + 双机会预测 + MACD 图，并可标注预测/真实高机会点。"""
    if not HAS_MATPLOTLIB:
        return None
    plot_df = _prepare_plot_df(df, max_points=max_points)
    if plot_df.empty:
        return None
    plot_df = annotate_high_opportunity_points(
        plot_df,
        sell_thresholds=sell_thresholds,
        buy_thresholds=buy_thresholds,
        high_quantile=high_quantile,
        local_peak_radius=local_peak_radius,
        min_marker_gap=min_marker_gap,
    )
    if include_true_targets:
        plot_df = annotate_realized_opportunity_points(
            plot_df,
            high_quantile=high_quantile,
            local_peak_radius=local_peak_radius,
            min_marker_gap=min_marker_gap,
        )
    x = plot_df["signal_time"]
    price = _series_or_nan(plot_df, ["current_ref_price", "bar_close", "bar_bar_vwap", "bar_vwap", "bar_session_vwap"])
    if price.notna().sum() == 0:
        price = pd.Series(np.arange(len(plot_df), dtype=float), index=plot_df.index)

    high_sell_mask = plot_df.get("high_sell_opp_point", 0).astype(int).astype(bool)
    high_buy_mask = plot_df.get("high_buy_opp_point", 0).astype(int).astype(bool)
    high_true_sell_mask = plot_df.get("high_true_sell_opp_point", 0).astype(int).astype(bool)
    high_true_buy_mask = plot_df.get("high_true_buy_opp_point", 0).astype(int).astype(bool)

    pred_sell = _series_or_nan(plot_df, ["pred_sell_opportunity"])
    pred_buy = _series_or_nan(plot_df, ["pred_buy_opportunity"])
    pred_edge = _series_or_nan(plot_df, ["pred_opportunity_edge"])
    true_sell = _series_or_nan(plot_df, ["sell_opportunity"])
    true_buy = _series_or_nan(plot_df, ["buy_opportunity"])

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(x, price, linewidth=1.2, label="price/current_ref")
    if show_pred_markers and high_sell_mask.any():
        ax1.scatter(x[high_sell_mask], price[high_sell_mask], marker="v", s=56, label="pred_high_sell_opp", zorder=5)
    if show_pred_markers and high_buy_mask.any():
        ax1.scatter(x[high_buy_mask], price[high_buy_mask], marker="^", s=56, label="pred_high_buy_opp", zorder=5)
    if show_true_markers and high_true_sell_mask.any():
        ax1.scatter(x[high_true_sell_mask], price[high_true_sell_mask], marker="1", s=90, label="true_high_sell_opp", zorder=6)
    if show_true_markers and high_true_buy_mask.any():
        ax1.scatter(x[high_true_buy_mask], price[high_true_buy_mask], marker="2", s=90, label="true_high_buy_opp", zorder=6)
    ax1.set_title(title)
    ax1.set_ylabel("price")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    if pred_sell.notna().any():
        ax2.plot(x, pred_sell, linewidth=1.0, label="pred_sell_opportunity")
    if pred_buy.notna().any():
        ax2.plot(x, pred_buy, linewidth=1.0, label="pred_buy_opportunity")
    if pred_edge.notna().any():
        ax2.plot(x, pred_edge, linewidth=1.0, label="pred_edge")
    if sell_thresholds and "q80" in sell_thresholds:
        ax2.axhline(float(sell_thresholds["q80"]), linewidth=0.8, linestyle="--", alpha=0.6, label="pred_sell_q80")
    if buy_thresholds and "q80" in buy_thresholds:
        ax2.axhline(float(buy_thresholds["q80"]), linewidth=0.8, linestyle=":", alpha=0.6, label="pred_buy_q80")
    if show_pred_markers and high_sell_mask.any() and pred_sell.notna().any():
        ax2.scatter(x[high_sell_mask], pred_sell[high_sell_mask], marker="v", s=46, label="pred_high_sell_point", zorder=5)
    if show_pred_markers and high_buy_mask.any() and pred_buy.notna().any():
        ax2.scatter(x[high_buy_mask], pred_buy[high_buy_mask], marker="^", s=46, label="pred_high_buy_point", zorder=5)
    if include_true_targets and true_sell.notna().any():
        ax2.plot(x, true_sell, linewidth=0.9, alpha=0.75, label="true_sell_opportunity")
    if include_true_targets and true_buy.notna().any():
        ax2.plot(x, true_buy, linewidth=0.9, alpha=0.75, label="true_buy_opportunity")
    if show_true_markers and include_true_targets and high_true_sell_mask.any() and true_sell.notna().any():
        ax2.scatter(x[high_true_sell_mask], true_sell[high_true_sell_mask], marker="1", s=90, label="true_high_sell_point", zorder=6)
    if show_true_markers and include_true_targets and high_true_buy_mask.any() and true_buy.notna().any():
        ax2.scatter(x[high_true_buy_mask], true_buy[high_true_buy_mask], marker="2", s=90, label="true_high_buy_point", zorder=6)
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_ylabel("opportunity")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", ncol=2)

    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    dif, dea, macd = _macd_from_price(price)
    ax3.plot(x, dif, linewidth=1.0, label="DIF")
    ax3.plot(x, dea, linewidth=1.0, label="DEA")
    ax3.bar(x, macd.fillna(0.0), width=0.002, alpha=0.35, label="MACD=2*(DIF-DEA)")
    ax3.axhline(0.0, linewidth=0.8)
    ax3.set_ylabel("MACD")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_price_with_trade_markers(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    max_points: int = 0,
    sell_thresholds: Optional[Dict[str, float]] = None,
    buy_thresholds: Optional[Dict[str, float]] = None,
    high_quantile: str = "Q5",
    local_peak_radius: int = 1,
    min_marker_gap: int = 3,
    show_pred_markers: bool = True,
    show_true_markers: bool = False,
    max_line_gap_minutes: int = 20,
) -> Optional[str]:
    """只画区间分钟价格线，并在价格线上标出预测/真实买卖点。

    关键修正：横轴使用“交易 bar 序号”，不是自然时间戳。
    这样 11:30->13:00、15:00->次日 09:35 不会按 24 小时拉开。
    """
    if not HAS_MATPLOTLIB:
        return None
    plot_df = _prepare_plot_df(df, max_points=max_points)
    if plot_df.empty:
        return None
    plot_df = annotate_high_opportunity_points(
        plot_df,
        sell_thresholds=sell_thresholds,
        buy_thresholds=buy_thresholds,
        high_quantile=high_quantile,
        local_peak_radius=local_peak_radius,
        min_marker_gap=min_marker_gap,
    )
    if show_true_markers:
        plot_df = annotate_realized_opportunity_points(
            plot_df,
            high_quantile=high_quantile,
            local_peak_radius=local_peak_radius,
            min_marker_gap=min_marker_gap,
        )

    plot_df = plot_df.copy()
    plot_df["signal_time"] = pd.to_datetime(plot_df["signal_time"], errors="coerce")
    plot_df = plot_df.dropna(subset=["signal_time"]).sort_values("signal_time").reset_index(drop=True)

    # 压缩交易时间轴：每个实际存在的 5 分钟交易 bar 占一个等距位置。
    # 不使用 datetime 作为 x，否则跨日/午休会被按自然时间拉出巨大空白。
    x_pos = pd.Series(np.arange(len(plot_df), dtype=float), index=plot_df.index)
    price = _series_or_nan(plot_df, ["current_ref_price", "bar_close", "bar_bar_vwap", "bar_vwap", "bar_session_vwap", "label_ref_price"])
    if price.notna().sum() == 0:
        price = pd.Series(np.arange(len(plot_df), dtype=float), index=plot_df.index)

    pred_sell_mask = plot_df.get("high_sell_opp_point", 0).astype(int).astype(bool)
    pred_buy_mask = plot_df.get("high_buy_opp_point", 0).astype(int).astype(bool)
    true_sell_mask = plot_df.get("high_true_sell_opp_point", 0).astype(int).astype(bool)
    true_buy_mask = plot_df.get("high_true_buy_opp_point", 0).astype(int).astype(bool)

    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_pos, price, linewidth=1.2, color="#1f77b4", label="minute_price")

    if show_pred_markers and pred_buy_mask.any():
        ax.scatter(x_pos[pred_buy_mask], price[pred_buy_mask], marker="^", s=90, color="#2ca02c", edgecolors="black", linewidths=0.5, label="pred_buy", zorder=5)
    if show_pred_markers and pred_sell_mask.any():
        ax.scatter(x_pos[pred_sell_mask], price[pred_sell_mask], marker="v", s=90, color="#d62728", edgecolors="black", linewidths=0.5, label="pred_sell", zorder=5)

    if show_true_markers and true_buy_mask.any():
        ax.scatter(x_pos[true_buy_mask], price[true_buy_mask], marker="o", s=78, color="#17becf", edgecolors="black", linewidths=0.6, label="true_buy", zorder=6)
    if show_true_markers and true_sell_mask.any():
        ax.scatter(x_pos[true_sell_mask], price[true_sell_mask], marker="X", s=82, color="#ff7f0e", edgecolors="black", linewidths=0.6, label="true_sell", zorder=6)

    # 日界线：不拉开自然时间，只在压缩轴上标记交易日切换。
    trade_day = plot_df["signal_time"].dt.normalize()
    day_change_idx = np.where(trade_day.ne(trade_day.shift()).to_numpy())[0]
    for j, idx in enumerate(day_change_idx):
        ax.axvline(float(idx), linewidth=0.8, linestyle="--", alpha=0.35)
        label = plot_df.loc[int(idx), "signal_time"].strftime("%m-%d")
        ax.text(float(idx), 1.01, label, transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=9, alpha=0.75)

    # 横轴 tick 用交易 bar 序号位置，但显示对应交易时刻。
    n = len(plot_df)
    if n <= 80:
        step = max(1, n // 12)
    elif n <= 240:
        step = 24
    else:
        step = max(24, n // 14)
    tick_idx = list(range(0, n, step))
    if n - 1 not in tick_idx:
        tick_idx.append(n - 1)
    tick_labels = []
    for idx in tick_idx:
        ts = plot_df.loc[int(idx), "signal_time"]
        # 多日区间显示日期+时刻；单日区间只显示时刻。
        if trade_day.nunique() > 1:
            tick_labels.append(ts.strftime("%m-%d\n%H:%M"))
        else:
            tick_labels.append(ts.strftime("%H:%M"))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=0)

    ax.set_title(title + " (compressed trading-time axis)")
    ax.set_ylabel("price")
    ax.set_xlabel("trading bar sequence")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_feature_importance_dual(importance_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> Optional[str]:
    if not HAS_MATPLOTLIB or importance_df is None or importance_df.empty:
        return None
    targets = list(importance_df["target"].dropna().unique()) if "target" in importance_df.columns else []
    if not targets:
        return None
    fig = plt.figure(figsize=(16, 10))
    for idx, target in enumerate(targets[:2], start=1):
        ax = fig.add_subplot(1, min(2, len(targets)), idx)
        sub = importance_df[importance_df["target"] == target].nlargest(top_n, "importance").iloc[::-1]
        ax.barh(sub["feature"], sub["importance"])
        ax.set_title(str(target))
        ax.grid(True, axis="x", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

def save_artifacts(
    out_dir: Path,
    sell_model: XGBRegressor,
    buy_model: XGBRegressor,
    schema: Dict[str, object],
    target_cfg: TargetConfig,
    rule_cfg: SignalRuleConfig,
    params: Dict[str, object],
    split_meta: Dict[str, object],
    diagnostics: Dict[str, object],
    thresholds: Dict[str, object],
) -> Dict[str, str]:
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sell_path = artifacts_dir / "xgb_sell_opportunity_model.json"
    buy_path = artifacts_dir / "xgb_buy_opportunity_model.json"
    schema_path = artifacts_dir / "feature_schema.json"
    meta_path = artifacts_dir / "model_artifacts.json"
    sell_model.save_model(sell_path)
    buy_model.save_model(buy_path)
    save_json(schema, schema_path)
    save_json({
        "target_config": asdict(target_cfg),
        "signal_rule_config": asdict(rule_cfg),
        "best_params": params,
        "split_meta": split_meta,
        "data_diagnostics": diagnostics,
        "thresholds": thresholds,
        "model_files": {
            "sell_opportunity": str(sell_path.name),
            "buy_opportunity": str(buy_path.name),
        },
    }, meta_path)
    return {
        "artifacts_dir": str(artifacts_dir),
        "sell_model_json": str(sell_path),
        "buy_model_json": str(buy_path),
        "feature_schema_json": str(schema_path),
        "model_artifacts_json": str(meta_path),
    }


def load_xgb_model(path: Path) -> XGBRegressor:
    model = XGBRegressor(objective="reg:squarederror")
    model.load_model(path)
    return model


def run_train(args: argparse.Namespace) -> Dict:
    t0 = time.perf_counter()
    helper = import_module_from_path(args.helper_py, "helper_mod_dual_train")
    out_dir = ensure_dir(args.output_dir)
    target_cfg = TargetConfig(
        target_mode=args.target_mode,
        horizon_bars=int(args.target_horizon_bars),
        price_field=args.target_price_field,
        future_extreme_mode=args.future_extreme_mode,
        min_future_bars=int(args.target_min_future_bars),
        min_complete_day_bars=int(args.min_complete_day_bars),
        round_trip_cost=args.round_trip_cost,
    )
    rule_cfg = SignalRuleConfig(
        min_action_edge=float(args.min_action_edge),
        min_action_opportunity=float(args.min_action_opportunity),
    )

    mod, cfg, meta, daily_feat, intraday_feat, fetch_logs, base_paths = helper.build_feature_frames(args)
    round_trip_cost = calc_round_trip_cost(args, cfg, target_cfg)
    target_cfg.round_trip_cost = round_trip_cost

    # 保存数据快照，便于你检查到底取了多少 BaoStock 数据。
    daily_feat.to_csv(out_dir / "daily_features.csv", index=False, encoding="utf-8-sig")
    intraday_feat.to_csv(out_dir / "intraday_features.csv", index=False, encoding="utf-8-sig")
    save_json({"base_paths": base_paths, "fetch_logs": fetch_logs}, out_dir / "data_fetch_log.json")

    samples = build_opportunity_samples(
        daily_feat=daily_feat,
        intraday_feat=intraday_feat,
        cfg=cfg,
        meta=meta,
        mod=mod,
        symbol=args.symbol,
        target_cfg=target_cfg,
        round_trip_cost=round_trip_cost,
        include_targets=True,
        require_complete_day_for_targets=not bool(getattr(args, "allow_partial_train_day", False)),
    )
    samples.to_csv(out_dir / "signal_samples.csv", index=False, encoding="utf-8-sig")

    model_cfg = XGBRegModelConfig(
        min_total_samples=int(args.min_total_samples),
        min_train_samples=int(args.min_train_samples),
        min_valid_samples=int(args.min_valid_samples),
        min_test_samples=int(args.min_test_samples),
        xgb_n_jobs=int(args.xgb_n_jobs),
        max_cv_param_combos=int(args.max_cv_param_combos),
    )
    train_df, valid_df, test_df, split_meta = chronological_split(samples, model_cfg, args.perf_start)

    # 特征 schema / 缺失值填充只能用训练集拟合，不能用 valid/test 的分布。
    num_cols, cat_cols = infer_feature_columns(train_df)
    X_train_schema_fit, feature_names, fill_values = build_design_matrix(train_df, num_cols, cat_cols)
    schema = make_feature_schema(num_cols, cat_cols, feature_names, fill_values)
    X_all = build_design_matrix_from_schema(samples, schema)

    idx_train = train_df["sample_idx"].to_numpy(dtype=int)
    idx_valid = valid_df["sample_idx"].to_numpy(dtype=int)
    idx_test = test_df["sample_idx"].to_numpy(dtype=int)
    X_train = X_all.iloc[idx_train]
    X_valid = X_all.iloc[idx_valid]
    X_test = X_all.iloc[idx_test]

    y_sell_all = samples["sell_opportunity"].astype(float).to_numpy()
    y_buy_all = samples["buy_opportunity"].astype(float).to_numpy()
    y_sell_train, y_sell_valid, y_sell_test = y_sell_all[idx_train], y_sell_all[idx_valid], y_sell_all[idx_test]
    y_buy_train, y_buy_valid, y_buy_test = y_buy_all[idx_train], y_buy_all[idx_valid], y_buy_all[idx_test]

    sell_params, sell_cv = cv_search_params(X_train, y_sell_train, model_cfg, "sell_opportunity")
    buy_params, buy_cv = cv_search_params(X_train, y_buy_train, model_cfg, "buy_opportunity")
    cv_df = pd.concat([sell_cv, buy_cv], axis=0, ignore_index=True)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False, encoding="utf-8-sig")

    sell_train_model = fit_one_xgb(X_train, y_sell_train, sell_params)
    buy_train_model = fit_one_xgb(X_train, y_buy_train, buy_params)
    valid_pred_df = valid_df.copy()
    valid_pred_df["pred_sell_opportunity"] = sell_train_model.predict(X_valid)
    valid_pred_df["pred_buy_opportunity"] = buy_train_model.predict(X_valid)
    valid_pred_df["pred_opportunity_edge"] = valid_pred_df["pred_sell_opportunity"] - valid_pred_df["pred_buy_opportunity"]
    valid_pred_df["signal_action"] = valid_pred_df.apply(lambda r: decide_action(r, rule_cfg), axis=1)
    valid_pred_df, sell_thresholds = add_prediction_quantiles(valid_pred_df, valid_pred_df, "pred_sell_opportunity", "sell", rule_cfg.q_edges)
    valid_pred_df, buy_thresholds = add_prediction_quantiles(valid_pred_df, valid_pred_df, "pred_buy_opportunity", "buy", rule_cfg.q_edges)
    valid_pred_df = annotate_high_opportunity_points(
        valid_pred_df,
        sell_thresholds=sell_thresholds,
        buy_thresholds=buy_thresholds,
        high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
        local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
        min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
    )
    valid_pred_df.to_csv(out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    train_valid_df = pd.concat([train_df, valid_df], axis=0).sort_values("signal_time").reset_index(drop=True)
    train_valid_idx = train_valid_df["sample_idx"].to_numpy(dtype=int)
    X_train_valid = X_all.iloc[train_valid_idx]
    sell_final = fit_one_xgb(X_train_valid, y_sell_all[train_valid_idx], sell_params)
    buy_final = fit_one_xgb(X_train_valid, y_buy_all[train_valid_idx], buy_params)

    test_pred_df = test_df.copy()
    test_pred_df["pred_sell_opportunity"] = sell_final.predict(X_test)
    test_pred_df["pred_buy_opportunity"] = buy_final.predict(X_test)
    test_pred_df["pred_opportunity_edge"] = test_pred_df["pred_sell_opportunity"] - test_pred_df["pred_buy_opportunity"]
    test_pred_df["signal_action"] = test_pred_df.apply(lambda r: decide_action(r, rule_cfg), axis=1)
    test_pred_df = add_prediction_quantiles_from_thresholds(test_pred_df, "pred_sell_opportunity", "sell", sell_thresholds)
    test_pred_df = add_prediction_quantiles_from_thresholds(test_pred_df, "pred_buy_opportunity", "buy", buy_thresholds)
    test_pred_df = annotate_high_opportunity_points(
        test_pred_df,
        sell_thresholds=sell_thresholds,
        buy_thresholds=buy_thresholds,
        high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
        local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
        min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
    )
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    all_scores = samples.copy()
    all_scores["pred_sell_opportunity"] = sell_final.predict(X_all)
    all_scores["pred_buy_opportunity"] = buy_final.predict(X_all)
    all_scores["pred_opportunity_edge"] = all_scores["pred_sell_opportunity"] - all_scores["pred_buy_opportunity"]
    all_scores["signal_action"] = all_scores.apply(lambda r: decide_action(r, rule_cfg), axis=1)

    # 分位阈值必须来自“只用 train 训练的模型”在 valid 上的预测，
    # 不能用 final(train+valid) 模型在 valid 上的预测，否则 valid 分位阈值被轻微污染。
    all_scores = add_prediction_quantiles_from_thresholds(all_scores, "pred_sell_opportunity", "sell", sell_thresholds)
    all_scores = add_prediction_quantiles_from_thresholds(all_scores, "pred_buy_opportunity", "buy", buy_thresholds)
    all_scores = annotate_high_opportunity_points(
        all_scores,
        sell_thresholds=sell_thresholds,
        buy_thresholds=buy_thresholds,
        high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
        local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
        min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
    )
    all_scores.to_csv(out_dir / "all_signal_scores.csv", index=False, encoding="utf-8-sig")

    importance_sell = pd.DataFrame({"feature": feature_names, "importance": sell_final.feature_importances_, "target": "sell_opportunity"})
    importance_buy = pd.DataFrame({"feature": feature_names, "importance": buy_final.feature_importances_, "target": "buy_opportunity"})
    importance = pd.concat([importance_sell, importance_buy], axis=0, ignore_index=True).sort_values(["target", "importance"], ascending=[True, False])
    importance.to_csv(out_dir / "feature_importance_dual.csv", index=False, encoding="utf-8-sig")

    plot_paths: Dict[str, Optional[str]] = {}
    if not bool(getattr(args, "no_plots", False)):
        plots_dir = out_dir / "plots"
        max_plot_points = int(getattr(args, "plot_max_points", 1200) or 1200)
        plot_paths["valid_overview_png"] = plot_dual_opportunity_predictions(
            valid_pred_df, plots_dir / "valid_predictions_overview.png",
            title=f"{args.symbol} valid dual-opportunity predictions",
            max_points=max_plot_points,
            include_true_targets=True,
            sell_thresholds=sell_thresholds,
            buy_thresholds=buy_thresholds,
            high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
            local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
            min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
        )
        plot_paths["test_overview_png"] = plot_dual_opportunity_predictions(
            test_pred_df, plots_dir / "test_predictions_overview.png",
            title=f"{args.symbol} test dual-opportunity predictions",
            max_points=max_plot_points,
            include_true_targets=True,
            sell_thresholds=sell_thresholds,
            buy_thresholds=buy_thresholds,
            high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
            local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
            min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
        )
        plot_paths["all_scores_overview_png"] = plot_dual_opportunity_predictions(
            all_scores, plots_dir / "all_signal_scores_overview.png",
            title=f"{args.symbol} all dual-opportunity scores (tail)",
            max_points=max_plot_points,
            include_true_targets=True,
            sell_thresholds=sell_thresholds,
            buy_thresholds=buy_thresholds,
            high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
            local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
            min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
        )
        plot_paths["feature_importance_png"] = plot_feature_importance_dual(
            importance, plots_dir / "feature_importance_dual_top30.png", top_n=30
        )

    eval_obj = {
        "valid_sell": evaluate_regression(y_sell_valid, valid_pred_df["pred_sell_opportunity"].to_numpy()),
        "valid_buy": evaluate_regression(y_buy_valid, valid_pred_df["pred_buy_opportunity"].to_numpy()),
        "test_sell": evaluate_regression(y_sell_test, test_pred_df["pred_sell_opportunity"].to_numpy()),
        "test_buy": evaluate_regression(y_buy_test, test_pred_df["pred_buy_opportunity"].to_numpy()),
    }
    save_json(eval_obj, out_dir / "eval_metrics.json")

    diagnostics = data_diagnostics(daily_feat, intraday_feat, samples, fetch_logs)
    save_json(diagnostics, out_dir / "data_diagnostics.json")

    artifacts = save_artifacts(
        out_dir=out_dir,
        sell_model=sell_final,
        buy_model=buy_final,
        schema=schema,
        target_cfg=target_cfg,
        rule_cfg=rule_cfg,
        params={"sell_opportunity": sell_params, "buy_opportunity": buy_params},
        split_meta=split_meta,
        diagnostics=diagnostics,
        thresholds={"sell_pred": sell_thresholds, "buy_pred": buy_thresholds},
    )

    summary = {
        "mode": "train",
        "symbol": args.symbol,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "data_source": diagnostics["data_source"],
        "daily_feature_rows": int(len(daily_feat)),
        "intraday_feature_rows": int(len(intraday_feat)),
        "sample_rows": int(len(samples)),
        "sample_dates": diagnostics["samples"],
        "round_trip_cost": round_trip_cost,
        "split_meta": split_meta,
        "eval_metrics": eval_obj,
        "artifacts": artifacts,
        "plots": plot_paths,
    }
    save_json(summary, out_dir / "summary.json")
    return summary


def _resolve_feature_paths(out_dir: Path, args: argparse.Namespace) -> Tuple[Path, Path]:
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else out_dir / "feature_cache"
    candidates_daily = [
        out_dir / "daily_features.csv",
        feature_cache_dir / "daily_features_cache.csv",
        out_dir / "data_cache" / "daily_features.csv",
    ]
    candidates_intra = [
        out_dir / "intraday_features.csv",
        feature_cache_dir / "intraday_features_cache.csv",
        out_dir / "data_cache" / "intraday_features.csv",
    ]
    daily_path = next((p for p in candidates_daily if p.exists()), None)
    intra_path = next((p for p in candidates_intra if p.exists()), None)
    if daily_path is None or intra_path is None:
        raise FileNotFoundError("未找到特征缓存。先运行 --mode train 或 --mode update_data。")
    return daily_path, intra_path


def _parse_prediction_window(args: argparse.Namespace, intraday_feat: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, bool]:
    """解析预测输出窗口。date-only 的 end 按整天包含处理。"""
    def _is_date_only(x: object) -> bool:
        s = str(x).strip()
        return bool(s) and (":" not in s) and ("T" not in s) and (len(s) <= 10)

    latest_dt = pd.to_datetime(intraday_feat["datetime"], errors="coerce").max()
    latest_day = pd.to_datetime(intraday_feat["date"], errors="coerce").max()
    raw_start = getattr(args, "predict_start", None)
    raw_end = getattr(args, "predict_end", None)
    range_mode = bool(raw_start or raw_end)

    if not range_mode:
        start_ts = pd.Timestamp(latest_day).normalize()
        end_ts = pd.Timestamp(latest_day).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return start_ts, min(end_ts, pd.Timestamp(latest_dt)), False

    if raw_start:
        start_ts = pd.to_datetime(raw_start)
        if _is_date_only(raw_start):
            start_ts = pd.Timestamp(start_ts).normalize()
    else:
        start_ts = pd.Timestamp(latest_day).normalize()

    if raw_end:
        end_ts = pd.to_datetime(raw_end)
        if _is_date_only(raw_end):
            end_ts = pd.Timestamp(end_ts).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        end_ts = pd.Timestamp(latest_dt)

    if end_ts < start_ts:
        raise ValueError(f"predict_end 早于 predict_start: {start_ts} > {end_ts}")
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts), True


def _safe_window_tag(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    return f"{pd.Timestamp(start_ts).strftime('%Y%m%d_%H%M%S')}_{pd.Timestamp(end_ts).strftime('%Y%m%d_%H%M%S')}"


def run_realtime_predict(args: argparse.Namespace) -> Dict:
    out_dir = ensure_dir(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else out_dir / "artifacts"
    schema = load_json(artifacts_dir / "feature_schema.json")
    artifact_meta = load_json(artifacts_dir / "model_artifacts.json")
    target_cfg = TargetConfig(**artifact_meta["target_config"])
    rule_cfg = SignalRuleConfig(**{k: v for k, v in artifact_meta.get("signal_rule_config", {}).items() if k in SignalRuleConfig.__dataclass_fields__})
    sell_model = load_xgb_model(artifacts_dir / "xgb_sell_opportunity_model.json")
    buy_model = load_xgb_model(artifacts_dir / "xgb_buy_opportunity_model.json")

    helper = import_module_from_path(args.helper_py, "helper_mod_dual_realtime")

    # realtime_live_predict：先走 BaoStock helper 更新原始/特征缓存。
    # realtime_predict：不联网、不更新，只读取 output_dir/feature_cache 等已存在缓存，便于复盘任意历史区间。
    fetch_logs: Dict[str, object] = {"mode": "cache_only"}
    base_paths: Dict[str, object] = {}
    if getattr(args, "mode", "") == "realtime_live_predict":
        mod, cfg, meta, daily_feat, intraday_feat, fetch_logs, base_paths = helper.build_feature_frames(args)
    else:
        mod = helper.import_backtest_module(args.backtest_py)
        cfg = mod.StrategyConfig(
            initial_shares=args.initial_shares,
            initial_cash=args.initial_cash,
            evaluation_start_date=args.perf_start or str(pd.to_datetime(args.intraday_start).date()),
            cost_buy_rate=args.cost_buy_rate,
            cost_sell_rate=args.cost_sell_rate,
            slippage_bps=args.slippage_bps,
            force_rebuy_at_close=not args.no_force_rebuy_close,
            verbose=not args.quiet,
        )
        meta = mod.MetaConfig(
            exchange=args.exchange,
            board=args.board,
            security_type=args.security_type,
            lot_size=args.lot_size,
            price_limit_ratio=args.price_limit_ratio,
            no_price_limit_default=False,
            t0_eligible=False,
        )
        daily_path, intra_path = _resolve_feature_paths(out_dir, args)
        daily_feat = pd.read_csv(daily_path)
        intraday_feat = pd.read_csv(intra_path)
        base_paths = {"daily_features_csv": str(daily_path), "intraday_features_csv": str(intra_path)}

    daily_feat["date"] = pd.to_datetime(daily_feat["date"], errors="coerce").dt.normalize()
    intraday_feat["datetime"] = pd.to_datetime(intraday_feat["datetime"], errors="coerce")
    if "date" in intraday_feat.columns:
        intraday_feat["date"] = pd.to_datetime(intraday_feat["date"], errors="coerce").dt.normalize()
    else:
        intraday_feat["date"] = intraday_feat["datetime"].dt.normalize()

    if args.intraday_end:
        intraday_feat = intraday_feat[intraday_feat["datetime"] <= pd.to_datetime(args.intraday_end)].copy()

    if intraday_feat.empty:
        raise ValueError("预测失败：intraday_features 为空。请检查 intraday_start/intraday_end 或先运行 update_data/train。")

    pred_start_ts, pred_end_ts, range_mode = _parse_prediction_window(args, intraday_feat)
    pred_start_day = pd.Timestamp(pred_start_ts).normalize()
    pred_end_day = pd.Timestamp(pred_end_ts).normalize()

    # 截断到预测窗口末尾，避免历史复盘时真实结果使用 predict_end 之后的数据。
    intraday_feat = intraday_feat[intraday_feat["datetime"] <= pred_end_ts].copy()
    if intraday_feat.empty:
        raise ValueError("预测失败：按 predict_end/intraday_end 截断后 intraday_features 为空。")

    unique_days = (
        pd.to_datetime(intraday_feat["date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not unique_days:
        raise ValueError("预测失败：无法从 intraday_features 提取交易日。")

    context_days = max(5, int(getattr(args, "realtime_context_days", 80) or 80))
    days = [pd.Timestamp(x).normalize() for x in unique_days]
    valid_idx = [i for i, d in enumerate(days) if d <= pred_end_day]
    if not valid_idx:
        raise ValueError("预测失败：预测区间结束日在特征缓存之前。")
    end_idx = max(valid_idx)
    start_candidates = [i for i, d in enumerate(days) if d >= pred_start_day]
    start_idx = min(start_candidates) if start_candidates else end_idx
    context_start_idx = max(0, start_idx - context_days)
    keep_days = set(days[context_start_idx:end_idx + 1])

    intraday_context = intraday_feat[pd.to_datetime(intraday_feat["date"], errors="coerce").dt.normalize().isin(keep_days)].copy()
    daily_context = daily_feat[daily_feat["date"] <= pred_end_day].copy()

    samples_context = build_opportunity_samples(
        daily_feat=daily_context,
        intraday_feat=intraday_context,
        cfg=cfg,
        meta=meta,
        mod=mod,
        symbol=args.symbol,
        target_cfg=target_cfg,
        round_trip_cost=float(target_cfg.round_trip_cost or calc_round_trip_cost(args, cfg, target_cfg)),
        include_targets=False,
        require_complete_day_for_targets=False,
    )
    sample_times = pd.to_datetime(samples_context["signal_time"], errors="coerce")
    samples = samples_context[(sample_times >= pred_start_ts) & (sample_times <= pred_end_ts)].copy().reset_index(drop=True)
    if samples.empty:
        raise ValueError("预测失败：指定预测区间没有构造出样本。请检查 predict_start/predict_end 与缓存数据范围。")

    X = build_design_matrix_from_schema(samples, schema)
    pred = samples.copy()
    pred["pred_sell_opportunity"] = sell_model.predict(X)
    pred["pred_buy_opportunity"] = buy_model.predict(X)
    pred["pred_opportunity_edge"] = pred["pred_sell_opportunity"] - pred["pred_buy_opportunity"]

    thresholds_meta = artifact_meta.get("thresholds", {}) if isinstance(artifact_meta, dict) else {}
    sell_thresholds = thresholds_meta.get("sell_pred", {}) if isinstance(thresholds_meta, dict) else {}
    buy_thresholds = thresholds_meta.get("buy_pred", {}) if isinstance(thresholds_meta, dict) else {}
    pred = add_prediction_quantiles_from_thresholds(pred, "pred_sell_opportunity", "sell", sell_thresholds)
    pred = add_prediction_quantiles_from_thresholds(pred, "pred_buy_opportunity", "buy", buy_thresholds)
    pred = annotate_high_opportunity_points(
        pred,
        sell_thresholds=sell_thresholds,
        buy_thresholds=buy_thresholds,
        high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
        local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
        min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
    )

    min_actionable_bars = max(1, int(getattr(target_cfg, "min_future_bars", 2) or 2))
    pred["is_actionable_bar"] = (pd.to_numeric(pred.get("remaining_bars_to_close", 0), errors="coerce") >= min_actionable_bars).astype(int)
    pred["signal_action_raw"] = pred.apply(lambda r: decide_action(r, rule_cfg), axis=1)
    pred["signal_action"] = np.where(pred["is_actionable_bar"].astype(bool), pred["signal_action_raw"], "WAIT")
    pred["action_block_reason"] = np.where(pred["is_actionable_bar"].astype(bool), "", "too_few_bars_to_close")

    # 根据当前/区间末尾已经可见的当日数据，计算真实后验机会；历史区间会逐日计算到该日可见末端。
    realized_truth_rows = 0
    try:
        realized_context = build_opportunity_samples(
            daily_feat=daily_context,
            intraday_feat=intraday_context,
            cfg=cfg,
            meta=meta,
            mod=mod,
            symbol=args.symbol,
            target_cfg=target_cfg,
            round_trip_cost=float(target_cfg.round_trip_cost or calc_round_trip_cost(args, cfg, target_cfg)),
            include_targets=True,
            require_complete_day_for_targets=False,
        )
        rt = pd.to_datetime(realized_context["signal_time"], errors="coerce")
        realized_range = realized_context[(rt >= pred_start_ts) & (rt <= pred_end_ts)].copy().reset_index(drop=True)
        if not realized_range.empty:
            realized_truth_rows = int(len(realized_range))
            merge_cols = [
                "signal_time", "label_ref_price", "execution_ref_price", "future_min_price", "future_max_price",
                "sell_opportunity", "buy_opportunity", "opportunity_edge",
                "target_end_time", "target_horizon_bars", "target_horizon_minutes", "available_future_bars_to_close",
            ]
            keep_cols = [c for c in merge_cols if c in realized_range.columns]
            pred = pred.merge(realized_range[keep_cols], on="signal_time", how="left", suffixes=("", "_realized"))
            pred = annotate_realized_opportunity_points(
                pred,
                high_quantile=str(getattr(args, "plot_high_quantile", "Q5") or "Q5"),
                local_peak_radius=int(getattr(args, "plot_peak_radius", 1) or 1),
                min_marker_gap=int(getattr(args, "plot_marker_gap", 3) or 3),
            )
    except Exception:
        realized_truth_rows = 0

    if range_mode:
        tag = _safe_window_tag(pred_start_ts, pred_end_ts)
        pred_path = out_dir / f"range_predictions_{tag}.csv"
        pred_plot_name = f"range_predictions_overview_{tag}.png"
        truth_plot_name = f"range_realized_truth_overview_{tag}.png"
        summary_path = out_dir / f"range_prediction_summary_{tag}.json"
    else:
        pred_path = out_dir / "realtime_predictions.csv"
        pred_plot_name = "realtime_predictions_overview.png"
        truth_plot_name = "realtime_realized_truth_overview.png"
        summary_path = out_dir / "realtime_prediction_summary.json"

    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")

    prediction_plot_path: Optional[str] = None
    realized_truth_plot_path: Optional[str] = None
    if not bool(getattr(args, "no_plots", False)):
        plot_high_quantile = str(getattr(args, "plot_high_quantile", "Q5") or "Q5")
        plot_peak_radius = int(getattr(args, "plot_peak_radius", 1) or 1)
        plot_marker_gap = int(getattr(args, "plot_marker_gap", 3) or 3)
        plot_max_points = int(getattr(args, "plot_max_points", 1200) or 1200)

        if range_mode:
            prediction_plot_path = plot_price_with_trade_markers(
                pred,
                out_dir / "plots" / pred_plot_name,
                title=f"{args.symbol} minute line with predicted buy/sell points",
                max_points=0,
                sell_thresholds=sell_thresholds,
                buy_thresholds=buy_thresholds,
                high_quantile=plot_high_quantile,
                local_peak_radius=plot_peak_radius,
                min_marker_gap=plot_marker_gap,
                show_pred_markers=True,
                show_true_markers=False,
            )
            if realized_truth_rows > 0:
                realized_truth_plot_path = plot_price_with_trade_markers(
                    pred,
                    out_dir / "plots" / truth_plot_name,
                    title=f"{args.symbol} minute line with realized true buy/sell points",
                    max_points=0,
                    sell_thresholds=sell_thresholds,
                    buy_thresholds=buy_thresholds,
                    high_quantile=plot_high_quantile,
                    local_peak_radius=plot_peak_radius,
                    min_marker_gap=plot_marker_gap,
                    show_pred_markers=False,
                    show_true_markers=True,
                )
        else:
            prediction_plot_path = plot_dual_opportunity_predictions(
                pred,
                out_dir / "plots" / pred_plot_name,
                title=f"{args.symbol} predicted high buy/sell opportunity points",
                max_points=plot_max_points,
                include_true_targets=False,
                sell_thresholds=sell_thresholds,
                buy_thresholds=buy_thresholds,
                high_quantile=plot_high_quantile,
                local_peak_radius=plot_peak_radius,
                min_marker_gap=plot_marker_gap,
                show_pred_markers=True,
                show_true_markers=False,
            )
            if realized_truth_rows > 0:
                realized_truth_plot_path = plot_dual_opportunity_predictions(
                    pred,
                    out_dir / "plots" / truth_plot_name,
                    title=f"{args.symbol} realized good buy/sell points within selected window",
                    max_points=plot_max_points,
                    include_true_targets=True,
                    sell_thresholds=sell_thresholds,
                    buy_thresholds=buy_thresholds,
                    high_quantile=plot_high_quantile,
                    local_peak_radius=plot_peak_radius,
                    min_marker_gap=plot_marker_gap,
                    show_pred_markers=False,
                    show_true_markers=True,
                )

    latest = pred.sort_values("signal_time").tail(1).iloc[0].to_dict() if not pred.empty else {}
    summary = {
        "mode": args.mode,
        "range_mode": bool(range_mode),
        "predict_start": str(pred_start_ts),
        "predict_end": str(pred_end_ts),
        "rows": int(len(pred)),
        "realized_truth_rows": int(realized_truth_rows),
        "prediction_csv": str(pred_path),
        "prediction_plot_png": prediction_plot_path,
        "realized_truth_plot_png": realized_truth_plot_path,
        "summary_json": str(summary_path),
        "latest_signal": latest,
        "data_source": fetch_logs.get("data_source", "cache_only") if isinstance(fetch_logs, dict) else "cache_only",
        "base_paths": base_paths,
    }
    save_json(summary, summary_path)
    return summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BaoStock 双机会 XGBoost 回归：sell_opportunity / buy_opportunity")
    p.add_argument("--mode", choices=["train", "update_data", "realtime_predict", "realtime_live_predict"], default="train")
    p.add_argument("--symbol", required=True)
    p.add_argument("--benchmark_symbol", default="000300")
    p.add_argument("--daily_start", default="2018-01-01")
    p.add_argument("--daily_end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--intraday_start", default="2018-01-01 09:30:00")
    p.add_argument("--intraday_end", default=pd.Timestamp.today().strftime("%Y-%m-%d 15:00:00"))
    p.add_argument("--predict_start", default=None, help="预测输出起点；支持 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS。为空时输出最新交易日。")
    p.add_argument("--predict_end", default=None, help="预测输出终点；date-only 会包含整天。为空时到 intraday_end/缓存最新时刻。")
    p.add_argument("--perf_start", default=None, help="测试集起点；为空则按 60/20/20 切分")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none", ""])

    p.add_argument("--backtest_py", default=str(SCRIPT_DIR / "t_strategy_backtest_cv5_split_eval.py"))
    p.add_argument("--helper_py", default=str(SCRIPT_DIR / "ashare_fetch_and_train_xgb_sell_signal_baostock_state_cache_helper_fix2.py"))
    p.add_argument("--output_dir", default="./ashare_xgb_dual_opportunity_baostock_out")
    p.add_argument("--artifacts_dir", default=None)

    p.add_argument("--cache_mode", choices=["incremental", "full"], default="incremental")
    p.add_argument("--force_refresh", action="store_true")
    p.add_argument("--raw_cache_dir", default=None)
    p.add_argument("--feature_cache_mode", choices=["incremental", "full"], default="incremental")
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--daily_feature_overlap_days", type=int, default=260)
    p.add_argument("--intraday_feature_overlap_days", type=int, default=40)
    p.add_argument("--daily_feature_context_days", type=int, default=0)
    p.add_argument("--intraday_feature_context_days", type=int, default=0)

    p.add_argument("--target_mode", choices=["opportunity", "fixed_horizon_opportunity"], default="opportunity")
    p.add_argument("--target_horizon_bars", type=int, default=12)
    p.add_argument("--target_price_field", choices=["bar_vwap", "close", "session_vwap", "vwap"], default="bar_vwap")
    p.add_argument("--future_extreme_mode", choices=["high_low", "price_field"], default="high_low")
    p.add_argument("--target_min_future_bars", type=int, default=2)
    p.add_argument("--min_complete_day_bars", type=int, default=40,
                   help="训练标签要求的完整交易日最少 5分钟 bar 数；默认 40，并且必须包含 15:00 bar。")
    p.add_argument("--round_trip_cost", type=float, default=None)
    p.add_argument("--allow_partial_train_day", action="store_true",
                   help="默认训练时剔除没有 15:00 收盘 bar 的不完整交易日；打开后允许使用部分交易日标签。")

    p.add_argument("--min_action_edge", type=float, default=0.0010)
    p.add_argument("--min_action_opportunity", type=float, default=0.0015)

    p.add_argument("--min_total_samples", type=int, default=300)
    p.add_argument("--min_train_samples", type=int, default=120)
    p.add_argument("--min_valid_samples", type=int, default=50)
    p.add_argument("--min_test_samples", type=int, default=50)
    p.add_argument("--xgb_n_jobs", type=int, default=4, help="XGBoost 并行线程数；Windows/笔记本上建议 1~4，避免 -1 占满 CPU 或卡死。")
    p.add_argument("--max_cv_param_combos", type=int, default=24,
                   help="CV最多评估的XGBoost参数组合数；0表示全网格。默认24，避免一次训练跑数百个XGB拟合。")

    p.add_argument("--exchange", default="SSE")
    p.add_argument("--board", default="MAIN")
    p.add_argument("--security_type", default="STOCK")
    p.add_argument("--lot_size", type=int, default=100)
    p.add_argument("--price_limit_ratio", type=float, default=0.10)
    p.add_argument("--initial_shares", type=int, default=2000)
    p.add_argument("--initial_cash", type=float, default=0.0)
    p.add_argument("--cost_buy_rate", type=float, default=0.00035)
    p.add_argument("--cost_sell_rate", type=float, default=0.00135)
    p.add_argument("--slippage_bps", type=float, default=2.0)
    p.add_argument("--no_force_rebuy_close", action="store_true")
    p.add_argument("--realtime_context_days", type=int, default=80,
                   help="实时预测时用于计算同 slot 历史统计的最近交易日上下文长度")
    p.add_argument("--no_plots", action="store_true", help="关闭 PNG 绘图输出。默认会输出 plots/*.png。")
    p.add_argument("--plot_max_points", type=int, default=1200, help="每张图最多绘制最近多少个样本点，避免长历史图过大。")
    p.add_argument("--plot_high_quantile", type=str, default="Q5", choices=["Q4", "Q5"], help="图上标注高机会点时使用的预测分位层级。Q5 更严格，Q4 更宽松。")
    p.add_argument("--plot_peak_radius", type=int, default=1, help="局部峰值窗口半径；越大越严格。")
    p.add_argument("--plot_marker_gap", type=int, default=3, help="相邻高机会点之间至少间隔多少个 bar，避免标记过密。")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "update_data":
        result = run_update_data(args)
    elif args.mode in {"realtime_predict", "realtime_live_predict"}:
        result = run_realtime_predict(args)
    else:
        result = run_train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
