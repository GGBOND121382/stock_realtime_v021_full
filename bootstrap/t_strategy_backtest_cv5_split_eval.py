#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T+1 做T 策略回测脚本（单文件版）

功能概览
--------
1. 基于前一交易日的日线特征，滚动训练一个逻辑回归分类器；在每个滚动窗口内，
   使用 5-fold 时间序列交叉验证选择正则强度，并输出下一交易日的 REVERSAL 概率 p_rev；
2. 仅在 REBOUND 环境中，允许盘中执行“先卖昨仓、再买回”的卖出型 T；
3. 使用 5 分钟数据计算 VWAP、5m ATR、5m RSI、同槽位平均成交量、累计量比、
   偏离 VWAP 的历史分位等；
4. 显式处理 A 股 T+1 约束、手数限制、涨跌停距离过滤、事件日过滤；
5. 输出逐笔成交、每日权益曲线、汇总统计。

输入数据格式
------------
1) 日线 CSV（必需）
   必需列：date, open, high, low, close, volume
   可选列：event_flag, no_price_limit_flag

2) 5 分钟 CSV（必需）
   必需列：datetime, open, high, low, close, volume

3) 基准日线 CSV（可选）
   必需列：date, close
   用于构造 RS5（个股相对基准 5 日收益）

示例
----
python t_strategy_backtest.py \
  --daily_csv daily.csv \
  --intraday_csv intraday_5m.csv \
  --benchmark_daily_csv index_daily.csv \
  --initial_shares 2000 \
  --initial_cash 0 \
  --lot_size 100 \
  --price_limit_ratio 0.10 \
  --cost_buy_rate 0.00035 \
  --cost_sell_rate 0.00135 \
  --output_dir ./bt_out

说明
----
- 该脚本默认使用“信号在 bar 收盘生成，下一根 bar 开盘成交”的执行模型，
  以避免 look-ahead；
- 默认 force_rebuy_at_close=True：如果当天卖出后未完全买回，则在收盘强制补回，
  以维持“核心底仓 + 做T”的使用场景；
- 逻辑回归标签是一个可编辑的 baseline 定义；你后续可以根据自己的股票风格
  进一步定制该标签。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


# =========================
# 配置
# =========================

@dataclass
class MetaConfig:
    exchange: str = "SSE"
    board: str = "MAIN"
    security_type: str = "STOCK"
    lot_size: int = 100
    price_limit_ratio: float = 0.10
    no_price_limit_default: bool = False
    t0_eligible: bool = False


@dataclass
class StrategyConfig:
    # 初始资金 / 仓位
    initial_shares: int = 2000
    initial_cash: float = 0.0

    # 评估窗口起点（长历史训练 + 短窗口评估）
    # None 表示默认从 intraday csv 的首个交易日开始评估
    evaluation_start_date: Optional[str] = None

    # 交易成本
    cost_buy_rate: float = 0.00035
    cost_sell_rate: float = 0.00135
    slippage_bps: float = 2.0
    safety_buffer: float = 0.0015

    # 日线特征 / 模型
    z_window: int = 120
    train_window: int = 500
    min_train_samples: int = 180
    logistic_lr: float = 0.05
    logistic_epochs: int = 200

    # 5-fold 时间序列交叉验证（用于选择逻辑回归正则强度）
    cv_folds: int = 5
    cv_min_val_samples: int = 20
    cv_l2_grid: tuple[float, ...] = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)

    # 逻辑回归标签定义（baseline，可按风格改）
    label_break_atr: float = 0.30
    label_close_atr: float = 0.20
    label_gap_atrp_mult: float = 0.50
    label_closepos_min: float = 0.60

    # 环境分类阈值
    reversal_prob_hi: float = 0.65
    reversal_prob_lo: float = 0.35

    # 盘中过滤 / 硬过滤
    skip_gap_atr_mult: float = 1.50
    min_prev_rvol20: float = 0.60
    min_dist_to_limit: float = 0.008

    # 分时历史窗口
    slot_lookback_days: int = 20
    slot_min_periods: int = 5

    # 卖出触发参数
    sell_dev_vwap_floor: float = 0.80
    sell_cum_vol_ratio_min: float = 1.10
    sell_slot_vol_ratio_min: float = 1.10
    sell_upper_shadow_min: float = 0.35
    sell_bar_close_pos_max: float = 0.35
    sell_amp_vs_open_extra: float = 0.001

    # 买回触发参数
    buy_pull_vwap_floor: float = 0.40
    buy_rsi_pos_max: float = 0.30
    buy_slot_vol_ratio_max: float = 0.90
    buy_second_leg_bar_close_pos_min: float = 0.60

    # 强制回补参数
    force_cover_atr_mult: float = 0.40
    force_cover_cum_vol_ratio: float = 1.40
    force_cover_consecutive_bars: int = 3

    # 仓位控制
    min_sell_pct: float = 0.10
    max_sell_pct: float = 0.35
    base_sell_pct: float = 0.10
    rebound_conf_sell_pct_span: float = 0.20
    vol_adj_min: float = 0.70
    vol_adj_max: float = 1.30
    liq_cap_slot_volume_pct: float = 0.02

    # 其他
    force_rebuy_at_close: bool = True
    allow_reversal_add: bool = False  # 默认关闭，聚焦卖出型 T
    verbose: bool = True


# =========================
# 工具函数
# =========================


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))



def floor_lot(qty: float, lot_size: int) -> int:
    if qty <= 0:
        return 0
    return int(np.floor(qty / lot_size) * lot_size)



def pct_rank_last(history: np.ndarray, value: float) -> float:
    history = history[np.isfinite(history)]
    if len(history) == 0 or not np.isfinite(value):
        return np.nan
    return float(np.mean(history <= value))



def rolling_pct_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        start = max(0, i - window)
        hist = vals[start:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= min_periods and np.isfinite(vals[i]):
            out[i] = np.mean(hist <= vals[i])
    return pd.Series(out, index=series.index)



def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())



def annualized_sharpe(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2 or daily_ret.std(ddof=0) < EPS:
        return np.nan
    return float(np.sqrt(252.0) * daily_ret.mean() / (daily_ret.std(ddof=0) + EPS))



def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi



def fit_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.05,
    epochs: int = 200,
) -> np.ndarray:
    """简单 numpy 版 L2 逻辑回归，用于滚动训练。"""
    n, m = X.shape
    beta = np.zeros(m + 1, dtype=float)
    for _ in range(epochs):
        z = beta[0] + X @ beta[1:]
        p = sigmoid(z)
        err = p - y
        grad0 = err.mean()
        grad = (X.T @ err) / n + (l2 / n) * beta[1:]
        beta[0] -= lr * grad0
        beta[1:] -= lr * grad
    return beta



def predict_logistic_prob(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return np.asarray([float(sigmoid(beta[0] + np.dot(X, beta[1:])) )])
    return sigmoid(beta[0] + X @ beta[1:])



def binary_logloss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))



def make_time_series_folds(n_samples: int, n_splits: int = 5, min_train_samples: int = 180, min_val_samples: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < (min_train_samples + min_val_samples):
        return []

    chosen_folds = []
    max_splits = max(1, n_splits)
    for splits in range(max_splits, 0, -1):
        test_size = max(min_val_samples, n_samples // (splits + 1))
        initial_train = n_samples - splits * test_size
        if initial_train < min_train_samples:
            continue
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for j in range(splits):
            train_end = initial_train + j * test_size
            val_start = train_end
            val_end = min(n_samples, val_start + test_size)
            if (train_end < min_train_samples) or (val_end - val_start < min_val_samples):
                continue
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            folds.append((train_idx, val_idx))
        if folds:
            chosen_folds = folds
            break
    return chosen_folds



def cross_validate_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    l2_grid: tuple[float, ...],
    n_splits: int,
    min_train_samples: int,
    min_val_samples: int,
    lr: float,
    epochs: int,
) -> Dict[str, object]:
    folds = make_time_series_folds(
        n_samples=len(X),
        n_splits=n_splits,
        min_train_samples=min_train_samples,
        min_val_samples=min_val_samples,
    )
    if not folds:
        return {"best_l2": None, "cv_score": np.nan, "folds_used": 0, "scores": {}}

    scores: Dict[float, float] = {}
    for l2 in l2_grid:
        fold_losses = []
        for tr_idx, va_idx in folds:
            beta = fit_logistic_l2(X[tr_idx], y[tr_idx], l2=float(l2), lr=lr, epochs=epochs)
            p_val = predict_logistic_prob(X[va_idx], beta)
            fold_losses.append(binary_logloss(y[va_idx], p_val))
        scores[float(l2)] = float(np.mean(fold_losses)) if fold_losses else np.nan

    valid = [(l2, sc) for l2, sc in scores.items() if np.isfinite(sc)]
    if not valid:
        return {"best_l2": None, "cv_score": np.nan, "folds_used": len(folds), "scores": scores}

    best_l2, best_score = min(valid, key=lambda kv: (kv[1], kv[0]))
    return {"best_l2": float(best_l2), "cv_score": float(best_score), "folds_used": len(folds), "scores": scores}



def next_bar_exec_price(raw_open: float, side: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side == "BUY":
        return raw_open * (1.0 + slip)
    if side == "SELL":
        return raw_open * (1.0 - slip)
    raise ValueError(f"unknown side={side}")



def close_exec_price(raw_close: float, side: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side == "BUY":
        return raw_close * (1.0 + slip)
    if side == "SELL":
        return raw_close * (1.0 - slip)
    raise ValueError(f"unknown side={side}")



def in_sell_window(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return (
        (pd.Timestamp("09:45").time() <= t <= pd.Timestamp("10:45").time())
        or (pd.Timestamp("13:00").time() <= t <= pd.Timestamp("14:00").time())
    )



def round_limit_price(price: float) -> float:
    return float(np.round(price + 1e-8, 2))


# =========================
# 数据读取 / 预处理
# =========================


def load_daily_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_map = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols_map]
    if missing:
        raise ValueError(f"daily csv 缺少列: {missing}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[cols_map["date"]]).dt.normalize(),
            "open": pd.to_numeric(df[cols_map["open"]], errors="coerce"),
            "high": pd.to_numeric(df[cols_map["high"]], errors="coerce"),
            "low": pd.to_numeric(df[cols_map["low"]], errors="coerce"),
            "close": pd.to_numeric(df[cols_map["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[cols_map["volume"]], errors="coerce"),
        }
    )
    out["event_flag"] = (
        pd.to_numeric(df[cols_map["event_flag"]], errors="coerce").fillna(0).astype(int)
        if "event_flag" in cols_map
        else 0
    )
    out["no_price_limit_flag"] = (
        pd.to_numeric(df[cols_map["no_price_limit_flag"]], errors="coerce").fillna(0).astype(int)
        if "no_price_limit_flag" in cols_map
        else 0
    )
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out



def load_intraday_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_map = {c.lower(): c for c in df.columns}
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols_map]
    if missing:
        raise ValueError(f"intraday csv 缺少列: {missing}")

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df[cols_map["datetime"]]),
            "open": pd.to_numeric(df[cols_map["open"]], errors="coerce"),
            "high": pd.to_numeric(df[cols_map["high"]], errors="coerce"),
            "low": pd.to_numeric(df[cols_map["low"]], errors="coerce"),
            "close": pd.to_numeric(df[cols_map["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[cols_map["volume"]], errors="coerce"),
        }
    )
    out["date"] = out["datetime"].dt.normalize()
    out = out.sort_values(["date", "datetime"]).reset_index(drop=True)
    out["bar_no"] = out.groupby("date").cumcount() + 1
    return out



def load_benchmark_daily_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_map = {c.lower(): c for c in df.columns}
    required = ["date", "close"]
    missing = [c for c in required if c not in cols_map]
    if missing:
        raise ValueError(f"benchmark daily csv 缺少列: {missing}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[cols_map["date"]]).dt.normalize(),
            "bench_close": pd.to_numeric(df[cols_map["close"]], errors="coerce"),
        }
    )
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


# =========================
# 日线特征 / 标签 / 模型
# =========================


def build_daily_features(
    daily: pd.DataFrame,
    cfg: StrategyConfig,
    meta: MetaConfig,
    bench_daily: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = daily.copy()

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema60"] = df["close"].ewm(span=60, adjust=False).mean()
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    df["atrp14"] = df["atr14"] / (df["close"] + EPS)
    df["rsi14"] = calc_rsi(df["close"], 14)
    df["rvol20"] = df["volume"] / (df["volume"].rolling(20, min_periods=20).mean() + EPS)
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + EPS)
    df["b10_prev"] = df["high"].rolling(10, min_periods=10).max().shift(1)
    df["r5"] = df["high"].rolling(5, min_periods=5).max()
    df["resistance"] = df[["ema20", "r5"]].max(axis=1)

    # OBV
    direction = np.sign(df["close"].diff()).fillna(0.0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma20"] = df["obv"].rolling(20, min_periods=20).mean()
    df["obv_std20"] = df["obv"].rolling(20, min_periods=20).std(ddof=0)

    # 相对强弱 RS5
    if bench_daily is not None:
        df = df.merge(bench_daily, on="date", how="left")
        df["bench_ret5"] = df["bench_close"] / df["bench_close"].shift(5) - 1.0
        df["ret5"] = df["close"] / df["close"].shift(5) - 1.0
        df["rs5"] = df["ret5"] - df["bench_ret5"]
    else:
        df["rs5"] = 0.0

    # 原始特征 x
    df["x1_price_vs_ema20"] = (df["close"] - df["ema20"]) / (df["atr14"] + EPS)
    df["x2_ema20_vs_ema60"] = (df["ema20"] - df["ema60"]) / (df["atr14"] + EPS)
    df["x3_break_b10"] = (df["close"] - df["b10_prev"]) / (df["atr14"] + EPS)
    df["x4_close_pos"] = df["close_pos"]
    df["x5_log_rvol20"] = np.log(df["rvol20"].clip(lower=1e-6))
    df["x6_obv_z"] = (df["obv"] - df["obv_ma20"]) / (df["obv_std20"] + EPS)
    df["x7_rs5"] = df["rs5"]
    df["x8_atrp_rank"] = df["atrp14"].rolling(120, min_periods=60).rank(pct=True)

    raw_feature_cols = [
        "x1_price_vs_ema20",
        "x2_ema20_vs_ema60",
        "x3_break_b10",
        "x4_close_pos",
        "x5_log_rvol20",
        "x6_obv_z",
        "x7_rs5",
        "x8_atrp_rank",
    ]

    # 标准化 z
    z_cols = []
    for c in raw_feature_cols:
        mu = df[c].rolling(cfg.z_window, min_periods=max(60, cfg.z_window // 2)).mean()
        sd = df[c].rolling(cfg.z_window, min_periods=max(60, cfg.z_window // 2)).std(ddof=0)
        zc = f"z_{c}"
        df[zc] = (df[c] - mu) / (sd + EPS)
        z_cols.append(zc)

    # baseline 标签：预测“次日更像趋势延续/启动”
    next_open = df["open"].shift(-1)
    next_high = df["high"].shift(-1)
    next_close = df["close"].shift(-1)
    next_close_pos = df["close_pos"].shift(-1)

    cond_break = ((next_high - np.maximum(df["high"], df["close"])) / (df["atr14"] + EPS)) >= cfg.label_break_atr
    cond_close = ((next_close - df["close"]) / (df["atr14"] + EPS)) >= cfg.label_close_atr
    cond_gap = ((next_open / (df["close"] + EPS)) - 1.0) >= (cfg.label_gap_atrp_mult * df["atrp14"])
    cond_strong_close = next_close_pos >= cfg.label_closepos_min

    df["label_rev"] = ((cond_break & cond_strong_close) | cond_close | cond_gap).astype(float)

    # 滚动训练逻辑回归：在每个滚动窗口内做 5-fold 时间序列交叉验证，
    # 选择最佳 L2 正则强度；随后在该窗口全量样本上重新拟合，并对当日生成 p_rev
    df["p_rev"] = np.nan
    df["cv_best_l2"] = np.nan
    df["cv_logloss"] = np.nan
    df["cv_folds_used"] = np.nan
    df["train_n"] = np.nan
    df["coef_intercept"] = np.nan
    for zc in z_cols:
        df[f"coef_{zc}"] = np.nan

    for i in range(len(df)):
        hist = df.iloc[max(0, i - cfg.train_window):i].copy()
        hist = hist.dropna(subset=z_cols + ["label_rev"])
        if len(hist) < cfg.min_train_samples:
            continue
        if df.loc[i, z_cols].isna().any():
            continue

        X = hist[z_cols].to_numpy(dtype=float)
        y = hist["label_rev"].to_numpy(dtype=float)

        cv_res = cross_validate_logistic_l2(
            X=X,
            y=y,
            l2_grid=cfg.cv_l2_grid,
            n_splits=cfg.cv_folds,
            min_train_samples=cfg.min_train_samples,
            min_val_samples=cfg.cv_min_val_samples,
            lr=cfg.logistic_lr,
            epochs=cfg.logistic_epochs,
        )

        best_l2 = cv_res["best_l2"]
        if best_l2 is None or not np.isfinite(best_l2):
            continue

        beta = fit_logistic_l2(
            X,
            y,
            l2=float(best_l2),
            lr=cfg.logistic_lr,
            epochs=cfg.logistic_epochs,
        )
        x_cur = df.loc[i, z_cols].to_numpy(dtype=float)
        df.loc[i, "p_rev"] = float(predict_logistic_prob(x_cur, beta)[0])
        df.loc[i, "cv_best_l2"] = float(best_l2)
        df.loc[i, "cv_logloss"] = float(cv_res["cv_score"])
        df.loc[i, "cv_folds_used"] = int(cv_res["folds_used"])
        df.loc[i, "train_n"] = int(len(hist))
        df.loc[i, "coef_intercept"] = float(beta[0])
        for j, zc in enumerate(z_cols, start=1):
            df.loc[i, f"coef_{zc}"] = float(beta[j])

    df["atrp14_med60"] = df["atrp14"].rolling(60, min_periods=20).median()
    df["price_limit_ratio"] = float(meta.price_limit_ratio)
    df["no_price_limit_flag"] = df["no_price_limit_flag"].fillna(int(meta.no_price_limit_default)).astype(int)
    return df


# =========================
# 分时特征
# =========================


def add_intraday_day_features(day_df: pd.DataFrame) -> pd.DataFrame:
    g = day_df.copy()
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            g["high"] - g["low"],
            (g["high"] - prev_close).abs(),
            (g["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    g["atr5m12"] = tr.rolling(12, min_periods=3).mean()
    g["rsi5m6"] = calc_rsi(g["close"], 6)
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    g["cum_volume"] = g["volume"].cumsum()
    g["cum_pv"] = (tp * g["volume"]).cumsum()
    g["vwap"] = g["cum_pv"] / (g["cum_volume"] + EPS)
    g["day_open"] = float(g["open"].iloc[0])
    g["cum_ret_from_open"] = g["close"] / (g["day_open"] + EPS) - 1.0
    g["bar_close_pos"] = (g["close"] - g["low"]) / (g["high"] - g["low"] + EPS)
    g["upper_shadow_ratio"] = (g["high"] - np.maximum(g["open"], g["close"])) / (g["high"] - g["low"] + EPS)
    g["dev_vwap"] = (g["close"] - g["vwap"]) / (g["atr5m12"] + EPS)
    g["pull_vwap"] = (g["vwap"] - g["close"]) / (g["atr5m12"] + EPS)
    return g



def build_intraday_features(intraday: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = intraday.copy()
    day_groups = []
    for _, g in df.groupby("date", sort=True):
        day_groups.append(add_intraday_day_features(g))
    df = pd.concat(day_groups, axis=0, ignore_index=True)

    # 同 bar_no 历史统计（过去 slot_lookback_days 天）
    lookback = cfg.slot_lookback_days
    minp = cfg.slot_min_periods

    grp = df.groupby("bar_no", group_keys=False)
    df["avg_slot_volume"] = grp["volume"].transform(lambda s: s.shift(1).rolling(lookback, min_periods=minp).mean())
    df["avg_slot_cum_volume"] = grp["cum_volume"].transform(lambda s: s.shift(1).rolling(lookback, min_periods=minp).mean())
    df["slot_vol_ratio"] = df["volume"] / (df["avg_slot_volume"] + EPS)
    df["cum_vol_ratio"] = df["cum_volume"] / (df["avg_slot_cum_volume"] + EPS)
    df["dev_vwap_q70"] = grp["dev_vwap"].transform(lambda s: s.shift(1).rolling(lookback, min_periods=minp).quantile(0.70))
    df["pull_vwap_q60"] = grp["pull_vwap"].transform(lambda s: s.shift(1).rolling(lookback, min_periods=minp).quantile(0.60))
    df["rsi_pos"] = grp["rsi5m6"].transform(lambda s: rolling_pct_rank(s, lookback, minp))
    return df


# =========================
# 信号函数
# =========================


def classify_regime(p_rev: float, cfg: StrategyConfig) -> str:
    if not np.isfinite(p_rev):
        return "NEUTRAL"
    if p_rev >= cfg.reversal_prob_hi:
        return "REVERSAL"
    if p_rev <= cfg.reversal_prob_lo:
        return "REBOUND"
    return "NEUTRAL"



def required_spread(cfg: StrategyConfig) -> float:
    return cfg.cost_buy_rate + cfg.cost_sell_rate + cfg.safety_buffer



def sell_signal(
    bar: pd.Series,
    prev_day: pd.Series,
    cfg: StrategyConfig,
    dist_to_limit_ok: bool,
) -> bool:
    if not dist_to_limit_ok:
        return False
    if not np.isfinite(bar["dev_vwap"]) or not np.isfinite(bar["atr5m12"]):
        return False
    if not np.isfinite(bar["avg_slot_volume"]) or not np.isfinite(bar["avg_slot_cum_volume"]):
        return False

    s1 = bar["dev_vwap"] >= max(cfg.sell_dev_vwap_floor, float(bar["dev_vwap_q70"]) if np.isfinite(bar["dev_vwap_q70"]) else -np.inf)
    s2 = (bar["cum_vol_ratio"] >= cfg.sell_cum_vol_ratio_min) and (bar["slot_vol_ratio"] >= cfg.sell_slot_vol_ratio_min)
    s3 = (bar["upper_shadow_ratio"] >= cfg.sell_upper_shadow_min) or (bar["bar_close_pos"] <= cfg.sell_bar_close_pos_max)
    s4 = bar["cum_ret_from_open"] >= max(required_spread(cfg) + cfg.sell_amp_vs_open_extra, 0.5 * float(prev_day["atrp14"]))
    return bool(s1 and s2 and s3 and s4)



def buy_first_leg_signal(bar: pd.Series, sell_price: float, cfg: StrategyConfig) -> bool:
    if not np.isfinite(bar["pull_vwap"]) or not np.isfinite(bar["rsi_pos"]):
        return False
    b1 = bar["pull_vwap"] >= max(cfg.buy_pull_vwap_floor, float(bar["pull_vwap_q60"]) if np.isfinite(bar["pull_vwap_q60"]) else -np.inf)
    b2 = bar["rsi_pos"] <= cfg.buy_rsi_pos_max
    b3 = bar["slot_vol_ratio"] <= cfg.buy_slot_vol_ratio_max
    b4 = ((sell_price - float(bar["close"])) / (sell_price + EPS)) >= required_spread(cfg)
    return bool(b1 and b2 and b3 and b4)



def buy_second_leg_signal(bar: pd.Series, sell_price: float, cfg: StrategyConfig) -> bool:
    b1 = bar["bar_close_pos"] >= cfg.buy_second_leg_bar_close_pos_min
    b2 = ((sell_price - float(bar["close"])) / (sell_price + EPS)) >= required_spread(cfg)
    return bool(b1 and b2)



def force_cover_signal(day_bars: pd.DataFrame, idx: int, sell_price: float, cfg: StrategyConfig) -> bool:
    bar = day_bars.iloc[idx]
    if not np.isfinite(bar["atr5m12"]):
        return False
    cond1 = float(bar["close"]) >= sell_price + cfg.force_cover_atr_mult * float(bar["atr5m12"])
    cond2 = float(bar["cum_vol_ratio"]) >= cfg.force_cover_cum_vol_ratio
    n = cfg.force_cover_consecutive_bars
    if idx + 1 < n:
        return False
    recent = day_bars.iloc[idx - n + 1: idx + 1]
    cond3 = bool((recent["close"] > recent["vwap"]).all())
    return bool(cond1 and cond2 and cond3)


# =========================
# 回测主逻辑
# =========================


def backtest_t_strategy(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    cfg: StrategyConfig,
    meta: MetaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    daily = daily.copy().reset_index(drop=True)
    intraday = intraday.copy().reset_index(drop=True)

    eval_start_date = pd.to_datetime(cfg.evaluation_start_date).normalize() if cfg.evaluation_start_date else None
    if eval_start_date is None and not intraday.empty:
        eval_start_date = pd.to_datetime(intraday["date"].min()).normalize()

    daily_idx = {d: i for i, d in enumerate(daily["date"])}

    cash = float(cfg.initial_cash)
    position = int(cfg.initial_shares)

    trades: List[Dict] = []
    equity_rows: List[Dict] = []

    completed_cycles = 0
    winning_cycles = 0
    forced_rebuys = 0

    eval_start_equity = np.nan
    eval_start_open = np.nan
    eval_start_trade_date = pd.NaT

    for trade_date, day_bars in intraday.groupby("date", sort=True):
        if eval_start_date is not None and pd.to_datetime(trade_date).normalize() < eval_start_date:
            continue
        if trade_date not in daily_idx:
            continue
        di = daily_idx[trade_date]
        if di < 1:
            # 第一天没有前一交易日上下文，直接记权益
            close_px = float(day_bars.iloc[-1]["close"])
            equity = cash + position * close_px
            equity_rows.append(
                {
                    "date": trade_date,
                    "cash": cash,
                    "position": position,
                    "close": close_px,
                    "equity": equity,
                }
            )
            continue

        prev_day = daily.iloc[di - 1]
        curr_day = daily.iloc[di]
        p_rev = float(prev_day["p_rev"]) if np.isfinite(prev_day["p_rev"]) else np.nan
        regime = classify_regime(p_rev, cfg)

        # T+1：今天可卖的是昨夜持有仓位；当天买回的仓不可再卖（本脚本每天最多做一轮）
        available_yday_shares = floor_lot(position, meta.lot_size)

        gap_abs = abs(float(curr_day["open"]) / (float(prev_day["close"]) + EPS) - 1.0)
        skip_day = (
            bool(int(curr_day.get("event_flag", 0)) == 1)
            or bool(int(curr_day.get("no_price_limit_flag", 0)) == 1)
            or bool(meta.no_price_limit_default)
            or (gap_abs >= cfg.skip_gap_atr_mult * float(prev_day["atrp14"]) if np.isfinite(prev_day["atrp14"]) else False)
            or (available_yday_shares < meta.lot_size)
            or (float(prev_day["rvol20"]) < cfg.min_prev_rvol20 if np.isfinite(prev_day["rvol20"]) else True)
        )

        prev_close_px = float(prev_day["close"])
        up_limit = round_limit_price(prev_close_px * (1.0 + float(curr_day["price_limit_ratio"])))
        down_limit = round_limit_price(prev_close_px * (1.0 - float(curr_day["price_limit_ratio"])))

        sold_qty = 0
        remaining_qty = 0
        first_leg_qty = 0
        sell_exec_price = np.nan
        cycle_pnl = 0.0
        cycle_open = False
        first_leg_done = False

        day_bars = day_bars.reset_index(drop=True)

        if not np.isfinite(eval_start_equity):
            eval_start_open = float(day_bars.iloc[0]["open"])
            eval_start_equity = float(cash + position * eval_start_open)
            eval_start_trade_date = pd.to_datetime(trade_date)

        if not skip_day and regime == "REBOUND":
            for i in range(len(day_bars) - 1):
                bar = day_bars.iloc[i]
                next_bar = day_bars.iloc[i + 1]

                # 当前 bar 是否过近涨跌停；只用于阻止新的卖出型 T
                dist_to_up = (up_limit - float(bar["close"])) / (prev_close_px + EPS)
                dist_to_down = (float(bar["close"]) - down_limit) / (prev_close_px + EPS)
                dist_to_limit_ok = min(dist_to_up, dist_to_down) >= cfg.min_dist_to_limit

                if (not cycle_open) and in_sell_window(pd.Timestamp(bar["datetime"])):
                    if sell_signal(bar, prev_day, cfg, dist_to_limit_ok=dist_to_limit_ok):
                        rebound_conf = float(np.clip((0.50 - p_rev) / 0.20, 0.0, 1.0)) if np.isfinite(p_rev) else 0.0
                        atrp_med60 = float(prev_day["atrp14_med60"]) if np.isfinite(prev_day["atrp14_med60"]) else float(prev_day["atrp14"])
                        atrp_now = float(prev_day["atrp14"]) if np.isfinite(prev_day["atrp14"]) else np.nan
                        if not np.isfinite(atrp_now) or atrp_now <= 0:
                            continue
                        vol_adj = float(np.clip(atrp_med60 / (atrp_now + EPS), cfg.vol_adj_min, cfg.vol_adj_max)) if np.isfinite(atrp_med60) else 1.0
                        sell_pct = float(np.clip((cfg.base_sell_pct + cfg.rebound_conf_sell_pct_span * rebound_conf) * vol_adj, cfg.min_sell_pct, cfg.max_sell_pct))
                        qty_by_pct = floor_lot(available_yday_shares * sell_pct, meta.lot_size)
                        liq_cap_qty = floor_lot(cfg.liq_cap_slot_volume_pct * float(bar["avg_slot_volume"]), meta.lot_size)
                        qty = min(qty_by_pct, liq_cap_qty, available_yday_shares)
                        if qty < meta.lot_size:
                            continue

                        exec_px = next_bar_exec_price(float(next_bar["open"]), side="SELL", slippage_bps=cfg.slippage_bps)
                        cash += qty * exec_px * (1.0 - cfg.cost_sell_rate)
                        position -= qty

                        sold_qty = qty
                        remaining_qty = qty
                        first_leg_qty = floor_lot(qty * 0.5, meta.lot_size)
                        if first_leg_qty == 0 and qty >= meta.lot_size:
                            first_leg_qty = meta.lot_size
                        sell_exec_price = exec_px
                        cycle_open = True

                        trades.append(
                            {
                                "date": trade_date,
                                "datetime": next_bar["datetime"],
                                "action": "SELL",
                                "qty": qty,
                                "price": exec_px,
                                "regime": regime,
                                "p_rev": p_rev,
                                "reason": "SELL_T",
                            }
                        )
                        continue

                if cycle_open and remaining_qty > 0:
                    # 强制回补优先级最高
                    if force_cover_signal(day_bars, i, sell_exec_price, cfg):
                        qty = remaining_qty
                        exec_px = next_bar_exec_price(float(next_bar["open"]), side="BUY", slippage_bps=cfg.slippage_bps)
                        cash -= qty * exec_px * (1.0 + cfg.cost_buy_rate)
                        position += qty
                        cycle_pnl += qty * (sell_exec_price * (1.0 - cfg.cost_sell_rate) - exec_px * (1.0 + cfg.cost_buy_rate))
                        remaining_qty = 0
                        cycle_open = False
                        first_leg_done = True
                        forced_rebuys += 1
                        trades.append(
                            {
                                "date": trade_date,
                                "datetime": next_bar["datetime"],
                                "action": "BUY",
                                "qty": qty,
                                "price": exec_px,
                                "regime": regime,
                                "p_rev": p_rev,
                                "reason": "FORCE_COVER",
                            }
                        )
                        completed_cycles += 1
                        if cycle_pnl > 0:
                            winning_cycles += 1
                        continue

                    if (not first_leg_done) and buy_first_leg_signal(bar, sell_exec_price, cfg):
                        qty = min(first_leg_qty, remaining_qty)
                        if qty >= meta.lot_size:
                            exec_px = next_bar_exec_price(float(next_bar["open"]), side="BUY", slippage_bps=cfg.slippage_bps)
                            cash -= qty * exec_px * (1.0 + cfg.cost_buy_rate)
                            position += qty
                            cycle_pnl += qty * (sell_exec_price * (1.0 - cfg.cost_sell_rate) - exec_px * (1.0 + cfg.cost_buy_rate))
                            remaining_qty -= qty
                            first_leg_done = True
                            trades.append(
                                {
                                    "date": trade_date,
                                    "datetime": next_bar["datetime"],
                                    "action": "BUY",
                                    "qty": qty,
                                    "price": exec_px,
                                    "regime": regime,
                                    "p_rev": p_rev,
                                    "reason": "BUYBACK_1",
                                }
                            )
                            if remaining_qty == 0:
                                cycle_open = False
                                completed_cycles += 1
                                if cycle_pnl > 0:
                                    winning_cycles += 1
                            continue

                    if first_leg_done and remaining_qty > 0 and buy_second_leg_signal(bar, sell_exec_price, cfg):
                        qty = remaining_qty
                        exec_px = next_bar_exec_price(float(next_bar["open"]), side="BUY", slippage_bps=cfg.slippage_bps)
                        cash -= qty * exec_px * (1.0 + cfg.cost_buy_rate)
                        position += qty
                        cycle_pnl += qty * (sell_exec_price * (1.0 - cfg.cost_sell_rate) - exec_px * (1.0 + cfg.cost_buy_rate))
                        remaining_qty = 0
                        cycle_open = False
                        trades.append(
                            {
                                "date": trade_date,
                                "datetime": next_bar["datetime"],
                                "action": "BUY",
                                "qty": qty,
                                "price": exec_px,
                                "regime": regime,
                                "p_rev": p_rev,
                                "reason": "BUYBACK_2",
                            }
                        )
                        completed_cycles += 1
                        if cycle_pnl > 0:
                            winning_cycles += 1
                        continue

        # 收盘强制回补，保证“底仓 + 做T”回测闭环
        if cycle_open and remaining_qty > 0 and cfg.force_rebuy_at_close:
            last_bar = day_bars.iloc[-1]
            qty = remaining_qty
            exec_px = close_exec_price(float(last_bar["close"]), side="BUY", slippage_bps=cfg.slippage_bps)
            cash -= qty * exec_px * (1.0 + cfg.cost_buy_rate)
            position += qty
            cycle_pnl += qty * (sell_exec_price * (1.0 - cfg.cost_sell_rate) - exec_px * (1.0 + cfg.cost_buy_rate))
            remaining_qty = 0
            cycle_open = False
            trades.append(
                {
                    "date": trade_date,
                    "datetime": last_bar["datetime"],
                    "action": "BUY",
                    "qty": qty,
                    "price": exec_px,
                    "regime": regime,
                    "p_rev": p_rev,
                    "reason": "FORCE_REBUY_CLOSE",
                }
            )
            completed_cycles += 1
            if cycle_pnl > 0:
                winning_cycles += 1

        close_px = float(day_bars.iloc[-1]["close"])
        equity = cash + position * close_px
        bh_equity = cfg.initial_cash + cfg.initial_shares * close_px
        equity_rows.append(
            {
                "date": trade_date,
                "cash": cash,
                "position": position,
                "close": close_px,
                "equity": equity,
                "buy_hold_equity": bh_equity,
                "alpha_vs_bh": equity - bh_equity,
                "regime": regime,
                "p_rev": p_rev,
                "skip_day": int(skip_day),
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    if not equity_df.empty:
        equity_df["daily_ret"] = equity_df["equity"].pct_change()

    summary = {
        "training_start_date": str(pd.to_datetime(daily["date"].min()).date()) if not daily.empty else None,
        "evaluation_start_date": str(pd.to_datetime(eval_start_trade_date).date()) if pd.notna(eval_start_trade_date) else None,
        "evaluation_start_open": float(eval_start_open) if np.isfinite(eval_start_open) else np.nan,
        "initial_equity": float(eval_start_equity) if np.isfinite(eval_start_equity) else np.nan,
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else np.nan,
        "total_return": float(equity_df["equity"].iloc[-1] / eval_start_equity - 1.0) if (not equity_df.empty and np.isfinite(eval_start_equity) and eval_start_equity != 0) else np.nan,
        "buy_hold_final": float(equity_df["buy_hold_equity"].iloc[-1]) if not equity_df.empty else np.nan,
        "alpha_vs_buy_hold": float(equity_df["alpha_vs_bh"].iloc[-1]) if not equity_df.empty else np.nan,
        "max_drawdown": compute_max_drawdown(equity_df["equity"]) if not equity_df.empty else np.nan,
        "annualized_sharpe": annualized_sharpe(equity_df["daily_ret"]) if not equity_df.empty else np.nan,
        "num_trades": int(len(trades_df)),
        "num_sell_cycles": int((trades_df["action"] == "SELL").sum()) if not trades_df.empty else 0,
        "completed_cycles": int(completed_cycles),
        "winning_cycles": int(winning_cycles),
        "cycle_win_rate": float(winning_cycles / completed_cycles) if completed_cycles > 0 else np.nan,
        "forced_rebuys": int(forced_rebuys),
        "ending_position": int(position),
        "ending_cash": float(cash),
    }
    return trades_df, equity_df, summary


# =========================
# 命令行 / 输出
# =========================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T+1 做T 策略回测脚本")
    p.add_argument("--daily_csv", required=True, help="日线 CSV 路径")
    p.add_argument("--intraday_csv", required=True, help="5 分钟 CSV 路径")
    p.add_argument("--benchmark_daily_csv", default=None, help="可选，基准日线 CSV")
    p.add_argument("--output_dir", default="./bt_out", help="输出目录")
    p.add_argument("--perf_start", default=None, help="评估起点 YYYY-MM-DD；默认使用 intraday csv 首个交易日")

    p.add_argument("--initial_shares", type=int, default=2000)
    p.add_argument("--initial_cash", type=float, default=0.0)
    p.add_argument("--lot_size", type=int, default=100)
    p.add_argument("--price_limit_ratio", type=float, default=0.10)
    p.add_argument("--cost_buy_rate", type=float, default=0.00035)
    p.add_argument("--cost_sell_rate", type=float, default=0.00135)
    p.add_argument("--slippage_bps", type=float, default=2.0)
    p.add_argument("--cv_folds", type=int, default=5, help="时间序列交叉验证折数")
    p.add_argument("--cv_l2_grid", type=str, default="0.01,0.1,0.3,1,3,10", help="L2 候选网格，逗号分隔")
    p.add_argument("--force_rebuy_at_close", type=int, default=1, help="1/0")
    p.add_argument("--verbose", type=int, default=1, help="1/0")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = MetaConfig(
        lot_size=args.lot_size,
        price_limit_ratio=args.price_limit_ratio,
    )
    cv_l2_grid = tuple(float(x.strip()) for x in str(args.cv_l2_grid).split(',') if str(x).strip())

    cfg = StrategyConfig(
        initial_shares=args.initial_shares,
        initial_cash=args.initial_cash,
        evaluation_start_date=args.perf_start,
        cost_buy_rate=args.cost_buy_rate,
        cost_sell_rate=args.cost_sell_rate,
        slippage_bps=args.slippage_bps,
        cv_folds=args.cv_folds,
        cv_l2_grid=cv_l2_grid,
        force_rebuy_at_close=bool(args.force_rebuy_at_close),
        verbose=bool(args.verbose),
    )

    daily = load_daily_csv(args.daily_csv)
    intraday = load_intraday_csv(args.intraday_csv)
    bench = load_benchmark_daily_csv(args.benchmark_daily_csv) if args.benchmark_daily_csv else None

    daily_feat = build_daily_features(daily, cfg=cfg, meta=meta, bench_daily=bench)
    intraday_feat = build_intraday_features(intraday, cfg=cfg)

    trades_df, equity_df, summary = backtest_t_strategy(
        daily=daily_feat,
        intraday=intraday_feat,
        cfg=cfg,
        meta=meta,
    )

    daily_feat.to_csv(outdir / "daily_features.csv", index=False)
    intraday_feat.to_csv(outdir / "intraday_features.csv", index=False)
    trades_df.to_csv(outdir / "trades.csv", index=False)
    equity_df.to_csv(outdir / "equity_curve.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"meta": asdict(meta), "strategy": asdict(cfg)}, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("回测完成")
    print(f"输出目录: {outdir.resolve()}")
    print("- daily_features.csv")
    print("- intraday_features.csv")
    print("- trades.csv")
    print("- equity_curve.csv")
    print("- summary.json")
    print("- config.json")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
