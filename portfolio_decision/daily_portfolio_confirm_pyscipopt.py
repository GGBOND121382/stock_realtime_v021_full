#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_decision/daily_portfolio_confirm_pyscipopt.py

PySCIPOpt-based daily portfolio confirmation optimizer.

It consumes normalized signal/metric/account/price/history files and outputs
final buy orders under:
  - max number of final positions
  - A-share 100-share lots
  - cash budget
  - min trade amount
  - stock/sector/model-type caps
  - volatility contribution constraint
  - historical scenario loss constraint
  - high-correlation blocking
  - optional covariance penalty

This is the low-level optimizer. In normal use, run:
  scripts/run_portfolio_confirm_from_signals.sh
or:
  python3 portfolio_decision/portfolio_confirm_from_buy_signals.py ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pyscipopt import Model, quicksum
except Exception as exc:
    Model = None
    quicksum = None
    _PYSCIPOPT_IMPORT_ERROR = exc
else:
    _PYSCIPOPT_IMPORT_ERROR = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "buy_commission_bps": 0.85,
    "sell_commission_bps": 0.85,
    "stamp_tax_bps": 10.0,
    "slippage_bps_buy": 2.0,
    "slippage_bps_sell": 2.0,
    "round_trip_cost_bps": None,
    "safety_margin_bps": 8.0,

    "min_ev_bps_close": 15.0,
    "min_ev_bps_hit": 10.0,
    "min_utility_bps": 0.0,

    "max_positions": 7,
    "max_daily_buy_pct_of_total_asset": 0.30,
    "max_daily_buy_pct_of_cash": 0.70,
    "max_policy_weight": 0.15,
    "min_trade_amount": 6000.0,
    "lot_size": 100,

    "max_new_hit_buys": 1,
    "max_new_observation_buys": 1,

    "vol_window": 60,
    "corr_window": 60,
    "scenario_window": 120,
    "max_pair_corr": 0.75,
    "max_new_position_vol_contribution": 0.0035,
    "max_scenario_loss_pct": 0.012,

    "use_covariance_penalty": False,
    "cov_risk_aversion": 3.0,
    "covariance_penalty_mode": "linear",
    "cov_linear_self_weight": 0.05,
    "vol_penalty_lambda": 0.02,
    "dd_penalty_lambda": 0.02,

    "default_profit_factor": 1.30,
    "default_trades": 100,
    "default_median_return_bps": 0.0,
    "default_max_drawdown": -0.12,
    "default_fail_loss_bps": 60.0,
    "default_hit_prob": 0.60,

    "tier_multiplier": {"1": 1.00, "2": 0.85, "3": 0.70, "4": 0.40},
    "default_tier": 2,
    "default_max_weight_by_tier": {"1": 0.20, "2": 0.15, "3": 0.12, "4": 0.05},
    "default_max_add_weight_by_tier": {"1": 0.15, "2": 0.10, "3": 0.08, "4": 0.05},
    "sector_limits": {},

    "stock_overrides": {
        "600312.SH": {"tier": 1, "max_weight": 0.20, "max_add_weight": 0.15},
        "601899.SH": {"tier": 1, "max_weight": 0.20, "max_add_weight": 0.15},
        "603308.SH": {"tier": 1, "max_weight": 0.15, "max_add_weight": 0.12},
        "600276.SH": {"tier": 2, "max_weight": 0.10, "max_add_weight": 0.08, "model_type": "hit"},
        "600096.SH": {"tier": 2, "max_weight": 0.15, "max_add_weight": 0.10},
        "002311.SZ": {"tier": 2, "max_weight": 0.15, "max_add_weight": 0.10},
        "601985.SH": {"tier": 3, "max_weight": 0.12, "max_add_weight": 0.08},
        "002714.SZ": {"tier": 4, "max_weight": 0.05, "max_add_weight": 0.05, "model_type": "observation"},
    },
}


@dataclass
class Candidate:
    stock_code: str
    model_name: str
    label_mode: str
    model_type: str
    sector: str
    price: float
    pred_return_bps: float
    pred_prob: float
    target_hit_bps: float
    fail_loss_bps: float
    entry_policy: str
    entry_vwap_premium_bps: float
    samples: str
    expected_return_col: str
    metadata_path: str
    ev_bps: float
    quality_weight: float
    tier: int
    tier_multiplier: float
    vol_daily: float
    vol_bps: float
    max_drawdown_abs: float
    max_drawdown_bps: float
    profit_factor: float
    trades: float
    median_return_bps: float
    utility_bps: float
    enabled: bool
    weight_multiplier: float
    override_reason: str
    current_value: float
    current_shares: int
    held: bool
    max_weight: float
    max_add_weight: float
    min_trade_amount: float
    min_lots: int
    max_lots: int
    reason: str = "ok"


def normalize_stock_code(x: Any) -> str:
    s = str(x).strip().upper()
    if not s:
        return s
    if s.isdigit() and len(s) == 6:
        return f"{s}.SH" if s.startswith(("6", "9")) else f"{s}.SZ"
    if s.startswith("SH."):
        return f"{s[3:]}.SH"
    if s.startswith("SZ."):
        return f"{s[3:]}.SZ"
    return s


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def parse_rate_decimal(x: Any, default: float = 0.0) -> float:
    s = str(default if x is None else x).strip().replace("%", "")
    val = as_float(s, default)
    if abs(val) > 1.5:
        val /= 100.0
    return val


def parse_bool_flag(x: Any, default: bool = True) -> bool:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip().lower()
    if s == "":
        return default
    if s in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def parse_drawdown_abs(x: Any, default: float = 0.12) -> float:
    val = as_float(x, default)
    if abs(val) > 1.5:
        val /= 100.0
    return abs(val)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def compute_round_trip_cost_bps(cfg: Dict[str, Any]) -> float:
    if cfg.get("round_trip_cost_bps") is not None:
        return float(cfg["round_trip_cost_bps"])
    return (
        float(cfg.get("buy_commission_bps", 0.85))
        + float(cfg.get("sell_commission_bps", 0.85))
        + float(cfg.get("stamp_tax_bps", 10.0))
        + float(cfg.get("slippage_bps_buy", 2.0))
        + float(cfg.get("slippage_bps_sell", 2.0))
    )


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df, ["stock_code", "model_name", "label_mode"], "signals")
    if df.empty:
        return df
    df["stock_code"] = df["stock_code"].map(normalize_stock_code)
    df["model_name"] = df["model_name"].astype(str)
    df["label_mode"] = df["label_mode"].astype(str)
    return df


def load_metrics(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["stock_code", "model_name"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["stock_code", "model_name"])
    ensure_columns(df, ["stock_code", "model_name"], "metrics")
    df["stock_code"] = df["stock_code"].map(normalize_stock_code)
    df["model_name"] = df["model_name"].astype(str)
    return df


def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["stock_code", "price"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["stock_code", "price"])
    ensure_columns(df, ["stock_code", "price"], "prices")
    df["stock_code"] = df["stock_code"].map(normalize_stock_code)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["stock_code", "price"]].dropna()


def normalize_holdings(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    holdings = raw.get("holdings", {})
    if isinstance(holdings, list):
        return {normalize_stock_code(h.get("stock_code", "")): dict(h) for h in holdings if h.get("stock_code")}
    if isinstance(holdings, dict):
        return {normalize_stock_code(k): dict(v or {}) for k, v in holdings.items()}
    return {}


def load_account(path: str) -> Dict[str, Any]:
    data = load_json(path)
    if "total_asset" not in data or "available_cash" not in data:
        raise ValueError("account.json must contain total_asset and available_cash")
    data["total_asset"] = float(data["total_asset"])
    data["available_cash"] = float(data["available_cash"])
    data["holdings"] = normalize_holdings(data)
    return data


def load_history(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    if "date" not in df.columns:
        raise ValueError("history CSV must contain date column")
    df["date"] = pd.to_datetime(df["date"])
    if {"stock_code", "close"}.issubset(df.columns):
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
        wide = df.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
    else:
        wide = df.set_index("date")
        wide.columns = [normalize_stock_code(c) for c in wide.columns]
    return wide.sort_index().apply(pd.to_numeric, errors="coerce")


def compute_returns(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    return history.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def compute_vol_map(returns: pd.DataFrame, window: int) -> Dict[str, float]:
    if returns.empty:
        return {}
    vols = returns.tail(window).std(skipna=True)
    return {normalize_stock_code(k): float(v) for k, v in vols.items() if pd.notna(v)}


def compute_corr_pairs(returns: pd.DataFrame, codes: List[str], window: int, threshold: float) -> List[Tuple[str, str, float]]:
    if returns.empty:
        return []
    codes = [c for c in codes if c in returns.columns]
    if len(codes) < 2:
        return []
    corr = returns[codes].tail(window).corr()
    out = []
    for i, ci in enumerate(codes):
        for cj in codes[i + 1:]:
            val = corr.loc[ci, cj]
            if pd.notna(val) and val >= threshold:
                out.append((ci, cj, float(val)))
    return out


def compute_cov_matrix(returns: pd.DataFrame, codes: List[str], window: int) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    codes = [c for c in codes if c in returns.columns]
    if not codes:
        return pd.DataFrame()
    cov = returns[codes].tail(window).cov().fillna(0.0)
    if cov.shape[0] > 1:
        diag = np.diag(np.diag(cov.values))
        cov_values = 0.90 * cov.values + 0.10 * diag
        cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)
    return cov


def get_scenario_returns(returns: pd.DataFrame, codes: List[str], window: int) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    codes = [c for c in codes if c in returns.columns]
    if not codes:
        return pd.DataFrame()
    return returns[codes].tail(window).fillna(0.0)


def merge_inputs(signals: pd.DataFrame, metrics: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = signals.copy()
    if not metrics.empty:
        df = df.merge(metrics, on=["stock_code", "model_name"], how="left", suffixes=("", "_metric"))
    if "price" not in df.columns:
        df["price"] = np.nan
    if not prices.empty:
        df = df.merge(prices, on="stock_code", how="left", suffixes=("", "_pricefile"))
        df["price"] = df["price"].where(df["price"].notna(), df["price_pricefile"])
        df = df.drop(columns=[c for c in ["price_pricefile"] if c in df.columns])
    return df


def get_row_field(row: pd.Series, name: str, default: Any = None) -> Any:
    if name in row.index and pd.notna(row[name]):
        return row[name]
    return default


def infer_model_type(label_mode: str, row: pd.Series, cfg: Dict[str, Any], stock_code: str) -> str:
    override = cfg.get("stock_overrides", {}).get(stock_code, {})
    if "model_type" in override:
        return str(override["model_type"]).lower()
    if "model_type" in row.index and pd.notna(row["model_type"]):
        return str(row["model_type"]).lower()
    label = str(label_mode).lower()
    if label == "hit" or label.startswith("hit"):
        return "hit"
    return "close"


def get_tier_and_caps(row: pd.Series, stock_code: str, model_type: str, cfg: Dict[str, Any]) -> Tuple[int, float, float]:
    override = cfg.get("stock_overrides", {}).get(stock_code, {})
    tier = int(override.get("tier", get_row_field(row, "tier", cfg["default_tier"])))
    tier_s = str(tier)
    default_max_w = cfg["default_max_weight_by_tier"].get(tier_s, 0.15)
    default_max_add = cfg["default_max_add_weight_by_tier"].get(tier_s, 0.10)
    max_weight = as_float(override.get("max_weight", get_row_field(row, "max_weight", default_max_w)), default_max_w)
    max_add_weight = as_float(override.get("max_add_weight", get_row_field(row, "max_add_weight", default_max_add)), default_max_add)
    max_policy_weight = cfg.get("max_policy_weight")
    if max_policy_weight is not None:
        max_weight = min(max_weight, as_float(max_policy_weight, max_weight))

    if model_type == "hit":
        max_weight = min(max_weight, 0.10)
        max_add_weight = min(max_add_weight, 0.08)
    if model_type == "observation":
        max_weight = min(max_weight, 0.05)
        max_add_weight = min(max_add_weight, 0.05)
    return tier, max_weight, max_add_weight


def calc_quality_weight(row: pd.Series, cfg: Dict[str, Any]) -> float:
    pf = as_float(get_row_field(row, "profit_factor", cfg["default_profit_factor"]), cfg["default_profit_factor"])
    trades = as_float(get_row_field(row, "trades", cfg["default_trades"]), cfg["default_trades"])
    median = as_float(get_row_field(row, "median_return_bps", cfg["default_median_return_bps"]), cfg["default_median_return_bps"])
    dd_abs = parse_drawdown_abs(get_row_field(row, "max_drawdown", cfg["default_max_drawdown"]), abs(cfg["default_max_drawdown"]))
    q_pf = clip(pf / 1.8, 0.50, 1.20)
    q_dd = clip(0.10 / max(dd_abs, 1e-6), 0.50, 1.20)
    q_n = clip(math.sqrt(max(trades, 1.0) / 150.0), 0.60, 1.10)
    q_median = 1.00 if median > 0 else 0.50
    return q_pf * q_dd * q_n * q_median


def build_candidates(df: pd.DataFrame, account: Dict[str, Any], vol_map: Dict[str, float], cfg: Dict[str, Any]) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    rejected: List[Dict[str, Any]] = []
    candidates: List[Candidate] = []
    total_asset = float(account["total_asset"])
    holdings = account.get("holdings", {})
    round_trip = compute_round_trip_cost_bps(cfg)
    margin = float(cfg.get("safety_margin_bps", 8.0))
    lot_size = int(cfg.get("lot_size", 100))
    min_trade_amount_default = float(cfg.get("min_trade_amount", 6000.0))

    for _, row in df.iterrows():
        code = normalize_stock_code(row["stock_code"])
        model_name = str(row["model_name"])
        label_mode = str(row["label_mode"])
        price = as_float(get_row_field(row, "price", np.nan), np.nan)
        if not np.isfinite(price) or price <= 0:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "missing_or_invalid_price"})
            continue

        enabled = parse_bool_flag(get_row_field(row, "enabled", 1), True)
        weight_multiplier = as_float(get_row_field(row, "weight_multiplier", 1.0), 1.0)
        override_reason = str(get_row_field(row, "model_override_reason", get_row_field(row, "recent_perf_note", "")) or "")
        if not enabled:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "disabled_by_override", "override_reason": override_reason})
            continue
        if weight_multiplier <= 0:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "non_positive_weight_multiplier", "weight_multiplier": weight_multiplier})
            continue

        model_type = infer_model_type(label_mode, row, cfg, code)
        sector = str(get_row_field(row, "sector", holdings.get(code, {}).get("sector", "UNKNOWN")))

        holding = holdings.get(code, {})
        current_value = as_float(holding.get("market_value", 0.0), 0.0)
        current_shares = int(as_float(holding.get("shares", 0), 0))
        held = current_value > 0 or current_shares > 0

        pred_return_bps = as_float(get_row_field(row, "pred_return_bps", np.nan), np.nan)
        if not np.isfinite(pred_return_bps):
            pred_return_bps = as_float(get_row_field(row, "score", 0.0), 0.0)

        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)
        entry_policy = as_text(get_row_field(row, "entry_policy", ""))
        entry_vwap_premium_bps = as_float(get_row_field(row, "entry_vwap_premium_bps", 50.0), 50.0)
        samples = as_text(get_row_field(row, "samples", ""))
        expected_return_col = as_text(get_row_field(row, "expected_return_col", ""))
        metadata_path = as_text(get_row_field(row, "metadata_path", ""))
        pred_prob = parse_rate_decimal(get_row_field(row, "pred_prob", get_row_field(row, "win_rate", cfg["default_hit_prob"])), cfg["default_hit_prob"])
        fail_loss_bps = as_float(get_row_field(row, "fail_loss_bps", cfg["default_fail_loss_bps"]), cfg["default_fail_loss_bps"])

        if model_type == "hit":
            ev_bps = pred_prob * target_hit_bps - (1 - pred_prob) * fail_loss_bps - round_trip - margin
            min_ev = float(cfg.get("min_ev_bps_hit", 10.0))
        else:
            ev_bps = pred_return_bps - round_trip - margin
            min_ev = float(cfg.get("min_ev_bps_close", 15.0))
        if ev_bps < min_ev:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "ev_below_threshold", "ev_bps": ev_bps, "threshold": min_ev})
            continue

        quality = calc_quality_weight(row, cfg)
        tier, max_weight, max_add_weight = get_tier_and_caps(row, code, model_type, cfg)
        max_weight_override = as_float(get_row_field(row, "max_weight_override", np.nan), np.nan)
        max_add_weight_override = as_float(get_row_field(row, "max_add_weight_override", np.nan), np.nan)
        if np.isfinite(max_weight_override) and max_weight_override > 0:
            max_weight = min(float(max_weight_override), as_float(cfg.get("max_policy_weight", max_weight), max_weight))
        if np.isfinite(max_add_weight_override) and max_add_weight_override > 0:
            max_add_weight = float(max_add_weight_override)
        tier_mult = as_float(cfg.get("tier_multiplier", {}).get(str(tier), 1.0), 1.0)

        vol_daily = float(vol_map.get(code, np.nan))
        if not np.isfinite(vol_daily) or vol_daily <= 0:
            vol_daily = 0.025
        vol_bps = vol_daily * 10000.0

        dd_abs = parse_drawdown_abs(get_row_field(row, "max_drawdown", cfg["default_max_drawdown"]), abs(cfg["default_max_drawdown"]))
        dd_bps = dd_abs * 10000.0

        pf = as_float(get_row_field(row, "profit_factor", cfg["default_profit_factor"]), cfg["default_profit_factor"])
        trades = as_float(get_row_field(row, "trades", cfg["default_trades"]), cfg["default_trades"])
        median_bps = as_float(get_row_field(row, "median_return_bps", cfg["default_median_return_bps"]), cfg["default_median_return_bps"])

        utility_bps = (
            ev_bps * quality * tier_mult
            - float(cfg.get("vol_penalty_lambda", 0.02)) * vol_bps
            - float(cfg.get("dd_penalty_lambda", 0.02)) * dd_bps
        )
        utility_bps *= weight_multiplier
        if utility_bps <= float(cfg.get("min_utility_bps", 0.0)):
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "utility_below_threshold", "utility_bps": utility_bps, "ev_bps": ev_bps})
            continue

        remaining_cap = max(max_weight * total_asset - current_value, 0.0)
        max_add_amount = max_add_weight * total_asset
        max_amount = min(remaining_cap, max_add_amount)

        min_lots = max(1, int(math.ceil(min_trade_amount_default / (lot_size * price))))
        max_lots = int(math.floor(max_amount / (lot_size * price)))
        if max_lots < min_lots:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "lot_bounds_infeasible", "min_lots": min_lots, "max_lots": max_lots})
            continue

        candidates.append(Candidate(
            stock_code=code, model_name=model_name, label_mode=label_mode, model_type=model_type,
            sector=sector, price=price, pred_return_bps=pred_return_bps, pred_prob=pred_prob,
            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps,
            entry_policy=entry_policy, entry_vwap_premium_bps=entry_vwap_premium_bps,
            samples=samples, expected_return_col=expected_return_col, metadata_path=metadata_path,
            ev_bps=ev_bps,
            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,
            vol_bps=vol_bps, max_drawdown_abs=dd_abs, max_drawdown_bps=dd_bps,
            profit_factor=pf, trades=trades, median_return_bps=median_bps,
            utility_bps=utility_bps, enabled=True, weight_multiplier=weight_multiplier,
            override_reason=override_reason, current_value=current_value, current_shares=current_shares,
            held=held, max_weight=max_weight, max_add_weight=max_add_weight,
            min_trade_amount=min_trade_amount_default, min_lots=min_lots, max_lots=max_lots,
        ))

    by_stock: Dict[str, Candidate] = {}
    for c in sorted(candidates, key=lambda z: z.utility_bps, reverse=True):
        if c.stock_code not in by_stock:
            by_stock[c.stock_code] = c
        else:
            rejected.append({"stock_code": c.stock_code, "model_name": c.model_name, "reason": "duplicate_stock_lower_utility", "utility_bps": c.utility_bps, "kept_model": by_stock[c.stock_code].model_name})
    return sorted(by_stock.values(), key=lambda z: z.utility_bps, reverse=True), rejected


def optimize_portfolio(candidates: List[Candidate], account: Dict[str, Any], corr_pairs, scenario_returns, cov_matrix, cfg: Dict[str, Any], time_limit_sec: float) -> Dict[str, Any]:
    if Model is None:
        raise RuntimeError(f"PySCIPOpt import failed: {_PYSCIPOPT_IMPORT_ERROR}")

    total_asset = float(account["total_asset"])
    available_cash = float(account["available_cash"])
    holdings = account.get("holdings", {})
    buy_commission = float(cfg.get("buy_commission_bps", 0.85)) / 10000.0
    cash_budget = min(
        available_cash * float(cfg.get("max_daily_buy_pct_of_cash", 0.70)),
        total_asset * float(cfg.get("max_daily_buy_pct_of_total_asset", 0.30)),
    )

    m = Model("daily_portfolio_confirm")
    m.hideOutput()
    m.setParam("limits/time", float(time_limit_sec))

    q, b, amount = {}, {}, {}
    lot_size = int(cfg.get("lot_size", 100))

    for c in candidates:
        code = c.stock_code
        q[code] = m.addVar(vtype="INTEGER", lb=0, ub=c.max_lots, name=f"lots_{code}")
        b[code] = m.addVar(vtype="BINARY", name=f"buy_{code}")
        amount[code] = lot_size * c.price * q[code]
        m.addCons(q[code] <= c.max_lots * b[code], name=f"bind_max_lots_{code}")
        m.addCons(q[code] >= c.min_lots * b[code], name=f"bind_min_lots_{code}")
        m.addCons(c.current_value + amount[code] <= c.max_weight * total_asset, name=f"max_final_weight_{code}")
        m.addCons(amount[code] <= c.max_add_weight * total_asset, name=f"max_add_weight_{code}")
        m.addCons(amount[code] >= c.min_trade_amount * b[code], name=f"min_trade_amount_{code}")

    held_codes = {
        normalize_stock_code(code)
        for code, h in holdings.items()
        if as_float(h.get("market_value", 0.0), 0.0) > 0 or as_float(h.get("shares", 0.0), 0.0) > 0
    }
    candidate_codes = {c.stock_code for c in candidates}
    existing_outside_count = len([x for x in held_codes if x not in candidate_codes])
    m.addCons(
        existing_outside_count + quicksum(b[c.stock_code] for c in candidates if c.stock_code not in held_codes)
        <= int(cfg.get("max_positions", 3)),
        name="max_final_positions",
    )

    m.addCons(
        quicksum(amount[c.stock_code] * (1.0 + buy_commission) for c in candidates) <= cash_budget,
        name="cash_budget",
    )

    hit_codes = [c.stock_code for c in candidates if c.model_type == "hit"]
    if hit_codes:
        m.addCons(quicksum(b[code] for code in hit_codes) <= int(cfg.get("max_new_hit_buys", 1)), name="max_new_hit_buys")

    obs_codes = [c.stock_code for c in candidates if c.model_type == "observation" or c.tier >= 4]
    if obs_codes:
        m.addCons(quicksum(b[code] for code in obs_codes) <= int(cfg.get("max_new_observation_buys", 1)), name="max_new_observation_buys")

    code_set = {c.stock_code for c in candidates}
    for i, j, corr in corr_pairs:
        if i in code_set and j in code_set:
            if i in held_codes and j not in held_codes:
                m.addCons(b[j] <= 0, name=f"corr_block_held_{i}_{j}")
            elif j in held_codes and i not in held_codes:
                m.addCons(b[i] <= 0, name=f"corr_block_held_{j}_{i}")
            elif i not in held_codes and j not in held_codes:
                m.addCons(b[i] + b[j] <= 1, name=f"corr_block_{i}_{j}")

    sector_limits = cfg.get("sector_limits", {}) or {}
    if sector_limits:
        current_sector_value: Dict[str, float] = {}
        for _, h in holdings.items():
            sec = str(h.get("sector", "UNKNOWN"))
            current_sector_value[sec] = current_sector_value.get(sec, 0.0) + as_float(h.get("market_value", 0.0), 0.0)
        for sec, limit in sector_limits.items():
            codes = [c.stock_code for c in candidates if c.sector == sec]
            if codes:
                m.addCons(
                    current_sector_value.get(sec, 0.0) + quicksum(amount[code] for code in codes)
                    <= float(limit) * total_asset,
                    name=f"sector_limit_{sec}",
                )

    max_vol_contrib = float(cfg.get("max_new_position_vol_contribution", 0.0035))
    if max_vol_contrib > 0:
        m.addCons(
            quicksum(amount[c.stock_code] * c.vol_daily for c in candidates) <= total_asset * max_vol_contrib,
            name="max_new_position_vol_contribution",
        )

    max_scenario_loss_pct = float(cfg.get("max_scenario_loss_pct", 0.012))
    if max_scenario_loss_pct > 0 and not scenario_returns.empty:
        for dt, row in scenario_returns.iterrows():
            loss_expr = quicksum(-float(row.get(c.stock_code, 0.0)) * amount[c.stock_code] for c in candidates)
            safe_name = str(dt).replace(" ", "_").replace("-", "").replace(":", "")
            m.addCons(loss_expr <= total_asset * max_scenario_loss_pct, name=f"scenario_loss_{safe_name}")

    obj = quicksum((c.utility_bps / 10000.0) * amount[c.stock_code] for c in candidates)


    if bool(cfg.get("use_covariance_penalty", False)) and not cov_matrix.empty:
        # PySCIPOpt setObjective() in this environment rejects nonlinear
        # objectives such as amount_i times amount_j. Use a linear marginal
        # covariance-risk proxy so the model remains MILP-compatible.
        risk_aversion = float(cfg.get("cov_risk_aversion", 3.0))
        self_weight = float(cfg.get("cov_linear_self_weight", 0.05))

        current_weights = {}
        for h_code, h in holdings.items():
            code = normalize_stock_code(h_code)
            mv = as_float(h.get("market_value", 0.0), 0.0)
            if mv > 0 and total_asset > 0:
                current_weights[code] = mv / total_asset

        cov_linear_penalty_bps = {}
        for c in candidates:
            code = c.stock_code
            if code not in cov_matrix.index:
                continue

            current_cov = 0.0
            for h_code, w_h in current_weights.items():
                if h_code in cov_matrix.columns:
                    current_cov += float(cov_matrix.loc[code, h_code]) * float(w_h)

            var_i = float(cov_matrix.loc[code, code]) if code in cov_matrix.columns else 0.0
            marginal_variance = max(0.0, 2.0 * current_cov + self_weight * var_i)
            cov_linear_penalty_bps[code] = 10000.0 * risk_aversion * marginal_variance

        if cov_linear_penalty_bps:
            obj = obj - quicksum(
                (cov_linear_penalty_bps.get(c.stock_code, 0.0) / 10000.0) * amount[c.stock_code]
                for c in candidates
            )
    m.setObjective(obj, "maximize")
    m.optimize()
    status = str(m.getStatus())

    try:
        obj_val = float(m.getObjVal())
    except Exception:
        obj_val = None

    orders, selected = [], []
    if status in {"optimal", "bestsollimit", "timelimit"}:
        for c in candidates:
            code = c.stock_code
            lots = int(round(m.getVal(q[code])))
            shares = lots * lot_size
            buy_amount = shares * c.price
            row = asdict(c)
            row.update({
                "selected_buy": lots > 0,
                "buy_lots": lots,
                "buy_shares": shares,
                "buy_amount": buy_amount,
                "buy_commission": buy_amount * buy_commission,
                "final_value_est": c.current_value + buy_amount,
                "final_weight_est": (c.current_value + buy_amount) / total_asset if total_asset > 0 else np.nan,
            })
            selected.append(row)
            if lots > 0:
                orders.append(row)

    return {
        "status": status,
        "objective": obj_val,
        "orders": orders,
        "selected": selected,
        "cash_budget": cash_budget,
        "existing_positions_count": len(held_codes),
        "existing_outside_count": existing_outside_count,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--account", required=True)
    ap.add_argument("--prices", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="portfolio_reports")
    ap.add_argument("--use-covariance-penalty", action="store_true")
    ap.add_argument("--cov-risk-aversion", type=float, default=None)
    ap.add_argument("--time-limit-sec", type=float, default=30.0)
    ap.add_argument("--print-candidates", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = deep_update(DEFAULT_CONFIG, load_json(args.config))
    if args.use_covariance_penalty:
        cfg["use_covariance_penalty"] = True
    if args.cov_risk_aversion is not None:
        cfg["cov_risk_aversion"] = float(args.cov_risk_aversion)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signals(args.signals)
    metrics = load_metrics(args.metrics)
    prices = load_prices(args.prices)
    account = load_account(args.account)
    history = load_history(args.history)
    returns = compute_returns(history)

    if signals.empty:
        report = {"date": args.date, "status": "empty_signals", "orders": []}
        (out_dir / f"daily_portfolio_report_{args.date}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(out_dir / f"daily_portfolio_orders_{args.date}.csv", index=False, encoding="utf-8-sig")
        print("Empty signals; no orders.")
        return 0

    merged = merge_inputs(signals, metrics, prices)
    vol_map = compute_vol_map(returns, int(cfg.get("vol_window", 60)))
    candidates, rejected = build_candidates(merged, account, vol_map, cfg)

    if args.print_candidates and candidates:
        print(pd.DataFrame([asdict(c) for c in candidates]).to_string(index=False))

    candidate_codes = [c.stock_code for c in candidates]
    corr_pairs = compute_corr_pairs(returns, candidate_codes, int(cfg.get("corr_window", 60)), float(cfg.get("max_pair_corr", 0.75)))
    scenario_returns = get_scenario_returns(returns, candidate_codes, int(cfg.get("scenario_window", 120)))
    cov_matrix = compute_cov_matrix(returns, candidate_codes, int(cfg.get("corr_window", 60)))

    if not candidates:
        result = {"status": "no_candidates", "objective": None, "orders": [], "selected": []}
    else:
        result = optimize_portfolio(candidates, account, corr_pairs, scenario_returns, cov_matrix, cfg, float(args.time_limit_sec))

    orders_df = pd.DataFrame(result.get("orders", []))
    selected_df = pd.DataFrame(result.get("selected", []))
    rejected_df = pd.DataFrame(rejected)

    orders_df.to_csv(out_dir / f"daily_portfolio_orders_{args.date}.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(out_dir / f"daily_portfolio_selected_{args.date}.csv", index=False, encoding="utf-8-sig")
    rejected_df.to_csv(out_dir / f"daily_portfolio_rejected_{args.date}.csv", index=False, encoding="utf-8-sig")

    report = {
        "date": args.date,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "cash_budget": result.get("cash_budget"),
        "orders": result.get("orders", []),
        "corr_pairs_blocked": [{"stock_i": i, "stock_j": j, "corr": corr} for i, j, corr in corr_pairs],
        "risk_summary": {
            "max_new_position_vol_contribution": cfg.get("max_new_position_vol_contribution"),
            "max_scenario_loss_pct": cfg.get("max_scenario_loss_pct"),
            "scenario_count": int(len(scenario_returns)) if not scenario_returns.empty else 0,
            "use_covariance_penalty": bool(cfg.get("use_covariance_penalty", False)),
            "cov_risk_aversion": cfg.get("cov_risk_aversion"),
        },
    }
    (out_dir / f"daily_portfolio_report_{args.date}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Status: {result.get('status')}")
    print(f"Orders: {len(result.get('orders', []))}")
    if not orders_df.empty:
        cols = [c for c in ["stock_code", "model_type", "buy_shares", "buy_amount", "ev_bps", "utility_bps", "final_weight_est"] if c in orders_df.columns]
        print(orders_df[cols].to_string(index=False))
    print(f"Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
