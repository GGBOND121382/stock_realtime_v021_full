#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_decision/portfolio_confirm_from_buy_signals.py

Adapter from trading-day signal pipeline output to portfolio optimizer.

Input:
  --signal-dir directory containing buy_signals.csv / all_scores.csv
  --saved-models saved_models
  --account account.json
  --history history_close.csv

Output:
  portfolio_reports/daily_portfolio_orders_<date>.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def normalize_stock_code(x: Any) -> str:
    s = str(x).strip().upper()
    if s.isdigit() and len(s) == 6:
        return f"{s}.SH" if s.startswith(("6", "9")) else f"{s}.SZ"
    if s.startswith("SH."):
        return f"{s[3:]}.SH"
    if s.startswith("SZ."):
        return f"{s[3:]}.SZ"
    return s


def as_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def find_metadata(saved_models: Path, stock_code: str, artifact_name: str) -> Optional[Path]:
    p = saved_models / stock_code / artifact_name / "metadata.json"
    if p.exists():
        return p
    matches = list(saved_models.glob(f"*/{artifact_name}/metadata.json"))
    return matches[0] if matches else None


def load_metadata(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_bps_from_return(x: Any) -> float:
    val = as_float(x, np.nan)
    if not np.isfinite(val):
        return np.nan
    if abs(val) < 2.0:
        return val * 10000.0
    return val


def build_inputs(signal_dir: Path, saved_models: Path, out_input_dir: Path, use_all_scores: bool = False) -> Dict[str, Path]:
    src = signal_dir / ("all_scores.csv" if use_all_scores else "buy_signals.csv")
    if not src.exists():
        raise FileNotFoundError(f"missing upstream signal file: {src}")

    raw = pd.read_csv(src)
    out_input_dir.mkdir(parents=True, exist_ok=True)

    signals_out = out_input_dir / "portfolio_signals.csv"
    metrics_out = out_input_dir / "portfolio_metrics.csv"
    prices_out = out_input_dir / "portfolio_prices.csv"

    if raw.empty:
        pd.DataFrame(columns=["stock_code", "model_name", "label_mode"]).to_csv(signals_out, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["stock_code", "model_name"]).to_csv(metrics_out, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["stock_code", "price"]).to_csv(prices_out, index=False, encoding="utf-8-sig")
        return {"signals": signals_out, "metrics": metrics_out, "prices": prices_out}

    required = {"stock_code", "artifact_name"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{src} missing required columns: {sorted(missing)}")

    sig_rows, met_rows, price_rows = [], [], []

    for _, r in raw.iterrows():
        stock_code = normalize_stock_code(r.get("stock_code", ""))
        artifact = str(r.get("artifact_name", "")).strip()
        if not stock_code or not artifact:
            continue

        meta_path = find_metadata(saved_models, stock_code, artifact)
        meta = load_metadata(meta_path)

        label_mode = str(meta.get("label_mode", r.get("label_mode", ""))).strip()
        if not label_mode:
            label_mode = "hit" if "hit" in artifact.lower() else "close_profit"

        validation = meta.get("validation_tail_trade_metrics", {}) or meta.get("validation_trade_metrics", {}) or {}
        avg_return_bps = metric_bps_from_return(validation.get("avg_return", np.nan))
        median_return_bps = metric_bps_from_return(validation.get("median_return", np.nan))
        trades = as_float(validation.get("trades", np.nan), np.nan)
        win_rate = as_float(validation.get("win_rate", np.nan), np.nan)
        max_drawdown = as_float(validation.get("max_drawdown", np.nan), np.nan)
        profit_factor = as_float(validation.get("profit_factor", np.nan), np.nan)

        target_hit_bps = as_float(meta.get("target_hit_bps", r.get("target_hit_bps", 80 if "80" in artifact else 50)), 50.0)

        price = as_float(r.get("close", np.nan), np.nan)
        if not np.isfinite(price) or price <= 0:
            price = as_float(r.get("daily_vwap", np.nan), np.nan)

        hit_score = as_float(r.get("hit_score", np.nan), np.nan)
        threshold = as_float(r.get("threshold", np.nan), np.nan)
        score_margin = as_float(r.get("score_margin", np.nan), np.nan)

        conf_mult = 1.0
        if np.isfinite(score_margin) and np.isfinite(threshold) and abs(threshold) > 1e-9:
            conf_mult = float(np.clip(1.0 + 0.20 * score_margin / max(abs(threshold), 1e-9), 0.80, 1.20))

        if label_mode == "hit":
            pred_prob = hit_score if np.isfinite(hit_score) else win_rate
            pred_return_bps = np.nan
        else:
            pred_return_bps = avg_return_bps * conf_mult if np.isfinite(avg_return_bps) else median_return_bps
            pred_prob = np.nan

        sector = r.get("sector", r.get("industry", "UNKNOWN"))

        sig_rows.append({
            "stock_code": stock_code,
            "model_name": artifact,
            "label_mode": label_mode,
            "pred_return_bps": pred_return_bps,
            "pred_prob": pred_prob,
            "target_hit_bps": target_hit_bps,
            "price": price,
            "sector": sector,
            "hit_score": hit_score,
            "threshold": threshold,
            "score_margin": score_margin,
            "metadata_path": str(meta_path) if meta_path else "",
        })

        met_rows.append({
            "stock_code": stock_code,
            "model_name": artifact,
            "label_mode": label_mode,
            "trades": trades,
            "win_rate": win_rate,
            "avg_return_bps": avg_return_bps,
            "median_return_bps": median_return_bps,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "target_hit_bps": target_hit_bps,
            "feature_group": meta.get("feature_group", ""),
            "base_model_name": meta.get("model_name", ""),
            "entry_policy": meta.get("entry_policy", ""),
            "sector": sector,
        })

        price_rows.append({"stock_code": stock_code, "price": price})

    pd.DataFrame(sig_rows).to_csv(signals_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(met_rows).to_csv(metrics_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(price_rows).drop_duplicates("stock_code", keep="last").to_csv(prices_out, index=False, encoding="utf-8-sig")

    return {"signals": signals_out, "metrics": metrics_out, "prices": prices_out}


def make_account_template(path: Path) -> None:
    sample = {
        "total_asset": 200000.0,
        "available_cash": 80000.0,
        "holdings": {
            "600312.SH": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "电网设备"},
            "601899.SH": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "贵金属"},
            "002311.SZ": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "农产品加工"},
            "603308.SH": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "通用设备"},
            "600096.SH": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "农化制品"},
            "601985.SH": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "电力"},
            "600276.SH": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "化学制药"},
            "002714.SZ": {"shares": 0, "market_value": 0.0, "cost_basis": 0.0, "sector": "养殖业"}
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote account template: {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=False, help="Decision date, e.g. 2026-05-12")
    ap.add_argument("--signal-dir", help="Directory containing buy_signals.csv/all_scores.csv")
    ap.add_argument("--account", help="account.json")
    ap.add_argument("--history", default=None)
    ap.add_argument("--saved-models", default="saved_models")
    ap.add_argument("--config", default="configs/portfolio_confirm_config.json")
    ap.add_argument("--out-dir", default="portfolio_reports")
    ap.add_argument("--optimizer-script", default="portfolio_decision/daily_portfolio_confirm_pyscipopt.py")
    ap.add_argument("--use-all-scores", action="store_true")
    ap.add_argument("--use-covariance-penalty", action="store_true")
    ap.add_argument("--cov-risk-aversion", type=float, default=None)
    ap.add_argument("--time-limit-sec", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-account-template", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.make_account_template:
        make_account_template(Path(args.make_account_template))
        return 0

    if not args.date or not args.signal_dir or not args.account:
        raise SystemExit("--date, --signal-dir and --account are required unless --make-account-template is used")

    out_dir = Path(args.out_dir)
    input_dir = out_dir / f"_portfolio_inputs_{args.date}"

    paths = build_inputs(
        signal_dir=Path(args.signal_dir),
        saved_models=Path(args.saved_models),
        out_input_dir=input_dir,
        use_all_scores=bool(args.use_all_scores),
    )

    cmd = [
        sys.executable,
        args.optimizer_script,
        "--date", args.date,
        "--signals", str(paths["signals"]),
        "--metrics", str(paths["metrics"]),
        "--account", args.account,
        "--prices", str(paths["prices"]),
        "--out-dir", str(out_dir),
        "--time-limit-sec", str(args.time_limit_sec),
    ]
    if args.history:
        cmd += ["--history", args.history]
    if args.config and Path(args.config).exists():
        cmd += ["--config", args.config]
    if args.use_covariance_penalty:
        cmd += ["--use-covariance-penalty"]
    if args.cov_risk_aversion is not None:
        cmd += ["--cov-risk-aversion", str(args.cov_risk_aversion)]

    print("[PORTFOLIO INPUTS]")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("[RUN]")
    print(" ".join(str(x) for x in cmd))

    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
