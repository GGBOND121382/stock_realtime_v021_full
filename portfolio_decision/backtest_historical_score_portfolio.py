#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical-data backtest: historical samples -> daily scores -> portfolio -> evaluation.

Input is historical sample data referenced by saved model metadata.

Flow:
    saved_models/*/*/metadata.json
      -> metadata["samples"] historical training sample CSV
      -> per historical date score all artifacts
      -> generated all_scores / buy_signals / rejected_scores
      -> existing portfolio_confirm_from_buy_signals.py
      -> existing daily_portfolio_confirm_pyscipopt.py
      -> rolling cash/position simulation and strategy metrics

No mandatory history_close.csv.  If --history is provided and exists, it is used
for risk/correlation and valuation.  Otherwise the valuation history is built
in memory from the same sample CSV close columns and is NOT written as another
project data table.

If metadata samples are missing, this script stops and prints the corresponding
run_premarket_history_update.py command to restore pipeline outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DEFAULT_EVAL_CONFIG = {
    "buy_commission_bps": 0.85,
    "sell_commission_bps": 0.85,
    "stamp_tax_bps": 10.0,
    "slippage_bps_buy": 2.0,
    "slippage_bps_sell": 2.0,
}


def normalize_stock_code(x: Any) -> str:
    s = str(x or "").strip().upper().replace("_", ".")
    if not s or s.lower() == "nan":
        return ""
    if s.startswith("SH."):
        return f"{s[3:]}.SH"
    if s.startswith("SZ."):
        return f"{s[3:]}.SZ"
    if "." in s:
        code, mkt = s.split(".", 1)
        digits = "".join(ch for ch in code if ch.isdigit()).zfill(6)
        return f"{digits}.{mkt.upper()}"
    digits = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    if not digits:
        return ""
    return f"{digits}.SH" if digits.startswith(("5", "6", "9")) else f"{digits}.SZ"


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


def load_json(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_eval_config(path: Optional[Path]) -> dict:
    cfg = dict(DEFAULT_EVAL_CONFIG)
    raw = load_json(path)
    for k in list(cfg):
        if k in raw:
            cfg[k] = raw[k]
    return cfg


def resolve_repo_path(raw: Any, stock_code: str = "") -> Optional[Path]:
    if raw is None:
        return None
    text = str(raw).strip().replace("\\", "/")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    p = Path(text)
    if p.exists():
        return p
    if not p.is_absolute():
        p2 = PROJECT_DIR / p
        if p2.exists():
            return p2

    for marker in ["stock_realtime_v021_full/", "stock_realtime/"]:
        if marker in text:
            cand = PROJECT_DIR / text.split(marker, 1)[1]
            if cand.exists():
                return cand

    if "saved_data/" in text:
        cand = PROJECT_DIR / text[text.index("saved_data/"):]
        if cand.exists():
            return cand

    # Conservative fallback: search stock-specific pipeline dirs first.
    code = normalize_stock_code(stock_code).split(".", 1)[0] if stock_code else ""
    name = Path(text).name
    roots = []
    if code:
        roots.extend([
            PROJECT_DIR / "saved_data" / f"{code}_pipeline_out",
            PROJECT_DIR / "saved_data" / f"{code}_base_out",
        ])
    roots.append(PROJECT_DIR / "saved_data")
    if name:
        for root in roots:
            if not root.exists():
                continue
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    return None


def load_history_close(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"history file is empty: {path}")
    if "date" not in df.columns:
        raise ValueError("history CSV must contain date column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if {"stock_code", "close"}.issubset(df.columns):
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
        wide = df.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
    else:
        wide = df.set_index("date")
        wide.columns = [normalize_stock_code(c) for c in wide.columns]
    wide = wide.sort_index().apply(pd.to_numeric, errors="coerce")
    return wide[[c for c in wide.columns if c]]


def price_on_or_before(history: pd.DataFrame, date: pd.Timestamp, stock: str) -> float:
    stock = normalize_stock_code(stock)
    if stock not in history.columns:
        return np.nan
    s = history[stock].loc[history.index <= date].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def next_trading_date(history: pd.DataFrame, date: pd.Timestamp, hold_days: int) -> Optional[pd.Timestamp]:
    idx = list(history.index)
    pos = int(np.searchsorted(idx, date))
    if pos >= len(idx):
        return None
    sell_pos = pos + int(hold_days)
    if sell_pos < len(idx):
        return pd.Timestamp(idx[sell_pos]).normalize()
    return None


def read_feature_columns(artifact_dir: Path) -> list[str]:
    p = artifact_dir / "feature_columns.txt"
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_feature_median(artifact_dir: Path, cols: list[str]) -> pd.Series:
    p = artifact_dir / "feature_median.csv"
    if not p.exists():
        return pd.Series({c: 0.0 for c in cols}, dtype=float)
    m = pd.read_csv(p, index_col=0)
    if "median" in m.columns:
        s = m["median"]
    elif len(m.columns) >= 1:
        s = m.iloc[:, 0]
    else:
        s = pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce")
    for c in cols:
        if c not in s.index:
            s.loc[c] = 0.0
    return s[cols].fillna(0.0)


def predict_positive(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(x), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


@dataclass
class ArtifactState:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    metadata: dict
    samples_path: Path
    samples: pd.DataFrame
    feature_columns: list[str]
    feature_median: pd.Series
    model: Any


@dataclass
class OpenLot:
    trade_id: str
    stock_code: str
    buy_date: str
    planned_sell_date: str
    buy_price: float
    shares: int
    buy_amount: float
    buy_fee: float
    model_name: str
    model_type: str
    sector: str
    ev_bps: float
    utility_bps: float


@dataclass
class TradeRecord:
    trade_id: str
    stock_code: str
    model_name: str
    model_type: str
    sector: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    shares: int
    buy_amount: float
    sell_amount: float
    buy_fee: float
    sell_fee: float
    stamp_tax: float
    pnl: float
    net_return: float
    hold_days_calendar: int
    ev_bps: float
    utility_bps: float
    exit_reason: str


def discover_artifacts(models_dir: Path, watchlist: Optional[set[str]], model_policy: str) -> list[Path]:
    meta_paths = []
    for p in sorted(models_dir.glob("*/*/metadata.json")):
        meta = load_json(p)
        stock = normalize_stock_code(meta.get("stock_code") or p.parent.parent.name)
        if watchlist and stock not in watchlist:
            continue
        meta_paths.append(p)

    if model_policy == "all":
        return meta_paths

    grouped: dict[str, list[Path]] = {}
    for p in meta_paths:
        meta = load_json(p)
        stock = normalize_stock_code(meta.get("stock_code") or p.parent.parent.name)
        grouped.setdefault(stock, []).append(p)
    chosen = []
    for _, items in grouped.items():
        items.sort(key=lambda p: (
            "close_profit" in p.parent.name,
            str(load_json(p).get("artifact_created_at") or ""),
            p.parent.stat().st_mtime,
        ), reverse=True)
        chosen.append(items[0])
    return sorted(chosen)


def restore_command(symbols: list[str], end_date: str, models_dir: Path, saved_data_dir: Path, context_config: Path) -> str:
    syms = ",".join(sorted(set(symbols)))
    return (
        "python3 pipelines/run_premarket_history_update.py "
        f"--models-dir {models_dir} "
        f"--saved-data-dir {saved_data_dir} "
        f"--context-config {context_config} "
        f"--symbols {syms} "
        f"--end-date {end_date} "
        "--keep-going"
    )


def load_artifact_states(
    models_dir: Path,
    watchlist: Optional[set[str]],
    model_policy: str,
    saved_data_dir: Path,
    context_config: Path,
    restore_end_date: str,
) -> list[ArtifactState]:
    states = []
    missing_samples: list[dict] = []

    for meta_path in discover_artifacts(models_dir, watchlist, model_policy):
        artifact_dir = meta_path.parent
        meta = load_json(meta_path)
        stock = normalize_stock_code(meta.get("stock_code") or artifact_dir.parent.name)
        artifact_name = str(meta.get("artifact_name") or artifact_dir.name)
        raw_samples = meta.get("samples")
        samples_path = resolve_repo_path(raw_samples, stock)
        model_path = artifact_dir / "model.joblib"

        if samples_path is None or not samples_path.exists():
            missing_samples.append({
                "stock_code": stock,
                "artifact_name": artifact_name,
                "metadata": str(meta_path),
                "metadata_samples": str(raw_samples),
            })
            continue

        if not model_path.exists():
            print(f"[SKIP] model.joblib missing for {stock} {artifact_name}", flush=True)
            continue

        try:
            samples = pd.read_csv(samples_path, parse_dates=["date"])
        except Exception as exc:
            print(f"[SKIP] read samples failed for {stock} {artifact_name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        if samples.empty or "date" not in samples.columns:
            continue
        samples["date"] = pd.to_datetime(samples["date"], errors="coerce").dt.normalize()
        samples = samples.dropna(subset=["date"]).sort_values("date")

        cols = read_feature_columns(artifact_dir)
        if not cols:
            print(f"[SKIP] feature_columns empty for {stock} {artifact_name}", flush=True)
            continue
        for c in [c for c in cols if c not in samples.columns]:
            samples[c] = np.nan

        med = read_feature_median(artifact_dir, cols)
        model = joblib.load(model_path)
        states.append(ArtifactState(
            stock_code=stock,
            artifact_name=artifact_name,
            artifact_dir=artifact_dir,
            metadata=meta,
            samples_path=samples_path,
            samples=samples,
            feature_columns=cols,
            feature_median=med,
            model=model,
        ))

    if missing_samples:
        by_stock = sorted({x["stock_code"] for x in missing_samples if x["stock_code"]})
        print("\n[ERROR] Some saved model metadata samples are missing.\n", file=sys.stderr)
        for x in missing_samples[:50]:
            print(f"- {x['stock_code']} | {x['artifact_name']}", file=sys.stderr)
            print(f"  metadata: {x['metadata']}", file=sys.stderr)
            print(f"  samples:  {x['metadata_samples']}", file=sys.stderr)
        if len(missing_samples) > 50:
            print(f"... and {len(missing_samples)-50} more", file=sys.stderr)
        print("\nRestore command:", file=sys.stderr)
        print(
            restore_command(
                by_stock,
                restore_end_date,
                models_dir=models_dir,
                saved_data_dir=saved_data_dir,
                context_config=context_config,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)

    return states


def build_history_close_from_states(states: list[ArtifactState]) -> pd.DataFrame:
    rows = []
    for st in states:
        price_col = None
        for c in ["close", "last_price", "daily_close", "adj_close"]:
            if c in st.samples.columns:
                price_col = c
                break
        if price_col is None:
            continue
        part = st.samples[["date", price_col]].copy()
        part["date"] = pd.to_datetime(part["date"], errors="coerce").dt.normalize()
        part["stock_code"] = st.stock_code
        part["close"] = pd.to_numeric(part[price_col], errors="coerce")
        part = part.dropna(subset=["date", "close"])
        part = part[part["close"] > 0]
        rows.append(part[["date", "stock_code", "close"]])

    if not rows:
        raise ValueError("cannot build in-memory close history: no close-like column found in model samples")

    long_df = pd.concat(rows, ignore_index=True)
    wide = long_df.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
    wide = wide.sort_index().apply(pd.to_numeric, errors="coerce")
    return wide[[c for c in wide.columns if c]]


def infer_entry_signal(row: pd.Series, meta: dict) -> bool:
    if "entry_signal" in row.index and pd.notna(row.get("entry_signal")):
        v = row.get("entry_signal")
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}
    policy = str(meta.get("entry_policy") or "all_days")
    if policy == "all_days":
        return True
    if policy == "vwap_low":
        close = as_float(row.get("close"), np.nan)
        vwap = as_float(row.get("daily_vwap"), np.nan)
        prem = as_float(meta.get("entry_vwap_premium_bps", 50.0), 50.0) / 10000.0
        return bool(np.isfinite(close) and np.isfinite(vwap) and vwap > 0 and close <= vwap * (1.0 + prem))
    return False


def first_available(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for n in names:
        if n in row.index and pd.notna(row.get(n)):
            return row.get(n)
    return default


def score_one_artifact_on_date(state: ArtifactState, date: pd.Timestamp, min_amount_yuan: float) -> Optional[dict]:
    part = state.samples[state.samples["date"] == date]
    if part.empty:
        return None
    row = part.iloc[-1]
    cols = state.feature_columns
    x = pd.DataFrame([row[cols].to_dict()]).apply(pd.to_numeric, errors="coerce")
    x = x.fillna(state.feature_median).replace([np.inf, -np.inf], np.nan).fillna(state.feature_median).fillna(0.0)
    score = float(predict_positive(state.model, x)[0])
    threshold = as_float(state.metadata.get("threshold"), 0.5)
    entry_signal = infer_entry_signal(row, state.metadata)
    score_pass = score >= threshold
    amount = as_float(first_available(row, ["amount", "daily_amount", "turnover_amount"], np.nan), np.nan)
    reject_reasons = []
    if not entry_signal:
        reject_reasons.append("entry_signal_false")
    if not score_pass:
        reject_reasons.append("score_below_threshold")
    if min_amount_yuan > 0 and np.isfinite(amount) and amount < min_amount_yuan:
        reject_reasons.append(f"amount_lt_{int(min_amount_yuan)}")

    label_mode = str(state.metadata.get("label_mode", "close_profit"))
    return_col = str(state.metadata.get("return_col") or ("trade_net_close_return" if label_mode == "close_profit" else "trade_target_or_close_return"))
    label_col = str(state.metadata.get("label_col") or ("trade_close_profit_label" if label_mode == "close_profit" else "trade_hit_label"))

    return {
        "rank": np.nan,
        "trade_date": date.strftime("%Y%m%d"),
        "date": date.strftime("%Y-%m-%d"),
        "stock_code": state.stock_code,
        "artifact_name": state.artifact_name,
        "artifact_dir": str(state.artifact_dir),
        "samples": str(state.samples_path),
        "entry_policy": state.metadata.get("entry_policy", ""),
        "label_mode": label_mode,
        "model_name": state.metadata.get("model_name", ""),
        "feature_group": state.metadata.get("feature_group", ""),
        "close": as_float(first_available(row, ["close", "last_price"], np.nan), np.nan),
        "open": as_float(row.get("open"), np.nan),
        "high": as_float(row.get("high"), np.nan),
        "low": as_float(row.get("low"), np.nan),
        "volume": as_float(row.get("volume"), np.nan),
        "amount": amount,
        "daily_vwap": as_float(row.get("daily_vwap"), np.nan),
        "hit_score": score,
        "threshold": threshold,
        "score_margin": score - threshold,
        "entry_signal": entry_signal,
        "signal_raw_score_pass": score_pass,
        "signal": bool(entry_signal and score_pass and not reject_reasons),
        "reject_reason": ";".join(reject_reasons),
        "target_hit_bps": as_float(state.metadata.get("target_hit_bps"), 50.0),
        "round_trip_cost_bps": as_float(state.metadata.get("round_trip_cost_bps"), np.nan),
        "expected_return_col": return_col,
        "realized_return": as_float(row.get(return_col), np.nan),
        "eval_label": as_float(row.get(label_col), np.nan),
        "sector": first_available(row, ["sector", "industry", "sector_symbol"], ""),
        "context_status": "historical_samples",
        "source_mode": "historical_samples",
    }


def generate_daily_scores(states: list[ArtifactState], dates: list[pd.Timestamp], out_signal_root: Path, min_amount_yuan: float) -> pd.DataFrame:
    summary_rows = []
    out_signal_root.mkdir(parents=True, exist_ok=True)
    for date in dates:
        ymd = date.strftime("%Y%m%d")
        day_dir = out_signal_root / ymd
        day_dir.mkdir(parents=True, exist_ok=True)
        rows = [r for st in states if (r := score_one_artifact_on_date(st, date, min_amount_yuan)) is not None]
        all_scores = pd.DataFrame(rows)
        if not all_scores.empty:
            all_scores = all_scores.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            all_scores["rank"] = np.arange(1, len(all_scores) + 1)
            buy = all_scores[(all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~all_scores.index.isin(buy.index)].copy()
        else:
            buy = pd.DataFrame()
            rejected = pd.DataFrame()
        all_scores.to_csv(day_dir / "all_scores.csv", index=False, encoding="utf-8-sig")
        buy.to_csv(day_dir / "buy_signals.csv", index=False, encoding="utf-8-sig")
        rejected.to_csv(day_dir / "rejected_scores.csv", index=False, encoding="utf-8-sig")
        run_summary = {
            "date": ymd,
            "source_mode": "historical_samples",
            "artifacts_scored": int(len(all_scores)),
            "buy_signals": int(len(buy)),
            "rejected": int(len(rejected)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (day_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(run_summary)
        print(f"[SCORED] {ymd} all={len(all_scores)} buy={len(buy)}", flush=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_signal_root / "historical_score_generation_summary.csv", index=False, encoding="utf-8-sig")
    return df


def account_from_state(cash: float, open_lots: list[OpenLot], history: pd.DataFrame, date: pd.Timestamp) -> dict:
    grouped: dict[str, dict] = {}
    total_market_value = 0.0
    for lot in open_lots:
        px = price_on_or_before(history, date, lot.stock_code)
        if not np.isfinite(px) or px <= 0:
            px = lot.buy_price
        mv = lot.shares * px
        total_market_value += mv
        g = grouped.setdefault(lot.stock_code, {"shares": 0, "market_value": 0.0, "cost_value": 0.0, "sector": lot.sector or "UNKNOWN"})
        g["shares"] += int(lot.shares)
        g["market_value"] += float(mv)
        g["cost_value"] += float(lot.buy_amount)

    holdings = {}
    for code, g in grouped.items():
        shares = int(g["shares"])
        holdings[code] = {
            "shares": shares,
            "market_value": float(g["market_value"]),
            "cost_basis": float(g["cost_value"] / shares) if shares > 0 else 0.0,
            "sector": g.get("sector", "UNKNOWN"),
        }
    return {"total_asset": float(cash + total_market_value), "available_cash": float(cash), "holdings": holdings}



def write_point_in_time_risk_history(
    history: pd.DataFrame,
    date: pd.Timestamp,
    out_dir: Path,
    include_current_day: bool = True,
    min_rows: int = 20,
) -> Optional[Path]:
    # Write the risk-history CSV visible to the optimizer on one backtest date.
    #
    # The optimizer is executed as a subprocess, so its covariance / correlation /
    # scenario-risk logic needs a CSV input. This function writes a point-in-time
    # wide close-price table clipped to the current decision date. It is an
    # intermediate risk input only, not a long-lived data source.
    if history is None or history.empty:
        return None

    cutoff = pd.Timestamp(date).normalize()
    hist = history.copy()
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    hist = hist[hist.index.notna()].sort_index()

    if include_current_day:
        hist = hist.loc[hist.index <= cutoff]
    else:
        hist = hist.loc[hist.index < cutoff]

    hist = hist.dropna(axis=1, how="all")
    if len(hist) < int(min_rows) or hist.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"risk_history_until_{cutoff.strftime('%Y%m%d')}.csv"

    export = hist.reset_index()
    first_col = export.columns[0]
    if first_col != "date":
        export = export.rename(columns={first_col: "date"})
    export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_portfolio_adapter(
    date: pd.Timestamp,
    signal_dir: Path,
    account_json: Path,
    history_path: Optional[Path],
    saved_models: Path,
    config: Optional[Path],
    out_dir: Path,
    optimizer_script: str,
    use_covariance_penalty: bool,
    cov_risk_aversion: Optional[float],
    time_limit_sec: float,
) -> Path:
    date_dash = date.strftime("%Y-%m-%d")
    cmd = [
        sys.executable, "portfolio_decision/portfolio_confirm_from_buy_signals.py",
        "--date", date_dash,
        "--signal-dir", str(signal_dir),
        "--account", str(account_json),
        "--saved-models", str(saved_models),
        "--out-dir", str(out_dir),
        "--optimizer-script", optimizer_script,
        "--time-limit-sec", str(time_limit_sec),
    ]
    if history_path is not None and history_path.exists():
        cmd += ["--history", str(history_path)]
    if config and config.exists():
        cmd += ["--config", str(config)]
    if use_covariance_penalty:
        cmd += ["--use-covariance-penalty"]
    if cov_risk_aversion is not None:
        cmd += ["--cov-risk-aversion", str(cov_risk_aversion)]
    print("[PORTFOLIO]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
    return out_dir / f"daily_portfolio_orders_{date_dash}.csv"


def read_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def calc_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    if equity.empty:
        return 0.0, equity
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()), dd


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not np.isfinite(b):
        return default
    return float(a / b)


def summarize(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash: float) -> dict:
    if equity_df.empty:
        return {"status": "empty", "initial_cash": initial_cash, "final_equity": initial_cash, "total_return": 0.0}
    eq = pd.to_numeric(equity_df["equity"], errors="coerce").dropna()
    daily_ret = pd.to_numeric(equity_df.get("daily_return", pd.Series(dtype=float)), errors="coerce").dropna()
    total_return = float(eq.iloc[-1] / initial_cash - 1.0) if initial_cash > 0 and not eq.empty else 0.0
    max_dd, _ = calc_drawdown(eq)
    ann_ret = float((1.0 + total_return) ** (252.0 / len(eq)) - 1.0) if len(eq) else 0.0
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) >= 2 else 0.0
    sharpe = safe_div(float(daily_ret.mean() * 252), ann_vol, 0.0) if len(daily_ret) >= 2 else 0.0

    if not trades_df.empty:
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
        ret = pd.to_numeric(trades_df["net_return"], errors="coerce")
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        profit_sum = float(pnl[pnl > 0].sum())
        loss_sum = float(-pnl[pnl < 0].sum())
        win_rate = safe_div(wins, len(trades_df), 0.0)
        pf = safe_div(profit_sum, loss_sum, float("inf") if profit_sum > 0 else 0.0)
        avg_ret = float(ret.mean())
        med_ret = float(ret.median())
    else:
        wins = losses = 0
        profit_sum = loss_sum = win_rate = pf = avg_ret = med_ret = 0.0

    return {
        "status": "ok",
        "initial_cash": float(initial_cash),
        "final_equity": float(eq.iloc[-1]) if not eq.empty else float(initial_cash),
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_dd,
        "trading_days": int(len(eq)),
        "realized_trades": int(len(trades_df)),
        "win_trades": wins,
        "loss_trades": losses,
        "win_rate": win_rate,
        "profit_sum": profit_sum,
        "loss_sum_abs": loss_sum,
        "profit_factor": pf,
        "avg_trade_return": avg_ret,
        "median_trade_return": med_ret,
        "avg_gross_exposure": float(pd.to_numeric(equity_df.get("gross_exposure", pd.Series(dtype=float)), errors="coerce").mean()) if "gross_exposure" in equity_df else 0.0,
    }


def simulate_portfolio(
    dates: list[pd.Timestamp],
    signal_root: Path,
    history: pd.DataFrame,
    history_path: Optional[Path],
    saved_models: Path,
    config_path: Optional[Path],
    out_dir: Path,
    optimizer_script: str,
    initial_cash: float,
    hold_days: int,
    close_open_at_end: bool,
    eval_cfg: dict,
    use_covariance_penalty: bool,
    cov_risk_aversion: Optional[float],
    time_limit_sec: float,
) -> dict:
    buy_cost_rate = (float(eval_cfg.get("buy_commission_bps", 0.85)) + float(eval_cfg.get("slippage_bps_buy", 2.0))) / 10000.0
    sell_cost_rate = (float(eval_cfg.get("sell_commission_bps", 0.85)) + float(eval_cfg.get("slippage_bps_sell", 2.0))) / 10000.0
    stamp_tax_rate = float(eval_cfg.get("stamp_tax_bps", 10.0)) / 10000.0

    cash = float(initial_cash)
    open_lots: list[OpenLot] = []
    trades: list[TradeRecord] = []
    equity_rows: list[dict] = []
    daily_rows: list[dict] = []

    for date in dates:
        ymd = date.strftime("%Y%m%d")
        date_dash = date.strftime("%Y-%m-%d")
        day_out = out_dir / "portfolio_runs" / ymd
        day_out.mkdir(parents=True, exist_ok=True)

        still_open = []
        realized_today = []
        for lot in open_lots:
            if pd.to_datetime(lot.planned_sell_date) <= date:
                sell_px = price_on_or_before(history, date, lot.stock_code)
                if not np.isfinite(sell_px) or sell_px <= 0:
                    still_open.append(lot)
                    continue
                sell_amount = lot.shares * sell_px
                sell_fee = sell_amount * sell_cost_rate
                stamp_tax = sell_amount * stamp_tax_rate
                pnl = sell_amount - sell_fee - stamp_tax - lot.buy_amount - lot.buy_fee
                cash += sell_amount - sell_fee - stamp_tax
                rec = TradeRecord(
                    trade_id=lot.trade_id,
                    stock_code=lot.stock_code,
                    model_name=lot.model_name,
                    model_type=lot.model_type,
                    sector=lot.sector,
                    buy_date=lot.buy_date,
                    sell_date=date_dash,
                    buy_price=lot.buy_price,
                    sell_price=sell_px,
                    shares=lot.shares,
                    buy_amount=lot.buy_amount,
                    sell_amount=sell_amount,
                    buy_fee=lot.buy_fee,
                    sell_fee=sell_fee,
                    stamp_tax=stamp_tax,
                    pnl=pnl,
                    net_return=pnl / (lot.buy_amount + lot.buy_fee) if lot.buy_amount + lot.buy_fee > 0 else np.nan,
                    hold_days_calendar=max(0, (date - pd.to_datetime(lot.buy_date)).days),
                    ev_bps=lot.ev_bps,
                    utility_bps=lot.utility_bps,
                    exit_reason="scheduled_hold_days",
                )
                trades.append(rec)
                realized_today.append(rec)
            else:
                still_open.append(lot)
        open_lots = still_open

        account = account_from_state(cash, open_lots, history, date)
        account_path = day_out / f"sim_account_{ymd}.json"
        account_path.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")

        risk_history_path = write_point_in_time_risk_history(history, date, day_out)

        orders_path = run_portfolio_adapter(
            date=date,
            signal_dir=signal_root / ymd,
            account_json=account_path,
            history_path=risk_history_path,
            saved_models=saved_models,
            config=config_path,
            out_dir=day_out,
            optimizer_script=optimizer_script,
            use_covariance_penalty=use_covariance_penalty,
            cov_risk_aversion=cov_risk_aversion,
            time_limit_sec=time_limit_sec,
        )
        orders = read_orders(orders_path)

        buy_count = 0
        buy_cost_today = 0.0
        skipped = []
        if not orders.empty:
            for i, r in orders.iterrows():
                shares = int(as_float(r.get("buy_shares", 0), 0))
                if shares <= 0:
                    continue
                code = normalize_stock_code(r.get("stock_code", ""))
                px = as_float(r.get("price"), np.nan)
                if not np.isfinite(px) or px <= 0:
                    px = price_on_or_before(history, date, code)
                if not code or not np.isfinite(px) or px <= 0:
                    skipped.append({"row": int(i), "stock_code": code, "reason": "invalid_price"})
                    continue
                sell_date = next_trading_date(history, date, hold_days)
                if sell_date is None:
                    skipped.append({"row": int(i), "stock_code": code, "reason": "no_future_sell_date"})
                    continue
                buy_amount = shares * px
                buy_fee = buy_amount * buy_cost_rate
                total_cost = buy_amount + buy_fee
                if total_cost > cash + 1e-6:
                    skipped.append({"row": int(i), "stock_code": code, "reason": "insufficient_cash", "need": total_cost, "cash": cash})
                    continue
                cash -= total_cost
                open_lots.append(OpenLot(
                    trade_id=f"{ymd}_{code}_{len(trades)+len(open_lots)+1:05d}",
                    stock_code=code,
                    buy_date=date_dash,
                    planned_sell_date=sell_date.strftime("%Y-%m-%d"),
                    buy_price=float(px),
                    shares=shares,
                    buy_amount=float(buy_amount),
                    buy_fee=float(buy_fee),
                    model_name=str(r.get("model_name", "")),
                    model_type=str(r.get("model_type", "")),
                    sector=str(r.get("sector", "")),
                    ev_bps=as_float(r.get("ev_bps", np.nan), np.nan),
                    utility_bps=as_float(r.get("utility_bps", np.nan), np.nan),
                ))
                buy_count += 1
                buy_cost_today += total_cost

        market_value = 0.0
        for lot in open_lots:
            px = price_on_or_before(history, date, lot.stock_code)
            if not np.isfinite(px) or px <= 0:
                px = lot.buy_price
            market_value += lot.shares * px
        equity = cash + market_value
        prev_equity = equity_rows[-1]["equity"] if equity_rows else initial_cash
        daily_return = equity / prev_equity - 1.0 if prev_equity else 0.0

        equity_rows.append({
            "date": date_dash,
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "daily_return": daily_return,
            "open_positions": len({lot.stock_code for lot in open_lots}),
            "open_lots": len(open_lots),
            "gross_exposure": market_value / equity if equity > 0 else 0.0,
        })
        daily_rows.append({
            "date": date_dash,
            "signal_dir": str(signal_root / ymd),
            "orders_path": str(orders_path),
            "orders": int(len(orders)),
            "executed_buys": buy_count,
            "realized_sells": int(len(realized_today)),
            "buy_cost": buy_cost_today,
            "realized_pnl": float(sum(x.pnl for x in realized_today)),
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "skipped_buys": json.dumps(skipped, ensure_ascii=False),
        })

    if open_lots and close_open_at_end and dates:
        final_date = dates[-1]
        final_dash = final_date.strftime("%Y-%m-%d")
        for lot in list(open_lots):
            sell_px = price_on_or_before(history, final_date, lot.stock_code)
            if not np.isfinite(sell_px) or sell_px <= 0:
                continue
            sell_amount = lot.shares * sell_px
            sell_fee = sell_amount * sell_cost_rate
            stamp_tax = sell_amount * stamp_tax_rate
            pnl = sell_amount - sell_fee - stamp_tax - lot.buy_amount - lot.buy_fee
            cash += sell_amount - sell_fee - stamp_tax
            trades.append(TradeRecord(
                trade_id=lot.trade_id,
                stock_code=lot.stock_code,
                model_name=lot.model_name,
                model_type=lot.model_type,
                sector=lot.sector,
                buy_date=lot.buy_date,
                sell_date=final_dash,
                buy_price=lot.buy_price,
                sell_price=sell_px,
                shares=lot.shares,
                buy_amount=lot.buy_amount,
                sell_amount=sell_amount,
                buy_fee=lot.buy_fee,
                sell_fee=sell_fee,
                stamp_tax=stamp_tax,
                pnl=pnl,
                net_return=pnl / (lot.buy_amount + lot.buy_fee) if lot.buy_amount + lot.buy_fee > 0 else np.nan,
                hold_days_calendar=max(0, (final_date - pd.to_datetime(lot.buy_date)).days),
                ev_bps=lot.ev_bps,
                utility_bps=lot.utility_bps,
                exit_reason="force_close_at_end",
            ))
            open_lots.remove(lot)
        if equity_rows:
            equity_rows[-1].update({"cash": cash, "market_value": 0.0, "equity": cash, "open_positions": 0, "open_lots": 0, "gross_exposure": 0.0})

    trades_df = pd.DataFrame([asdict(x) for x in trades])
    equity_df = pd.DataFrame(equity_rows)
    daily_df = pd.DataFrame(daily_rows)
    open_df = pd.DataFrame([asdict(x) for x in open_lots])

    if not equity_df.empty:
        equity_df["equity_peak"] = pd.to_numeric(equity_df["equity"], errors="coerce").cummax()
        equity_df["drawdown"] = pd.to_numeric(equity_df["equity"], errors="coerce") / equity_df["equity_peak"] - 1.0

    return {
        "summary": summarize(equity_df, trades_df, initial_cash),
        "trades": trades_df,
        "equity": equity_df,
        "daily": daily_df,
        "open_lots": open_df,
    }



def _profit_factor_from_pnl(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    profit = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    if loss <= 0:
        return float("inf") if profit > 0 else 0.0
    return profit / loss


def write_perf_summaries(trades_df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    paths = {
        "model_perf": out_dir / "model_perf_summary.csv",
        "stock_perf": out_dir / "stock_perf_summary.csv",
        "suggested_recent_perf": out_dir / "suggested_portfolio_model_recent_perf.csv",
    }
    if trades_df.empty:
        for path in paths.values():
            pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        return {k: str(v) for k, v in paths.items()}

    df = trades_df.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")

    def summarize_group(g: pd.DataFrame) -> pd.Series:
        pnl = pd.to_numeric(g["pnl"], errors="coerce")
        ret = pd.to_numeric(g["net_return"], errors="coerce")
        return pd.Series({
            "trades": int(len(g)),
            "pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()) if len(g) else 0.0,
            "profit_factor": _profit_factor_from_pnl(pnl),
            "avg_return": float(ret.mean()) if len(g) else 0.0,
            "median_return": float(ret.median()) if len(g) else 0.0,
            "max_loss": float(pnl.min()) if len(g) else 0.0,
        })

    model_cols = [c for c in ["stock_code", "model_name"] if c in df.columns]
    model_perf = df.groupby(model_cols, dropna=False).apply(summarize_group).reset_index() if model_cols else pd.DataFrame()
    stock_perf = df.groupby(["stock_code"], dropna=False).apply(summarize_group).reset_index() if "stock_code" in df.columns else pd.DataFrame()

    suggested = model_perf.copy()
    if not suggested.empty:
        def mult(row):
            trades = float(row.get("trades", 0))
            pf = float(row.get("profit_factor", 0))
            pnl = float(row.get("pnl", 0))
            if trades >= 10 and (pf < 0.9 or pnl < 0):
                return 0.30
            if trades >= 10 and pf < 1.1:
                return 0.60
            if trades >= 20 and pf > 1.5 and pnl > 0:
                return 1.10
            return 1.00
        suggested["artifact_pattern"] = suggested.get("model_name", "")
        suggested["enabled"] = 1
        suggested["weight_multiplier"] = suggested.apply(mult, axis=1)
        suggested["notes"] = "generated_from_backtest_perf; review before using as RECENT_PERF"
        suggested = suggested[["stock_code", "artifact_pattern", "enabled", "weight_multiplier", "trades", "pnl", "win_rate", "profit_factor", "avg_return", "median_return", "notes"]]

    model_perf.to_csv(paths["model_perf"], index=False, encoding="utf-8-sig")
    stock_perf.to_csv(paths["stock_perf"], index=False, encoding="utf-8-sig")
    suggested.to_csv(paths["suggested_recent_perf"], index=False, encoding="utf-8-sig")
    return {k: str(v) for k, v in paths.items()}


def read_watchlist(path: Optional[Path]) -> Optional[set[str]]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    out = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.add(normalize_stock_code(token))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate historical daily scores, portfolio decisions, and strategy evaluation")
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--saved-data-dir", default="saved_data")
    ap.add_argument("--context-config", default="configs/realtime_context_sources.toml")
    ap.add_argument("--watchlist", default=None)
    ap.add_argument("--model-policy", choices=["all", "preferred"], default="all")
    ap.add_argument("--history", default=None, help="Optional close history file; not required.")
    ap.add_argument("--config", default="configs/portfolio_confirm_config.json")
    ap.add_argument("--out-dir", default="portfolio_reports/backtests/historical_score_portfolio")
    ap.add_argument("--generated-signal-root", default=None)
    ap.add_argument("--optimizer-script", default="portfolio_decision/daily_portfolio_confirm_pyscipopt.py")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--restore-end-date", default="today")
    ap.add_argument("--initial-cash", type=float, default=200000.0)
    ap.add_argument("--hold-days", type=int, default=1)
    ap.add_argument("--min-amount-yuan", type=float, default=50000000.0)
    ap.add_argument("--close-open-at-end", action="store_true")
    ap.add_argument("--use-covariance-penalty", action="store_true")
    ap.add_argument("--cov-risk-aversion", type=float, default=None)
    ap.add_argument("--time-limit-sec", type=float, default=30.0)
    ap.add_argument("--score-only", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir)
    saved_data_dir = Path(args.saved_data_dir)
    context_config = Path(args.context_config)
    config_path = Path(args.config) if args.config else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    watchlist = read_watchlist(Path(args.watchlist)) if args.watchlist else None
    states = load_artifact_states(
        models_dir=models_dir,
        watchlist=watchlist,
        model_policy=args.model_policy,
        saved_data_dir=saved_data_dir,
        context_config=context_config,
        restore_end_date=str(args.restore_end_date),
    )
    if not states:
        raise SystemExit("no usable saved model artifacts found")

    history_path = Path(args.history) if args.history else None
    if history_path is not None and history_path.exists():
        history = load_history_close(history_path)
        portfolio_history_path = history_path
        history_source = str(history_path)
    else:
        history = build_history_close_from_states(states)
        portfolio_history_path = None
        history_source = "in_memory_from_model_samples"

    eval_cfg = load_eval_config(config_path)

    start_ts = pd.to_datetime(args.start_date).normalize() if args.start_date else None
    end_ts = pd.to_datetime(args.end_date).normalize() if args.end_date else None
    date_set = set()
    for st in states:
        for d in st.samples["date"].dropna().unique():
            dt = pd.Timestamp(d).normalize()
            if start_ts is not None and dt < start_ts:
                continue
            if end_ts is not None and dt > end_ts:
                continue
            if dt in history.index:
                date_set.add(dt)
    dates = sorted(date_set)
    if not dates:
        raise SystemExit("no overlapping dates among model samples and close history in requested range")

    signal_root = Path(args.generated_signal_root) if args.generated_signal_root else out_dir / "generated_signals"
    score_summary = generate_daily_scores(states, dates, signal_root, float(args.min_amount_yuan))

    if args.score_only:
        print(f"[SCORE_ONLY] generated signals under {signal_root}")
        return 0

    sim = simulate_portfolio(
        dates=dates,
        signal_root=signal_root,
        history=history,
        history_path=portfolio_history_path,
        saved_models=models_dir,
        config_path=config_path,
        out_dir=out_dir,
        optimizer_script=args.optimizer_script,
        initial_cash=float(args.initial_cash),
        hold_days=int(args.hold_days),
        close_open_at_end=bool(args.close_open_at_end),
        eval_cfg=eval_cfg,
        use_covariance_penalty=bool(args.use_covariance_penalty),
        cov_risk_aversion=args.cov_risk_aversion,
        time_limit_sec=float(args.time_limit_sec),
    )

    perf_summary_paths = write_perf_summaries(sim["trades"], out_dir)
    summary = sim["summary"]
    summary.update({
        "perf_summary_paths": perf_summary_paths,
        "source_mode": "historical_samples_generated_scores",
        "model_policy": args.model_policy,
        "models_dir": str(models_dir),
        "history_source": history_source,
        "risk_history_mode": "point_in_time_daily_csv",
        "risk_history_include_current_day": True,
        "signal_root": str(signal_root),
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "scored_days": int(len(score_summary)),
        "artifacts_loaded": int(len(states)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Replay of current saved model library on historical samples; not strict walk-forward unless artifacts are point-in-time.",
    })

    paths = {
        "summary": out_dir / "historical_score_portfolio_backtest_summary.json",
        "equity": out_dir / "historical_score_portfolio_backtest_equity.csv",
        "daily": out_dir / "historical_score_portfolio_backtest_daily.csv",
        "trades": out_dir / "historical_score_portfolio_backtest_trades.csv",
        "open_lots": out_dir / "historical_score_portfolio_backtest_open_lots.csv",
        "score_summary": signal_root / "historical_score_generation_summary.csv",
    }
    sim["equity"].to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    sim["daily"].to_csv(paths["daily"], index=False, encoding="utf-8-sig")
    sim["trades"].to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    sim["open_lots"].to_csv(paths["open_lots"], index=False, encoding="utf-8-sig")
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OUTPUTS]")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
