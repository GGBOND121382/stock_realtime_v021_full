#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rolling pre-trade retrain backtest.

For each trade date T and each saved artifact configuration, train a fresh
model using only rows whose labels would be known before T, choose the threshold
on the trailing validation window, score T, then run the existing portfolio
simulator from generated buy_signals.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_saving.save_nextday_model import get_model_template
from model_training.optimize_nextday_vwap_model import choose_valid_threshold, prepare_x_by_median, trade_metrics
from model_training.search_walk_forward_model_complexity import clone_model, predict_positive
from portfolio_decision.backtest_historical_score_portfolio import (
    as_float,
    build_history_close_from_states,
    build_history_high_from_states,
    load_artifact_states,
    load_eval_config,
    read_watchlist,
    simulate_portfolio,
    summarize,
)


def retrain_cutoff_for_date(
    trade_date: pd.Timestamp,
    all_dates: list[pd.Timestamp],
    date_pos: dict[pd.Timestamp, int],
    period_start_by_date: dict[pd.Timestamp, pd.Timestamp],
) -> pd.Timestamp | None:
    train_asof = period_start_by_date.get(trade_date, trade_date)
    pos = date_pos.get(train_asof)
    if pos is None or pos < 2:
        return None
    return all_dates[pos - 2]


def period_starts(dates: list[pd.Timestamp], frequency: str) -> dict[pd.Timestamp, pd.Timestamp]:
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    first_by_period: dict[Any, pd.Timestamp] = {}
    for d in dates:
        if frequency == "daily":
            key = d
        elif frequency == "weekly":
            iso = d.isocalendar()
            key = (int(iso.year), int(iso.week))
        elif frequency == "monthly":
            key = (int(d.year), int(d.month))
        else:
            raise ValueError(f"unknown retrain_frequency={frequency}")
        first_by_period.setdefault(key, d)
        out[d] = first_by_period[key]
    return out


def fit_classifier_model(
    template: Any,
    fit_df: pd.DataFrame,
    cols: list[str],
    label_col: str,
    min_train_entries: int,
) -> tuple[Any, pd.Series] | None:
    if len(fit_df) < min_train_entries or fit_df[label_col].nunique() < 2:
        return None
    median = fit_df[cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x_fit = fit_df[cols].apply(pd.to_numeric, errors="coerce")
    x_fit = x_fit.fillna(median).replace([np.inf, -np.inf], np.nan).fillna(median).fillna(0.0)
    y = fit_df[label_col].to_numpy(int)
    model = clone_model(template)
    params = model.get_params() if hasattr(model, "get_params") else {}
    if "scale_pos_weight" in params:
        pos_n = max(float(np.sum(y == 1)), 1.0)
        neg_n = max(float(np.sum(y == 0)), 1.0)
        model.set_params(scale_pos_weight=neg_n / pos_n)
    model.fit(x_fit, y)
    return model, median


def apply_median(frame: pd.DataFrame, cols: list[str], median: pd.Series) -> pd.DataFrame:
    x = frame[cols].apply(pd.to_numeric, errors="coerce")
    return x.fillna(median).replace([np.inf, -np.inf], np.nan).fillna(median).fillna(0.0)


def oof_validation_scores(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    template: Any,
    cols: list[str],
    label_col: str,
    return_col: str,
    args: argparse.Namespace,
) -> pd.DataFrame | None:
    scored_parts = []
    valid_positions = list(valid.index)
    folds = [fold for fold in np.array_split(valid_positions, int(args.oof_folds)) if len(fold) > 0]
    for fold in folds:
        fold_start_idx = int(fold[0])
        fold_valid = history.loc[fold].copy()
        fold_train = history.loc[history.index < fold_start_idx].copy()
        fold_fit = fold_train.loc[fold_train["entry_signal"].to_numpy(bool)].copy()
        fitted = fit_classifier_model(template, fold_fit, cols, label_col, int(args.min_train_entries))
        if fitted is None:
            return None
        model, median = fitted
        x_fold = apply_median(fold_valid, cols, median)
        fold_scored = fold_valid[["date", "entry_signal", return_col, label_col]].copy()
        fold_scored = fold_scored.rename(columns={return_col: "selected_eval_return", label_col: "eval_label"})
        fold_scored["hit_score"] = predict_positive(model, x_fold)
        scored_parts.append(fold_scored)
    if not scored_parts:
        return None
    return pd.concat(scored_parts, ignore_index=False).sort_values("date")


def train_score_one(
    state: Any,
    trade_date: pd.Timestamp,
    all_dates: list[pd.Timestamp],
    date_pos: dict[pd.Timestamp, int],
    period_start_by_date: dict[pd.Timestamp, pd.Timestamp],
    model_cache: dict[tuple[str, pd.Timestamp], dict[str, Any]],
    args: argparse.Namespace,
) -> dict | None:
    df = state.samples.sort_values("date").reset_index(drop=True)
    label_cutoff = retrain_cutoff_for_date(trade_date, all_dates, date_pos, period_start_by_date)
    if label_cutoff is None:
        return None

    # Before trade date T, the latest fully known next-day label is T-2's row:
    # T-1's next_day_close is T close, unavailable before T.
    meta = state.metadata
    label_mode = str(meta.get("label_mode", "close_profit"))
    if label_mode == "hit":
        label_col = "trade_hit_label"
        return_col = "trade_target_or_close_return"
    else:
        label_col = "trade_close_profit_label"
        return_col = "trade_net_close_return"

    cols = state.feature_columns
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{state.stock_code} {state.artifact_name} missing features: {missing[:30]} total={len(missing)}")

    cache_key = (str(state.artifact_dir), label_cutoff)
    cached = model_cache.get(cache_key)
    if cached is None:
        history = df[df["date"] <= label_cutoff].dropna(subset=["trade_net_close_return"]).copy()
        if history.empty:
            return None
        if label_col not in history.columns:
            history[label_col] = (history["trade_net_close_return"] > 0).astype(int)
        configured_valid_rows = (
            int(meta.get("valid_rows_for_threshold") or args.valid_rows)
            if args.use_metadata_valid_rows
            else int(args.valid_rows)
        )
        valid_rows = min(configured_valid_rows, max(1, len(history) // 5))
        if len(history) <= valid_rows + 1:
            return None
        train = history.iloc[:-valid_rows].copy()
        valid = history.iloc[-valid_rows:].copy()
        fit_train = train.loc[train["entry_signal"].to_numpy(bool)].copy()
        if len(fit_train) < args.min_train_entries or fit_train[label_col].nunique() < 2:
            return None

        template = get_model_template(str(meta.get("model_name")))
        fitted = fit_classifier_model(template, fit_train, cols, label_col, int(args.min_train_entries))
        if fitted is None:
            return None
        threshold_model, threshold_median = fitted
        if args.threshold_mode == "oof":
            valid_scored = oof_validation_scores(history, valid, template, cols, label_col, return_col, args)
            if valid_scored is None:
                return None
        else:
            x_valid = apply_median(valid, cols, threshold_median)
            valid_scored = valid[["date", "entry_signal", return_col, label_col]].copy()
            valid_scored = valid_scored.rename(columns={return_col: "selected_eval_return", label_col: "eval_label"})
            valid_scored["hit_score"] = predict_positive(threshold_model, x_valid)
        threshold_info = choose_valid_threshold(
            valid_scored,
            "hit_score",
            [float(x) for x in args.quantiles.split(",") if x.strip()],
            args.min_valid_trades,
            "selected_eval_return",
            "none",
        )
        if threshold_info is None:
            return None
        threshold = float(threshold_info["threshold"])
        selected_ret = valid_scored.loc[
            valid_scored["entry_signal"].to_numpy(bool) & (valid_scored["hit_score"] >= threshold),
            "selected_eval_return",
        ].dropna()
        validation_metrics = trade_metrics(selected_ret)

        if args.final_refit_with_valid or args.threshold_mode == "oof":
            final_fit = history.loc[history["entry_signal"].to_numpy(bool)].copy()
            fitted = fit_classifier_model(template, final_fit, cols, label_col, int(args.min_train_entries))
            if fitted is None:
                return None
            model, median = fitted
            final_fit_rows = int(len(final_fit))
        else:
            model = threshold_model
            median = threshold_median
            final_fit_rows = int(len(fit_train))
        cached = {
            "model": model,
            "median": median,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "valid_scored": valid_scored,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "fit_train_rows": int(len(fit_train)),
            "final_fit_rows": final_fit_rows,
            "threshold_mode": str(args.threshold_mode),
            "valid_auc": float(roc_auc_score(valid_scored["eval_label"], valid_scored["hit_score"]))
            if valid_scored["eval_label"].nunique() > 1 else np.nan,
        }
        model_cache[cache_key] = cached
    model = cached["model"]
    median = cached["median"]
    threshold = float(cached["threshold"])

    day = df[df["date"] == trade_date]
    if day.empty:
        return None
    row = day.iloc[-1]
    x_day = pd.DataFrame([row[cols].to_dict()]).apply(pd.to_numeric, errors="coerce")
    x_day = x_day.fillna(median).replace([np.inf, -np.inf], np.nan).fillna(median).fillna(0.0)
    score = float(predict_positive(model, x_day)[0])
    entry_signal = bool(row.get("entry_signal"))
    score_pass = score >= threshold
    amount = as_float(row.get("amount", row.get("daily_amount", np.nan)), np.nan)
    reject_reasons = []
    if not entry_signal:
        reject_reasons.append("entry_signal_false")
    if not score_pass:
        reject_reasons.append("score_below_threshold")
    if args.min_amount_yuan > 0 and np.isfinite(amount) and amount < args.min_amount_yuan:
        reject_reasons.append(f"amount_lt_{int(args.min_amount_yuan)}")

    validation_metrics = cached["validation_metrics"]
    pred_return_bps = as_float(validation_metrics.get("avg_return"), np.nan) * 10000.0

    return {
        "rank": np.nan,
        "trade_date": trade_date.strftime("%Y%m%d"),
        "date": trade_date.strftime("%Y-%m-%d"),
        "stock_code": state.stock_code,
        "artifact_name": state.artifact_name,
        "artifact_dir": str(state.artifact_dir),
        "samples": str(state.samples_path),
        "entry_policy": meta.get("entry_policy", ""),
        "label_mode": label_mode,
        "model_name": meta.get("model_name", ""),
        "feature_group": meta.get("feature_group", ""),
        "close": as_float(row.get("close"), np.nan),
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
        "target_hit_bps": as_float(meta.get("target_hit_bps"), 50.0),
        "round_trip_cost_bps": as_float(meta.get("round_trip_cost_bps"), np.nan),
        "expected_return_col": return_col,
        "realized_return": as_float(row.get(return_col), np.nan),
        "eval_label": as_float(row.get(label_col), np.nan),
        "sector": row.get("sector", row.get("industry", "")),
        "context_status": "rolling_retrain",
        "source_mode": "rolling_retrain",
        "rolling_label_cutoff": label_cutoff.strftime("%Y-%m-%d"),
        "rolling_threshold_mode": str(cached.get("threshold_mode", args.threshold_mode)),
        "rolling_train_rows": int(cached["train_rows"]),
        "rolling_valid_rows": int(cached["valid_rows"]),
        "rolling_fit_train_entry_rows": int(cached["fit_train_rows"]),
        "rolling_final_fit_entry_rows": int(cached.get("final_fit_rows", cached["fit_train_rows"])),
        "rolling_valid_auc": as_float(cached["valid_auc"], np.nan),
        "rolling_valid_trades": int(validation_metrics.get("trades", 0)),
        "rolling_valid_avg_return": as_float(validation_metrics.get("avg_return"), np.nan),
        "rolling_valid_max_drawdown": as_float(validation_metrics.get("max_drawdown"), np.nan),
        "pred_return_bps_override": pred_return_bps,
    }


def generate_scores(states: list[Any], dates: list[pd.Timestamp], signal_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    signal_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_dates = sorted({pd.Timestamp(d).normalize() for st in states for d in st.samples["date"].dropna().unique()})
    date_pos = {d: i for i, d in enumerate(all_dates)}
    period_start_by_date = period_starts(dates, args.retrain_frequency)
    model_cache: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for date in dates:
        day_dir = signal_root / date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for st in states:
            r = train_score_one(st, date, all_dates, date_pos, period_start_by_date, model_cache, args)
            if r is not None:
                rows.append(r)
        all_scores = pd.DataFrame(rows)
        if not all_scores.empty:
            all_scores = all_scores.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            all_scores["rank"] = np.arange(1, len(all_scores) + 1)
            buy_mask = (all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")
            buy = all_scores[buy_mask].copy().reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~buy_mask].copy()
        else:
            buy = pd.DataFrame()
            rejected = pd.DataFrame()
        all_scores.to_csv(day_dir / "all_scores.csv", index=False, encoding="utf-8-sig")
        buy.to_csv(day_dir / "buy_signals.csv", index=False, encoding="utf-8-sig")
        rejected.to_csv(day_dir / "rejected_scores.csv", index=False, encoding="utf-8-sig")
        row = {
            "date": date.strftime("%Y%m%d"),
            "source_mode": "rolling_retrain",
            "artifacts_scored": int(len(all_scores)),
            "buy_signals": int(len(buy)),
            "rejected": int(len(rejected)),
            "retrain_frequency": args.retrain_frequency,
            "threshold_mode": args.threshold_mode,
            "period_train_start": period_start_by_date.get(date, date).strftime("%Y%m%d"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (day_dir / "run_summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(row)
        print(f"[ROLLING SCORED] {date:%Y%m%d} all={len(all_scores)} buy={len(buy)}", flush=True)
    out = pd.DataFrame(summary_rows)
    out.to_csv(signal_root / "historical_score_generation_summary.csv", index=False, encoding="utf-8-sig")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling retrain portfolio backtest")
    p.add_argument("--models-dir", default="saved_models")
    p.add_argument("--saved-data-dir", default="saved_data")
    p.add_argument("--context-config", default="configs/realtime_context_sources.toml")
    p.add_argument("--config", default="configs/portfolio_confirm_config.json")
    p.add_argument("--watchlist", default=None)
    p.add_argument("--model-policy", choices=["all", "preferred"], default="all")
    p.add_argument("--out-dir", default="portfolio_reports/backtests/rolling_retrain_portfolio")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", default=None)
    p.add_argument("--initial-cash", type=float, default=200000.0)
    p.add_argument("--hold-days", type=int, default=1)
    p.add_argument("--min-amount-yuan", type=float, default=50000000.0)
    p.add_argument("--valid-rows", type=int, default=252)
    p.add_argument("--use-metadata-valid-rows", action="store_true")
    p.add_argument("--final-refit-with-valid", action="store_true")
    p.add_argument("--threshold-mode", choices=["tail", "oof"], default="tail")
    p.add_argument("--oof-folds", type=int, default=3)
    p.add_argument("--retrain-frequency", choices=["daily", "weekly", "monthly"], default="weekly")
    p.add_argument("--min-train-entries", type=int, default=80)
    p.add_argument("--min-valid-trades", type=int, default=8)
    p.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    p.add_argument("--close-open-at-end", action="store_true")
    p.add_argument("--time-limit-sec", type=float, default=30.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    watchlist = read_watchlist(Path(args.watchlist)) if args.watchlist else None
    states = load_artifact_states(
        models_dir=Path(args.models_dir),
        watchlist=watchlist,
        model_policy=args.model_policy,
        saved_data_dir=Path(args.saved_data_dir),
        context_config=Path(args.context_config),
        restore_end_date=str(args.end_date or "today"),
    )
    history = build_history_close_from_states(states)
    high_history = build_history_high_from_states(states)
    start_ts = pd.to_datetime(args.start_date).normalize()
    end_ts = pd.to_datetime(args.end_date).normalize() if args.end_date else None
    date_set = set()
    for st in states:
        for d in st.samples["date"].dropna().unique():
            dt = pd.Timestamp(d).normalize()
            if dt < start_ts or (end_ts is not None and dt > end_ts):
                continue
            if dt in history.index:
                date_set.add(dt)
    dates = sorted(date_set)
    if not dates:
        raise SystemExit("no dates to score")

    signal_root = out_dir / "generated_signals"
    score_summary = generate_scores(states, dates, signal_root, args)
    sim = simulate_portfolio(
        dates=dates,
        signal_root=signal_root,
        history=history,
        high_history=high_history,
        history_path=None,
        saved_models=Path(args.models_dir),
        config_path=Path(args.config),
        out_dir=out_dir,
        optimizer_script="portfolio_decision/daily_portfolio_confirm_pyscipopt.py",
        initial_cash=float(args.initial_cash),
        hold_days=int(args.hold_days),
        close_open_at_end=bool(args.close_open_at_end),
        eval_cfg=load_eval_config(Path(args.config)),
        use_covariance_penalty=False,
        cov_risk_aversion=None,
        time_limit_sec=float(args.time_limit_sec),
    )
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
    summary = sim["summary"]
    summary.update({
        "source_mode": "rolling_retrain",
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "scored_days": int(len(score_summary)),
        "artifacts_loaded": int(len(states)),
        "retrain_frequency": args.retrain_frequency,
        "threshold_mode": args.threshold_mode,
        "oof_folds": int(args.oof_folds),
        "label_cutoff_rule": "train rows date <= two trading sessions before trade date",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OUTPUTS]")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
