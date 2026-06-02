#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rolling retrain portfolio backtest for model-retention active sets.

This is the 14:55-as-of counterpart of rolling_retrain_portfolio_backtest.py.
It does not use the historical sample path stored in saved model metadata.
Instead, each retention-table row is rebuilt from canonical per-stock pipeline
outputs under saved_data/<code>_pipeline_out.

By default this script runs two portfolio backtests and compares them:
1. A_Active only.
2. A_Active + B_Backup.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_training.optimize_nextday_vwap_model import add_market_state_features, add_trade_returns, feature_groups
from portfolio_decision.backtest_historical_score_portfolio import (
    as_float,
    build_history_close_from_states,
    build_history_high_from_states,
    load_eval_config,
    read_watchlist,
    simulate_portfolio,
)
from portfolio_decision.backtest_historical_score_portfolio import normalize_stock_code
from scripts.rolling_retrain_portfolio_backtest import period_starts, train_score_one


FEATURE_GROUP_ALIASES = {
    "reversal": "reversal_fundamental_regime",
    "sector": "reversal_fundamental_regime_sector",
    "sector_external": "reversal_fundamental_regime_sector_external",
    "reversal_external": "reversal_fundamental_regime_external",
    "all_no_ak": "all_no_ak",
}


SCORE_COLUMNS = [
    "rank", "trade_date", "date", "stock_code", "artifact_name", "artifact_dir",
    "samples", "entry_policy", "label_mode", "model_name", "feature_group",
    "close", "open", "high", "low", "volume", "amount", "daily_vwap",
    "hit_score", "threshold", "score_margin", "entry_signal",
    "signal_raw_score_pass", "signal", "reject_reason", "target_hit_bps",
    "round_trip_cost_bps", "expected_return_col", "realized_return",
    "eval_label", "sector", "context_status", "source_mode",
    "rolling_label_cutoff", "rolling_threshold_mode", "rolling_train_rows",
    "rolling_valid_rows", "rolling_train_pool_rows",
    "rolling_configured_train_rows", "rolling_train_window_mode",
    "rolling_fit_train_entry_rows", "rolling_final_fit_entry_rows",
    "rolling_valid_auc", "rolling_valid_trades", "rolling_valid_avg_return",
    "rolling_valid_max_drawdown", "pred_return_bps_override",
]


EXTERNAL_STAGE_BY_NAME = {
    "storage_power": "external_storage_power",
    "power_utility_rate": "external_power_utility_rate",
    "ai_compute": "external_ai_compute",
    "feed": "external_feed",
    "hog": "external_hog",
    "fertilizer": "external_fertilizer",
    "zijin_external": "external_zijin_external",
    "optical_cable_grid": "external_optical_cable_grid",
    "aero_nuclear_equipment": "external_aero_nuclear_equipment",
}


@dataclass
class ActiveState:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    metadata: dict[str, Any]
    samples_path: Path
    samples: pd.DataFrame
    feature_columns: list[str]
    feature_median: pd.Series
    model: Any = None


def safe_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def repo_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(str(value))
    if p.exists():
        return p
    marker = PROJECT_DIR.name
    parts = p.parts
    if marker in parts:
        cand = PROJECT_DIR.joinpath(*parts[parts.index(marker) + 1 :])
        if cand.exists():
            return cand
    cand = PROJECT_DIR / str(value)
    return cand if cand.exists() else None


def raw_code(symbol: str) -> str:
    return normalize_stock_code(symbol).split(".", 1)[0]


def label_info(label: Any) -> tuple[str, float]:
    text = str(label or "").strip().lower()
    if text.startswith("hit"):
        digits = "".join(ch for ch in text if ch.isdigit())
        return "hit", float(digits or 50)
    return "close_profit", 50.0


def mapped_feature_group(value: Any) -> str:
    key = safe_name(value)
    return FEATURE_GROUP_ALIASES.get(key, key)


def load_pipeline_paths(asof_data_root: Path, stock_code: str) -> tuple[Path, Path, dict[str, Any]]:
    pipeline_dir = asof_data_root / f"{raw_code(stock_code)}_pipeline_out"
    meta_path = pipeline_dir / "pipeline_summary.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing pipeline_summary.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samples = repo_path(meta.get("final_samples"))
    intraday = repo_path(meta.get("intraday_bars"))
    if samples is None or not samples.exists():
        raise FileNotFoundError(f"missing final_samples for {stock_code}: {meta.get('final_samples')}")
    if intraday is None or not intraday.exists():
        raise FileNotFoundError(f"missing intraday_bars for {stock_code}: {meta.get('intraday_bars')}")
    return samples, intraday, meta


def asof_data_available(asof_data_root: Path, stock_code: str) -> bool:
    try:
        load_pipeline_paths(asof_data_root, stock_code)
        return True
    except Exception:
        return False


def load_retention_table(path: Path, sheets: list[str]) -> pd.DataFrame:
    frames = []
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet)
        df = df.dropna(subset=["Stock", "Model", "FeatureGroup", "Label", "Entry"]).copy()
        df["Stock"] = df["Stock"].map(normalize_stock_code)
        df = df[df["Stock"].astype(str) != ""].reset_index(drop=True)
        df["_source_sheet"] = sheet
        df["_source_row"] = np.arange(1, len(df) + 1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sheet_list(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def external_steps(external: Any) -> list[str]:
    text = str(external or "").strip()
    if not text or text == "-":
        return []
    return [x.strip() for x in text.split(",") if x.strip() and x.strip() != "-"]


def source_pipeline_summary_from_origin(row: pd.Series) -> Path | None:
    origins = str(row.get("Origins") or "")
    for origin in origins.split(";"):
        token = origin.strip()
        if not token:
            continue
        pipeline_name = token.split("/", 1)[0].strip()
        if not pipeline_name:
            continue
        cand = PROJECT_DIR / "saved_data" / pipeline_name / "pipeline_summary.json"
        if cand.exists():
            return cand
    stock = raw_code(str(row.get("Stock") or ""))
    hits = sorted((PROJECT_DIR / "saved_data").glob(f"{stock}_pipeline_out/pipeline_summary.json"))
    return hits[-1] if hits else None


def asof_stage_names(external: Any) -> list[str]:
    stages = ["update_data", "samples", "asof_samples", "fundamental", "sector"]
    for ext in external_steps(external):
        stage = EXTERNAL_STAGE_BY_NAME.get(ext)
        if not stage:
            raise ValueError(f"unsupported external profile for auto asof build: {ext}")
        stages.append(stage)
    return stages


def build_asof_data_from_retention_row(args: argparse.Namespace, row: pd.Series) -> None:
    stock = str(row["Stock"])
    src_summary = source_pipeline_summary_from_origin(row)
    if src_summary is None:
        raise FileNotFoundError(f"cannot infer source pipeline for {stock}; no usable Origins pipeline_summary.json")
    src_meta = json.loads(src_summary.read_text(encoding="utf-8"))
    src_args = src_meta.get("args") or {}
    sector_symbol = src_args.get("sector_symbol")
    if not sector_symbol:
        raise ValueError(f"cannot infer sector_symbol for {stock} from {src_summary}")
    external = str(row.get("External") if pd.notna(row.get("External")) else src_args.get("external", "")).strip()
    if external == "-":
        external = ""
    out_dir = Path(args.asof_data_root) / f"{raw_code(stock)}_pipeline_out"
    cmd = [
        args.python,
        "pipelines/run_nextday_pipeline.py",
        "--symbol",
        stock,
        "--sector-symbol",
        str(sector_symbol),
        "--out-root",
        str(out_dir),
        "--start-date",
        str(args.asof_start_date),
        "--end-date",
        str(args.asof_end_date),
        "--feature-time-mode",
        "asof1455",
        "--feature-cutoff-time",
        str(args.cutoff_time),
        "--feature-pipeline",
        "fundamental,sector",
        "--only-stages",
        ",".join(asof_stage_names(external)),
        "--skip-akshare-fund-flow",
        "--continue-on-error",
        "--resume",
    ]
    if external:
        cmd.extend(["--external", external])
    print(f"[BUILD ASOF1455 FROM ORIGIN] {stock} origin={src_summary.parent}", flush=True)
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def ensure_asof_data(args: argparse.Namespace, sheets: list[str]) -> None:
    if args.no_build_missing_asof_data:
        return
    table = load_retention_table(Path(args.retention_xlsx), sheets)
    watchlist = read_watchlist(Path(args.watchlist)) if args.watchlist else None
    if watchlist:
        table = table[table["Stock"].isin(watchlist)].reset_index(drop=True)
    stocks = sorted(set(str(x) for x in table["Stock"].dropna() if str(x)))
    missing = [s for s in stocks if not asof_data_available(Path(args.asof_data_root), s)]
    if not missing:
        return

    for stock in missing:
        rows = table[table["Stock"] == stock]
        if rows.empty:
            continue
        build_asof_data_from_retention_row(args, rows.iloc[0])

    still_missing = [s for s in missing if not asof_data_available(Path(args.asof_data_root), s)]
    if still_missing:
        raise FileNotFoundError(f"asof1455 data still missing after build: {still_missing}")


def make_artifact_name(row: pd.Series, row_num: int) -> str:
    parts = [
        safe_name(row.get("_source_sheet", "retention")),
        f"{int(row.get('_source_row', row_num)):03d}",
        safe_name(row.get("Strategy")),
        safe_name(row.get("Label")),
        safe_name(row.get("Entry")),
        safe_name(row.get("FeatureGroup")),
        safe_name(row.get("Model")),
    ]
    return "_".join(parts)


def active_validation_metrics(row: pd.Series) -> dict[str, float]:
    return {
        "trades": as_float(row.get("Trades"), np.nan),
        "win_rate": as_float(row.get("WinRate"), np.nan),
        "avg_return": as_float(row.get("AvgReturn"), np.nan),
        "median_return": as_float(row.get("MedianReturn"), np.nan),
        "max_drawdown": as_float(row.get("MaxDrawdown"), np.nan),
        "profit_factor": as_float(row.get("ProfitFactor"), np.nan),
    }


def load_active_states(args: argparse.Namespace, metadata_root: Path, sheets: list[str]) -> tuple[list[ActiveState], pd.DataFrame]:
    table = load_retention_table(Path(args.retention_xlsx), sheets)
    watchlist = read_watchlist(Path(args.watchlist)) if args.watchlist else None
    if watchlist:
        table = table[table["Stock"].isin(watchlist)].reset_index(drop=True)

    sample_cache: dict[str, tuple[pd.DataFrame, Path, Path, dict[str, Any]]] = {}
    states: list[ActiveState] = []
    report_rows = []

    for i, row in table.iterrows():
        stock = str(row["Stock"])
        try:
            if stock not in sample_cache:
                samples_path, intraday_path, pipe_meta = load_pipeline_paths(Path(args.asof_data_root), stock)
                samples = pd.read_csv(samples_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
                samples["date"] = pd.to_datetime(samples["date"], errors="coerce").dt.normalize()
                samples = samples.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
                samples = add_market_state_features(samples)
                samples = samples.replace([np.inf, -np.inf], np.nan)
                sample_cache[stock] = (samples, samples_path, intraday_path, pipe_meta)
            samples, samples_path, intraday_path, pipe_meta = sample_cache[stock]

            label_mode, target_hit_bps = label_info(row.get("Label"))
            prepared = add_trade_returns(
                samples,
                cost_bps=float(args.round_trip_cost_bps),
                target_bps=target_hit_bps,
                entry_policy=str(row.get("Entry")).strip(),
                entry_vwap_premium_bps=float(args.entry_vwap_premium_bps),
                feature_time_mode="asof1455",
            )
            prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["trade_net_close_return", "trade_net_high_return", "trade_target_or_close_return"]
            ).reset_index(drop=True)

            group_name = mapped_feature_group(row.get("FeatureGroup"))
            groups = feature_groups(prepared, max_missing=float(args.max_missing), feature_time_mode="asof1455")
            cols = groups.get(group_name, [])
            if not cols:
                raise ValueError(f"empty feature group after asof1455 filtering: {group_name}")

            artifact_name = make_artifact_name(row, i + 1)
            artifact_dir = metadata_root / stock / artifact_name
            metadata = {
                "artifact_created_at": datetime.now().isoformat(timespec="seconds"),
                "stock_code": stock,
                "artifact_name": artifact_name,
                "samples": str(samples_path.resolve()),
                "intraday_bars": str(intraday_path.resolve()),
                "feature_group": group_name,
                "model_name": str(row.get("Model")).strip(),
                "label_mode": label_mode,
                "entry_policy": str(row.get("Entry")).strip(),
                "entry_vwap_premium_bps": float(args.entry_vwap_premium_bps),
                "feature_time_mode": "asof1455",
                "feature_cutoff_time": str(args.cutoff_time),
                "target_hit_bps": target_hit_bps,
                "round_trip_cost_bps": float(args.round_trip_cost_bps),
                "feature_count": int(len(cols)),
                "rows": int(len(prepared)),
                "validation_tail_trade_metrics": active_validation_metrics(row),
                "a_active_row": int(i + 1),
                "retention_sheet": row.get("_source_sheet", ""),
                "retention_sheet_row": int(row.get("_source_row", i + 1)),
                "a_active_tier": row.get("Tier", ""),
                "a_active_strategy": row.get("Strategy", ""),
                "a_active_origin": row.get("Origins", ""),
                "asof_pipeline": str((Path(args.asof_data_root) / f"{raw_code(stock)}_pipeline_out").resolve()),
                "usage_note": "Synthetic metadata for A_Active rolling asof1455 backtest.",
            }
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            states.append(
                ActiveState(
                    stock_code=stock,
                    artifact_name=artifact_name,
                    artifact_dir=artifact_dir,
                    metadata=metadata,
                    samples_path=samples_path,
                    samples=prepared,
                    feature_columns=cols,
                    feature_median=pd.Series(dtype=float),
                )
            )
            report_rows.append({
                "status": "ok",
                "stock_code": stock,
                "artifact_name": artifact_name,
                "retention_sheet": row.get("_source_sheet", ""),
                "retention_sheet_row": int(row.get("_source_row", i + 1)),
                "feature_group": group_name,
                "feature_count": len(cols),
                "rows": len(prepared),
                "samples": str(samples_path),
                "intraday_bars": str(intraday_path),
            })
        except Exception as exc:
            report_rows.append({
                "status": "skip",
                "stock_code": stock,
                "row": int(i + 1),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            if not args.keep_going:
                raise

    return states, pd.DataFrame(report_rows)


def generate_scores(states: list[ActiveState], dates: list[pd.Timestamp], signal_root: Path, args: argparse.Namespace) -> pd.DataFrame:
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
                r["context_status"] = f"{args.run_name}_asof1455_rolling_retrain"
                r["source_mode"] = f"{args.run_name}_asof1455_rolling_retrain"
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
            all_scores = pd.DataFrame(columns=SCORE_COLUMNS)
            buy = pd.DataFrame(columns=SCORE_COLUMNS)
            rejected = pd.DataFrame(columns=SCORE_COLUMNS)
        all_scores.to_csv(day_dir / "all_scores.csv", index=False, encoding="utf-8-sig")
        buy.to_csv(day_dir / "buy_signals.csv", index=False, encoding="utf-8-sig")
        rejected.to_csv(day_dir / "rejected_scores.csv", index=False, encoding="utf-8-sig")
        row = {
            "date": date.strftime("%Y%m%d"),
            "source_mode": f"{args.run_name}_asof1455_rolling_retrain",
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
        print(f"[{args.run_name.upper()} ASOF1455 SCORED] {date:%Y%m%d} all={len(all_scores)} buy={len(buy)}", flush=True)
    out = pd.DataFrame(summary_rows)
    out.to_csv(signal_root / "historical_score_generation_summary.csv", index=False, encoding="utf-8-sig")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling retrain portfolio backtest for retention-table models using asof1455 features")
    p.add_argument("--retention-xlsx", default="docs/model_retention_tier_tables.xlsx")
    p.add_argument("--sheet", default="A_Active", help="Used when --run-set custom.")
    p.add_argument(
        "--run-set",
        choices=["compare", "a_active", "a_plus_backup", "custom"],
        default="compare",
        help="compare runs A_Active and A_Active+B_Backup, then writes comparison files.",
    )
    p.add_argument("--asof-data-root", default="saved_data")
    p.add_argument("--asof-start-date", default="2025-01-01")
    p.add_argument("--asof-end-date", default=date.today().isoformat(), help="End date used when auto-building missing asof1455 data.")
    p.add_argument("--no-build-missing-asof-data", action="store_true", help="Fail on missing asof1455 data instead of building it first.")
    p.add_argument("--python", default=sys.executable, help="Python executable used for auto-building missing asof1455 data.")
    p.add_argument("--config", default="configs/portfolio_confirm_config.json")
    p.add_argument("--watchlist", default=None)
    p.add_argument("--out-dir", default="portfolio_reports/backtests/a_active_asof1455_rolling_retrain")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", default=None)
    p.add_argument("--initial-cash", type=float, default=200000.0)
    p.add_argument("--hold-days", type=int, default=1)
    p.add_argument("--min-amount-yuan", type=float, default=50000000.0)
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--valid-rows", type=int, default=126)
    p.add_argument("--train-rows", type=int, default=0)
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)
    p.add_argument("--cutoff-time", default="14:55")
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
    p.add_argument("--keep-going", action="store_true", help="Skip invalid A_Active rows instead of failing.")
    return p.parse_args()


def sheets_for_run(args: argparse.Namespace, run_set: str) -> list[str]:
    if run_set == "a_active":
        return ["A_Active"]
    if run_set == "a_plus_backup":
        return ["A_Active", "B_Backup"]
    if run_set == "custom":
        return sheet_list(args.sheet)
    raise ValueError(f"unsupported run_set={run_set}")


def run_one_backtest(args: argparse.Namespace, run_name: str, sheets: list[str], out_dir: Path) -> dict[str, Any]:
    args.run_name = run_name
    out_dir = Path(args.out_dir)
    if run_name:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_asof_data(args, sheets)
    metadata_root = out_dir / "generated_active_metadata"
    states, load_report = load_active_states(args, metadata_root, sheets)
    load_report.to_csv(out_dir / "retention_load_report.csv", index=False, encoding="utf-8-sig")
    if not states:
        raise SystemExit(f"no usable retention rows loaded for {run_name}")

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
        saved_models=metadata_root,
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
        "load_report": out_dir / "retention_load_report.csv",
        "generated_metadata": metadata_root,
    }
    sim["equity"].to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    sim["daily"].to_csv(paths["daily"], index=False, encoding="utf-8-sig")
    sim["trades"].to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    sim["open_lots"].to_csv(paths["open_lots"], index=False, encoding="utf-8-sig")
    summary = sim["summary"]
    summary.update({
        "source_mode": f"{run_name}_asof1455_rolling_retrain",
        "run_name": run_name,
        "retention_xlsx": str(Path(args.retention_xlsx)),
        "retention_sheets": ",".join(sheets),
        "asof_data_root": str(Path(args.asof_data_root)),
        "feature_time_mode": "asof1455",
        "feature_cutoff_time": str(args.cutoff_time),
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "scored_days": int(len(score_summary)),
        "retention_rows_loaded": int(len(states)),
        "unique_stocks": int(len({st.stock_code for st in states})),
        "retrain_frequency": args.retrain_frequency,
        "threshold_mode": args.threshold_mode,
        "oof_folds": int(args.oof_folds),
        "train_rows": int(args.train_rows),
        "valid_rows": int(args.valid_rows),
        "label_cutoff_rule": "train rows date <= two trading sessions before trade date",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OUTPUTS]")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"run_name": run_name, "out_dir": str(out_dir), "summary": summary, "paths": {k: str(v) for k, v in paths.items()}}


def comparison_row(result: dict[str, Any]) -> dict[str, Any]:
    s = result["summary"]
    keys = [
        "run_name", "retention_sheets", "retention_rows_loaded", "unique_stocks",
        "start_date", "end_date", "scored_days", "trading_days", "realized_trades",
        "final_equity", "total_return", "annualized_return", "annualized_volatility",
        "sharpe_rf0", "max_drawdown", "win_rate", "profit_factor",
        "avg_trade_return", "median_trade_return", "avg_gross_exposure",
    ]
    row = {"out_dir": result["out_dir"]}
    for k in keys:
        row[k] = s.get(k)
    return row


def write_comparison(root: Path, results: list[dict[str, Any]]) -> None:
    rows = [comparison_row(r) for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(root / "a_active_vs_a_plus_backup_comparison.csv", index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": rows,
        "delta_a_plus_backup_minus_a_active": {},
    }
    if len(rows) >= 2:
        base = rows[0]
        other = rows[1]
        for k, v in other.items():
            bv = base.get(k)
            if isinstance(v, (int, float)) and isinstance(bv, (int, float)) and np.isfinite(v) and np.isfinite(bv):
                payload["delta_a_plus_backup_minus_a_active"][k] = float(v) - float(bv)
    (root / "a_active_vs_a_plus_backup_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)
    if args.run_set == "compare":
        runs = [
            ("a_active", ["A_Active"], root_out / "a_active"),
            ("a_active_plus_b_backup", ["A_Active", "B_Backup"], root_out / "a_active_plus_b_backup"),
        ]
        results = []
        original_out = args.out_dir
        for run_name, sheets, run_out in runs:
            print(f"[RUN BACKTEST] {run_name} sheets={','.join(sheets)} out={run_out}", flush=True)
            args.out_dir = str(run_out)
            results.append(run_one_backtest(args, run_name, sheets, run_out))
        args.out_dir = original_out
        write_comparison(root_out, results)
        print(f"[COMPARISON] {root_out / 'a_active_vs_a_plus_backup_comparison.csv'}")
        print(f"[COMPARISON] {root_out / 'a_active_vs_a_plus_backup_comparison.json'}")
        return 0

    run_name = {
        "a_active": "a_active",
        "a_plus_backup": "a_active_plus_b_backup",
        "custom": safe_name(args.sheet),
    }[args.run_set]
    run_one_backtest(args, run_name, sheets_for_run(args, args.run_set), root_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
