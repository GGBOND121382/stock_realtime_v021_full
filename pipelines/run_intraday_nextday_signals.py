#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect intraday data and emit pre-close next-day model buy signals.

This is a thin orchestration layer:
  - candidates come from selected_watchlist.txt
  - intraday collection is delegated to collect_akshare_l1_cache.py
  - scoring uses stock-specific artifacts under saved_models/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_training.optimize_nextday_vwap_model import (
    add_market_state_features,
    add_reversal_daily_features,
    compute_entry_signal,
    segment_ret,
    segment_vwap,
)


ROOT = PROJECT_DIR
PYTHON = Path(sys.executable)
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
COLLECT_AKSHARE_SCRIPT = "data_collection/collect_akshare_l1_cache.py"
UPDATE_BAOSTOCK_SCRIPT = "data_collection/update_baostock_raw_cache.py"


@dataclass
class ModelArtifact:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    metadata: dict
    created_at: str


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace("_", ".")
    if not s or s.startswith("#"):
        return ""
    if "." in s:
        a, b = s.split(".", 1)
        if a in {"SH", "SZ"}:
            market, code = a, b
        else:
            code, market = a, b
        return f"{code.zfill(6)}.{market}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def read_watchlist(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = normalize_symbol(line)
        if symbol:
            symbols.append(symbol)
    return list(dict.fromkeys(symbols))


def parse_hhmm(value: str) -> dtime:
    hh, mm = value.split(":", 1)
    return dtime(int(hh), int(mm))


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def yyyymmdd_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("RUN", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_until(target: dtime) -> None:
    while datetime.now().time() < target:
        remaining = datetime.combine(datetime.today(), target) - datetime.now()
        sleep_s = max(1, min(30, int(remaining.total_seconds())))
        print(f"waiting for signal time {target.strftime('%H:%M')}, {int(remaining.total_seconds())}s left")
        time.sleep(sleep_s)


def load_artifacts(models_dir: Path, watchlist: set[str], policy: str) -> list[ModelArtifact]:
    artifacts: list[ModelArtifact] = []
    for meta_path in sorted(models_dir.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stock = normalize_symbol(meta.get("stock_code", ""))
        if not stock or stock not in watchlist:
            continue
        artifacts.append(
            ModelArtifact(
                stock_code=stock,
                artifact_name=str(meta.get("artifact_name") or meta_path.parent.name),
                artifact_dir=meta_path.parent,
                metadata=meta,
                created_at=str(meta.get("artifact_created_at") or ""),
            )
        )
    if policy == "all":
        return artifacts

    chosen: list[ModelArtifact] = []
    for stock in sorted({a.stock_code for a in artifacts}):
        stock_artifacts = [a for a in artifacts if a.stock_code == stock]
        stock_artifacts.sort(
            key=lambda a: (
                "close_profit" in a.artifact_name,
                a.created_at,
                a.artifact_dir.stat().st_mtime,
            ),
            reverse=True,
        )
        chosen.append(stock_artifacts[0])
    return chosen


def resolve_repo_path(raw: Optional[str], stock_code: str = "") -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p

    text = str(raw).replace("\\", "/")
    marker = "stock_realtime/"
    if marker in text:
        candidate = ROOT / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate

    name = Path(text).name
    if name:
        hits = list(ROOT.rglob(name))
        if stock_code:
            code6 = stock_code.split(".", 1)[0]
            preferred = [h for h in hits if code6 in str(h)]
            if preferred:
                return preferred[0]
        if hits:
            return hits[0]
    return None


def cache_symbol_dir(cache_dir: Path, trade_date: str, stock_code: str) -> Path:
    return cache_dir / "pending" / trade_date / normalize_symbol(stock_code)


def infer_raw_cache_dir(artifact: ModelArtifact) -> Path:
    intraday = resolve_repo_path(artifact.metadata.get("intraday_bars"), artifact.stock_code)
    if intraday is not None and intraday.name.endswith("_5m_raw.csv"):
        return intraday.parent

    code6 = artifact.stock_code.split(".", 1)[0]
    hits = list(ROOT.rglob(f"{code6}_5m_raw.csv"))
    if hits:
        return hits[0].parent

    for candidate in [
        ROOT / "saved_data" / f"{code6}_pipeline_out" / "00_base" / "raw_cache",
        ROOT / "saved_data" / f"{code6}_base_out" / "raw_cache",
        ROOT / "saved_data" / f"zijin_{code6}_base_out" / "raw_cache",
        ROOT / f"{code6}_base_out" / "raw_cache",
        ROOT / f"zijin_{code6}_base_out" / "raw_cache",
    ]:
        if candidate.exists():
            return candidate
    return ROOT / "saved_data" / f"{code6}_base_out" / "raw_cache"


def resolve_intraday_path(artifact: ModelArtifact, trade_date: str, cache_dir: Path) -> Optional[Path]:
    code6 = artifact.stock_code.split(".", 1)[0]
    baostock = infer_raw_cache_dir(artifact) / f"{code6}_5m_raw.csv"
    if baostock.exists():
        return baostock
    return resolve_repo_path(artifact.metadata.get("intraday_bars"), artifact.stock_code)


def baostock_update_window(raw_dir: Path, stock_code: str, end_date: str, start_override: Optional[str]) -> Optional[tuple[str, str]]:
    if start_override:
        return start_override, end_date
    code6 = stock_code.split(".", 1)[0]
    intraday_path = raw_dir / f"{code6}_5m_raw.csv"
    if not intraday_path.exists():
        return end_date, end_date
    try:
        existing = pd.read_csv(intraday_path, usecols=["datetime"], parse_dates=["datetime"])
    except Exception:
        return end_date, end_date
    if existing.empty or existing["datetime"].dropna().empty:
        return end_date, end_date
    last_date = existing["datetime"].max().date()
    end_dt = pd.to_datetime(end_date).date()
    if last_date >= end_dt:
        return None
    start_dt = last_date + timedelta(days=1)
    return start_dt.isoformat(), end_date


def intraday_feature_cache_path(cache_dir: Path, stock_code: str) -> Path:
    return cache_dir / "feature_cache" / f"{normalize_symbol(stock_code)}_intraday_reversal_features.csv"


def build_intraday_rows(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.replace([np.inf, -np.inf], np.nan).dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    if bars.empty:
        return pd.DataFrame()
    bars["date"] = bars["datetime"].dt.normalize()
    bars["time_str"] = bars["datetime"].dt.strftime("%H:%M:%S")
    rows = []
    for date, g in bars.groupby("date", sort=True):
        g = g.sort_values("datetime")
        row = {
            "date": date,
            "bar_count": int(len(g)),
            "first_30m_ret": segment_ret(g, "09:35:00", "10:00:00"),
            "first_60m_ret": segment_ret(g, "09:35:00", "10:30:00"),
            "morning_ret": segment_ret(g, "09:35:00", "11:30:00"),
            "afternoon_ret": segment_ret(g, "13:05:00", "15:00:00"),
            "last_30m_ret": segment_ret(g, "14:35:00", "15:00:00"),
            "last_60m_ret": segment_ret(g, "14:05:00", "15:00:00"),
            "morning_vwap": segment_vwap(g, "09:35:00", "11:30:00"),
            "afternoon_vwap": segment_vwap(g, "13:05:00", "15:00:00"),
            "last_30m_vwap": segment_vwap(g, "14:35:00", "15:00:00"),
        }
        if "volume" in g.columns:
            total_volume = max(pd.to_numeric(g["volume"], errors="coerce").sum(), 1e-12)
            row["first_60m_volume_share"] = pd.to_numeric(g[(g["time_str"] >= "09:35:00") & (g["time_str"] <= "10:30:00")]["volume"], errors="coerce").sum() / total_volume
            row["last_30m_volume_share"] = pd.to_numeric(g[(g["time_str"] >= "14:35:00") & (g["time_str"] <= "15:00:00")]["volume"], errors="coerce").sum() / total_volume
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date")


def load_intraday_feature_cache(intraday_path: Optional[Path], cache_dir: Path, stock_code: str) -> pd.DataFrame:
    if intraday_path is None or not intraday_path.exists():
        return pd.DataFrame()
    cache_path = intraday_feature_cache_path(cache_dir, stock_code)
    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
    min_rebuild_date = cached["date"].max() if not cached.empty else None
    bars = pd.read_csv(intraday_path, parse_dates=["datetime"])
    if min_rebuild_date is not None:
        bars = bars[pd.to_datetime(bars["datetime"]).dt.normalize() >= min_rebuild_date]
    new_rows = build_intraday_rows(bars)
    if cached.empty:
        out = new_rows
    elif new_rows.empty:
        out = cached
    else:
        out = pd.concat([cached, new_rows], ignore_index=True)
        out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    if not out.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def add_scoring_features(df: pd.DataFrame, intraday_path: Optional[Path], cache_dir: Path, stock_code: str) -> pd.DataFrame:
    out = add_reversal_daily_features(df)
    intra = load_intraday_feature_cache(intraday_path, cache_dir, stock_code)
    if not intra.empty:
        out = out.merge(intra, on="date", how="left")
        for col in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
            if col in out.columns and "close" in out.columns:
                out[f"{col}_to_close"] = out["close"] / out[col].replace(0, np.nan) - 1.0
        if {"morning_ret", "afternoon_ret"}.issubset(out.columns):
            out["morning_afternoon_reversal"] = -out["morning_ret"] * out["afternoon_ret"]
        if {"first_60m_ret", "last_30m_ret"}.issubset(out.columns):
            out["first60_last30_reversal"] = -out["first_60m_ret"] * out["last_30m_ret"]
    return add_market_state_features(out)


def overlay_current_day_from_cache(df: pd.DataFrame, stock_code: str, trade_date: str, cache_dir: Path) -> pd.DataFrame:
    daily_path = cache_symbol_dir(cache_dir, trade_date, stock_code) / "daily_features.csv"
    if not daily_path.exists():
        return df
    daily = pd.read_csv(daily_path)
    if daily.empty:
        return df
    current = daily.iloc[-1]
    out = df.sort_values("date").copy()
    row = out.iloc[-1].copy()
    row["date"] = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    for src, dst in [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("amount", "amount"),
        ("daily_vwap", "daily_vwap"),
        ("volume", "daily_vwap_volume"),
        ("amount", "daily_vwap_pv"),
        ("snapshots", "n_intraday_bars"),
    ]:
        if src in current.index and dst in out.columns:
            row[dst] = pd.to_numeric(current[src], errors="coerce")
    for col in ["next_date", "next_day_vwap", "next_day_close", "next_day_high", "next_day_low"]:
        if col in out.columns:
            row[col] = pd.NA
    target_date = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    keep = out[pd.to_datetime(out["date"]) != target_date].copy()
    return pd.concat([keep, pd.DataFrame([row])], ignore_index=True).sort_values("date").reset_index(drop=True)


def overlay_current_day_from_intraday(df: pd.DataFrame, trade_date: str, intraday_path: Optional[Path]) -> pd.DataFrame:
    if intraday_path is None or not intraday_path.exists():
        return df
    bars = pd.read_csv(intraday_path, parse_dates=["datetime"])
    if bars.empty:
        return df
    target_date = pd.to_datetime(yyyymmdd_to_iso(trade_date)).normalize()
    bars = bars[pd.to_datetime(bars["datetime"]).dt.normalize() == target_date].sort_values("datetime")
    if bars.empty:
        return df
    out = df.sort_values("date").copy()
    row = out.iloc[-1].copy()
    row["date"] = target_date
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in bars.columns:
            values = pd.to_numeric(bars[col], errors="coerce")
            if col == "open":
                row[col] = values.dropna().iloc[0] if values.dropna().size else pd.NA
            elif col == "high":
                row[col] = values.max()
            elif col == "low":
                row[col] = values.min()
            elif col == "close":
                row[col] = values.dropna().iloc[-1] if values.dropna().size else pd.NA
            elif col in out.columns:
                row[col] = values.sum()
    volume = pd.to_numeric(bars.get("volume"), errors="coerce").sum() if "volume" in bars.columns else np.nan
    amount = pd.to_numeric(bars.get("amount"), errors="coerce").sum() if "amount" in bars.columns else np.nan
    if "daily_vwap" in out.columns and volume and np.isfinite(volume):
        row["daily_vwap"] = amount / volume
    if "daily_vwap_volume" in out.columns:
        row["daily_vwap_volume"] = volume
    if "daily_vwap_pv" in out.columns:
        row["daily_vwap_pv"] = amount
    if "n_intraday_bars" in out.columns:
        row["n_intraday_bars"] = len(bars)
    for col in ["next_date", "next_day_vwap", "next_day_close", "next_day_high", "next_day_low"]:
        if col in out.columns:
            row[col] = pd.NA
    keep = out[pd.to_datetime(out["date"]) != target_date].copy()
    return pd.concat([keep, pd.DataFrame([row])], ignore_index=True).sort_values("date").reset_index(drop=True)


def score_artifact(artifact: ModelArtifact, trade_date: str, out_dir: Path, cache_dir: Path, strict_features: bool) -> dict:
    model = joblib.load(artifact.artifact_dir / "model.joblib")
    cols = [
        line.strip()
        for line in (artifact.artifact_dir / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    med = pd.read_csv(artifact.artifact_dir / "feature_median.csv", index_col=0)["median"]

    samples_path = resolve_repo_path(artifact.metadata.get("samples"), artifact.stock_code)
    intraday_path = resolve_intraday_path(artifact, trade_date, cache_dir)
    if samples_path is None:
        raise FileNotFoundError(f"samples not found for {artifact.stock_code}: {artifact.metadata.get('samples')}")

    df = pd.read_csv(samples_path, parse_dates=["date"])
    df = overlay_current_day_from_cache(df, artifact.stock_code, trade_date, cache_dir)
    df = overlay_current_day_from_intraday(df, trade_date, intraday_path)
    df = add_scoring_features(df, intraday_path, cache_dir, artifact.stock_code)
    df = df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    target_date = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    day_df = df[df["date"] == target_date].copy()
    if day_df.empty:
        day_df = df.tail(1).copy()
        used_date = str(pd.to_datetime(day_df["date"].iloc[0]).date())
        date_status = "fallback_latest_sample"
    else:
        used_date = str(target_date.date())
        date_status = "exact_trade_date"

    missing = [c for c in cols if c not in day_df.columns]
    if missing:
        if strict_features:
            raise ValueError(f"missing {len(missing)} features, first={missing[:10]}")
        for col in missing:
            day_df[col] = np.nan

    x = day_df[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    score = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
    threshold = float(artifact.metadata["threshold"])
    entry_policy = str(artifact.metadata.get("entry_policy") or "vwap_low")
    entry_vwap_premium_bps = float(artifact.metadata.get("entry_vwap_premium_bps", 50.0))
    # Always recompute from artifact metadata; do not trust an entry_signal
    # column inherited from saved samples because it may have been generated
    # under a different entry policy.
    entry_signal = bool(compute_entry_signal(day_df, entry_policy, entry_vwap_premium_bps).iloc[-1])
    signal_raw_score_pass = bool(float(score[-1]) >= threshold)
    signal = entry_signal and signal_raw_score_pass

    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": yyyymmdd_to_iso(trade_date),
        "sample_date": used_date,
        "used_sample_date": used_date,
        "used_live_bar_date": yyyymmdd_to_iso(trade_date) if intraday_path else "",
        "date_status": date_status,
        "feature_status": "live_overlay_partial",
        "stock_code": artifact.stock_code,
        "artifact_name": artifact.artifact_name,
        "entry_policy": entry_policy,
        "entry_vwap_premium_bps": entry_vwap_premium_bps,
        "entry_signal": entry_signal,
        "close": float(pd.to_numeric(day_df["close"], errors="coerce").iloc[-1]) if "close" in day_df.columns else np.nan,
        "daily_vwap": float(pd.to_numeric(day_df["daily_vwap"], errors="coerce").iloc[-1]) if "daily_vwap" in day_df.columns else np.nan,
        "hit_score": float(score[-1]),
        "threshold": threshold,
        "signal_raw_score_pass": signal_raw_score_pass,
        "signal": signal,
        "missing_feature_count": int(len(missing)),
        "strict_features": bool(strict_features),
        "samples": str(samples_path),
        "intraday_bars": str(intraday_path) if intraday_path else "",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stock_out = out_dir / f"{artifact.stock_code}_{artifact.artifact_name}_score.csv"
    pd.DataFrame([row]).to_csv(stock_out, index=False, encoding="utf-8-sig")
    return row


def collect_once(args: argparse.Namespace) -> None:
    cmd = [
        str(PYTHON),
        COLLECT_AKSHARE_SCRIPT,
        "collect-once",
        "--symbols-file",
        str(args.watchlist),
        "--out-dir",
        str(args.cache_dir),
        "--allow-l1-only",
    ]
    if not args.enable_five_level:
        cmd.append("--disable-em-bid-ask")
    if args.max_symbols:
        cmd.extend(["--max-symbols", str(args.max_symbols)])
    run_cmd(cmd, args.dry_run)


def collect_trades(args: argparse.Namespace, trade_date: str) -> None:
    cmd = [
        str(PYTHON),
        COLLECT_AKSHARE_SCRIPT,
        "collect-trades",
        "--symbols-file",
        str(args.watchlist),
        "--out-dir",
        str(args.cache_dir),
        "--trades-date",
        trade_date,
        "--allow-l1-only",
    ]
    if not args.enable_five_level:
        cmd.append("--disable-em-bid-ask")
    if args.max_symbols:
        cmd.extend(["--max-symbols", str(args.max_symbols)])
    run_cmd(cmd, args.dry_run)


def build_bars_and_validate(args: argparse.Namespace, trade_date: str) -> None:
    for subcmd in ("build-bars", "validate-day"):
        cmd = [
            str(PYTHON),
            COLLECT_AKSHARE_SCRIPT,
            subcmd,
            "--out-dir",
            str(args.cache_dir),
            "--date",
            trade_date,
        ]
        run_cmd(cmd, args.dry_run)


def collect_loop_until_signal(args: argparse.Namespace) -> None:
    cmd = [
        str(PYTHON),
        COLLECT_AKSHARE_SCRIPT,
        "collect-loop",
        "--symbols-file",
        str(args.watchlist),
        "--out-dir",
        str(args.cache_dir),
        "--interval-seconds",
        str(args.interval_seconds),
        "--until",
        args.signal_time,
        "--allow-l1-only",
    ]
    if not args.enable_five_level:
        cmd.append("--disable-em-bid-ask")
    if args.with_trades:
        cmd.extend(["--with-trades", "--trades-interval-seconds", str(args.trades_interval_seconds)])
    if args.max_symbols:
        cmd.extend(["--max-symbols", str(args.max_symbols)])
    run_cmd(cmd, args.dry_run)


def update_baostock_for_artifacts(args: argparse.Namespace, artifacts: list[ModelArtifact], trade_date: str) -> None:
    if args.skip_baostock_update:
        return
    end_date = args.baostock_end_date or yyyymmdd_to_iso(trade_date)
    for artifact in artifacts:
        raw_dir = infer_raw_cache_dir(artifact)
        window = baostock_update_window(raw_dir, artifact.stock_code, end_date, args.baostock_start_date)
        if window is None:
            print(f"SKIP BaoStock {artifact.stock_code}: raw cache already covers {end_date}")
            continue
        start_date, final_end_date = window
        cmd = [
            str(PYTHON),
            UPDATE_BAOSTOCK_SCRIPT,
            "--symbol",
            artifact.stock_code,
            "--start-date",
            start_date,
            "--end-date",
            final_end_date,
            "--raw-cache-dir",
            str(raw_dir),
        ]
        run_cmd(cmd, args.dry_run)


def score_models(args: argparse.Namespace, trade_date: str) -> pd.DataFrame:
    watchlist = set(read_watchlist(Path(args.watchlist)))
    artifacts = load_artifacts(Path(args.models_dir), watchlist, args.model_policy)
    rows = []
    for artifact in artifacts:
        try:
            row = score_artifact(
                artifact,
                trade_date,
                Path(args.signal_out_dir) / trade_date,
                Path(args.cache_dir),
                args.strict_features,
            )
            print(f"{row['stock_code']} {row['artifact_name']} score={row['hit_score']:.6f} threshold={row['threshold']:.6f} signal={row['signal']}")
            rows.append(row)
        except Exception as exc:
            row = {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "trade_date": yyyymmdd_to_iso(trade_date),
                "stock_code": artifact.stock_code,
                "artifact_name": artifact.artifact_name,
                "signal": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"ERROR {artifact.stock_code} {artifact.artifact_name}: {row['error']}")
            rows.append(row)
    out_dir = Path(args.signal_out_dir) / trade_date
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "buy_signals.csv", index=False, encoding="utf-8-sig")
    return summary


def prepare_baostock_at_start(args: argparse.Namespace, trade_date: str) -> None:
    watchlist = set(read_watchlist(Path(args.watchlist)))
    artifacts = load_artifacts(Path(args.models_dir), watchlist, args.model_policy)
    update_baostock_for_artifacts(args, artifacts, trade_date)


def list_models(args: argparse.Namespace) -> None:
    watchlist = set(read_watchlist(Path(args.watchlist)))
    artifacts = load_artifacts(Path(args.models_dir), watchlist, args.model_policy)
    rows = [
        {
            "stock_code": a.stock_code,
            "artifact_name": a.artifact_name,
            "created_at": a.created_at,
            "threshold": a.metadata.get("threshold"),
            "feature_group": a.metadata.get("feature_group"),
            "artifact_dir": str(a.artifact_dir),
        }
        for a in artifacts
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday collection and pre-close next-day buy signals")
    p.add_argument("cmd", choices=["list-models", "collect-and-score", "score-now"])
    p.add_argument("--watchlist", default="selected_watchlist.txt")
    p.add_argument("--models-dir", default=str(SAVED_MODELS_DIR))
    p.add_argument("--cache-dir", default=str(SAVED_DATA_DIR / "akshare_realtime_cache"))
    p.add_argument("--signal-out-dir", default=str(SAVED_DATA_DIR / "intraday_nextday_signals"))
    p.add_argument("--date", help="YYYYMMDD; defaults to today")
    p.add_argument("--signal-time", default="14:55")
    p.add_argument("--interval-seconds", type=int, default=30)
    p.add_argument("--with-trades", action="store_true")
    p.add_argument("--trades-interval-seconds", type=int, default=300)
    p.add_argument("--model-policy", choices=["preferred", "all"], default="preferred")
    p.add_argument("--max-symbols", type=int, help="Debug only: limit collection symbols")
    p.add_argument("--skip-final-collect", action="store_true")
    p.add_argument("--skip-build-bars", action="store_true")
    p.add_argument("--skip-baostock-update", action="store_true")
    p.add_argument("--baostock-start-date", help="YYYY-MM-DD; defaults to signal date")
    p.add_argument("--baostock-end-date", help="YYYY-MM-DD; defaults to signal date")
    p.add_argument("--enable-five-level", action="store_true", help="Also fetch Eastmoney five-level quotes")
    p.add_argument("--strict-features", action="store_true", help="Fail a model when live rows miss trained features")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = args.date or today_yyyymmdd()
    if args.cmd == "list-models":
        list_models(args)
        return
    prepare_baostock_at_start(args, trade_date)
    if args.cmd == "collect-and-score":
        collect_loop_until_signal(args)
        if not args.dry_run:
            wait_until(parse_hhmm(args.signal_time))
    if not args.skip_final_collect:
        collect_once(args)
        if args.with_trades:
            collect_trades(args, trade_date)
    if not args.skip_build_bars:
        build_bars_and_validate(args, trade_date)
    score_models(args, trade_date)


if __name__ == "__main__":
    main()
