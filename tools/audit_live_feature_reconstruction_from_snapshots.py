#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare live reconstructed scoring features against full training samples.

The audit replays historical snapshot caches as if they were same-day live data:
only samples strictly before the replay date are fed into the live overlay path.
The full sample row for the replay date is used only as the comparison target.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.run_intraday_nextday_signals import (  # noqa: E402
    ModelArtifact,
    add_market_state_features,
    add_reversal_daily_features,
    add_scoring_features,
    cache_symbol_dir,
    load_intraday_feature_cache,
    normalize_symbol,
    overlay_current_day_from_cache,
    recompute_stock_vs_sector_features,
    resolve_intraday_path,
    yyyymmdd_to_iso,
)


def load_snapshots(path: Path, cutoff_time: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return df
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(
            df["trade_date"].astype(str) + df["trade_time"].astype(str).str.zfill(6),
            errors="coerce",
        )
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    if cutoff_time:
        cutoff_dt = pd.to_datetime(f"{yyyymmdd_to_iso(str(df['trade_date'].iloc[-1]))} {cutoff_time}:00")
        df = df[df["datetime"] <= cutoff_dt].copy()
    return df


def write_bars_from_snapshots(src_snap: Path, dst_sym_dir: Path, trade_date: str, cutoff_time: str | None) -> bool:
    df = load_snapshots(src_snap, cutoff_time)
    if df.empty:
        return False
    px = pd.to_numeric(df.get("last_price"), errors="coerce").ffill()
    if px.dropna().empty:
        return False
    dst_sym_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_snap, dst_sym_dir / "snapshot_5level.csv")

    base = pd.DataFrame(index=df["datetime"])
    base["open"] = px.to_numpy()
    base["high"] = px.to_numpy()
    base["low"] = px.to_numpy()
    base["close"] = px.to_numpy()
    for col in ["volume", "amount"]:
        if col in df.columns:
            base[col] = pd.to_numeric(df[col], errors="coerce").to_numpy()

    symbol = normalize_symbol(str(df.get("symbol", pd.Series([dst_sym_dir.name])).iloc[0]))
    wrote = False
    for freq in ["1min", "5min"]:
        bars = base.resample(freq).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "last") if "volume" in base.columns else ("close", "count"),
            amount=("amount", "last") if "amount" in base.columns else ("close", "count"),
        )
        bars = bars.dropna(subset=["open", "high", "low", "close"], how="all")
        if bars.empty:
            continue
        if "volume" in bars.columns:
            bars["bar_volume"] = pd.to_numeric(bars["volume"], errors="coerce").diff()
            bars.loc[bars["bar_volume"] < 0, "bar_volume"] = np.nan
        if "amount" in bars.columns:
            bars["bar_amount"] = pd.to_numeric(bars["amount"], errors="coerce").diff()
            bars.loc[bars["bar_amount"] < 0, "bar_amount"] = np.nan
        bars = bars.reset_index()
        bars.insert(0, "trade_date", trade_date)
        bars.insert(0, "symbol", symbol)
        bars.to_csv(dst_sym_dir / f"minute_bars_{freq}.csv", index=False, encoding="utf-8-sig")
        wrote = True
    return wrote


def iter_artifacts(saved_models: Path) -> list[ModelArtifact]:
    artifacts: list[ModelArtifact] = []
    for meta_path in sorted(saved_models.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stock_code = normalize_symbol(str(meta.get("stock_code", meta_path.parent.parent.name)))
        artifacts.append(
            ModelArtifact(
                stock_code=stock_code,
                artifact_name=str(meta.get("artifact_name", meta_path.parent.name)),
                artifact_dir=meta_path.parent,
                metadata=meta,
                created_at=str(meta.get("artifact_created_at", "")),
            )
        )
    return artifacts


def resolve_metadata_samples_path(value: object) -> Path | None:
    if not value:
        return None
    text = str(value)
    p = Path(text)
    if p.exists():
        return p
    marker = "stock_realtime_v021_full"
    parts = p.parts
    if marker in parts:
        candidate = ROOT.joinpath(*parts[parts.index(marker) + 1 :])
        if candidate.exists():
            return candidate
    candidate = ROOT / text
    return candidate if candidate.exists() else None


def add_complete_training_features(
    df: pd.DataFrame,
    intraday_path: Path | None,
    cache_dir: Path,
    stock_code: str,
) -> pd.DataFrame:
    out = add_reversal_daily_features(df)
    intra = load_intraday_feature_cache(intraday_path, cache_dir, stock_code)
    if not intra.empty:
        out = out.merge(intra, on="date", how="left")
        if "bar_count" in out.columns:
            if "n_intraday_bars" not in out.columns:
                out["n_intraday_bars"] = np.nan
            bar_count = pd.to_numeric(out["bar_count"], errors="coerce")
            out["n_intraday_bars"] = out["n_intraday_bars"].where(bar_count.isna(), bar_count)
        for col in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
            if col in out.columns and "close" in out.columns:
                out[f"{col}_to_close"] = out["close"] / out[col].replace(0, np.nan) - 1.0
        if {"morning_ret", "afternoon_ret"}.issubset(out.columns):
            out["morning_afternoon_reversal"] = -out["morning_ret"] * out["afternoon_ret"]
        if {"first_60m_ret", "last_30m_ret"}.issubset(out.columns):
            out["first60_last30_reversal"] = -out["first_60m_ret"] * out["last_30m_ret"]
    return add_market_state_features(out)


def compare_one(
    artifact: ModelArtifact,
    trade_date: str,
    src_cache: Path,
    work_cache: Path,
    target_cache: Path,
    cutoff_time: str | None,
):
    src_snap = cache_symbol_dir(src_cache, trade_date, artifact.stock_code) / "snapshot_5level.csv"
    if not src_snap.exists():
        return None, []

    dst_sym_dir = cache_symbol_dir(work_cache, trade_date, artifact.stock_code)
    if not write_bars_from_snapshots(src_snap, dst_sym_dir, trade_date, cutoff_time):
        return None, []

    samples_path = resolve_metadata_samples_path(artifact.metadata.get("samples"))
    if samples_path is None:
        return None, []
    full = pd.read_csv(samples_path, parse_dates=["date"])
    target_ts = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    hist = full[pd.to_datetime(full["date"]).dt.normalize() < target_ts].copy()
    through_target = full[pd.to_datetime(full["date"]).dt.normalize() <= target_ts].copy()
    if through_target.empty or hist.empty:
        return None, []

    cols = [
        line.strip()
        for line in (artifact.artifact_dir / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    intraday_path = resolve_intraday_path(artifact, trade_date, work_cache)

    target_features = add_complete_training_features(
        through_target,
        intraday_path,
        target_cache,
        artifact.stock_code,
    ).replace([np.inf, -np.inf], np.nan)
    target = target_features[pd.to_datetime(target_features["date"]).dt.normalize() == target_ts].copy()
    if target.empty:
        return None, []

    replay = overlay_current_day_from_cache(hist, artifact.stock_code, trade_date, work_cache, cutoff_time=cutoff_time)
    replay = add_scoring_features(replay, intraday_path, work_cache, artifact.stock_code, cutoff_time=cutoff_time)
    day = replay[pd.to_datetime(replay["date"]).dt.normalize() == target_ts].copy()
    if day.empty:
        return None, []
    day = recompute_stock_vs_sector_features(day, replay, cols).replace([np.inf, -np.inf], np.nan)

    live = day.iloc[-1]
    train = target.iloc[-1]
    rows = []
    abs_rels = []
    missing_live = 0
    missing_train = 0
    comparable = 0
    exactish = 0
    for col in cols:
        lv = pd.to_numeric(pd.Series([live[col] if col in day.columns else np.nan]), errors="coerce").iloc[0]
        tv = pd.to_numeric(pd.Series([train[col] if col in target.columns else np.nan]), errors="coerce").iloc[0]
        lna = pd.isna(lv)
        tna = pd.isna(tv)
        missing_live += int(lna)
        missing_train += int(tna)
        if lna or tna:
            rows.append([trade_date, artifact.stock_code, artifact.artifact_name, col, lv, tv, np.nan, np.nan, lna, tna])
            continue
        diff = float(lv) - float(tv)
        rel = abs(diff) / max(abs(float(tv)), 1e-9)
        abs_rels.append(rel)
        comparable += 1
        exactish += int(abs(diff) <= 1e-9 or rel <= 1e-6)
        rows.append([trade_date, artifact.stock_code, artifact.artifact_name, col, lv, tv, diff, rel, False, False])

    detail_cols = [
        "trade_date", "stock_code", "artifact_name", "feature", "live_value", "train_value",
        "diff", "abs_rel_diff", "live_missing", "train_missing",
    ]
    summary = {
        "trade_date": trade_date,
        "stock_code": artifact.stock_code,
        "artifact_name": artifact.artifact_name,
        "feature_count": len(cols),
        "comparable_features": comparable,
        "live_missing": missing_live,
        "train_missing": missing_train,
        "exactish_features": exactish,
        "exactish_share": exactish / comparable if comparable else np.nan,
        "median_abs_rel_diff": float(np.nanmedian(abs_rels)) if abs_rels else np.nan,
        "p90_abs_rel_diff": float(np.nanpercentile(abs_rels, 90)) if abs_rels else np.nan,
        "p99_abs_rel_diff": float(np.nanpercentile(abs_rels, 99)) if abs_rels else np.nan,
        "max_abs_rel_diff": float(np.nanmax(abs_rels)) if abs_rels else np.nan,
        "snapshot_count": int(pd.read_csv(dst_sym_dir / "snapshot_5level.csv", usecols=["trade_time"]).shape[0]),
        "n_intraday_bars_live": float(pd.to_numeric(pd.Series([live.get("n_intraday_bars", np.nan)]), errors="coerce").iloc[0]),
        "n_intraday_bars_train": float(pd.to_numeric(pd.Series([train.get("n_intraday_bars", np.nan)]), errors="coerce").iloc[0]),
        "close_live": float(pd.to_numeric(pd.Series([live.get("close", np.nan)]), errors="coerce").iloc[0]),
        "close_train": float(pd.to_numeric(pd.Series([train.get("close", np.nan)]), errors="coerce").iloc[0]),
        "daily_vwap_live": float(pd.to_numeric(pd.Series([live.get("daily_vwap", np.nan)]), errors="coerce").iloc[0]),
        "daily_vwap_train": float(pd.to_numeric(pd.Series([train.get("daily_vwap", np.nan)]), errors="coerce").iloc[0]),
    }
    return summary, pd.DataFrame(rows, columns=detail_cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-cache", default=r"D:\VSCodeWorkspace\stockAnalysis\stock_realtime\akshare_realtime_cache")
    ap.add_argument("--saved-models", default=str(ROOT / "saved_models"))
    ap.add_argument("--out-dir", default=str(ROOT / "saved_data" / "feature_reconstruction_audit"))
    ap.add_argument("--cutoff-time", default="14:55")
    args = ap.parse_args()

    src_cache = Path(args.snapshot_cache)
    out_dir = Path(args.out_dir)
    work_cache = out_dir / "cache_from_snapshots"
    target_cache = out_dir / "cache_from_training_intraday"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = iter_artifacts(Path(args.saved_models))
    pending = src_cache / "pending"
    dates = [p.name for p in sorted(pending.iterdir()) if p.is_dir()]
    summaries = []
    details = []
    for trade_date in dates:
        for artifact in artifacts:
            summary, detail = compare_one(artifact, trade_date, src_cache, work_cache, target_cache, args.cutoff_time)
            if summary is None:
                continue
            summaries.append(summary)
            details.append(detail)

    summary_df = pd.DataFrame(summaries).sort_values(["trade_date", "stock_code", "artifact_name"])
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_path = out_dir / "feature_reconstruction_summary.csv"
    detail_path = out_dir / "feature_reconstruction_feature_diffs.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"WROTE {summary_path} rows={len(summary_df)}")
    print(f"WROTE {detail_path} rows={len(detail_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
