#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill sector_range_z20 using sampled realtime sector prices + training history.

Purpose
-------
Keep the existing THS summary realtime source.  Do NOT switch interfaces.
For each row in saved_data/realtime_context/YYYYMMDD/context_features_asof.csv:
  1) Estimate today's sector_range_pct from context_snapshots.csv by using
     max/min of the sampled sector price sequence before cutoff.
  2) Read the row's training samples path from realtime_context_plan.csv
     / metadata-derived samples column.
  3) Compute sector_range_z20 using the last 20 historical sector_range_pct
     values strictly before the trade date.
  4) Remove sector_range_z20 from missing_context_features when filled.

This script is intentionally a post-processor so it is robust to the existing
collector internals and can be called after collect_realtime_context.py
build-features and before score-now.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def normalize_symbol(symbol: object) -> str:
    s = str(symbol or "").strip().upper().replace("_", ".")
    if not s or s.lower() == "nan":
        return ""
    if "." in s:
        a, b = s.split(".", 1)
        if a in {"SH", "SZ"}:
            market, code = a, b
        else:
            code, market = a, b
        digits = "".join(ch for ch in code if ch.isdigit()).zfill(6)
        return f"{digits}.{market}"
    digits = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    if not digits:
        return ""
    market = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{market}"


def yyyymmdd_to_iso(value: str) -> str:
    value = str(value)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def resolve_path(raw: object, stock_code: str = "") -> Optional[Path]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    p = Path(text)
    if p.exists():
        return p
    if not p.is_absolute():
        p2 = PROJECT_DIR / p
        if p2.exists():
            return p2
    text2 = text.replace("\\", "/")
    marker = "stock_realtime/"
    if marker in text2:
        p3 = PROJECT_DIR / text2.split(marker, 1)[1]
        if p3.exists():
            return p3
    # Conservative fallback: search likely pipeline dirs for this stock only.
    code6 = normalize_symbol(stock_code).split(".", 1)[0] if stock_code else ""
    if code6:
        roots = [
            SAVED_DATA_DIR / f"{code6}_pipeline_out",
            SAVED_DATA_DIR / f"{code6}_base_out",
        ]
        name = Path(text2).name
        for root in roots:
            if root.exists() and name:
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]
        # Last resort: choose a training_samples csv containing sector_range_pct.
        for root in roots:
            if not root.exists():
                continue
            for cand in root.rglob("training_samples*.csv"):
                try:
                    head = pd.read_csv(cand, nrows=2)
                except Exception:
                    continue
                if "sector_range_pct" in head.columns:
                    return cand
    return None


def split_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item and item.lower() != "nan":
            out.append(item)
    return out


def finite_float(value: object) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def infer_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["datetime", "snapshot_time", "context_snapshot_time", "collected_at", "time"]:
        if c in df.columns:
            return c
    return None


def price_column(df: pd.DataFrame) -> Optional[str]:
    # context_snapshots.csv from current collector uses close.  Keep fallbacks
    # for older files or hand-built debug data.
    for c in [
        "close", "sector_close", "latest", "last", "price", "current",
        "均价", "最新价", "现价", "收盘", "收盘价",
    ]:
        if c in df.columns:
            return c
    return None


def filter_cutoff(df: pd.DataFrame, trade_date: str, cutoff_time: str) -> pd.DataFrame:
    if df.empty:
        return df
    tcol = infer_time_col(df)
    if tcol is None:
        return df
    out = df.copy()
    dtv = pd.to_datetime(out[tcol], errors="coerce")
    # context collector may only write HH:MM:SS.  If so, attach trade date.
    if dtv.isna().all():
        dtv = pd.to_datetime(yyyymmdd_to_iso(trade_date) + " " + out[tcol].astype(str), errors="coerce")
    cutoff = pd.to_datetime(f"{yyyymmdd_to_iso(trade_date)} {cutoff_time}:00", errors="coerce")
    out = out.loc[dtv.notna()].copy()
    out["__dt"] = dtv.loc[out.index]
    if pd.notna(cutoff):
        out = out[out["__dt"] <= cutoff]
    return out.sort_values("__dt")


def estimate_sector_range_pct(
    snapshots: pd.DataFrame,
    stock_code: str,
    artifact_name: str,
    sector_symbols: Iterable[str],
    trade_date: str,
    cutoff_time: str,
) -> tuple[float, dict]:
    """Estimate sector_range_pct from sampled sector price max/min.

    For multiple sector symbols, compute a range per sector and average them.
    This avoids mixing different price levels across unrelated sector indices.
    """
    info = {"n_ticks": 0, "n_symbols": 0, "symbols_used": ""}
    if snapshots.empty:
        return np.nan, info
    part = snapshots.copy()
    if "kind" in part.columns:
        part = part[part["kind"].astype(str).str.lower().eq("sector")]
    if "status" in part.columns:
        part = part[part["status"].astype(str).str.lower().eq("ok")]

    # Prefer exact stock/artifact matching if those columns exist.  Some files
    # may only have sector rows without artifact_name; the sector_symbols filter
    # below is the fallback.
    nstock = normalize_symbol(stock_code)
    if "stock_code" in part.columns and nstock:
        tmp = part[part["stock_code"].map(normalize_symbol).eq(nstock)]
        if not tmp.empty:
            part = tmp
    if "artifact_name" in part.columns and artifact_name:
        tmp = part[part["artifact_name"].astype(str).eq(str(artifact_name))]
        if not tmp.empty:
            part = tmp

    symbols = [s for s in sector_symbols if s and str(s).lower() != "nan"]
    sym_col = None
    for c in ["context_symbol", "symbol", "sector_symbol"]:
        if c in part.columns:
            sym_col = c
            break
    if symbols and sym_col:
        part = part[part[sym_col].astype(str).isin(symbols)]

    part = filter_cutoff(part, trade_date, cutoff_time)
    pcol = price_column(part)
    if pcol is None or part.empty:
        return np.nan, info

    ranges: list[float] = []
    used: list[str] = []
    if sym_col and sym_col in part.columns:
        groups = part.groupby(part[sym_col].astype(str), dropna=False)
    else:
        groups = [("__all__", part)]

    for sym, g in groups:
        px = pd.to_numeric(g[pcol], errors="coerce").dropna()
        px = px[np.isfinite(px) & (px > 0)]
        if len(px) < 2:
            continue
        lo = float(px.min())
        hi = float(px.max())
        if lo > 0 and hi >= lo:
            ranges.append(hi / lo - 1.0)
            used.append(str(sym))
            info["n_ticks"] += int(len(px))

    if not ranges:
        return np.nan, info
    info["n_symbols"] = len(ranges)
    info["symbols_used"] = ",".join(used)
    return float(np.nanmean(ranges)), info


def compute_z20(samples_path: Path, trade_date: str, today_range_pct: float, min_periods: int = 10) -> tuple[float, dict]:
    info = {"hist_n": 0, "hist_mean": np.nan, "hist_std": np.nan, "samples": str(samples_path)}
    if not samples_path or not samples_path.exists():
        return np.nan, info
    try:
        df = pd.read_csv(samples_path, parse_dates=["date"])
    except Exception as exc:
        info["error"] = f"read_samples:{type(exc).__name__}:{exc}"
        return np.nan, info
    if "sector_range_pct" not in df.columns:
        info["error"] = "missing sector_range_pct in samples"
        return np.nan, info
    today = pd.to_datetime(yyyymmdd_to_iso(trade_date), errors="coerce")
    hist = df.loc[pd.to_datetime(df["date"], errors="coerce") < today, "sector_range_pct"]
    hist = pd.to_numeric(hist, errors="coerce").dropna().tail(20)
    info["hist_n"] = int(len(hist))
    if len(hist) < min_periods:
        info["error"] = f"hist_n<{min_periods}"
        return np.nan, info
    mu = float(hist.mean())
    sd = float(hist.std())
    info["hist_mean"] = mu
    info["hist_std"] = sd
    if not np.isfinite(sd) or sd <= 0:
        info["error"] = "hist_std<=0"
        return np.nan, info
    x = finite_float(today_range_pct)
    if x is None:
        info["error"] = "today_range_pct invalid"
        return np.nan, info
    return float((x - mu) / sd), info


def remove_missing_feature(text: object, feature: str) -> str:
    parts = [p.strip() for p in str(text or "").split(",") if p.strip() and p.strip().lower() != "nan"]
    parts = [p for p in parts if p != feature]
    return ",".join(parts)


def build_plan_lookup(plan: pd.DataFrame) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if plan.empty:
        return out
    for _, r in plan.iterrows():
        key = (normalize_symbol(r.get("stock_code", "")), str(r.get("artifact_name", "")))
        out[key] = r.to_dict()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill sector_range_z20 from sampled sector prices and training history")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--context-dir", default=str(SAVED_DATA_DIR / "realtime_context"))
    ap.add_argument("--cutoff-time", default="14:55")
    ap.add_argument("--min-periods", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="overwrite existing finite sector_range_z20")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    day_dir = Path(args.context_dir) / args.date
    feat_path = day_dir / "context_features_asof.csv"
    snap_path = day_dir / "context_snapshots.csv"
    plan_path = day_dir / "realtime_context_plan.csv"
    summary_path = day_dir / "context_summary.json"

    if not feat_path.exists():
        raise SystemExit(f"context_features_asof.csv not found: {feat_path}")
    if not snap_path.exists():
        raise SystemExit(f"context_snapshots.csv not found: {snap_path}")
    if not plan_path.exists():
        raise SystemExit(f"realtime_context_plan.csv not found: {plan_path}")

    features = pd.read_csv(feat_path)
    snapshots = pd.read_csv(snap_path)
    plan = pd.read_csv(plan_path)
    plan_lookup = build_plan_lookup(plan)

    for col in [
        "sector_range_z20", "sector_range_pct", "sector_range_z20_source",
        "sector_range_pct_source", "sector_range_z20_hist_n", "sector_range_z20_hist_mean",
        "sector_range_z20_hist_std", "sector_range_est_n_ticks", "sector_range_est_symbols",
    ]:
        if col not in features.columns:
            features[col] = np.nan if col not in {"sector_range_z20_source", "sector_range_pct_source", "sector_range_est_symbols"} else ""

    filled = 0
    pct_estimated = 0
    failed = 0
    details = []

    for idx, row in features.iterrows():
        stock = normalize_symbol(row.get("stock_code", ""))
        art = str(row.get("artifact_name", ""))
        plan_row = plan_lookup.get((stock, art), {})
        samples_raw = row.get("samples", "") if "samples" in features.columns else ""
        if not str(samples_raw or "").strip():
            samples_raw = plan_row.get("samples", "")
        samples = resolve_path(samples_raw, stock)

        # Skip rows that neither require sector nor mention sector_range_z20.
        required = str(row.get("required_context_features", plan_row.get("required_context_features", "")) or "")
        missing = str(row.get("missing_context_features", "") or "")
        wants_sector_range = ("sector_range_z20" in required) or ("sector_range_z20" in missing)
        if not wants_sector_range and not args.force:
            continue

        existing_z = finite_float(row.get("sector_range_z20"))
        if existing_z is not None and not args.force:
            continue

        current_pct = finite_float(row.get("sector_range_pct"))
        pct_source = "existing"
        # Existing 0 is usually fake when THS summary filled OHLC with current price.
        if current_pct is None or current_pct <= 0:
            sectors = split_list(row.get("sector_symbols", plan_row.get("sector_symbols", "")))
            est_pct, est_info = estimate_sector_range_pct(
                snapshots=snapshots,
                stock_code=stock,
                artifact_name=art,
                sector_symbols=sectors,
                trade_date=args.date,
                cutoff_time=args.cutoff_time,
            )
            if np.isfinite(est_pct) and est_pct >= 0:
                current_pct = float(est_pct)
                pct_source = "sampled_snapshot_estimate"
                features.at[idx, "sector_range_pct"] = current_pct
                features.at[idx, "sector_range_pct_source"] = pct_source
                features.at[idx, "sector_range_est_n_ticks"] = est_info.get("n_ticks", 0)
                features.at[idx, "sector_range_est_symbols"] = est_info.get("symbols_used", "")
                pct_estimated += 1
            else:
                features.at[idx, "sector_range_pct_source"] = "unavailable"

        z, zinfo = (np.nan, {})
        if samples is not None and current_pct is not None and np.isfinite(current_pct):
            z, zinfo = compute_z20(samples, args.date, current_pct, args.min_periods)

        if np.isfinite(z):
            features.at[idx, "sector_range_z20"] = float(z)
            features.at[idx, "sector_range_z20_source"] = f"training_history_{pct_source}"
            features.at[idx, "sector_range_z20_hist_n"] = zinfo.get("hist_n", np.nan)
            features.at[idx, "sector_range_z20_hist_mean"] = zinfo.get("hist_mean", np.nan)
            features.at[idx, "sector_range_z20_hist_std"] = zinfo.get("hist_std", np.nan)
            if "missing_context_features" in features.columns:
                features.at[idx, "missing_context_features"] = remove_missing_feature(row.get("missing_context_features", ""), "sector_range_z20")
            # If this was the only missing context feature, make context_status ok.
            if "context_status" in features.columns:
                remain = str(features.at[idx, "missing_context_features"] if "missing_context_features" in features.columns else "")
                if not remain.strip() or remain.strip().lower() == "nan":
                    features.at[idx, "context_status"] = "ok"
            filled += 1
            details.append({"stock_code": stock, "artifact_name": art, "status": "filled", "sector_range_pct": current_pct, "sector_range_z20": float(z), "samples": str(samples)})
        else:
            failed += 1
            details.append({"stock_code": stock, "artifact_name": art, "status": "failed", "reason": zinfo.get("error", "unknown"), "samples": str(samples) if samples else ""})

    if not args.no_backup:
        backup = feat_path.with_suffix(feat_path.suffix + ".before_sector_range_z20_history_patch")
        if not backup.exists():
            backup.write_bytes(feat_path.read_bytes())

    features.to_csv(feat_path, index=False, encoding="utf-8-sig")

    diag = {
        "date": args.date,
        "context_features_asof": str(feat_path),
        "rows": int(len(features)),
        "filled_sector_range_z20": int(filled),
        "estimated_sector_range_pct": int(pct_estimated),
        "failed_sector_range_z20": int(failed),
        "details": details[:200],
    }
    (day_dir / "sector_range_z20_history_patch_summary.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        summary["sector_range_z20_history_patch"] = {k: v for k, v in diag.items() if k != "details"}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps({k: v for k, v in diag.items() if k != "details"}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
