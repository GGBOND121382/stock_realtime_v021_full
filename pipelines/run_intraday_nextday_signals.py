#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect intraday data and emit pre-close next-day model buy signals.

This is a thin orchestration layer:
  - candidates come from selected_watchlist.txt
  - intraday collection is delegated to collect_akshare_l1_cache.py
  - scoring uses stock-specific artifacts under saved_models/

Outputs are intentionally split:
  - all_scores.csv: every artifact score, diagnostics included
  - buy_signals.csv: only rows that pass signal + safety gates
"""
from __future__ import annotations

import argparse
import json
import re
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
COLLECT_CONTEXT_SCRIPT = "data_collection/collect_realtime_context.py"
UPDATE_BAOSTOCK_SCRIPT = "data_collection/update_baostock_raw_cache.py"
DEFAULT_BENCHMARK_SYMBOLS = "000300.SH,000001.SH,399001.SZ,399006.SZ"


@dataclass
class ModelArtifact:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    metadata: dict
    created_at: str


REQUIRED_REALTIME_CORE_FIELDS = {"close", "open", "high", "low", "volume", "amount"}
PRICE_SNAPSHOT_FEATURES = {
    "close", "open", "high", "low", "volume", "amount", "daily_vwap",
    "daily_vwap_volume", "daily_vwap_pv", "close_to_vwap", "open_to_vwap",
    "intraday_ret", "range_pct", "close_range_pos", "high_to_close", "close_to_low",
    "down_day_below_vwap", "overnight_intraday_reversal", "overnight_plus_intraday",
}
INTRADAY_BAR_FEATURES = {
    "bar_count", "first_30m_ret", "first_60m_ret", "morning_ret", "afternoon_ret",
    "last_30m_ret", "last_60m_ret", "morning_vwap", "afternoon_vwap",
    "last_30m_vwap", "first_60m_volume_share", "last_30m_volume_share",
    "morning_vwap_to_close", "afternoon_vwap_to_close", "last_30m_vwap_to_close",
    "morning_afternoon_reversal", "first60_last30_reversal", "n_intraday_bars",
}
BID_ASK_FEATURE_PREFIXES = (
    "bid", "ask", "spread", "depth_imbalance", "weighted_bid", "weighted_ask", "mid_price",
)
REALTIME_FUND_FLOW_PREFIXES = ("ak_fund_",)
HISTORICAL_CONTEXT_PREFIXES = (
    "sector_", "stock_vs_sector_", "bench_", "profit_", "operation_", "growth_",
    "solvency_", "cashflow_", "dupont_", "zijin_", "gold_", "copper_", "hk_",
)
HISTORICAL_CONTEXT_COLUMNS = {"peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "fund_days_since_effective"}

REALTIME_CONTEXT_PREFIXES = (
    "sector_", "stock_vs_sector_",
    "gold_", "copper_", "silver_", "zijin_hk_", "zijin_a_h_", "stock_vs_gold_", "stock_vs_copper_",
    "precious_", "industrial_metal_", "minor_metal_",
    "hog_", "feed_",
    # New stock-external profiles from build_stock_external_features.py.
    "ai_", "mwb_", "pur_", "fert_", "sp_", "ane_", "ocg_",
)
NON_CRITICAL_REALTIME_CONTEXT_FEATURES = {"sector_range_z20"}

REALTIME_CONTEXT_EXACT_FEATURES = {"gold_silver_ratio", "gold_copper_ratio", "feed_cost_index", "feed_soymeal_corn_ratio", "feed_hog_cost_ratio"}
STOCK_EXTERNAL_RELATIVE_FAMILIES = (
    "stock_basket", "etf_basket", "future_basket", "board_basket", "us_basket",
)



@dataclass
class RuntimeRequirement:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    requires_price_snapshot: bool
    requires_intraday_bars: bool
    requires_bid_ask: bool
    requires_realtime_fund_flow: bool
    uses_historical_context: bool
    requires_realtime_context: bool
    required_context_features: list[str]
    unsupported_realtime_features: list[str]
    required_raw_fields: list[str]
    required_derived_features: list[str]


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


def run_cmd(cmd: list[str], dry_run: bool = False, timeout_seconds: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess with optional timeout.

    The intraday signal path is time-sensitive.  A slow data source should not
    make the whole 14:55 signal job hang indefinitely, so callers can pass a
    timeout for final collection / bar building.
    """
    print("RUN", " ".join(cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0)
    try:
        return subprocess.run(cmd, cwd=ROOT, check=check, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout_seconds}s: {' '.join(cmd)}") from exc


def wait_until(target: dtime) -> None:
    while datetime.now().time() < target:
        remaining = datetime.combine(datetime.today(), target) - datetime.now()
        sleep_s = max(1, min(30, int(remaining.total_seconds())))
        print(f"waiting for signal time {target.strftime('%H:%M')}, {int(remaining.total_seconds())}s left", flush=True)
        time.sleep(sleep_s)


def is_after_hhmm(value: Optional[str]) -> bool:
    if not value:
        return False
    return datetime.now().time() >= parse_hhmm(value)


def append_reject_reason(row: dict, reason: str) -> None:
    existing = str(row.get("reject_reason") or "")
    parts = [p for p in existing.split(";") if p]
    parts.append(reason)
    row["reject_reason"] = ";".join(parts)


def snapshot_age_seconds(snapshot_time: str, trade_date: str, cutoff_time: Optional[str]) -> float:
    if not snapshot_time or not cutoff_time:
        return np.nan
    snap_ts = pd.to_datetime(snapshot_time, errors="coerce")
    cutoff_ts = parse_cutoff_dt(trade_date, cutoff_time)
    if pd.isna(snap_ts) or cutoff_ts is None or pd.isna(cutoff_ts):
        return np.nan
    return float((cutoff_ts - snap_ts).total_seconds())


def as_bool_series_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


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



def artifact_feature_columns(artifact: ModelArtifact) -> list[str]:
    path = artifact.artifact_dir / "feature_columns.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_realtime_context_feature(col: str) -> bool:
    return col in REALTIME_CONTEXT_EXACT_FEATURES or col.startswith(REALTIME_CONTEXT_PREFIXES)


def context_dependencies_for_model_features(cols: list[str]) -> list[str]:
    deps: set[str] = set()
    for col in cols:
        if col in NON_CRITICAL_REALTIME_CONTEXT_FEATURES:
            # sector_range_z20 needs a 20-day historical sector_range_pct window.
            # The realtime THS summary snapshot alone cannot compute it correctly,
            # so do not let this single feature hard-block sector models.
            continue
        if not is_realtime_context_feature(col):
            continue
        if col.startswith("stock_vs_sector_ret"):
            suffix = col.replace("stock_vs_sector_ret", "")
            if suffix.isdigit():
                deps.add(f"sector_ret{suffix}")
            continue
        if col.startswith("stock_vs_") and "_ret" in col:
            body = col[len("stock_vs_"):]
            ctx, _, suffix = body.partition("_ret")
            if ctx and suffix.isdigit():
                deps.add(f"{ctx}_ret{suffix}")
            continue
        if col.startswith("sector_vs_bench_ret"):
            suffix = col.replace("sector_vs_bench_ret", "")
            if suffix.isdigit():
                deps.add(f"sector_ret{suffix}")
            continue
        m = re.fullmatch(r"([A-Za-z0-9]+)_stock_vs_(stock_basket|etf_basket|future_basket|board_basket|us_basket)_ret(\d+)", col)
        if m:
            pfx, family, suffix = m.group(1), m.group(2), m.group(3)
            deps.add(f"{pfx}_{family}_close_ret{suffix}")
            continue
        deps.add(col)
    return sorted(deps)


def infer_runtime_requirement(artifact: ModelArtifact) -> RuntimeRequirement:
    cols = artifact_feature_columns(artifact)
    colset = set(cols)
    requires_price = bool(colset & PRICE_SNAPSHOT_FEATURES)
    requires_intraday = bool(colset & INTRADAY_BAR_FEATURES)
    bid_ask_cols = [c for c in cols if c.startswith(BID_ASK_FEATURE_PREFIXES)]
    fund_flow_cols = [c for c in cols if c.startswith(REALTIME_FUND_FLOW_PREFIXES)]
    context_cols = [c for c in cols if c in HISTORICAL_CONTEXT_COLUMNS or c.startswith(HISTORICAL_CONTEXT_PREFIXES)]
    realtime_context_cols = [c for c in cols if is_realtime_context_feature(c)]
    realtime_context_deps = context_dependencies_for_model_features(cols)
    unsupported = []
    unsupported.extend(fund_flow_cols)
    derived = sorted((colset & PRICE_SNAPSHOT_FEATURES) | (colset & INTRADAY_BAR_FEATURES))
    return RuntimeRequirement(
        stock_code=artifact.stock_code,
        artifact_name=artifact.artifact_name,
        artifact_dir=artifact.artifact_dir,
        requires_price_snapshot=requires_price or requires_intraday,
        requires_intraday_bars=requires_intraday,
        requires_bid_ask=bool(bid_ask_cols),
        requires_realtime_fund_flow=bool(fund_flow_cols),
        uses_historical_context=bool(context_cols),
        requires_realtime_context=bool(realtime_context_cols),
        required_context_features=sorted(set(realtime_context_deps)),
        unsupported_realtime_features=unsupported,
        required_raw_fields=sorted(REQUIRED_REALTIME_CORE_FIELDS) if (requires_price or requires_intraday) else [],
        required_derived_features=derived,
    )


def runtime_requirement_rows(artifacts: list[ModelArtifact]) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        req = infer_runtime_requirement(artifact)
        rows.append({
            "stock_code": req.stock_code,
            "artifact_name": req.artifact_name,
            "artifact_dir": str(req.artifact_dir),
            "requires_price_snapshot": req.requires_price_snapshot,
            "requires_intraday_bars": req.requires_intraday_bars,
            "requires_bid_ask": req.requires_bid_ask,
            "requires_realtime_fund_flow": req.requires_realtime_fund_flow,
            "uses_historical_context": req.uses_historical_context,
            "requires_realtime_context": req.requires_realtime_context,
            "required_context_features": ",".join(req.required_context_features),
            "required_raw_fields": ",".join(req.required_raw_fields),
            "required_derived_features": ",".join(req.required_derived_features),
            "unsupported_realtime_features": ",".join(req.unsupported_realtime_features),
        })
    return pd.DataFrame(rows)


def prepare_runtime_plan(args: argparse.Namespace, trade_date: str) -> tuple[list[ModelArtifact], Path]:
    selected = set(read_watchlist(Path(args.watchlist)))
    models_dir = Path(args.models_dir)
    artifacts_all: list[ModelArtifact] = []
    all_model_symbols: set[str] = set()
    for meta_path in sorted(models_dir.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stock = normalize_symbol(meta.get("stock_code", ""))
        if not stock:
            continue
        artifacts_all.append(
            ModelArtifact(
                stock_code=stock,
                artifact_name=str(meta.get("artifact_name") or meta_path.parent.name),
                artifact_dir=meta_path.parent,
                metadata=meta,
                created_at=str(meta.get("artifact_created_at") or ""),
            )
        )
        all_model_symbols.add(stock)

    candidate_symbols = (selected & all_model_symbols) if selected else all_model_symbols
    artifacts = [a for a in artifacts_all if a.stock_code in candidate_symbols]
    if args.model_policy != "all":
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
        artifacts = chosen

    out_dir = Path(args.signal_out_dir) / trade_date
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_symbols = sorted({a.stock_code for a in artifacts})
    effective_watchlist = out_dir / "effective_watchlist.txt"
    effective_watchlist.write_text("\n".join(effective_symbols) + ("\n" if effective_symbols else ""), encoding="utf-8")

    skipped = [{"stock_code": sym, "reason": "no_saved_model"} for sym in sorted(selected - all_model_symbols)]
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_symbols_no_model.csv", index=False, encoding="utf-8-sig")

    req_df = runtime_requirement_rows(artifacts)
    req_df.to_csv(out_dir / "runtime_feature_requirements.csv", index=False, encoding="utf-8-sig")

    unsupported_by_stock: dict[str, list[str]] = {}
    if not req_df.empty:
        for _, row in req_df.iterrows():
            unsupported = str(row.get("unsupported_realtime_features") or "")
            if unsupported:
                unsupported_by_stock.setdefault(str(row["stock_code"]), []).append(unsupported)
    eff_rows = []
    for sym in sorted(selected | all_model_symbols):
        has_model = sym in all_model_symbols
        in_selected = sym in selected
        collect_required = sym in effective_symbols
        unsupported = ";".join(unsupported_by_stock.get(sym, []))
        eff_rows.append({
            "stock_code": sym,
            "in_selected_watchlist": in_selected,
            "has_saved_model": has_model,
            "collect_required": collect_required,
            "unsupported_realtime_features": unsupported,
            "reason": "ok" if collect_required else ("no_saved_model" if in_selected and not has_model else "not_selected"),
        })
    pd.DataFrame(eff_rows).to_csv(out_dir / "effective_watchlist.csv", index=False, encoding="utf-8-sig")

    print(f"[MODEL] saved model symbols: {len(all_model_symbols)}", flush=True)
    print(f"[WATCHLIST] selected symbols: {len(selected)}", flush=True)
    print(f"[COLLECT] symbols with models: {len(effective_symbols)}", flush=True)
    print(f"[SKIP] symbols without models: {len(skipped)}", flush=True)
    args._effective_watchlist = str(effective_watchlist)
    args._runtime_artifacts = artifacts
    return artifacts, effective_watchlist

def resolve_repo_path(raw: Optional[str], stock_code: str = "") -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p

    text = str(raw).replace("\\", "/")
    for marker in ["stock_realtime_v021_full/", "stock_realtime/"]:
        if marker in text:
            candidate = ROOT / text.split(marker, 1)[1]
            if candidate.exists():
                return candidate

    name = Path(text).name
    if name and stock_code:
        code6 = stock_code.split(".", 1)[0]
        roots = sorted((ROOT / "saved_data").glob(f"{code6}_pipeline_out*"))
        roots.extend([
            ROOT / "saved_data" / f"{code6}_base_out",
            ROOT / "saved_data" / f"zijin_{code6}_base_out",
        ])
        hits = []
        for root in roots:
            if root.exists():
                hits.extend(root.rglob(name))
        hits = sorted(dict.fromkeys(h.resolve() for h in hits))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(f"ambiguous path for {stock_code} {name}: {[str(h) for h in hits[:10]]}")
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
    bars = bars[
        ((bars["time_str"] >= "09:35:00") & (bars["time_str"] <= "11:30:00"))
        | ((bars["time_str"] >= "13:05:00") & (bars["time_str"] <= "15:00:00"))
    ].copy()
    if bars.empty:
        return pd.DataFrame()
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


def _normalize_realtime_minute_bars(bars: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    out = bars.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif {"trade_date", "trade_time"}.issubset(out.columns):
        out["datetime"] = pd.to_datetime(
            out["trade_date"].astype(str) + out["trade_time"].astype(str).str.zfill(6),
            errors="coerce",
        )
    else:
        # Last resort: use date plus row order minutes; caller will likely drop if invalid.
        out["datetime"] = pd.NaT
    for c in ["open", "high", "low", "close", "volume", "amount", "bar_volume", "bar_amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # If minute bars store cumulative volume/amount, convert to bar increments when available.
    if "bar_volume" in out.columns and out["bar_volume"].notna().any():
        out["volume"] = out["bar_volume"].fillna(0)
    if "bar_amount" in out.columns and out["bar_amount"].notna().any():
        out["amount"] = out["bar_amount"].fillna(0)
    out = normalize_volume_unit_if_lot(out)
    return out.dropna(subset=["datetime", "open", "high", "low", "close"])


def infer_volume_units(volume: pd.Series, amount: pd.Series, price: pd.Series) -> pd.DataFrame:
    vol = pd.to_numeric(volume, errors="coerce")
    amt = pd.to_numeric(amount, errors="coerce")
    px = pd.to_numeric(price, errors="coerce")
    ratio = (amt / vol.replace(0, np.nan)) / px.replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    unit = pd.Series("unknown", index=vol.index, dtype="object")
    unit = unit.mask(ratio.between(0.5, 2.0), "shares")
    unit = unit.mask(ratio.between(50.0, 150.0), "lots")
    multiplier = pd.Series(np.nan, index=vol.index, dtype="float64")
    multiplier = multiplier.mask(unit.eq("shares"), 1.0)
    multiplier = multiplier.mask(unit.eq("lots"), 100.0)
    return pd.DataFrame({
        "raw_volume": vol,
        "raw_amount": amt,
        "vwap_ratio_raw": ratio,
        "volume_unit_inferred": unit,
        "volume_unit_multiplier": multiplier,
        "normalized_volume": vol * multiplier,
    }, index=vol.index)


def normalize_volume_unit_if_lot(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize realtime volume row by row; unknown units become missing."""
    if not {"volume", "amount", "close"}.issubset(df.columns):
        return df
    out = df.copy()
    unit_df = infer_volume_units(out["volume"], out["amount"], out["close"])
    for c in ["raw_volume", "raw_amount", "vwap_ratio_raw", "volume_unit_inferred", "volume_unit_multiplier"]:
        if c not in out.columns:
            out[c] = unit_df[c]
    out["volume"] = unit_df["normalized_volume"]
    return out


def normalize_pct_chg_value(raw_pct, close, prev_close, source_text: str = "") -> dict:
    raw = pd.to_numeric(pd.Series([raw_pct]), errors="coerce").iloc[0]
    close_v = pd.to_numeric(pd.Series([close]), errors="coerce").iloc[0]
    prev_v = pd.to_numeric(pd.Series([prev_close]), errors="coerce").iloc[0]
    source = str(source_text or "").lower()
    if np.isfinite(raw) and any(token in source for token in ("ths", "xq", "em", "eastmoney")):
        return {
            "pct_chg_raw": float(raw),
            "pct_chg_norm": float(raw) / 100.0,
            "pct_chg_source": source_text,
            "pct_chg_unit": "percent",
        }
    if np.isfinite(close_v) and np.isfinite(prev_v) and prev_v > 0:
        return {
            "pct_chg_raw": float(raw) if np.isfinite(raw) else np.nan,
            "pct_chg_norm": float(close_v) / float(prev_v) - 1.0,
            "pct_chg_source": source_text or "derived_close_prev_close",
            "pct_chg_unit": "ratio",
        }
    if np.isfinite(raw):
        return {
            "pct_chg_raw": float(raw),
            "pct_chg_norm": float(raw) / 100.0,
            "pct_chg_source": source_text,
            "pct_chg_unit": "percent_assumed",
        }
    return {
        "pct_chg_raw": np.nan,
        "pct_chg_norm": np.nan,
        "pct_chg_source": source_text,
        "pct_chg_unit": "unknown",
    }


def summarize_volume_unit_diagnostics(unit_df: pd.DataFrame) -> dict:
    if unit_df.empty or "volume_unit_inferred" not in unit_df.columns:
        return {}
    ratio = pd.to_numeric(unit_df.get("vwap_ratio_raw"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    counts = unit_df["volume_unit_inferred"].fillna("unknown").value_counts()
    known_units = [u for u in ("shares", "lots") if int(counts.get(u, 0)) > 0]
    return {
        "volume_unit_inferred": "mixed" if len(known_units) > 1 else (known_units[0] if known_units else "unknown"),
        "volume_unit_ratio_median": float(ratio.median()) if len(ratio) else np.nan,
        "volume_unit_ratio_p05": float(ratio.quantile(0.05)) if len(ratio) else np.nan,
        "volume_unit_ratio_p95": float(ratio.quantile(0.95)) if len(ratio) else np.nan,
        "volume_unit_lots_rows": int(counts.get("lots", 0)),
        "volume_unit_shares_rows": int(counts.get("shares", 0)),
        "volume_unit_unknown_rows": int(counts.get("unknown", 0)),
        "volume_unit_mixed": bool(len(known_units) > 1),
    }


def _load_pending_intraday_rows(
    cache_dir: Path,
    stock_code: str,
    trade_date: str,
    cutoff_time: Optional[str] = None,
) -> pd.DataFrame:
    base = cache_symbol_dir(cache_dir, trade_date, stock_code)
    path = base / "minute_bars_5min.csv"
    if not path.exists():
        return pd.DataFrame()
    cutoff_dt = parse_cutoff_dt(trade_date, cutoff_time)
    try:
        bars = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if bars.empty:
        return pd.DataFrame()
    if "source" not in bars.columns or not (bars["source"].astype(str) == "local_snapshot_5m_right_endpoint").all():
        return pd.DataFrame()
    if "bar_freq" in bars.columns and not (bars["bar_freq"].astype(str) == "5min").all():
        return pd.DataFrame()
    if "bar_label" in bars.columns and not (bars["bar_label"].astype(str) == "right").all():
        return pd.DataFrame()
    bars = _normalize_realtime_minute_bars(bars, trade_date)
    if bars.empty:
        return pd.DataFrame()
    times = pd.to_datetime(bars["datetime"], errors="coerce").dropna().sort_values()
    if len(times) >= 2:
        prev_times = times.shift(1)
        diffs = times - prev_times
        regular = diffs == pd.Timedelta(minutes=5)
        lunch_break = (
            prev_times.dt.strftime("%H:%M:%S").eq("11:30:00")
            & times.dt.strftime("%H:%M:%S").eq("13:05:00")
        )
        invalid = ~(regular | lunch_break | diffs.isna())
        if invalid.any():
            return pd.DataFrame()
    if cutoff_dt is not None:
        bars = bars[bars["datetime"] <= cutoff_dt].copy()
    if bars.empty:
        return pd.DataFrame()
    return build_intraday_rows(bars)


def add_scoring_features(
    df: pd.DataFrame,
    intraday_path: Optional[Path],
    cache_dir: Path,
    stock_code: str,
    cutoff_time: Optional[str] = None,
    scoring_trade_date: Optional[str] = None,
    benchmark_symbols: str = DEFAULT_BENCHMARK_SYMBOLS,
) -> pd.DataFrame:
    out = add_reversal_daily_features(df)
    if scoring_trade_date and {"date", "prev_close_for_feature"}.issubset(out.columns):
        scoring_date = pd.to_datetime(yyyymmdd_to_iso(scoring_trade_date)).normalize()
        mask = pd.to_datetime(out["date"], errors="coerce").dt.normalize() == scoring_date
        pcf = pd.to_numeric(out.loc[mask, "prev_close_for_feature"], errors="coerce")
        valid = pcf.notna() & (pcf > 0)
        if valid.any():
            idx = pcf[valid].index
            out.loc[idx, "prev_close"] = pcf.loc[idx]
            if "open" in out.columns:
                out.loc[idx, "overnight_ret"] = pd.to_numeric(out.loc[idx, "open"], errors="coerce") / pcf.loc[idx] - 1.0
            if "close" in out.columns:
                pct = pd.to_numeric(out.loc[idx, "close"], errors="coerce") / pcf.loc[idx] - 1.0
                out.loc[idx, "pct_chg"] = pct
                if "pct_chg_norm" in out.columns:
                    out.loc[idx, "pct_chg_norm"] = pct

    intra = load_intraday_feature_cache(intraday_path, cache_dir, stock_code)
    # Merge current-day pending minute bars.  The historical feature cache is
    # normally only complete up to T-1, so without this the same-day row misses
    # morning/afternoon/last-30m segment features and triggers filled_features_gt_*.
    try:
        if "date" in out.columns and not out.empty:
            if scoring_trade_date:
                scoring_date = pd.to_datetime(yyyymmdd_to_iso(scoring_trade_date)).normalize()
            else:
                scoring_date = pd.to_datetime(out["date"].max()).normalize()
            trade_date = scoring_date.strftime("%Y%m%d")
            if not intra.empty:
                intra_dates = pd.to_datetime(intra["date"], errors="coerce").dt.normalize()
                intra = intra.loc[intra_dates < scoring_date].copy()
            pending_intra = _load_pending_intraday_rows(cache_dir, stock_code, trade_date, cutoff_time=cutoff_time)
            if not pending_intra.empty:
                if intra.empty:
                    intra = pending_intra
                else:
                    intra = pd.concat([intra, pending_intra], ignore_index=True)
                    intra["date"] = pd.to_datetime(intra["date"], errors="coerce")
                    intra = intra.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    except Exception as exc:
        print(f"[WARN] failed to merge pending intraday bars for {stock_code}: {type(exc).__name__}: {exc}", flush=True)

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
    bench_trade_date = scoring_trade_date or (pd.to_datetime(out["date"].max()).strftime("%Y%m%d") if "date" in out.columns and not out.empty else "")
    out = overlay_benchmark_asof(out, bench_trade_date, cache_dir, cutoff_time, benchmark_symbols)
    out = add_market_state_features(out)
    return add_benchmark_asof_aliases(out)

def parse_cutoff_dt(trade_date: str, cutoff_time: Optional[str]) -> Optional[pd.Timestamp]:
    if not cutoff_time:
        return None
    return pd.to_datetime(f"{yyyymmdd_to_iso(trade_date)} {cutoff_time}:00", errors="coerce")


def live_daily_from_snapshots(stock_code: str, trade_date: str, cache_dir: Path, cutoff_time: Optional[str]) -> dict:
    """Return latest live daily fields from pending snapshots.

    Important fixes:
      - keep true cumulative amount/volume for liquidity gates;
      - use vendor daily open/high/low when available instead of only sampled px range;
      - expose prev_close, pct_chg and overnight_ret for same-day scoring features.
    """
    snap_path = cache_symbol_dir(cache_dir, trade_date, stock_code) / "snapshot_5level.csv"
    if not snap_path.exists():
        return {}
    try:
        snap = pd.read_csv(snap_path, encoding="utf-8-sig")
    except Exception:
        return {}
    if snap.empty:
        return {}

    if "datetime" in snap.columns:
        snap["datetime"] = pd.to_datetime(snap["datetime"], errors="coerce")
    else:
        snap["datetime"] = pd.to_datetime(
            snap.get("trade_date", "").astype(str) + snap.get("trade_time", "").astype(str).str.zfill(6),
            errors="coerce",
        )
    snap = snap.dropna(subset=["datetime"]).sort_values("datetime")
    cutoff_dt = parse_cutoff_dt(trade_date, cutoff_time)
    if cutoff_dt is not None:
        snap = snap[snap["datetime"] <= cutoff_dt].copy()
    if snap.empty:
        return {"core_complete": False, "missing_core_fields": "no_snapshot_before_cutoff"}

    def last_num(col: str):
        if col not in snap.columns:
            return np.nan
        ss = pd.to_numeric(snap[col], errors="coerce").dropna()
        return float(ss.iloc[-1]) if len(ss) else np.nan

    px = pd.to_numeric(snap.get("last_price"), errors="coerce") if "last_price" in snap.columns else pd.Series(dtype=float)
    valid_px = px[px > 0].dropna()
    if valid_px.empty:
        return {"core_complete": False, "missing_core_fields": "close"}

    last = snap.iloc[-1]
    close = float(valid_px.iloc[-1])
    open_v = last_num("open")
    high_v = last_num("high")
    low_v = last_num("low")
    if not np.isfinite(open_v) or open_v <= 0:
        open_v = float(valid_px.iloc[0])
    if not np.isfinite(high_v) or high_v <= 0:
        high_v = float(valid_px.max())
    if not np.isfinite(low_v) or low_v <= 0:
        low_v = float(valid_px.min())

    volume = last_num("volume")
    amount = last_num("amount")
    raw_volume = volume
    prev_close = last_num("prev_close")
    pct_diag = normalize_pct_chg_value(
        last_num("pct_chg"),
        close,
        prev_close,
        str(last.get("spot_source_used") or last.get("quote_source") or ""),
    )
    pct_chg = pct_diag["pct_chg_norm"]
    unit_diag = {}
    if {"volume", "amount"}.issubset(snap.columns):
        unit_df = infer_volume_units(snap["volume"], snap["amount"], px.reindex(snap.index))
        unit_diag = summarize_volume_unit_diagnostics(unit_df)
        norm_vol = pd.to_numeric(unit_df["normalized_volume"], errors="coerce")
        valid_norm = norm_vol.dropna()
        if len(valid_norm):
            volume = float(valid_norm.iloc[-1])
            last_idx = valid_norm.index[-1]
            unit_diag["vwap_ratio_raw"] = float(unit_df.loc[last_idx, "vwap_ratio_raw"])
            unit_diag["volume_unit_multiplier"] = float(unit_df.loc[last_idx, "volume_unit_multiplier"])
            unit_diag["volume_unit_last"] = str(unit_df.loc[last_idx, "volume_unit_inferred"])
        else:
            volume = np.nan

    row = {
        "open": open_v,
        "high": high_v,
        "low": low_v,
        "close": close,
        "raw_volume": raw_volume,
        "volume": volume,
        "amount": amount,
        "live_prev_close": prev_close,
        "prev_close": prev_close,
        "prev_close_for_feature": prev_close,
        "pct_chg": pct_chg,
        "pct_chg_raw": pct_diag["pct_chg_raw"],
        "pct_chg_norm": pct_diag["pct_chg_norm"],
        "pct_chg_source": pct_diag["pct_chg_source"],
        "pct_chg_unit": pct_diag["pct_chg_unit"],
        "snapshots": int(len(snap)),
        "snapshot_time": str(last["datetime"]),
        "source_used": str(last.get("spot_source_used") or last.get("quote_source") or ""),
    }
    row.update(unit_diag)
    if np.isfinite(amount) and np.isfinite(volume) and volume > 0:
        row["daily_vwap"] = amount / volume
    if np.isfinite(prev_close) and prev_close > 0 and np.isfinite(open_v):
        row["overnight_ret"] = open_v / prev_close - 1.0

    missing = []
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        v = row.get(field)
        if v is None or not np.isfinite(v) or float(v) <= 0:
            missing.append(field)
    row["core_complete"] = len(missing) == 0
    row["missing_core_fields"] = ",".join(missing)
    return row


def live_benchmark_close_from_snapshots(symbol: str, trade_date: str, cache_dir: Path, cutoff_time: Optional[str]) -> dict:
    snap_path = cache_symbol_dir(cache_dir, trade_date, symbol) / "snapshot_5level.csv"
    if not snap_path.exists():
        return {}
    try:
        snap = pd.read_csv(snap_path, encoding="utf-8-sig")
    except Exception:
        return {}
    if snap.empty:
        return {}
    if "datetime" in snap.columns:
        snap["datetime"] = pd.to_datetime(snap["datetime"], errors="coerce")
    else:
        snap["datetime"] = pd.to_datetime(
            snap.get("trade_date", "").astype(str) + snap.get("trade_time", "").astype(str).str.zfill(6),
            errors="coerce",
        )
    snap = snap.dropna(subset=["datetime"]).sort_values("datetime")
    cutoff_dt = parse_cutoff_dt(trade_date, cutoff_time)
    if cutoff_dt is not None:
        snap = snap[snap["datetime"] <= cutoff_dt].copy()
    if snap.empty or "last_price" not in snap.columns:
        return {}
    close_s = pd.to_numeric(snap["last_price"], errors="coerce")
    valid = close_s[close_s > 0].dropna()
    if valid.empty:
        return {}
    last_idx = valid.index[-1]
    raw_pct = pd.to_numeric(snap.get("pct_chg"), errors="coerce").dropna() if "pct_chg" in snap.columns else pd.Series(dtype=float)
    source = str(snap.loc[last_idx].get("spot_source_used") or snap.loc[last_idx].get("quote_source") or "")
    return {
        "bench_symbol_asof": normalize_symbol(symbol),
        "bench_close_asof": float(valid.iloc[-1]),
        "bench_snapshot_time": str(snap.loc[last_idx, "datetime"]),
        "bench_pct_chg_raw": float(raw_pct.iloc[-1]) if len(raw_pct) else np.nan,
        "bench_source_used": source,
    }


def overlay_benchmark_asof(df: pd.DataFrame, trade_date: str, cache_dir: Path, cutoff_time: Optional[str], benchmark_symbols: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns or "bench_close" not in df.columns:
        return df
    live = {}
    for sym in [s.strip() for s in str(benchmark_symbols or "").replace(";", ",").split(",") if s.strip()]:
        live = live_benchmark_close_from_snapshots(sym, trade_date, cache_dir, cutoff_time)
        if live:
            break
    if not live:
        return df
    out = df.sort_values("date").copy()
    trade_ts = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    if not (pd.to_datetime(out["date"], errors="coerce").dt.normalize() == trade_ts).any():
        return out
    idx = out.index[pd.to_datetime(out["date"], errors="coerce").dt.normalize() == trade_ts][-1]
    for col in ["bench_close_asof", "bench_symbol_asof", "bench_snapshot_time", "bench_pct_chg_raw", "bench_source_used"]:
        if col not in out.columns:
            out[col] = np.nan
    out.at[idx, "bench_close"] = live["bench_close_asof"]
    for col, val in live.items():
        out.at[idx, col] = val
    return out


def add_benchmark_asof_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty or "date" not in out.columns:
        return out
    for src, dst in [
        ("bench_close", "bench_close_asof"),
        ("bench_ret1", "bench_ret1_asof"),
        ("bench_ret5", "bench_ret5_asof"),
        ("bench_ret20", "bench_ret20_asof"),
        ("bench_ma20_gap", "bench_ma20_gap_asof"),
        ("bench_ma60_gap", "bench_ma60_gap_asof"),
        ("regime_market_ok", "regime_market_ok_asof"),
        ("regime_market_strict", "regime_market_strict_asof"),
    ]:
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
        elif src in out.columns:
            out[dst] = out[src]
    if "bench_close" in out.columns:
        close = pd.to_numeric(out["bench_close"], errors="coerce")
        out["bench_ret1"] = close / close.shift(1) - 1.0
        out["bench_ret5"] = close / close.shift(5) - 1.0
        out["bench_ret1_asof"] = out["bench_ret1"]
        out["bench_ret5_asof"] = out["bench_ret5"]
    return out


def overlay_current_day_from_cache(df: pd.DataFrame, stock_code: str, trade_date: str, cache_dir: Path, cutoff_time: Optional[str] = None) -> pd.DataFrame:
    """Overlay current-day live daily fields onto the last scoring row.

    Fix: do not require live fields to already exist in the historical training
    sample.  Liquidity gates and diagnostics need amount/volume even if the
    saved model feature set never used them.
    """
    live = live_daily_from_snapshots(stock_code, trade_date, cache_dir, cutoff_time)
    if not live:
        daily_path = cache_symbol_dir(cache_dir, trade_date, stock_code) / "daily_features.csv"
        if not daily_path.exists():
            return df
        daily = pd.read_csv(daily_path)
        if daily.empty:
            return df
        current = daily.iloc[-1].to_dict()
        if cutoff_time and "last_time" in current:
            try:
                last_time = pd.to_datetime(current["last_time"]).time()
                hh, mm = cutoff_time.split(":", 1)
                if last_time > dtime(int(hh), int(mm)):
                    return df
            except Exception:
                pass
        live = dict(current)

    out = df.sort_values("date").copy()
    trade_ts = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    hist_before = out[pd.to_datetime(out["date"], errors="coerce").dt.normalize() < trade_ts].sort_values("date")
    hist_prev_close = np.nan
    if not hist_before.empty and "close" in hist_before.columns:
        hist_close = pd.to_numeric(hist_before["close"], errors="coerce").dropna()
        if len(hist_close):
            hist_prev_close = float(hist_close.iloc[-1])

    # Ensure diagnostics/liquidity/raw live columns always exist.
    live_columns = {
        "open", "high", "low", "close", "raw_volume", "volume", "amount", "daily_vwap",
        "daily_vwap_volume", "daily_vwap_pv", "n_intraday_bars", "snapshot_count",
        "prev_close", "live_prev_close", "hist_prev_close", "prev_close_for_feature",
        "pct_chg", "pct_chg_raw", "pct_chg_norm", "pct_chg_source", "pct_chg_unit", "overnight_ret",
        "volume_unit_ratio_median", "volume_unit_ratio_p05", "volume_unit_ratio_p95",
        "volume_unit_lots_rows", "volume_unit_shares_rows", "volume_unit_unknown_rows",
        "volume_unit_multiplier",
        "volume_unit_inferred", "volume_unit_last", "volume_unit_mixed",
        "_snapshot_time", "_source_used", "_core_complete", "_missing_core_fields",
    }
    for c in live_columns:
        if c not in out.columns:
            out[c] = np.nan

    row = out.iloc[-1].copy()
    row["date"] = trade_ts
    live_prev_close = pd.to_numeric(pd.Series([live.get("live_prev_close", live.get("prev_close"))]), errors="coerce").iloc[0]
    prev_close_for_feature = live_prev_close if np.isfinite(live_prev_close) and live_prev_close > 0 else hist_prev_close
    row["live_prev_close"] = live_prev_close
    row["hist_prev_close"] = hist_prev_close
    row["prev_close_for_feature"] = prev_close_for_feature
    mapping = [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("raw_volume", "raw_volume"),
        ("volume", "volume"),
        ("amount", "amount"),
        ("daily_vwap", "daily_vwap"),
        ("volume", "daily_vwap_volume"),
        ("amount", "daily_vwap_pv"),
        ("snapshots", "snapshot_count"),
        ("prev_close", "prev_close"),
        ("live_prev_close", "live_prev_close"),
        ("prev_close_for_feature", "prev_close_for_feature"),
        ("pct_chg", "pct_chg"),
        ("pct_chg_raw", "pct_chg_raw"),
        ("pct_chg_norm", "pct_chg_norm"),
        ("overnight_ret", "overnight_ret"),
        ("volume_unit_ratio_median", "volume_unit_ratio_median"),
        ("volume_unit_ratio_p05", "volume_unit_ratio_p05"),
        ("volume_unit_ratio_p95", "volume_unit_ratio_p95"),
        ("volume_unit_lots_rows", "volume_unit_lots_rows"),
        ("volume_unit_shares_rows", "volume_unit_shares_rows"),
        ("volume_unit_unknown_rows", "volume_unit_unknown_rows"),
        ("volume_unit_multiplier", "volume_unit_multiplier"),
    ]
    for src, dst in mapping:
        if src in live:
            row[dst] = pd.to_numeric(live[src], errors="coerce")
    for src in ["pct_chg_source", "pct_chg_unit"]:
        if src in live:
            row[src] = live[src]
    row["hist_prev_close"] = hist_prev_close
    if np.isfinite(prev_close_for_feature) and prev_close_for_feature > 0:
        row["prev_close_for_feature"] = prev_close_for_feature
        row["prev_close"] = prev_close_for_feature

    # If prev_close exists but pct/overnight were not supplied, derive them.
    try:
        if pd.notna(row.get("prev_close_for_feature")) and float(row["prev_close_for_feature"]) > 0:
            if pd.notna(row.get("close")):
                row["pct_chg"] = float(row["close"]) / float(row["prev_close_for_feature"]) - 1.0
            if pd.notna(row.get("open")):
                row["overnight_ret"] = float(row["open"]) / float(row["prev_close_for_feature"]) - 1.0
    except Exception:
        pass

    row["_snapshot_time"] = live.get("snapshot_time", live.get("last_time", ""))
    row["_source_used"] = live.get("source_used", live.get("source", ""))
    row["_core_complete"] = bool(live.get("core_complete", False))
    row["_missing_core_fields"] = live.get("missing_core_fields", "")
    row["volume_unit_inferred"] = live.get("volume_unit_inferred", "")
    row["volume_unit_last"] = live.get("volume_unit_last", "")
    row["volume_unit_mixed"] = bool(live.get("volume_unit_mixed", False))

    out = out[pd.to_datetime(out["date"], errors="coerce").dt.normalize() != trade_ts]
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)

def overlay_current_day_from_intraday(df: pd.DataFrame, trade_date: str, intraday_path: Optional[Path], cutoff_time: Optional[str] = None) -> pd.DataFrame:
    if intraday_path is None or not intraday_path.exists():
        return df
    bars = pd.read_csv(intraday_path, parse_dates=["datetime"])
    if bars.empty:
        return df
    target_date = pd.to_datetime(yyyymmdd_to_iso(trade_date)).normalize()
    bars = bars[pd.to_datetime(bars["datetime"]).dt.normalize() == target_date].sort_values("datetime")
    cutoff_dt = parse_cutoff_dt(trade_date, cutoff_time)
    if cutoff_dt is not None:
        bars = bars[bars["datetime"] <= cutoff_dt].copy()
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



def context_day_dir(args: argparse.Namespace, trade_date: str) -> Path:
    return Path(args.context_dir) / trade_date


def load_realtime_context_row(args: argparse.Namespace, artifact: ModelArtifact, trade_date: str, req: RuntimeRequirement) -> tuple[dict, dict]:
    """Return (feature_values, metadata) from context_features_asof.csv.

    Missing realtime context is a first-class diagnostic.  We do not silently
    treat required sector/external columns as ordinary missing model features.
    """
    meta = {
        "context_status": "not_required" if not req.requires_realtime_context else "missing_file",
        "context_mode": "not_required" if not req.requires_realtime_context else "estimated_asof_cutoff",
        "context_snapshot_time": "",
        "missing_context_features": ",".join(req.required_context_features) if req.requires_realtime_context else "",
        "required_context_features": ",".join(req.required_context_features),
        "context_has_daily_fallback": "",
        "context_stale_features": "",
        "context_source_modes": "",
    }
    if not req.requires_realtime_context:
        return {}, meta
    path = context_day_dir(args, trade_date) / "context_features_asof.csv"
    if not path.exists():
        return {}, meta
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        meta["context_status"] = "read_error"
        meta["context_error"] = f"{type(exc).__name__}: {exc}"
        return {}, meta
    if df.empty:
        meta["context_status"] = "missing"
        return {}, meta
    part = df[(df.get("stock_code", "").astype(str) == artifact.stock_code) & (df.get("artifact_name", "").astype(str) == artifact.artifact_name)]
    if part.empty:
        meta["context_status"] = "missing_artifact_context_row"
        return {}, meta
    row = part.iloc[-1].to_dict()
    for k in [
        "context_status", "context_mode", "context_snapshot_time", "missing_context_features",
        "required_context_features", "context_has_daily_fallback", "context_stale_features",
        "context_source_modes",
    ]:
        if k in row:
            meta[k] = str(row.get(k) or "")
    values = {}
    for col in req.required_context_features:
        if col in row:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                values[col] = float(val)
    # Keep also any columns that match trained feature names.  This lets the
    # generic context builder fill derived columns not explicitly classified.
    return values, meta


def apply_realtime_context_to_df(df: pd.DataFrame, artifact: ModelArtifact, trade_date: str, req: RuntimeRequirement, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    values, meta = load_realtime_context_row(args, artifact, trade_date, req)
    if not values:
        return df, meta
    out = df.copy()
    target_date = pd.to_datetime(yyyymmdd_to_iso(trade_date))
    if "date" not in out.columns:
        return out, meta
    mask = pd.to_datetime(out["date"]) == target_date
    if not mask.any():
        return out, meta
    idx = out.index[mask][-1]
    for col, val in values.items():
        if col not in out.columns:
            out[col] = np.nan
        out.at[idx, col] = val
    return out, meta


def recompute_stock_vs_sector_features(day_df: pd.DataFrame, full_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Best-effort recomputation for cross features after realtime sector estimate.

    stock_vs_sector_retN depends on current stock return and current sector_retN.
    sector_vs_bench_retN depends on current sector_retN and benchmark return.  The
    benchmark leg is currently reused from the latest sample row; the sector leg
    is estimated as-of cutoff by collect_realtime_context.py.
    """
    if day_df.empty:
        return day_df
    out = day_df.copy()
    if "date" not in full_df.columns or "date" not in out.columns:
        return out
    hist = full_df[pd.to_datetime(full_df["date"]) < pd.to_datetime(out["date"].iloc[-1])].sort_values("date")
    if hist.empty:
        return out

    cur_close = pd.to_numeric(out["close"], errors="coerce").iloc[-1] if "close" in out.columns else np.nan
    hclose = pd.to_numeric(hist["close"], errors="coerce").dropna() if "close" in hist.columns else pd.Series(dtype=float)
    for n in [1, 5, 20, 60]:
        col = f"stock_vs_sector_ret{n}"
        sec_col = f"sector_ret{n}"
        if col in cols and sec_col in out.columns and np.isfinite(cur_close) and len(hclose) >= n and hclose.iloc[-n] != 0:
            sec = pd.to_numeric(out[sec_col], errors="coerce").iloc[-1]
            if np.isfinite(sec):
                out[col] = cur_close / hclose.iloc[-n] - 1.0 - sec

        svb_col = f"sector_vs_bench_ret{n}"
        bench_col = f"bench_ret{n}"
        if svb_col in cols and sec_col in out.columns:
            sec = pd.to_numeric(out[sec_col], errors="coerce").iloc[-1]
            # Prefer same-row benchmark if present; otherwise latest completed sample.
            bench = np.nan
            if bench_col in out.columns:
                bench = pd.to_numeric(out[bench_col], errors="coerce").iloc[-1]
            if (not np.isfinite(bench)) and bench_col in hist.columns:
                hb = pd.to_numeric(hist[bench_col], errors="coerce").dropna()
                if not hb.empty:
                    bench = hb.iloc[-1]
            if np.isfinite(sec) and np.isfinite(bench):
                out[svb_col] = sec - bench

        # Generic external cross features, e.g. stock_vs_gold_ret20,
        # stock_vs_copper_ret5.  The realtime context builder produces
        # gold_ret20/copper_ret5 as-of cutoff; this computes the stock leg
        # from current close and subtracts the context return.
        for col in [c for c in cols if c.startswith("stock_vs_") and c.endswith(f"ret{n}") and c != f"stock_vs_sector_ret{n}"]:
            body = col[len("stock_vs_"):]
            ctx, _, suffix = body.partition("_ret")
            ctx_ret_col = f"{ctx}_ret{suffix}"
            if ctx and suffix == str(n) and ctx_ret_col in out.columns and np.isfinite(cur_close) and len(hclose) >= n and hclose.iloc[-n] != 0:
                ctx_ret = pd.to_numeric(out[ctx_ret_col], errors="coerce").iloc[-1]
                if np.isfinite(ctx_ret):
                    out[col] = cur_close / hclose.iloc[-n] - 1.0 - ctx_ret

        # New stock-external relative features from build_stock_external_features.py,
        # e.g. ane_stock_vs_stock_basket_ret5 and fert_stock_vs_future_basket_ret20.
        for col in [c for c in cols if c.endswith(f"ret{n}")]:
            m = re.fullmatch(r"([A-Za-z0-9]+)_stock_vs_(stock_basket|etf_basket|future_basket|board_basket|us_basket)_ret(\d+)", col)
            if not m or m.group(3) != str(n):
                continue
            pfx, family = m.group(1), m.group(2)
            ctx_ret_col = f"{pfx}_{family}_close_ret{n}"
            if ctx_ret_col in out.columns and np.isfinite(cur_close) and len(hclose) >= n and hclose.iloc[-n] != 0:
                ctx_ret = pd.to_numeric(out[ctx_ret_col], errors="coerce").iloc[-1]
                if np.isfinite(ctx_ret):
                    out[col] = cur_close / hclose.iloc[-n] - 1.0 - ctx_ret
    return out

def score_artifact(artifact: ModelArtifact, trade_date: str, out_dir: Path, cache_dir: Path, args: argparse.Namespace) -> dict:
    model = joblib.load(artifact.artifact_dir / "model.joblib")
    cols = [
        line.strip()
        for line in (artifact.artifact_dir / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    med = pd.read_csv(artifact.artifact_dir / "feature_median.csv", index_col=0)["median"]
    req = infer_runtime_requirement(artifact)

    samples_path = resolve_repo_path(artifact.metadata.get("samples"), artifact.stock_code)
    intraday_path = resolve_intraday_path(artifact, trade_date, cache_dir)
    if samples_path is None:
        raise FileNotFoundError(f"samples not found for {artifact.stock_code}: {artifact.metadata.get('samples')}")

    df = pd.read_csv(samples_path, parse_dates=["date"])
    df = overlay_current_day_from_cache(df, artifact.stock_code, trade_date, cache_dir, cutoff_time=args.cutoff_time)
    df = add_scoring_features(
        df,
        intraday_path,
        cache_dir,
        artifact.stock_code,
        cutoff_time=args.cutoff_time,
        scoring_trade_date=trade_date,
        benchmark_symbols=args.benchmark_symbols,
    )
    df, context_meta = apply_realtime_context_to_df(df, artifact, trade_date, req, args)
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

    day_df = recompute_stock_vs_sector_features(day_df, df, cols)

    missing = [c for c in cols if c not in day_df.columns]
    if missing:
        if args.strict_features:
            raise ValueError(f"missing {len(missing)} features, first={missing[:10]}")
        for col in missing:
            day_df[col] = np.nan

    raw_x = day_df[cols].apply(pd.to_numeric, errors="coerce")
    filled_feature_count = int(raw_x.isna().sum(axis=1).iloc[-1]) if not raw_x.empty else len(cols)
    intraday_cols_used = [c for c in cols if c in INTRADAY_BAR_FEATURES]
    intraday_missing_count = int(raw_x[intraday_cols_used].isna().sum(axis=1).iloc[-1]) if intraday_cols_used and not raw_x.empty else 0
    x = raw_x.fillna(med)
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

    close_value = float(pd.to_numeric(day_df["close"], errors="coerce").iloc[-1]) if "close" in day_df.columns else np.nan
    daily_vwap_value = float(pd.to_numeric(day_df["daily_vwap"], errors="coerce").iloc[-1]) if "daily_vwap" in day_df.columns else np.nan
    amount_value = float(pd.to_numeric(day_df["amount"], errors="coerce").iloc[-1]) if "amount" in day_df.columns else np.nan
    volume_value = float(pd.to_numeric(day_df["volume"], errors="coerce").iloc[-1]) if "volume" in day_df.columns else np.nan
    n_intraday_bars = int(pd.to_numeric(day_df["n_intraday_bars"], errors="coerce").fillna(0).iloc[-1]) if "n_intraday_bars" in day_df.columns else 0
    pct_chg_value = float(pd.to_numeric(day_df["pct_chg"], errors="coerce").iloc[-1]) if "pct_chg" in day_df.columns else np.nan
    live_prev_close = float(pd.to_numeric(day_df["live_prev_close"], errors="coerce").iloc[-1]) if "live_prev_close" in day_df.columns else np.nan
    hist_prev_close = float(pd.to_numeric(day_df["hist_prev_close"], errors="coerce").iloc[-1]) if "hist_prev_close" in day_df.columns else np.nan
    prev_close_for_feature = float(pd.to_numeric(day_df["prev_close_for_feature"], errors="coerce").iloc[-1]) if "prev_close_for_feature" in day_df.columns else np.nan
    pct_chg_raw = float(pd.to_numeric(day_df["pct_chg_raw"], errors="coerce").iloc[-1]) if "pct_chg_raw" in day_df.columns else np.nan
    pct_chg_norm = float(pd.to_numeric(day_df["pct_chg_norm"], errors="coerce").iloc[-1]) if "pct_chg_norm" in day_df.columns else pct_chg_value
    pct_chg_source = str(day_df["pct_chg_source"].iloc[-1]) if "pct_chg_source" in day_df.columns else ""
    pct_chg_unit = str(day_df["pct_chg_unit"].iloc[-1]) if "pct_chg_unit" in day_df.columns else ""
    bench_close_asof = float(pd.to_numeric(day_df["bench_close_asof"], errors="coerce").iloc[-1]) if "bench_close_asof" in day_df.columns else np.nan
    bench_ret1_asof = float(pd.to_numeric(day_df["bench_ret1_asof"], errors="coerce").iloc[-1]) if "bench_ret1_asof" in day_df.columns else np.nan
    bench_ret5_asof = float(pd.to_numeric(day_df["bench_ret5_asof"], errors="coerce").iloc[-1]) if "bench_ret5_asof" in day_df.columns else np.nan
    bench_ret20_asof = float(pd.to_numeric(day_df["bench_ret20_asof"], errors="coerce").iloc[-1]) if "bench_ret20_asof" in day_df.columns else np.nan
    regime_market_ok_asof = float(pd.to_numeric(day_df["regime_market_ok_asof"], errors="coerce").iloc[-1]) if "regime_market_ok_asof" in day_df.columns else np.nan
    regime_market_strict_asof = float(pd.to_numeric(day_df["regime_market_strict_asof"], errors="coerce").iloc[-1]) if "regime_market_strict_asof" in day_df.columns else np.nan
    sample_prev_close = np.nan
    try:
        hist_before = df[pd.to_datetime(df["date"], errors="coerce") < target_date].sort_values("date")
        if not hist_before.empty and "close" in hist_before.columns:
            sample_prev_close = float(pd.to_numeric(hist_before["close"], errors="coerce").dropna().iloc[-1])
    except Exception:
        sample_prev_close = np.nan
    prev_close_rel_diff = (
        abs(live_prev_close / sample_prev_close - 1.0)
        if np.isfinite(live_prev_close) and np.isfinite(sample_prev_close) and sample_prev_close > 0
        else np.nan
    )

    feature_status = "live_overlay_partial" if date_status == "exact_trade_date" else "fallback_latest_sample"
    snapshot_time = str(day_df["_snapshot_time"].iloc[-1]) if "_snapshot_time" in day_df.columns else ""
    snapshot_age = snapshot_age_seconds(snapshot_time, trade_date, args.cutoff_time)
    source_used = str(day_df["_source_used"].iloc[-1]) if "_source_used" in day_df.columns else ""
    core_complete = bool(day_df["_core_complete"].iloc[-1]) if "_core_complete" in day_df.columns else False
    missing_core_fields = str(day_df["_missing_core_fields"].iloc[-1]) if "_missing_core_fields" in day_df.columns else ""
    volume_unit_inferred = str(day_df["volume_unit_inferred"].iloc[-1]) if "volume_unit_inferred" in day_df.columns else ""
    volume_unit_last = str(day_df["volume_unit_last"].iloc[-1]) if "volume_unit_last" in day_df.columns else ""
    volume_unit_mixed = bool(day_df["volume_unit_mixed"].iloc[-1]) if "volume_unit_mixed" in day_df.columns else False
    volume_unit_unknown_rows = int(pd.to_numeric(day_df["volume_unit_unknown_rows"], errors="coerce").fillna(0).iloc[-1]) if "volume_unit_unknown_rows" in day_df.columns else 0
    volume_unit_ratio_median = float(pd.to_numeric(day_df["volume_unit_ratio_median"], errors="coerce").iloc[-1]) if "volume_unit_ratio_median" in day_df.columns else np.nan
    unsupported = list(req.unsupported_realtime_features)
    if req.requires_bid_ask and not args.enable_five_level:
        unsupported.append("bid_ask")

    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": yyyymmdd_to_iso(trade_date),
        "sample_date": used_date,
        "used_sample_date": used_date,
        "used_live_bar_date": yyyymmdd_to_iso(trade_date) if date_status == "exact_trade_date" and intraday_path else "",
        "date_status": date_status,
        "feature_status": feature_status,
        "snapshot_time": snapshot_time,
        "snapshot_age_seconds": snapshot_age,
        "max_snapshot_age_seconds": int(args.max_snapshot_age_seconds) if args.max_snapshot_age_seconds is not None else "",
        "cutoff_time": args.cutoff_time or "",
        "source_used": source_used,
        "core_complete": core_complete,
        "missing_core_fields": missing_core_fields,
        "requires_price_snapshot": req.requires_price_snapshot,
        "requires_intraday_bars": req.requires_intraday_bars,
        "requires_bid_ask": req.requires_bid_ask,
        "uses_historical_context": req.uses_historical_context,
        "requires_realtime_context": req.requires_realtime_context,
        "context_status": context_meta.get("context_status", ""),
        "context_mode": context_meta.get("context_mode", ""),
        "context_snapshot_time": context_meta.get("context_snapshot_time", ""),
        "context_has_daily_fallback": context_meta.get("context_has_daily_fallback", ""),
        "context_stale_features": context_meta.get("context_stale_features", ""),
        "context_source_modes": context_meta.get("context_source_modes", ""),
        "required_context_features": context_meta.get("required_context_features", ""),
        "missing_context_features": context_meta.get("missing_context_features", ""),
        "unsupported_realtime_features": ",".join(unsupported),
        "stock_code": artifact.stock_code,
        "artifact_name": artifact.artifact_name,
        "entry_policy": entry_policy,
        "entry_vwap_premium_bps": entry_vwap_premium_bps,
        "entry_signal": entry_signal,
        "close": close_value,
        "daily_vwap": daily_vwap_value,
        "amount": amount_value,
        "volume": volume_value,
        "volume_unit_inferred": volume_unit_inferred,
        "volume_unit_last": volume_unit_last,
        "volume_unit_mixed": volume_unit_mixed,
        "volume_unit_unknown_rows": volume_unit_unknown_rows,
        "volume_unit_ratio_median": volume_unit_ratio_median,
        "pct_chg": pct_chg_value,
        "pct_chg_raw": pct_chg_raw,
        "pct_chg_norm": pct_chg_norm,
        "pct_chg_source": pct_chg_source,
        "pct_chg_unit": pct_chg_unit,
        "live_prev_close": live_prev_close,
        "hist_prev_close": hist_prev_close,
        "prev_close_for_feature": prev_close_for_feature,
        "sample_prev_close": sample_prev_close,
        "prev_close_rel_diff": prev_close_rel_diff,
        "bench_close_asof": bench_close_asof,
        "bench_ret1_asof": bench_ret1_asof,
        "bench_ret5_asof": bench_ret5_asof,
        "bench_ret20_asof": bench_ret20_asof,
        "regime_market_ok_asof": regime_market_ok_asof,
        "regime_market_strict_asof": regime_market_strict_asof,
        "n_intraday_bars": n_intraday_bars,
        "min_intraday_bars": int(args.min_intraday_bars) if args.min_intraday_bars is not None else "",
        "max_intraday_bars": int(args.max_intraday_bars) if args.max_intraday_bars is not None else "",
        "intraday_feature_count": int(len(intraday_cols_used)),
        "intraday_missing_count": int(intraday_missing_count),
        "hit_score": float(score[-1]),
        "threshold": threshold,
        "score_margin": float(score[-1]) - threshold,
        "signal_raw_score_pass": signal_raw_score_pass,
        "signal": signal,
        "missing_column_count": int(len(missing)),
        "missing_feature_count": int(filled_feature_count),
        "strict_features": bool(args.strict_features),
        "samples": str(samples_path),
        "intraday_bars": str(intraday_path) if intraday_path else "",
        "reject_reason": "",
    }

    # Safety gates for live trading output.  They do not affect all_scores.csv,
    # but they prevent stale rows / low-quality feature rows from entering
    # buy_signals.csv.
    if args.require_exact_date and date_status != "exact_trade_date":
        row["signal"] = False
        append_reject_reason(row, "not_exact_trade_date")
    if args.require_core_complete and req.requires_price_snapshot and not core_complete:
        row["signal"] = False
        append_reject_reason(row, "missing_core_fields")
    if (
        req.requires_price_snapshot
        and args.max_snapshot_age_seconds is not None
        and np.isfinite(snapshot_age)
        and snapshot_age > args.max_snapshot_age_seconds
    ):
        row["signal"] = False
        append_reject_reason(row, f"stale_snapshot_gt_{args.max_snapshot_age_seconds}s")
    if req.requires_price_snapshot and volume_unit_mixed:
        row["signal"] = False
        append_reject_reason(row, "volume_unit_mixed")
    if req.requires_price_snapshot and volume_unit_last == "unknown":
        row["signal"] = False
        append_reject_reason(row, "volume_unit_unknown")
    if unsupported:
        row["signal"] = False
        append_reject_reason(row, "unsupported_realtime_features")
    if args.require_realtime_context and req.requires_realtime_context and str(context_meta.get("context_status", "")) not in {"ok", "not_required"}:
        row["signal"] = False
        append_reject_reason(row, "missing_required_realtime_context")
    if (
        req.requires_intraday_bars
        and args.min_intraday_bars is not None
        and (not np.isfinite(n_intraday_bars) or n_intraday_bars < args.min_intraday_bars)
    ):
        row["signal"] = False
        append_reject_reason(row, f"intraday_bars_lt_{args.min_intraday_bars}")
    if (
        req.requires_intraday_bars
        and args.max_intraday_bars is not None
        and np.isfinite(n_intraday_bars)
        and n_intraday_bars > args.max_intraday_bars
    ):
        row["signal"] = False
        append_reject_reason(row, f"intraday_bars_gt_{args.max_intraday_bars}")
    if req.requires_intraday_bars and intraday_missing_count > 0:
        row["signal"] = False
        append_reject_reason(row, f"intraday_features_missing_{intraday_missing_count}")
    if (
        args.max_prev_close_rel_diff is not None
        and np.isfinite(prev_close_rel_diff)
        and prev_close_rel_diff > args.max_prev_close_rel_diff
    ):
        row["signal"] = False
        append_reject_reason(row, f"prev_close_mismatch_gt_{args.max_prev_close_rel_diff:g}")
    if args.max_missing_features is not None and int(filled_feature_count) > args.max_missing_features:
        row["signal"] = False
        append_reject_reason(row, f"filled_features_gt_{args.max_missing_features}")
    if args.min_amount_yuan and (not np.isfinite(amount_value) or amount_value < args.min_amount_yuan):
        row["signal"] = False
        append_reject_reason(row, f"amount_lt_{args.min_amount_yuan:g}")
    if args.max_abs_pct_chg is not None and np.isfinite(pct_chg_value) and abs(pct_chg_value) > args.max_abs_pct_chg:
        row["signal"] = False
        append_reject_reason(row, f"abs_pct_chg_gt_{args.max_abs_pct_chg:g}")
    if not entry_signal:
        append_reject_reason(row, "entry_signal_false")
    if not signal_raw_score_pass:
        append_reject_reason(row, "score_below_threshold")
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
        str(getattr(args, "_effective_watchlist", args.watchlist)),
        "--out-dir",
        str(args.cache_dir),
        "--allow-l1-only",
    ]
    if not args.enable_five_level:
        cmd.append("--disable-em-bid-ask")
    cmd.extend(["--spot-source-priority", args.spot_source_priority])
    cmd.extend(["--benchmark-symbols", args.benchmark_symbols or ""])
    if args.enable_source_short_circuit:
        cmd.append("--enable-source-short-circuit")
    if args.xq_only_missing:
        cmd.append("--xq-only-missing")
    cmd.extend(["--required-fields", args.required_fields])
    if args.xq_max_symbols_per_round is not None:
        cmd.extend(["--xq-max-symbols-per-round", str(args.xq_max_symbols_per_round)])
    if args.max_symbols:
        cmd.extend(["--max-symbols", str(args.max_symbols)])
    run_cmd(cmd, args.dry_run, timeout_seconds=args.final_collect_timeout_seconds)


def collect_trades(args: argparse.Namespace, trade_date: str) -> None:
    cmd = [
        str(PYTHON),
        COLLECT_AKSHARE_SCRIPT,
        "collect-trades",
        "--symbols-file",
        str(getattr(args, "_effective_watchlist", args.watchlist)),
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
    run_cmd(cmd, args.dry_run, timeout_seconds=args.trades_collect_timeout_seconds)


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
        if args.cutoff_time:
            cmd.extend(["--cutoff-time", args.cutoff_time])
        cmd.extend(["--benchmark-symbols", args.benchmark_symbols or ""])
        if getattr(args, "_effective_watchlist", None) and subcmd == "build-bars":
            cmd.extend(["--symbols-file", str(args._effective_watchlist)])
        run_cmd(cmd, args.dry_run, timeout_seconds=args.build_bars_timeout_seconds)


def collect_loop_until_signal(args: argparse.Namespace) -> None:
    cmd = [
        str(PYTHON),
        COLLECT_AKSHARE_SCRIPT,
        "collect-loop",
        "--symbols-file",
        str(getattr(args, "_effective_watchlist", args.watchlist)),
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
    cmd.extend(["--spot-source-priority", args.spot_source_priority])
    cmd.extend(["--benchmark-symbols", args.benchmark_symbols or ""])
    if args.enable_source_short_circuit:
        cmd.append("--enable-source-short-circuit")
    if args.xq_only_missing:
        cmd.append("--xq-only-missing")
    cmd.extend(["--required-fields", args.required_fields])
    if args.xq_max_symbols_per_round is not None:
        cmd.extend(["--xq-max-symbols-per-round", str(args.xq_max_symbols_per_round)])
    if args.max_symbols:
        cmd.extend(["--max-symbols", str(args.max_symbols)])
    run_cmd(cmd, args.dry_run)


def update_baostock_for_artifacts(args: argparse.Namespace, artifacts: list[ModelArtifact], trade_date: str) -> None:
    if not args.baostock_update:
        print("SKIP BaoStock update: disabled for intraday signal run", flush=True)
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


def write_run_summary(out_dir: Path, trade_date: str, summary: pd.DataFrame, buy: pd.DataFrame, artifacts: list[ModelArtifact], started_at: datetime) -> None:
    def count_col_value(df: pd.DataFrame, col: str) -> dict:
        if df.empty or col not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}

    data = {
        "trade_date": yyyymmdd_to_iso(trade_date),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_artifacts": len(artifacts),
        "n_all_scores": int(len(summary)),
        "n_buy_signals": int(len(buy)),
        "signal_counts": count_col_value(summary, "signal"),
        "date_status_counts": count_col_value(summary, "date_status"),
        "feature_status_counts": count_col_value(summary, "feature_status"),
        "source_usage": count_col_value(summary, "source_used"),
        "core_complete_counts": count_col_value(summary, "core_complete"),
        "context_status_counts": count_col_value(summary, "context_status"),
        "reject_reason_counts": count_col_value(summary, "reject_reason"),
        "error_count": int(summary.get("error", pd.Series(dtype=object)).notna().sum()) if not summary.empty else 0,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def score_models(args: argparse.Namespace, trade_date: str) -> pd.DataFrame:
    started_at = datetime.now()
    artifacts = getattr(args, "_runtime_artifacts", None)
    if artifacts is None:
        artifacts, _ = prepare_runtime_plan(args, trade_date)
    rows = []
    for artifact in artifacts:
        try:
            row = score_artifact(
                artifact,
                trade_date,
                Path(args.signal_out_dir) / trade_date,
                Path(args.cache_dir),
                args,
            )
            print(
                f"{row['stock_code']} {row['artifact_name']} "
                f"score={row['hit_score']:.6f} threshold={row['threshold']:.6f} "
                f"raw_pass={row['signal_raw_score_pass']} entry={row['entry_signal']} "
                f"signal={row['signal']} reject={row.get('reject_reason','')}",
                flush=True,
            )
            rows.append(row)
        except Exception as exc:
            row = {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "trade_date": yyyymmdd_to_iso(trade_date),
                "stock_code": artifact.stock_code,
                "artifact_name": artifact.artifact_name,
                "signal": False,
                "reject_reason": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"ERROR {artifact.stock_code} {artifact.artifact_name}: {row['error']}", flush=True)
            rows.append(row)
    out_dir = Path(args.signal_out_dir) / trade_date
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    if "error" not in summary.columns:
        summary["error"] = ""
    summary.to_csv(out_dir / "all_scores.csv", index=False, encoding="utf-8-sig")

    # buy_signals.csv is intentionally a clean tradable candidate list.
    # Full diagnostics stay in all_scores.csv.
    if summary.empty:
        buy = summary.copy()
    else:
        signal_bool = summary["signal"].apply(as_bool_series_value) if "signal" in summary.columns else pd.Series(False, index=summary.index)
        err_blank = summary["error"].isna() | (summary["error"].astype(str).str.strip() == "")
        buy = summary[signal_bool & err_blank].copy()
        if not buy.empty and "score_margin" in buy.columns:
            buy["score_margin"] = pd.to_numeric(buy["score_margin"], errors="coerce")
            sort_cols = [c for c in ["score_margin", "hit_score"] if c in buy.columns]
            buy = buy.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else buy
        if not buy.empty:
            buy.insert(0, "rank", range(1, len(buy) + 1))
    buy.to_csv(out_dir / "buy_signals.csv", index=False, encoding="utf-8-sig")

    # Keep a quick rejected/non-tradable diagnostic file for 14:55 review.
    rejected = summary.drop(index=buy.index, errors="ignore") if not summary.empty else summary.copy()
    rejected.to_csv(out_dir / "rejected_scores.csv", index=False, encoding="utf-8-sig")
    write_run_summary(out_dir, trade_date, summary, buy, artifacts, started_at)
    print(f"WROTE {out_dir / 'all_scores.csv'} rows={len(summary)}", flush=True)
    print(f"WROTE {out_dir / 'buy_signals.csv'} rows={len(buy)}", flush=True)
    return summary


def prepare_baostock_at_start(args: argparse.Namespace, trade_date: str) -> None:
    artifacts = getattr(args, "_runtime_artifacts", None)
    if artifacts is None:
        artifacts, _ = prepare_runtime_plan(args, trade_date)
    update_baostock_for_artifacts(args, artifacts, trade_date)


def list_models(args: argparse.Namespace, trade_date: str) -> None:
    artifacts = getattr(args, "_runtime_artifacts", None)
    if artifacts is None:
        artifacts, _ = prepare_runtime_plan(args, trade_date)
    rows = [
        {
            "stock_code": a.stock_code,
            "artifact_name": a.artifact_name,
            "created_at": a.created_at,
            "threshold": a.metadata.get("threshold"),
            "feature_group": a.metadata.get("feature_group"),
            "entry_policy": a.metadata.get("entry_policy"),
            "artifact_dir": str(a.artifact_dir),
        }
        for a in artifacts
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday collection and pre-close next-day buy signals")
    p.add_argument("cmd", choices=["list-models", "plan", "collect-and-score", "score-now"])
    p.add_argument("--watchlist", default="selected_watchlist.txt")
    p.add_argument("--models-dir", default=str(SAVED_MODELS_DIR))
    p.add_argument("--cache-dir", default=str(SAVED_DATA_DIR / "akshare_realtime_cache"))
    p.add_argument("--context-dir", default=str(SAVED_DATA_DIR / "realtime_context"), help="Directory containing realtime sector/external context_features_asof.csv")
    p.add_argument("--signal-out-dir", default=str(SAVED_DATA_DIR / "intraday_nextday_signals"))
    p.add_argument("--date", help="YYYYMMDD; defaults to today")
    p.add_argument("--signal-time", default="14:55")
    p.add_argument("--interval-seconds", type=int, default=30)
    p.add_argument("--cutoff-time", default="14:55", help="HH:MM. Score/build-bars use only snapshots at or before this time.")
    p.add_argument("--spot-source-priority", default="sina_batch,ths_etf,xq")
    p.add_argument("--benchmark-symbols", default=DEFAULT_BENCHMARK_SYMBOLS, help="Comma separated benchmark index symbols collected/scored as-of cutoff; empty disables")
    p.add_argument("--enable-source-short-circuit", action="store_true", default=True, help="Use later sources only for symbols still missing required fields")
    p.add_argument("--no-source-short-circuit", dest="enable_source_short_circuit", action="store_false")
    p.add_argument("--required-fields", default="close,open,high,low,volume,amount")
    p.add_argument("--xq-only-missing", action="store_true", default=True, help="Use Xueqiu only for symbols still missing required fields")
    p.add_argument("--no-xq-only-missing", dest="xq_only_missing", action="store_false")
    p.add_argument("--xq-max-symbols-per-round", type=int, default=10)
    p.add_argument("--with-trades", action="store_true")
    p.add_argument("--trades-interval-seconds", type=int, default=300)
    p.add_argument("--model-policy", choices=["preferred", "all"], default="preferred")
    p.add_argument("--max-symbols", type=int, help="Debug only: limit collection symbols")
    p.add_argument("--enable-final-collect", dest="final_collect", action="store_true", help="Explicitly run one more collect-once before scoring. Disabled by default for punctual 14:55 signals.")
    p.add_argument("--skip-final-collect", dest="final_collect", action="store_false", help="Compatibility alias; final collect is already skipped by default.")
    p.set_defaults(final_collect=False)
    p.add_argument("--enable-build-bars", dest="build_bars", action="store_true", help="Explicitly run build-bars/validate-day before scoring. Disabled by default for punctual 14:55 signals.")
    p.add_argument("--skip-build-bars", dest="build_bars", action="store_false", help="Compatibility alias; build-bars is already skipped by default.")
    p.set_defaults(build_bars=False)
    p.add_argument("--enable-baostock-update", dest="baostock_update", action="store_true", help="Update BaoStock raw cache before scoring. Disabled by default for 14:55 runs.")
    p.add_argument("--skip-baostock-update", dest="baostock_update", action="store_false", help="Compatibility alias; BaoStock update is already skipped by default.")
    p.set_defaults(baostock_update=False)
    p.add_argument("--baostock-start-date", help="YYYY-MM-DD; defaults to signal date")
    p.add_argument("--baostock-end-date", help="YYYY-MM-DD; defaults to signal date")
    p.add_argument("--enable-five-level", action="store_true", help="Also fetch Eastmoney five-level quotes")
    p.add_argument("--require-realtime-context", dest="require_realtime_context", action="store_true", default=True, help="Force signal=False when a model uses sector/external realtime context but context_features_asof.csv is missing/incomplete")
    p.add_argument("--allow-missing-realtime-context", dest="require_realtime_context", action="store_false")
    p.add_argument("--strict-features", action="store_true", help="Fail a model when live rows miss trained features")
    p.add_argument("--allow-fallback-date", dest="require_exact_date", action="store_false", help="Allow signals from fallback latest sample when current trade date is absent. Not recommended for live trading.")
    p.set_defaults(require_exact_date=True)
    p.add_argument("--require-core-complete", dest="require_core_complete", action="store_true", default=True, help="Force signal=False when required raw realtime fields are missing")
    p.add_argument("--allow-missing-core", dest="require_core_complete", action="store_false")
    p.add_argument("--max-missing-features", type=int, default=5, help="Force signal=False when live row misses more than this many trained features. Use -1 to disable.")
    p.add_argument("--min-intraday-bars", type=int, default=40, help="Force signal=False for models using intraday-bar features when reconstructed same-day bars are below this count. Use -1 to disable.")
    p.add_argument("--max-intraday-bars", type=int, default=60, help="Force signal=False for models using intraday-bar features when same-day bars exceed this count. Use -1 to disable.")
    p.add_argument("--max-snapshot-age-seconds", type=int, default=180, help="Force signal=False when latest snapshot is older than cutoff by this many seconds. Use -1 to disable.")
    p.add_argument("--max-prev-close-rel-diff", type=float, default=0.02, help="Force signal=False when live prev_close differs from latest sample close by more than this ratio. Use -1 to disable.")
    p.add_argument("--min-amount-yuan", type=float, default=0.0, help="Optional liquidity gate for buy_signals.csv; force signal=False below this traded amount.")
    p.add_argument("--max-abs-pct-chg", type=float, default=None, help="Optional risk gate; force signal=False if abs(pct_chg) exceeds this value when pct_chg exists.")
    p.add_argument("--deadline-time", help="HH:MM. After this time, skip optional slow steps such as trade-detail collection.")
    p.add_argument("--final-collect-timeout-seconds", type=int, default=180)
    p.add_argument("--trades-collect-timeout-seconds", type=int, default=180)
    p.add_argument("--build-bars-timeout-seconds", type=int, default=180)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_missing_features is not None and args.max_missing_features < 0:
        args.max_missing_features = None
    if args.min_intraday_bars is not None and args.min_intraday_bars < 0:
        args.min_intraday_bars = None
    if args.max_intraday_bars is not None and args.max_intraday_bars < 0:
        args.max_intraday_bars = None
    if args.max_snapshot_age_seconds is not None and args.max_snapshot_age_seconds < 0:
        args.max_snapshot_age_seconds = None
    if args.max_prev_close_rel_diff is not None and args.max_prev_close_rel_diff < 0:
        args.max_prev_close_rel_diff = None
    trade_date = args.date or today_yyyymmdd()
    if args.cmd == "list-models":
        list_models(args, trade_date)
        return
    artifacts, effective_watchlist = prepare_runtime_plan(args, trade_date)
    if args.cmd == "plan":
        print(f"WROTE runtime plan under {Path(args.signal_out_dir) / trade_date}")
        return
    if not artifacts:
        print("NO saved model artifacts for effective watchlist; write empty signal outputs and exit", flush=True)
        score_models(args, trade_date)
        return
    prepare_baostock_at_start(args, trade_date)
    if args.cmd == "collect-and-score":
        collect_loop_until_signal(args)
        if not args.dry_run:
            wait_until(parse_hhmm(args.signal_time))
    if args.final_collect:
        try:
            collect_once(args)
        except Exception as exc:
            print(f"WARN final collect failed; continue with cached snapshots: {type(exc).__name__}: {exc}", flush=True)
        if args.with_trades:
            if is_after_hhmm(args.deadline_time):
                print(f"SKIP trade-detail collection: deadline {args.deadline_time} reached", flush=True)
            else:
                try:
                    collect_trades(args, trade_date)
                except Exception as exc:
                    print(f"WARN trade-detail collection failed; continue: {type(exc).__name__}: {exc}", flush=True)
    if args.build_bars:
        try:
            build_bars_and_validate(args, trade_date)
        except Exception as exc:
            print(f"WARN build-bars/validate-day failed; continue with available sample/cache data: {type(exc).__name__}: {exc}", flush=True)
    score_models(args, trade_date)


if __name__ == "__main__":
    main()
