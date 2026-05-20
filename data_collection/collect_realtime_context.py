#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic realtime sector/external context collector for 14:55 scoring.

The collector is deliberately configuration-driven.  It does not hard-code
Zijin/Muyuan/Haida logic in Python.  It scans saved_models, reads every
artifact's feature_columns.txt, maps required feature prefixes to configured
sector/external contexts, collects only those contexts, and builds as-of-cutoff
feature estimates for scoring.

Outputs under saved_data/realtime_context/YYYYMMDD/:
  - realtime_context_plan.csv
  - context_snapshots.csv
  - context_features_asof.csv
  - context_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover - py3.10 compat
    import tomli as tomllib

SAVED_DATA_DIR = PROJECT_DIR / "saved_data"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "realtime_context_sources.toml"
DEFAULT_OUT_DIR = SAVED_DATA_DIR / "realtime_context"

SECTOR_FEATURE_PREFIXES = ("sector_", "stock_vs_sector_")
# External feature prefixes produced by feature_building/build_stock_external_features.py.
# Keep this list in sync with model_training/search feature groups and the realtime
# scoring adapter.  These prefixes are used only for deciding whether a feature
# needs realtime context; exact collection still comes from realtime_context_sources.toml.
STOCK_EXTERNAL_PREFIXES = (
    "ai_", "mwb_", "pur_", "fert_", "sp_", "ane_", "ocg_",
)
STOCK_EXTERNAL_RELATIVE_FAMILIES = (
    "stock_basket", "etf_basket", "future_basket", "board_basket", "us_basket",
)
NON_CRITICAL_REALTIME_CONTEXT_FEATURES = {"sector_range_z20"}

METADATA_COLS = {
    "stock_code", "artifact_name", "artifact_dir", "samples", "cutoff_time",
    "context_status", "context_mode", "missing_context_features",
    "required_context_features", "context_groups", "sector_symbols",
    "snapshot_time", "context_snapshot_time", "context_errors",
}


def is_lagged_daily_external_feature(feature: str) -> bool:
    """External families whose training merge uses completed daily data."""
    text = str(feature)
    return bool(
        re.match(r"^[A-Za-z0-9]+_(?:fut_|future_basket_|us_|us_basket_)", text)
        or re.fullmatch(r"[A-Za-z0-9]+_stock_vs_(future_basket|us_basket)_ret\d+", text)
    )


def is_lagged_daily_external_relative_feature(feature: str) -> bool:
    m = re.fullmatch(r"[A-Za-z0-9]+_stock_vs_(future_basket|us_basket)_ret\d+", str(feature))
    return bool(m)


def context_status_for_missing(required: list[str], missing: list[str], has_daily_fallback: bool = False) -> str:
    if not required:
        return "not_required"
    if not missing:
        return "ok_with_daily_fallback" if has_daily_fallback else "ok"
    if any(is_lagged_daily_external_feature(f) for f in missing):
        return "lagged_daily_missing"
    derived_suffixes = (
        "_ret1", "_ret3", "_ret5", "_ret20", "_ret60",
        "_ma20_gap", "_ma60_gap", "_z20", "_z60", "_vol20", "_shock20",
    )
    if all(f.endswith(derived_suffixes) or "_range_pct_" in f for f in missing):
        return "derived_missing"
    raw_suffixes = ("_open", "_high", "_low", "_close", "_volume", "_amount", "_range_pct")
    if all(f.endswith(raw_suffixes) for f in missing):
        return "raw_missing"
    return "source_unavailable"


def split_csv_cell(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [x.strip() for x in str(value).split(",") if x.strip() and x.strip().lower() != "nan"]


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


def read_watchlist(path: Optional[str | Path]) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        sym = normalize_symbol(line)
        if sym:
            out.append(sym)
    return list(dict.fromkeys(out))


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def yyyymmdd_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def parse_hhmm(value: Optional[str]) -> Optional[dtime]:
    if not value:
        return None
    hh, mm = value.split(":", 1)
    return dtime(int(hh), int(mm))


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"stocks": {}, "contexts": {}, "defaults": {}}
    with p.open("rb") as fh:
        return tomllib.load(fh)


def resolve_repo_path(raw: Optional[str], stock_code: str = "") -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    text = str(raw).replace("\\", "/")
    for marker in ["stock_realtime_v021_full/", "stock_realtime/"]:
        if marker in text:
            cand = PROJECT_DIR / text.split(marker, 1)[1]
            if cand.exists():
                return cand
    name = Path(text).name
    if name and stock_code:
        code6 = stock_code.split(".", 1)[0]
        roots = sorted((PROJECT_DIR / "saved_data").glob(f"{code6}_pipeline_out*"))
        roots.extend([
            PROJECT_DIR / "saved_data" / f"{code6}_base_out",
            PROJECT_DIR / "saved_data" / f"zijin_{code6}_base_out",
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


@dataclass
class ArtifactInfo:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    metadata: dict
    feature_columns: list[str]


def load_artifacts(models_dir: Path, watchlist: set[str], policy: str = "all") -> list[ArtifactInfo]:
    artifacts: list[ArtifactInfo] = []
    for meta_path in sorted(models_dir.rglob("metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stock = normalize_symbol(meta.get("stock_code", ""))
        if watchlist and stock not in watchlist:
            continue
        fpath = meta_path.parent / "feature_columns.txt"
        cols = []
        if fpath.exists():
            cols = [x.strip() for x in fpath.read_text(encoding="utf-8").splitlines() if x.strip()]
        artifacts.append(ArtifactInfo(stock, str(meta.get("artifact_name") or meta_path.parent.name), meta_path.parent, meta, cols))
    if policy == "all":
        return artifacts
    chosen = []
    for stock in sorted({a.stock_code for a in artifacts}):
        cand = [a for a in artifacts if a.stock_code == stock]
        cand.sort(key=lambda a: ("close_profit" in a.artifact_name, str(a.metadata.get("artifact_created_at") or ""), a.artifact_dir.stat().st_mtime), reverse=True)
        chosen.append(cand[0])
    return chosen


def matches_prefix(feature: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if feature == prefix or feature.startswith(prefix):
            return True
    return False


def is_sector_feature(feature: str) -> bool:
    return feature.startswith(SECTOR_FEATURE_PREFIXES)


def context_dependencies_for_features(features: list[str]) -> list[str]:
    """Return context columns that must be estimated before scoring.

    Saved models may contain derived features such as stock_vs_sector_ret5.
    Those should not be required directly from context_features_asof.csv; they
    can be recomputed in score_artifact once sector_ret5 and the current stock
    price are available.  This function maps model features to the minimal
    real-time context dependencies that need to be produced by this module.
    """
    deps: set[str] = set()
    for feat in features:
        if not feat:
            continue
        if feat in NON_CRITICAL_REALTIME_CONTEXT_FEATURES:
            # Needs historical 20-day sector range.  Realtime THS summary
            # alone cannot compute it correctly, so do not hard-block.
            continue
            continue
        if feat.startswith("stock_vs_sector_ret"):
            suffix = feat.replace("stock_vs_sector_ret", "")
            if suffix.isdigit():
                deps.add(f"sector_ret{suffix}")
            continue
        if feat.startswith("stock_vs_") and "_ret" in feat:
            body = feat[len("stock_vs_"):]
            ctx, _, suffix = body.partition("_ret")
            if ctx and suffix.isdigit():
                deps.add(f"{ctx}_ret{suffix}")
            continue
        if feat.startswith("sector_vs_bench_ret"):
            suffix = feat.replace("sector_vs_bench_ret", "")
            if suffix.isdigit():
                deps.add(f"sector_ret{suffix}")
            # score stage recomputes sector_vs_bench_ret* using sector_ret* and
            # bench_ret* from the current sample row.
            continue
        # New stock-external relative features are generated by
        # build_stock_external_features.py, e.g.
        #   ane_stock_vs_stock_basket_ret5
        #   fert_stock_vs_future_basket_ret20
        # The score stage recomputes the current stock leg; realtime context only
        # needs the external basket return.
        m = re.fullmatch(r"([A-Za-z0-9]+)_stock_vs_(stock_basket|etf_basket|future_basket|board_basket|us_basket)_ret(\d+)", feat)
        if m:
            pfx, family, suffix = m.group(1), m.group(2), m.group(3)
            if family in {"future_basket", "us_basket"}:
                continue
            deps.add(f"{pfx}_{family}_close_ret{suffix}")
            continue
        if is_lagged_daily_external_feature(feat):
            continue
        deps.add(feat)
    return sorted(deps)


def context_groups_for_features(features: list[str], stock_cfg: dict, contexts_cfg: dict) -> tuple[list[str], dict[str, list[str]], list[str]]:
    configured_groups = list(stock_cfg.get("context_groups", []))
    group_features: dict[str, list[str]] = {}
    missing_config_features: list[str] = []
    for feat in features:
        if is_sector_feature(feat):
            continue
        if is_lagged_daily_external_feature(feat) or is_lagged_daily_external_relative_feature(feat):
            continue
        matched_any = False
        for group in configured_groups:
            spec = contexts_cfg.get(group, {})
            if matches_prefix(feat, spec.get("feature_prefixes", [])):
                group_features.setdefault(group, []).append(feat)
                matched_any = True
        if (feat.startswith(("gold_", "copper_", "silver_", "zijin_hk_", "hog_", "feed_") + STOCK_EXTERNAL_PREFIXES) or "_hk_" in feat) and not matched_any:
            missing_config_features.append(feat)
    return sorted(group_features), group_features, missing_config_features


def create_plan(args: argparse.Namespace) -> pd.DataFrame:
    cfg = load_config(args.config)
    watchlist = set(read_watchlist(args.watchlist))
    artifacts = load_artifacts(Path(args.models_dir), watchlist, args.model_policy)
    rows = []
    plan_columns = [
        "stock_code", "artifact_name", "artifact_dir", "samples",
        "requires_sector_context", "sector_symbols", "context_groups",
        "required_context_features", "missing_context_config_features",
    ]
    for art in artifacts:
        stock_cfg = cfg.get("stocks", {}).get(art.stock_code, {})
        sector_model_features = [f for f in art.feature_columns if is_sector_feature(f)]
        sector_features = context_dependencies_for_features(sector_model_features)
        sector_symbols = list(stock_cfg.get("sector_symbols", [])) if sector_model_features else []
        groups, group_features, missing_cfg = context_groups_for_features(art.feature_columns, stock_cfg, cfg.get("contexts", {}))
        external_features = context_dependencies_for_features([f for fs in group_features.values() for f in fs])
        req_feats = sorted(set(sector_features + external_features))
        samples_path = resolve_repo_path(art.metadata.get("samples"), art.stock_code)
        rows.append({
            "stock_code": art.stock_code,
            "artifact_name": art.artifact_name,
            "artifact_dir": str(art.artifact_dir),
            "samples": str(samples_path or art.metadata.get("samples") or ""),
            "requires_sector_context": bool(sector_features),
            "sector_symbols": ",".join(sector_symbols),
            "context_groups": ",".join(groups),
            "required_context_features": ",".join(req_feats),
            "missing_context_config_features": ",".join(sorted(set(missing_cfg))),
        })
    return pd.DataFrame(rows, columns=plan_columns)


def out_day_dir(out_dir: str | Path, date: str) -> Path:
    p = Path(out_dir) / date
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_plan(args: argparse.Namespace) -> pd.DataFrame:
    df = create_plan(args)
    od = out_day_dir(args.out_dir, args.date)
    df.to_csv(od / "realtime_context_plan.csv", index=False, encoding="utf-8-sig")
    # machine-readable collection list
    collection_rows = []
    for _, row in df.iterrows():
        for sec in [x.strip() for x in str(row.get("sector_symbols") or "").split(",") if x.strip()]:
            collection_rows.append({"kind": "sector", "stock_code": row["stock_code"], "artifact_name": row["artifact_name"], "context_group": "sector", "symbol": sec})
        for group in [x.strip() for x in str(row.get("context_groups") or "").split(",") if x.strip()]:
            spec = load_config(args.config).get("contexts", {}).get(group, {})
            for sym in spec.get("symbols", []):
                collection_rows.append({"kind": "external", "stock_code": row["stock_code"], "artifact_name": row["artifact_name"], "context_group": group, "symbol": sym})
    cdf = pd.DataFrame(collection_rows).drop_duplicates() if collection_rows else pd.DataFrame(columns=["kind","stock_code","artifact_name","context_group","symbol"])
    cdf.to_csv(od / "context_collection_plan.csv", index=False, encoding="utf-8-sig")
    print(f"WROTE {od / 'realtime_context_plan.csv'} rows={len(df)}")
    print(f"WROTE {od / 'context_collection_plan.csv'} rows={len(cdf)}")
    return df


def load_akshare():
    import akshare as ak
    return ak


def to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        value = value.replace(",", "").replace("%", "").strip()
        if value in {"", "-", "--"}:
            return None
    try:
        return float(value)
    except Exception:
        return None


def pick(row: pd.Series | dict, names: Iterable[str]) -> Optional[float]:
    for name in names:
        if isinstance(row, pd.Series):
            if name in row.index:
                v = to_float(row[name])
                if v is not None:
                    return v
        else:
            if name in row:
                v = to_float(row[name])
                if v is not None:
                    return v
    return None


def first_data_row(raw: Any) -> pd.Series | dict | None:
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return None
        return raw.iloc[-1]
    if isinstance(raw, dict):
        return raw
    return None


def normalize_snapshot(kind: str, context_group: str, symbol: str, provider: str, raw_row: pd.Series | dict | None, canonical_prefix: str = "") -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    if raw_row is None:
        return {"datetime": now, "kind": kind, "context_group": context_group, "context_symbol": symbol, "provider": provider, "canonical_prefix": canonical_prefix, "status": "empty"}
    # THS sector summary reports stock-average price ("均价"), not the THS
    # industry/concept index close used during training.  Never map it into
    # *_close/open/high/low; only pct_chg/ret1 is scale-compatible.
    provider_name = str(provider)
    summary_provider = provider_name.startswith("ths_sector_summary") or provider_name.startswith("stock_board_") and provider_name.endswith("_summary_ths")
    price_names = ["最新", "最新价", "现价", "收盘", "收盘价", "close", "last", "price"]
    if not summary_provider:
        price_names.append("均价")
    price = pick(raw_row, price_names)
    open_v = pick(raw_row, ["今开", "开盘", "开盘价", "open"])
    high_v = pick(raw_row, ["最高", "最高价", "high"])
    low_v = pick(raw_row, ["最低", "最低价", "low"])
    raw_volume = pick(raw_row, ["总成交量", "成交量", "volume", "总成交量(万手)"])
    raw_amount = pick(raw_row, ["总成交额", "成交额", "amount", "总成交额(亿元)"])
    volume = raw_volume
    amount = raw_amount
    amount_source = ""
    volume_source = ""
    summary_amount_yuan = None
    summary_volume_shares = None
    if summary_provider:
        if raw_amount is not None:
            summary_amount_yuan = raw_amount * 1e8
            amount = summary_amount_yuan
            amount_source = "ths_summary_total_amount_yuan"
        if raw_volume is not None:
            summary_volume_shares = raw_volume * 10000.0 * 100.0
            volume = summary_volume_shares
            volume_source = "ths_summary_total_volume_shares"
    # Some realtime summary sources only provide current price/average price.
    # Fill OHLC with current level to avoid rejecting models for unavailable
    # intraday high/low from a provider that never exposes them.
    if price is not None:
        open_v = open_v if open_v is not None else price
        high_v = high_v if high_v is not None else price
        low_v = low_v if low_v is not None else price
    return {
        "datetime": now,
        "kind": kind,
        "context_group": context_group,
        "context_symbol": symbol,
        "provider": provider,
        "canonical_prefix": canonical_prefix,
        "status": "ok" if (price is not None or pick(raw_row, ["涨跌幅", "pct_chg", "changepercent"]) is not None) else "no_price",
        "level_scale_status": "ret_only" if summary_provider else "level_ok",
        "open": open_v,
        "high": high_v,
        "low": low_v,
        "close": price,
        "volume": volume,
        "amount": amount,
        "summary_amount_raw": raw_amount if summary_provider else None,
        "summary_volume_raw": raw_volume if summary_provider else None,
        "summary_amount_yuan": summary_amount_yuan,
        "summary_volume_shares": summary_volume_shares,
        "summary_amount_unit": "亿元" if summary_provider and raw_amount is not None else "",
        "summary_volume_unit": "万手" if summary_provider and raw_volume is not None else "",
        "amount_source": amount_source,
        "volume_source": volume_source,
        "pct_chg": pick(raw_row, ["涨跌幅", "pct_chg", "changepercent"]),
    }




def _symbol_digits(symbol: str) -> str:
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits


def context_canonical_prefix(spec: dict, group: str, symbol: str) -> str:
    """Resolve the canonical prefix for one configured context symbol.

    Supported TOML fields:
      canonical_prefix = "ane_fut_ni0"
      canonical_prefix_template = "ane_stk_{code}"
      canonical_prefixes = ["ane_board_b01", ...]  # aligned with symbols

    This allows one context group to collect many A-share/ETF/board symbols
    while still emitting individual feature prefixes that match the training
    builder, e.g. ane_stk_600893_close_ret1.
    """
    symbols = [str(x) for x in spec.get("symbols", [])]
    prefixes = list(spec.get("canonical_prefixes", []) or [])
    try:
        idx = symbols.index(str(symbol))
    except ValueError:
        idx = -1
    if idx >= 0 and idx < len(prefixes) and str(prefixes[idx]).strip():
        return str(prefixes[idx]).strip()

    template = str(spec.get("canonical_prefix_template") or "").strip()
    if template:
        code = _symbol_digits(symbol)
        norm = normalize_symbol(symbol) if code else str(symbol).strip().upper()
        token = re.sub(r"[^0-9A-Za-z]+", "_", str(symbol).strip().lower()).strip("_") or "asset"
        return template.format(symbol=str(symbol), code=code, norm=norm, token=token)

    return str(spec.get("canonical_prefix") or group)


def attach_context_metadata(snapshot: dict, spec: dict, group: str, symbol: str) -> dict:
    snapshot["canonical_prefix"] = context_canonical_prefix(spec, group, symbol)
    if spec.get("aggregate_prefix"):
        snapshot["aggregate_prefix"] = str(spec.get("aggregate_prefix"))
    return snapshot

def _sector_error_snapshot(symbol: str, provider: str, error: str) -> dict:
    return {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "kind": "sector",
        "context_group": "sector",
        "context_symbol": symbol,
        "provider": provider,
        "canonical_prefix": "sector",
        "status": "error",
        "error": error,
    }


def _find_sector_name_col(raw: pd.DataFrame) -> Optional[str]:
    if raw is None or raw.empty:
        return None
    return next(
        (
            c
            for c in raw.columns
            if str(c) in {"板块", "名称", "name"} or "板块" in str(c)
        ),
        None,
    )


def _fetch_ths_sector_summary_worker(queue) -> None:
    """Worker used by multiprocessing.

    Import akshare inside the child process so a stuck request can be killed
    without poisoning the parent 14:55 pipeline process.
    """
    try:
        import akshare as ak  # type: ignore

        fn = getattr(ak, "stock_board_industry_summary_ths", None)
        if fn is None:
            queue.put({"ok": False, "error": "stock_board_industry_summary_ths missing"})
            return
        raw = fn()
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            queue.put({"ok": False, "error": "empty dataframe"})
            return
        name_col = _find_sector_name_col(raw)
        if name_col is None:
            queue.put({"ok": False, "error": "no sector name column"})
            return
        queue.put({"ok": True, "data": raw})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}:{exc}"})


def _fetch_ths_sector_summary_once_subprocess(timeout_seconds: float = 5.0) -> tuple[Optional[pd.DataFrame], str]:
    """Fetch THS sector summary once with a hard subprocess timeout."""
    import multiprocessing as mp

    queue = mp.Queue()
    proc = mp.Process(target=_fetch_ths_sector_summary_worker, args=(queue,))
    proc.start()
    proc.join(max(0.1, float(timeout_seconds)))

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, f"timeout>{timeout_seconds}s"

    if queue.empty():
        return None, "no child result"

    result = queue.get()
    if result.get("ok") and isinstance(result.get("data"), pd.DataFrame):
        return result["data"], "ok"
    return None, str(result.get("error") or "unknown error")


def _start_ths_sector_summary_proc(idx: int):
    """Start one THS sector summary subprocess request."""
    import multiprocessing as mp

    queue = mp.Queue()
    proc = mp.Process(target=_fetch_ths_sector_summary_worker, args=(queue,))
    proc.start()
    return {
        "idx": idx,
        "queue": queue,
        "proc": proc,
        "started_monotonic": time.monotonic(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "done": False,
    }


def _finish_ths_sector_proc(item: dict, timeout_seconds: float) -> tuple[bool, Optional[pd.DataFrame], str]:
    """Poll one subprocess.

    Returns:
      (finished, dataframe_or_none, message)

    If finished=True and dataframe is not None, the request succeeded.
    If finished=True and dataframe is None, the request failed or timed out.
    If finished=False, it is still running.
    """
    proc = item["proc"]
    queue = item["queue"]
    idx = item["idx"]

    if not queue.empty():
        try:
            result = queue.get_nowait()
        except Exception as exc:
            result = {"ok": False, "error": f"queue_get:{type(exc).__name__}:{exc}"}

        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=0.5)
        item["done"] = True

        if result.get("ok") and isinstance(result.get("data"), pd.DataFrame) and not result["data"].empty:
            return True, result["data"], f"worker{idx}:ok"
        return True, None, f"worker{idx}:{result.get('error') or 'empty'}"

    if not proc.is_alive():
        proc.join(timeout=0.5)
        item["done"] = True
        return True, None, f"worker{idx}:no_result"

    elapsed = time.monotonic() - float(item["started_monotonic"])
    if elapsed >= float(timeout_seconds):
        proc.terminate()
        proc.join(timeout=0.5)
        item["done"] = True
        return True, None, f"worker{idx}:timeout>{timeout_seconds}s"

    return False, None, f"worker{idx}:running"


def _terminate_ths_sector_processes(items: list[dict]) -> None:
    for item in items:
        proc = item.get("proc")
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)


def fetch_ths_sector_summary_hedged(
    timeout_seconds: float = 5.0,
    hedge_workers: int = 2,
    hedge_delay_seconds: float = 1.5,
) -> tuple[Optional[pd.DataFrame], dict]:
    """Low-concurrency hedged fetch for THS sector summary.

    Optimized behavior:
      1. Start the first subprocess request immediately.
      2. Wait up to `hedge_delay_seconds`.
      3. If the first request has already succeeded, return immediately and DO
         NOT launch the second request.
      4. If the first request is still running or has failed, start the second
         subprocess request.
      5. Return the first non-empty DataFrame and terminate the remaining
         subprocesses.

    This avoids the previous wasteful behavior where both hedged tasks were
    scheduled up front and the context manager could still wait for the delayed
    second task even when worker 0 had already succeeded.
    """
    hedge_workers = max(1, min(int(hedge_workers), 3))
    timeout_seconds = max(0.5, float(timeout_seconds))
    hedge_delay_seconds = max(0.0, float(hedge_delay_seconds))

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()
    next_launch_time = t0
    active: list[dict] = []
    errors: list[str] = []
    launched = 0

    global_deadline = t0 + timeout_seconds + hedge_delay_seconds * (hedge_workers - 1) + 1.0

    try:
        while time.monotonic() <= global_deadline:
            now_mono = time.monotonic()

            if launched < hedge_workers and (launched == 0 or now_mono >= next_launch_time):
                active.append(_start_ths_sector_summary_proc(launched))
                launched += 1
                next_launch_time = t0 + hedge_delay_seconds * launched

            still_active: list[dict] = []
            for item in active:
                if item.get("done"):
                    continue
                finished, df, msg = _finish_ths_sector_proc(item, timeout_seconds)
                if finished:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        _terminate_ths_sector_processes(active)
                        return df, {
                            "provider": "stock_board_industry_summary_ths",
                            "status": "ok",
                            "winner": item["idx"],
                            "started_at": started_at,
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                            "rows": int(len(df)),
                            "errors": errors,
                            "launched_workers": launched,
                        }
                    errors.append(msg)
                else:
                    still_active.append(item)
            active = still_active

            # If all launched workers have failed before the scheduled hedge
            # delay, launch the next one immediately instead of idling.
            if not active and launched < hedge_workers:
                next_launch_time = time.monotonic()

            if not active and launched >= hedge_workers:
                break

            time.sleep(0.05)

    finally:
        _terminate_ths_sector_processes(active)

    return None, {
        "provider": "stock_board_industry_summary_ths",
        "status": "error",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "rows": 0,
        "errors": errors or ["no successful THS sector summary response"],
        "launched_workers": launched,
    }


def sector_snapshot_from_summary(raw: Optional[pd.DataFrame], symbol: str, meta: Optional[dict] = None) -> dict:
    """Build one sector snapshot row from a previously fetched THS summary table."""
    meta = meta or {}
    provider = meta.get("provider", "stock_board_industry_summary_ths")
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return _sector_error_snapshot(symbol, provider, "sector summary unavailable:" + ";".join(map(str, meta.get("errors", []))))

    name_col = _find_sector_name_col(raw)
    if name_col is None:
        return _sector_error_snapshot(symbol, provider, "no sector name column")

    part = raw[raw[name_col].astype(str).eq(str(symbol))]
    if part.empty:
        # Exact match failed; contains fallback is still limited to the 90-row
        # THS summary table and never selects a whole market table.
        part = raw[raw[name_col].astype(str).str.contains(str(symbol), na=False, regex=False)]
    if part.empty:
        return _sector_error_snapshot(symbol, provider, f"sector symbol not found: {symbol}")

    row = part.iloc[0]
    snap = normalize_snapshot("sector", "sector", symbol, provider, row, canonical_prefix="sector")
    if snap.get("status") == "ok":
        snap["fetch_status"] = meta.get("status")
        snap["fetch_winner"] = meta.get("winner")
        return snap

    return _sector_error_snapshot(symbol, provider, f"normalize status={snap.get('status')}")




def external_board_snapshot_from_summary(raw: Optional[pd.DataFrame], symbol: str, group: str, spec: dict, meta: Optional[dict] = None) -> dict:
    """Build one external-board snapshot from the shared THS summary table."""
    meta = meta or {}
    provider = meta.get("provider", "stock_board_industry_summary_ths")
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "kind": "external",
            "context_group": group,
            "context_symbol": symbol,
            "provider": provider,
            "canonical_prefix": context_canonical_prefix(spec, group, symbol),
            "aggregate_prefix": str(spec.get("aggregate_prefix") or ""),
            "status": "error",
            "error": "sector summary unavailable:" + ";".join(map(str, meta.get("errors", []))),
        }

    name_col = _find_sector_name_col(raw)
    if name_col is None:
        return {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "kind": "external",
            "context_group": group,
            "context_symbol": symbol,
            "provider": provider,
            "canonical_prefix": context_canonical_prefix(spec, group, symbol),
            "aggregate_prefix": str(spec.get("aggregate_prefix") or ""),
            "status": "error",
            "error": "no sector name column",
        }

    part = raw[raw[name_col].astype(str).eq(str(symbol))]
    if part.empty:
        part = raw[raw[name_col].astype(str).str.contains(str(symbol), na=False, regex=False)]
    if part.empty:
        return {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "kind": "external",
            "context_group": group,
            "context_symbol": symbol,
            "provider": provider,
            "canonical_prefix": context_canonical_prefix(spec, group, symbol),
            "aggregate_prefix": str(spec.get("aggregate_prefix") or ""),
            "status": "error",
            "error": f"board symbol not found: {symbol}",
        }

    snap = normalize_snapshot("external", group, symbol, provider, part.iloc[0], canonical_prefix=context_canonical_prefix(spec, group, symbol))
    snap = attach_context_metadata(snap, spec, group, symbol)
    if snap.get("status") == "ok":
        snap["fetch_status"] = meta.get("status")
        snap["fetch_winner"] = meta.get("winner")
    return snap

def fetch_sector_snapshot(ak, symbol: str, timeout_seconds: float = 5.0, hedge_workers: int = 2, hedge_delay_seconds: float = 1.5) -> dict:
    """Backward-compatible wrapper.

    Prefer collect_once(), which fetches the 90-row THS summary once and then
    filters all required sectors from that cached table.
    """
    raw, meta = fetch_ths_sector_summary_hedged(
        timeout_seconds=timeout_seconds,
        hedge_workers=hedge_workers,
        hedge_delay_seconds=hedge_delay_seconds,
    )
    return sector_snapshot_from_summary(raw, symbol, meta)




def _sina_market_symbol(symbol: str) -> str:
    norm = normalize_symbol(symbol)
    code, market = norm.split(".", 1)
    return f"{market.lower()}{code}"


def _parse_sina_a_payload(symbol: str, raw: str) -> dict:
    parts = [x.strip() for x in str(raw).strip().strip('";').split(',')]
    out = {"代码": normalize_symbol(symbol).split(".", 1)[0], "symbol": normalize_symbol(symbol), "raw": raw, "n_fields": len(parts)}
    if not parts or all(x == "" for x in parts):
        return out
    # Common Sina A-share/ETF format:
    # name,open,prev_close,current,high,low,bid,ask,volume,amount,...,date,time,status
    out.update({
        "名称": parts[0] if len(parts) > 0 else None,
        "今开": parts[1] if len(parts) > 1 else None,
        "昨收": parts[2] if len(parts) > 2 else None,
        "最新价": parts[3] if len(parts) > 3 else None,
        "最高": parts[4] if len(parts) > 4 else None,
        "最低": parts[5] if len(parts) > 5 else None,
        "成交量": parts[8] if len(parts) > 8 else None,
        "成交额": parts[9] if len(parts) > 9 else None,
    })
    prev = to_float(out.get("昨收"))
    last = to_float(out.get("最新价"))
    if prev is not None and last is not None and prev != 0:
        out["涨跌幅"] = (last / prev - 1.0) * 100.0
    if len(parts) > 30:
        out["行情日期"] = parts[30]
    if len(parts) > 31:
        out["行情时间"] = parts[31]
    return out


def fetch_sina_a_realtime_batch(symbols: list[str], timeout: float = 4.0) -> pd.DataFrame:
    """Fetch a small targeted A-share/ETF quote batch from Sina hq.

    This deliberately avoids full-market Eastmoney/EM spot tables.  It is used
    for realtime external peer/ETF contexts such as ane_stocks/ane_etfs.
    """
    normed = []
    for s0 in symbols:
        try:
            ns = normalize_symbol(str(s0))
        except Exception:
            continue
        if ns and ns not in normed:
            normed.append(ns)
    if not normed:
        return pd.DataFrame()

    market_codes = [_sina_market_symbol(s0) for s0 in normed]
    url = "https://hq.sinajs.cn/list=" + ",".join(market_codes)
    headers = {"Referer": "https://finance.sina.com.cn/", "User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        body = resp.content.decode("gbk")
    except Exception:
        body = resp.text

    market_to_symbol = {_sina_market_symbol(s0): s0 for s0 in normed}
    rows = []
    for line in body.splitlines():
        m = re.search(r"hq_str_([a-z]{2}\d{6})=\"(.*)\";?", line.strip(), flags=re.I)
        if not m:
            continue
        market_code, raw = m.group(1).lower(), m.group(2)
        sym = market_to_symbol.get(market_code)
        if not sym:
            continue
        row = _parse_sina_a_payload(sym, raw)
        if row.get("名称") or row.get("最新价"):
            rows.append(row)
    return pd.DataFrame(rows)


def _filter_a_table(raw: Any, symbol: str) -> Any:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return raw
    norm = normalize_symbol(symbol)
    code = norm.split(".", 1)[0]
    mask = pd.Series(False, index=raw.index)
    for col in raw.columns:
        name = str(col)
        series = raw[col].astype(str)
        if name in {"symbol", "代码"} or "代码" in name:
            mask = mask | series.str.contains(code, na=False, regex=False) | series.str.contains(norm, na=False, regex=False)
    return raw[mask].copy() if mask.any() else raw.iloc[0:0].copy()

def _normalize_hk_code_variants(symbol: str) -> list[str]:
    raw = str(symbol).strip().upper().replace(".HK", "").replace("HK", "")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return [str(symbol)]
    code5 = digits.zfill(5)
    code4 = str(int(digits)) if digits.lstrip("0") else digits
    out = []
    for x in [code5, code4, f"HK{code4}", f"{code4}.HK"]:
        if x and x not in out:
            out.append(x)
    return out

def _is_number_like(x: str) -> bool:
    try:
        float(str(x).replace(",", "").replace("%", "").strip())
        return True
    except Exception:
        return False


def _parse_sina_hk_payload(code5: str, raw: str) -> dict:
    """Parse one Sina HK hq payload.

    Sina HK real-time fields have had a few variants.  The common variants are:
      name,open,prev_close,high,low,last,change,pct,bid,ask,volume,amount,date,time
    or:
      en_name,cn_name,open,prev_close,high,low,last,change,pct,bid,ask,volume,amount,date,time

    We keep the raw payload in the output.  If field ordering changes, tests can
    reveal it without silently pulling a full-market table.
    """
    parts = [p.strip() for p in str(raw).strip().strip('";').split(",")]
    out = {"代码": code5, "raw": raw, "n_fields": len(parts)}
    if not parts or all(p == "" for p in parts):
        return out

    # Detect whether the second field is a non-numeric Chinese/English name.
    if len(parts) >= 15 and not _is_number_like(parts[1]) and _is_number_like(parts[2]):
        name = parts[1] or parts[0]
        offset = 2
    elif len(parts) >= 14 and _is_number_like(parts[1]):
        name = parts[0]
        offset = 1
    elif len(parts) >= 15 and _is_number_like(parts[2]):
        name = parts[1] or parts[0]
        offset = 2
    else:
        name = parts[0]
        offset = 1

    vals = parts[offset:]
    out["名称"] = name
    # Best-effort mapping for common Sina HK format.
    field_map = [
        ("今开", 0),
        ("昨收", 1),
        ("最高", 2),
        ("最低", 3),
        ("最新价", 4),
        ("涨跌额", 5),
        ("涨跌幅", 6),
        ("买入", 7),
        ("卖出", 8),
        # Sina HK hq fields after ask are:
        #   amount, volume
        # Example:
        #   ...,41.30000,41.38000,134343826,3207517,...
        # means amount=134343826, volume=3207517.
        # Do NOT reverse them.
        ("成交额", 9),
        ("成交量", 10),
    ]
    for key, idx in field_map:
        if idx < len(vals):
            out[key] = vals[idx]
    if len(vals) >= 13:
        out["行情日期"] = vals[-2]
        out["行情时间"] = vals[-1]
    return out


def fetch_sina_hk_realtime_batch(codes: list[str], timeout: float = 3.0) -> pd.DataFrame:
    """Fetch a small batch of HK quotes from Sina hq by code.

    This is the critical replacement for stock_hk_spot()/stock_hk_spot_em():
    those functions pull the whole HK market and can timeout or disconnect.
    Here we request only the codes the model needs, e.g. 02714/01610/00288.
    """
    codes5 = []
    for c in codes:
        digits = "".join(ch for ch in str(c) if ch.isdigit())
        if digits:
            code5 = digits.zfill(5)
            if code5 not in codes5:
                codes5.append(code5)
    if not codes5:
        return pd.DataFrame()

    symbols = ",".join(f"hk{c}" for c in codes5)
    url = f"https://hq.sinajs.cn/list={symbols}"
    headers = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    # Sina HK is usually GBK/GB18030 encoded.
    try:
        text = r.content.decode("gbk")
    except Exception:
        text = r.text

    rows = []
    for line in text.splitlines():
        m = re.search(r"hq_str_hk(\d{5})=\"(.*)\";?", line)
        if not m:
            continue
        code5, raw = m.group(1), m.group(2)
        rows.append(_parse_sina_hk_payload(code5, raw))
    return pd.DataFrame(rows)


def _filter_hk_table(raw: Any, symbol: str, group: str = "") -> Any:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return raw
    variants = _normalize_hk_code_variants(symbol)
    code5 = variants[0]
    digits = str(int(code5)) if code5.isdigit() else code5

    mask = pd.Series(False, index=raw.index)
    for col in raw.columns:
        name = str(col)
        series = raw[col].astype(str)
        if "代码" in name or name.lower() in {"symbol", "code", "证券代码"}:
            s_norm = series.str.upper().str.replace(".HK", "", regex=False).str.replace("HK", "", regex=False)
            mask |= s_norm.str.zfill(5).eq(code5)
            mask |= s_norm.eq(digits)
        if "名称" in name or name.lower() in {"name", "证券简称"}:
            if symbol in {"02714", "2714"} or group in {"muyuan_hk"}:
                mask |= series.str.contains("牧原", na=False)
    hit = raw[mask].copy()
    return hit if not hit.empty else raw


def fetch_external_snapshot(ak, group: str, spec: dict, symbol: str) -> dict:
    provider = str(spec.get("provider") or "")
    canonical = context_canonical_prefix(spec, group, symbol)
    errors = []

    if provider == "ths_sector_summary":
        # collect_once handles THS board contexts in one shared summary fetch.
        return {"datetime": datetime.now().isoformat(timespec="seconds"), "kind": "external", "context_group": group, "context_symbol": symbol, "provider": provider, "canonical_prefix": canonical, "status": "deferred_to_collect_once"}

    if provider in {"sina_a_spot_or_daily", "sina_a_realtime"}:
        try:
            sym = normalize_symbol(symbol)
            raw = fetch_sina_a_realtime_batch([sym], timeout=float(spec.get("timeout", 4.0)))
            raw = _filter_a_table(raw, sym)
            row = first_data_row(raw)
            snap = normalize_snapshot("external", group, symbol, "sina_a_targeted", row, canonical_prefix=canonical)
            snap = attach_context_metadata(snap, spec, group, symbol)
            if snap.get("status") == "ok":
                return snap
            errors.append(f"sina_a_targeted:{snap.get('status')}")
        except Exception as exc:
            errors.append(f"sina_a_targeted:{type(exc).__name__}:{exc}")
        return {"datetime": datetime.now().isoformat(timespec="seconds"), "kind": "external", "context_group": group, "context_symbol": symbol, "provider": "none", "canonical_prefix": canonical, "aggregate_prefix": str(spec.get("aggregate_prefix") or ""), "status": "error", "error": ";".join(errors)}

    if provider in {"hk_spot_or_hist", "hk_ah_spot_or_daily", "sina_hq_batch_or_daily"}:
        # HK realtime must never pull the whole HK market in the 14:55 path.
        # Use small targeted Sina hq requests first.  stock_hk_spot() and
        # stock_hk_spot_em() are intentionally not used here.
        code5 = _normalize_hk_code_variants(symbol)[0]

        # 1. Targeted HK realtime quote, usually sub-second to a few seconds.
        try:
            raw = fetch_sina_hk_realtime_batch([code5], timeout=float(spec.get("timeout", 3.0)))
            raw = _filter_hk_table(raw, code5, group)
            row = first_data_row(raw)
            snap = normalize_snapshot("external", group, symbol, "sina_hq_batch", row, canonical_prefix=canonical)
            snap = attach_context_metadata(snap, spec, group, symbol)
            if snap.get("status") == "ok":
                return snap
            errors.append(f"sina_hq_batch:{snap.get('status')}")
        except Exception as exc:
            errors.append(f"sina_hq_batch:{type(exc).__name__}:{exc}")

        # 2. A+H realtime fallback for A+H names such as 02714.  This table is
        # much smaller than full HK spot; still keep it as fallback only.
        if group in {"muyuan_hk", "zijin_hk"} or code5 in {"02714", "02899"}:
            try:
                raw = ak.stock_zh_ah_spot()
                raw = _filter_hk_table(raw, code5, group)
                row = first_data_row(raw)
                snap = normalize_snapshot("external", group, symbol, "stock_zh_ah_spot", row, canonical_prefix=canonical)
                snap = attach_context_metadata(snap, spec, group, symbol)
                if snap.get("status") == "ok":
                    return snap
                errors.append(f"stock_zh_ah_spot:{snap.get('status')}")
            except Exception as exc:
                errors.append(f"stock_zh_ah_spot:{type(exc).__name__}:{exc}")

        # 3. Latest daily fallback.  This is not true 14:55 realtime, but it is
        # acceptable as an explicitly marked fallback row rather than blocking.
        for prov_name, fn_name, kwargs in [
            ("stock_hk_daily_latest", "stock_hk_daily", {"symbol": code5}),
            ("stock_zh_ah_daily_latest", "stock_zh_ah_daily", {"symbol": code5, "adjust": ""}),
        ]:
            try:
                fn = getattr(ak, fn_name)
                raw = fn(**kwargs)
                raw = _filter_hk_table(raw, code5, group)
                row = first_data_row(raw)
                snap = normalize_snapshot("external", group, symbol, prov_name, row, canonical_prefix=canonical)
                snap = attach_context_metadata(snap, spec, group, symbol)
                if snap.get("status") == "ok":
                    snap["context_mode"] = "latest_daily_fallback"
                    return snap
                errors.append(f"{prov_name}:{snap.get('status')}")
            except Exception as exc:
                errors.append(f"{prov_name}:{type(exc).__name__}:{exc}")

        return {"datetime": datetime.now().isoformat(timespec="seconds"), "kind": "external", "context_group": group, "context_symbol": symbol, "provider": "none", "canonical_prefix": canonical, "aggregate_prefix": str(spec.get("aggregate_prefix") or ""), "status": "error", "error": ";".join(errors)}

    attempts = [
        ("futures_zh_spot", "futures_zh_spot", {"symbol": symbol}),
        ("futures_main_sina", "futures_main_sina", {"symbol": symbol}),
        ("futures_zh_daily_sina", "futures_zh_daily_sina", {"symbol": symbol}),
    ]

    for prov_name, fn_name, kwargs in attempts:
        try:
            fn = getattr(ak, fn_name)
            raw = fn(**kwargs) if kwargs else fn()
            row = first_data_row(raw)
            snap = normalize_snapshot("external", group, symbol, prov_name, row, canonical_prefix=canonical)
            snap = attach_context_metadata(snap, spec, group, symbol)
            if snap.get("status") == "ok":
                return snap
            errors.append(f"{prov_name}:{snap.get('status')}")
        except Exception as exc:
            errors.append(f"{prov_name}:{type(exc).__name__}:{exc}")

    return {"datetime": datetime.now().isoformat(timespec="seconds"), "kind": "external", "context_group": group, "context_symbol": symbol, "provider": "none", "canonical_prefix": canonical, "aggregate_prefix": str(spec.get("aggregate_prefix") or ""), "status": "error", "error": ";".join(errors)}

def collect_once(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    od = out_day_dir(args.out_dir, args.date)
    plan_path = od / "realtime_context_plan.csv"
    if not plan_path.exists() or args.refresh_plan:
        write_plan(args)
    plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    ak = load_akshare()
    rows = []

    # Sector snapshots: fetch the small THS summary table ONCE per collection
    # round using low-concurrency hedged subprocesses, then filter every required
    # sector from that single table.  This avoids N repeated THS requests for N
    # sectors and prevents one stuck request from blocking 14:55.
    sector_seen: list[str] = []
    seen_set = set()
    contexts = cfg.get("contexts", {})
    external_ths_items: list[tuple[str, dict, str]] = []
    for _, row in plan.iterrows():
        for sec in [x.strip() for x in str(row.get("sector_symbols") or "").split(",") if x.strip()]:
            if sec in seen_set:
                continue
            seen_set.add(sec)
            sector_seen.append(sec)
        for group in [x.strip() for x in str(row.get("context_groups") or "").split(",") if x.strip()]:
            spec = contexts.get(group, {})
            if str(spec.get("provider") or "") != "ths_sector_summary":
                continue
            for sym in spec.get("symbols", []):
                if sym not in seen_set:
                    seen_set.add(sym)
                    sector_seen.append(sym)
                external_ths_items.append((group, spec, sym))

    if sector_seen:
        summary_df, summary_meta = fetch_ths_sector_summary_hedged(
            timeout_seconds=float(getattr(args, "sector_request_timeout_seconds", 5.0)),
            hedge_workers=int(getattr(args, "sector_hedge_workers", 2)),
            hedge_delay_seconds=float(getattr(args, "sector_hedge_delay_seconds", 1.5)),
        )
        normal_sectors = set()
        for _, row in plan.iterrows():
            for sec in [x.strip() for x in str(row.get("sector_symbols") or "").split(",") if x.strip()]:
                normal_sectors.add(sec)
        for sec in sorted(normal_sectors):
            rows.append(sector_snapshot_from_summary(summary_df, sec, summary_meta))
        board_seen = set()
        for group, spec, sym in external_ths_items:
            key = (group, sym)
            if key in board_seen:
                continue
            board_seen.add(key)
            rows.append(external_board_snapshot_from_summary(summary_df, sym, group, spec, summary_meta))

    ext_seen = set()
    for _, row in plan.iterrows():
        for group in [x.strip() for x in str(row.get("context_groups") or "").split(",") if x.strip()]:
            spec = contexts.get(group, {})
            if str(spec.get("provider") or "") == "ths_sector_summary":
                continue
            for sym in spec.get("symbols", []):
                key = (group, sym)
                if key in ext_seen:
                    continue
                ext_seen.add(key)
                rows.append(fetch_external_snapshot(ak, group, spec, sym))
    if not rows:
        print("No realtime context required by current saved models.")
        return
    out = pd.DataFrame(rows)
    path = od / "context_snapshots.csv"
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"WROTE {path} rows={len(out)} new={len(rows)}")


def collect_loop(args: argparse.Namespace) -> None:
    until = parse_hhmm(args.until)
    while True:
        if until and datetime.now().time() > until:
            print(f"context collect-loop reached until={args.until}")
            return
        collect_once(args)
        if until and datetime.now().time() > until:
            print(f"context collect-loop reached until={args.until}")
            return
        time.sleep(max(1, int(args.interval_seconds)))


def load_snapshots(path: Path, cutoff_time: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    cutoff = parse_hhmm(cutoff_time)
    if cutoff is not None:
        df = df[df["datetime"].dt.time <= cutoff].copy()
    return df



def snapshots_to_flat_current(
    snapshots: pd.DataFrame,
    sector_symbols: list[str],
    context_groups: list[str],
    hist: Optional[pd.DataFrame] = None,
    target_date: Optional[pd.Timestamp] = None,
) -> dict[str, Any]:
    """Build per-artifact current context map from snapshots.

    Generic sector features use the first configured sector symbol as the
    model's primary sector.  External contexts are selected by context group.

    Special handling:
      - hog_hk_proxy aggregates multiple HK proxy rows into
        hog_hk_proxy_close_mean / open_mean / high_mean / low_mean / volume_mean / amount_mean.
        This matches historical builder features such as
        hog_hk_proxy_close_mean_ret1 / hog_hk_proxy_close_mean_ret20.
      - individual HK proxy groups, e.g. hog_hk_yurun_food, keep their own
        canonical prefix so features such as hog_hk_yurun_food_volume_ret1 can
        be estimated as-of cutoff.
    """
    flat: dict[str, Any] = {}
    if snapshots.empty:
        return flat
    ok = snapshots[snapshots.get("status", "") == "ok"].copy()
    if ok.empty:
        return flat

    hist_before = pd.DataFrame()
    if hist is not None and not hist.empty and "date" in hist.columns:
        target = target_date if target_date is not None else pd.Timestamp.max
        hist_before = hist[pd.to_datetime(hist["date"], errors="coerce") < target].sort_values("date")

    def previous_close(prefix: str) -> float:
        col = f"{prefix}_close"
        if hist_before.empty or col not in hist_before.columns:
            return np.nan
        s = pd.to_numeric(hist_before[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    def add_sampled_index_ohlc(part: pd.DataFrame, prefix: str) -> None:
        """Build sampled index OHLC from pct_chg snapshots and historical close."""
        prev = previous_close(prefix)
        if not np.isfinite(prev) or prev <= 0 or part.empty:
            return
        pct = pd.to_numeric(part.get("pct_chg"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        tmp = part.assign(_ret1=pct / 100.0).dropna(subset=["_ret1"]).sort_values("datetime")
        if tmp.empty:
            return
        px = prev * (1.0 + pd.to_numeric(tmp["_ret1"], errors="coerce"))
        px = px.replace([np.inf, -np.inf], np.nan).dropna()
        if px.empty:
            return
        flat[f"{prefix}_open"] = float(px.iloc[0])
        flat[f"{prefix}_high"] = float(px.max())
        flat[f"{prefix}_low"] = float(px.min())
        flat[f"{prefix}_close"] = float(px.iloc[-1])
        flat[f"{prefix}_sampled_ohlc"] = True

    def add_sampled_aggregate_ohlc(part: pd.DataFrame, prefix: str) -> None:
        prev = previous_close(prefix)
        if not np.isfinite(prev) or prev <= 0 or part.empty:
            return
        tmp = part.copy()
        tmp["_pct"] = pd.to_numeric(tmp.get("pct_chg"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        tmp = tmp.dropna(subset=["datetime", "_pct"]).sort_values("datetime")
        if tmp.empty:
            return
        pct_by_time = tmp.groupby("datetime")["_pct"].mean().dropna()
        if pct_by_time.empty:
            return
        px = prev * (1.0 + pct_by_time / 100.0)
        px = px.replace([np.inf, -np.inf], np.nan).dropna()
        if px.empty:
            return
        flat[f"{prefix}_open"] = float(px.iloc[0])
        flat[f"{prefix}_high"] = float(px.max())
        flat[f"{prefix}_low"] = float(px.min())
        flat[f"{prefix}_close"] = float(px.iloc[-1])
        flat[f"{prefix}_sampled_ohlc"] = True

    def add_row(r: pd.Series, prefix: str, part: Optional[pd.DataFrame] = None) -> None:
        ret_only = (
            str(r.get("level_scale_status") or "") == "ret_only"
            or str(r.get("provider") or "").startswith("ths_sector_summary")
            or (str(r.get("provider") or "").startswith("stock_board_") and str(r.get("provider") or "").endswith("_summary_ths"))
        )
        summary_provider = (
            str(r.get("provider") or "").startswith("ths_sector_summary")
            or (str(r.get("provider") or "").startswith("stock_board_") and str(r.get("provider") or "").endswith("_summary_ths"))
        )
        amount_val = to_float(r.get("amount"))
        volume_val = to_float(r.get("volume"))
        amount_source = str(r.get("amount_source") or "")
        volume_source = str(r.get("volume_source") or "")
        summary_amount_raw = to_float(r.get("summary_amount_raw"))
        summary_volume_raw = to_float(r.get("summary_volume_raw"))
        if summary_provider and not amount_source and amount_val is not None:
            summary_amount_raw = amount_val
            amount_val = amount_val * 1e8
            amount_source = "ths_summary_total_amount_yuan"
        if summary_provider and not volume_source and volume_val is not None:
            summary_volume_raw = volume_val
            volume_val = volume_val * 10000.0 * 100.0
            volume_source = "ths_summary_total_volume_shares"
        if not ret_only:
            flat[f"{prefix}_open"] = to_float(r.get("open"))
            flat[f"{prefix}_high"] = to_float(r.get("high"))
            flat[f"{prefix}_low"] = to_float(r.get("low"))
            flat[f"{prefix}_close"] = to_float(r.get("close"))
        elif part is not None:
            add_sampled_index_ohlc(part, prefix)
        flat[f"{prefix}_volume"] = volume_val
        flat[f"{prefix}_amount"] = amount_val
        flat[f"{prefix}_amount_source"] = amount_source
        flat[f"{prefix}_volume_source"] = volume_source
        flat[f"{prefix}_summary_amount_raw"] = summary_amount_raw
        flat[f"{prefix}_summary_volume_raw"] = summary_volume_raw
        flat[f"{prefix}_summary_amount_yuan"] = amount_val if amount_source == "ths_summary_total_amount_yuan" else to_float(r.get("summary_amount_yuan"))
        flat[f"{prefix}_summary_volume_shares"] = volume_val if volume_source == "ths_summary_total_volume_shares" else to_float(r.get("summary_volume_shares"))
        flat[f"{prefix}_level_scale_status"] = "ret_only" if ret_only else "level_ok"
        is_daily_fallback = str(r.get("context_mode") or "") == "latest_daily_fallback"
        flat[f"{prefix}_source_mode"] = "daily_fallback" if is_daily_fallback else ("snapshot_ret_only" if ret_only else "snapshot_level")
        flat[f"{prefix}_is_daily_fallback"] = bool(is_daily_fallback)
        flat[f"{prefix}_pct_chg"] = to_float(r.get("pct_chg"))
        pct = to_float(r.get("pct_chg"))
        if pct is not None and np.isfinite(pct):
            # Most CN/HK realtime APIs report pct_chg in percent units.
            ret1 = pct / 100.0
            flat[f"{prefix}_ret1"] = ret1
            flat[f"{prefix}_close_ret1"] = ret1
        flat[f"{prefix}_snapshot_time"] = str(r.get("datetime"))

    def add_proxy_mean(part: pd.DataFrame) -> None:
        if part.empty:
            return
        # Use latest row per proxy symbol before aggregating.
        latest_rows = []
        for _, g in part.sort_values("datetime").groupby("context_symbol", dropna=False):
            latest_rows.append(g.iloc[-1])
        if not latest_rows:
            return
        latest = pd.DataFrame(latest_rows)
        for field in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
            vals = pd.to_numeric(latest.get(field), errors="coerce") if field in latest.columns else pd.Series(dtype=float)
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                flat[f"hog_hk_proxy_{field}_mean"] = float(vals.mean())
        flat["hog_hk_proxy_snapshot_time"] = str(latest["datetime"].max()) if "datetime" in latest.columns else ""
        flat["hog_hk_proxy_symbol_count"] = int(len(latest))
        flat["hog_hk_proxy_source_mode"] = "daily_fallback" if latest.get("context_mode", pd.Series(dtype=str)).astype(str).eq("latest_daily_fallback").any() else "snapshot_level"
        flat["hog_hk_proxy_is_daily_fallback"] = bool(latest.get("context_mode", pd.Series(dtype=str)).astype(str).eq("latest_daily_fallback").any())

    # Primary sector: first configured sector symbol that has an OK snapshot.
    for sec in sector_symbols:
        part = ok[(ok["kind"].astype(str) == "sector") & (ok["context_symbol"].astype(str) == sec)]
        if not part.empty:
            add_row(part.sort_values("datetime").iloc[-1], "sector", part)
            break

    def add_aggregate(part: pd.DataFrame, prefix: str) -> None:
        if part.empty or not prefix:
            return
        latest_rows = []
        for _, g in part.sort_values("datetime").groupby("context_symbol", dropna=False):
            latest_rows.append(g.iloc[-1])
        if not latest_rows:
            return
        latest = pd.DataFrame(latest_rows)
        # For basket indices constructed from mean component returns, the most
        # consistent realtime input is the mean pct_chg.  estimate_feature()
        # uses {prefix}_close_ret1 to update the historical basket close scale.
        if "pct_chg" in latest.columns:
            pct_vals = pd.to_numeric(latest["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not pct_vals.empty:
                flat[f"{prefix}_close_ret1"] = float(pct_vals.mean()) / 100.0
        add_sampled_aggregate_ohlc(part, prefix)
        for field in ["volume", "amount"]:
            if field in latest.columns:
                vals = pd.to_numeric(latest[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not vals.empty:
                    flat[f"{prefix}_{field}_mean"] = float(vals.mean())
        flat[f"{prefix}_snapshot_time"] = str(latest["datetime"].max()) if "datetime" in latest.columns else ""
        flat[f"{prefix}_symbol_count"] = int(len(latest))
        is_daily_fallback = latest.get("context_mode", pd.Series(dtype=str)).astype(str).eq("latest_daily_fallback").any()
        flat[f"{prefix}_source_mode"] = "daily_fallback" if is_daily_fallback else "snapshot_aggregate"
        flat[f"{prefix}_is_daily_fallback"] = bool(is_daily_fallback)

    for group in context_groups:
        part = ok[ok["context_group"].astype(str) == group]
        if part.empty:
            continue
        if group == "hog_hk_proxy":
            add_proxy_mean(part)
            continue
        latest_rows = []
        for _, g in part.sort_values("datetime").groupby("context_symbol", dropna=False):
            latest_rows.append(g.iloc[-1])
        latest = pd.DataFrame(latest_rows) if latest_rows else pd.DataFrame()
        if latest.empty:
            continue
        # Emit individual context prefixes, e.g. ane_stk_600893_close.
        for _, r in latest.iterrows():
            prefix = str(r.get("canonical_prefix") or group)
            ppart = part[part["context_symbol"].astype(str) == str(r.get("context_symbol"))]
            add_row(r, prefix, ppart)
        # Emit optional basket prefix, e.g. ane_stock_basket_close_ret1.
        agg_prefixes = [x for x in latest.get("aggregate_prefix", pd.Series(dtype=str)).astype(str).unique().tolist() if x and x != "nan"]
        for agg_prefix in agg_prefixes:
            add_aggregate(latest[latest.get("aggregate_prefix").astype(str) == agg_prefix], agg_prefix)

    def val(k):
        x = flat.get(k)
        return x if x is not None and np.isfinite(x) else np.nan
    if np.isfinite(val("gold_close")) and np.isfinite(val("silver_close")) and val("silver_close") != 0:
        flat["gold_silver_ratio"] = val("gold_close") / val("silver_close")
    if np.isfinite(val("gold_close")) and np.isfinite(val("copper_close")) and val("copper_close") != 0:
        flat["gold_copper_ratio"] = val("gold_close") / val("copper_close")
    if np.isfinite(val("feed_soymeal_close")) and np.isfinite(val("feed_corn_close")):
        idx = 0.45 * val("feed_soymeal_close") + 0.35 * val("feed_corn_close")
        if np.isfinite(val("feed_rapeseed_meal_close")):
            idx += 0.20 * val("feed_rapeseed_meal_close")
        flat["feed_cost_index"] = idx
    if np.isfinite(val("feed_soymeal_close")) and np.isfinite(val("feed_corn_close")) and val("feed_corn_close") != 0:
        flat["feed_soymeal_corn_ratio"] = val("feed_soymeal_close") / val("feed_corn_close")
    if np.isfinite(val("hog_fut_close")) and np.isfinite(flat.get("feed_cost_index", np.nan)) and flat.get("feed_cost_index") != 0:
        flat["feed_hog_cost_ratio"] = val("hog_fut_close") / flat["feed_cost_index"]
    return flat

def load_hist_samples(path: str, stock_code: str) -> pd.DataFrame:
    p = resolve_repo_path(path, stock_code)
    if p is None or not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    except Exception:
        return pd.DataFrame()


def estimate_feature(feature: str, current: dict[str, Any], hist: pd.DataFrame, target_date: pd.Timestamp) -> Any:
    """Estimate a context feature as of cutoff_time.

    The goal is to use cutoff-time current values whenever the feature is meant
    to reflect the current trading day, while using historical completed days
    only as rolling baselines.  This mirrors the offline sector feature formulas
    in feature_building/build_sector_features.py.
    """
    if feature in current and current[feature] is not None:
        try:
            x = float(current[feature])
            if np.isfinite(x):
                return x
        except Exception:
            return current[feature]

    if hist is None or hist.empty or "date" not in hist.columns:
        hist_before = pd.DataFrame()
    else:
        hist_before = hist[pd.to_datetime(hist["date"]) < target_date].sort_values("date")

    def num_series(col: str) -> pd.Series:
        if hist_before.empty or col not in hist_before.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(hist_before[col], errors="coerce").dropna()

    def cur_value(col: str) -> float:
        v = current.get(col)
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            pass
        if col.endswith("_range_pct"):
            prefix = col[: -len("_range_pct")]
            hi = cur_value(f"{prefix}_high")
            lo = cur_value(f"{prefix}_low")
            if np.isfinite(hi) and np.isfinite(lo) and lo != 0:
                return float(hi / lo - 1.0)
        # Basket contexts may only expose current ret1.  Convert it back to the
        # historical basket index scale so direct close/ma/z/retN features remain
        # comparable with training-time basket_close columns.
        r1 = current.get(f"{col}_ret1")
        try:
            r1x = float(r1)
        except Exception:
            r1x = np.nan
        h = num_series(col)
        if np.isfinite(r1x) and not h.empty and np.isfinite(h.iloc[-1]):
            return float(h.iloc[-1]) * (1.0 + r1x)
        return np.nan

    # Special handling for sector_retN.  THS realtime summary gives sector_pct_chg
    # but not a realtime index OHLC.  Estimate today's sector close from the
    # last completed sector_close and current sector_ret1, then compute retN.
    m = re.fullmatch(r"sector_ret(1|5|20|60)", feature)
    if m:
        n = int(m.group(1))
        cur_ret1 = current.get("sector_ret1")
        try:
            cur_ret1 = float(cur_ret1)
        except Exception:
            cur_ret1 = np.nan
        if n == 1 and np.isfinite(cur_ret1):
            return cur_ret1
        hclose = num_series("sector_close")
        if np.isfinite(cur_ret1) and len(hclose) >= n and hclose.iloc[-n] != 0:
            cur_close = hclose.iloc[-1] * (1.0 + cur_ret1)
            return cur_close / hclose.iloc[-n] - 1.0

    if feature == "sector_close":
        cur_ret1 = current.get("sector_ret1")
        try:
            cur_ret1 = float(cur_ret1)
        except Exception:
            cur_ret1 = np.nan
        hclose = num_series("sector_close")
        if np.isfinite(cur_ret1) and not hclose.empty and np.isfinite(hclose.iloc[-1]):
            return float(hclose.iloc[-1]) * (1.0 + cur_ret1)

    if feature.endswith("_close"):
        cur = cur_value(feature)
        if np.isfinite(cur):
            return cur

    for suffix, window, minp in [("sector_ma20_gap", 20, 10), ("sector_ma60_gap", 60, 30)]:
        if feature == suffix:
            cur_ret1 = current.get("sector_ret1")
            try:
                cur_ret1 = float(cur_ret1)
            except Exception:
                cur_ret1 = np.nan
            hclose = num_series("sector_close")
            h = hclose.tail(window)
            if np.isfinite(cur_ret1) and len(h) >= minp and h.mean() != 0:
                cur_close = hclose.iloc[-1] * (1.0 + cur_ret1)
                return cur_close / h.mean() - 1.0

    # retN convention: base_ret1/base_ret3/base_ret5/base_ret20/base_ret60
    for suffix, n in [("_ret1", 1), ("_ret3", 3), ("_ret5", 5), ("_ret20", 20), ("_ret60", 60)]:
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            cur = cur_value(base)
            h = num_series(base)
            if not np.isfinite(cur) or len(h) < n or h.iloc[-n] == 0:
                return np.nan
            return cur / h.iloc[-n] - 1.0

    # maN_gap convention: base / previous rolling mean - 1
    for suffix, window, minp in [("_ma20_gap", 20, 10), ("_ma60_gap", 60, 30)]:
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            cur = cur_value(base)
            h = num_series(base).tail(window)
            if not np.isfinite(cur) or len(h) < minp or h.mean() == 0:
                return np.nan
            return cur / h.mean() - 1.0

    # sector_vol20 in offline code is std(previous sector_ret1, 20), so it
    # intentionally does not use today's partial return.
    if feature.endswith("_vol20"):
        prefix = feature[: -len("_vol20")]
        ret_col = f"{prefix}_ret1"
        h = num_series(ret_col).tail(20)
        sd = h.std()
        if len(h) < 10 or not np.isfinite(sd):
            return np.nan
        return sd

    # shock/z-score use today's as-of value over previous completed days.
    if feature.endswith("_shock20"):
        base = feature[: -len("_shock20")]
        cur = cur_value(base)
        h = num_series(base).tail(20)
        if not np.isfinite(cur) or len(h) < 10 or h.mean() == 0:
            return np.nan
        return cur / h.mean() - 1.0

    for suffix, window, minp in [("_z20", 20, 10), ("_z60", 60, 20)]:
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            cur = cur_value(base)
            h = num_series(base).tail(window)
            sd = h.std()
            if not np.isfinite(cur) or len(h) < minp or sd == 0 or not np.isfinite(sd):
                return np.nan
            return (cur - h.mean()) / sd

    if feature.endswith("_range_pct"):
        return cur_value(feature)

    if feature.endswith("_range_z20"):
        prefix = feature[: -len("_range_z20")]
        cur_range = estimate_feature(f"{prefix}_range_pct", current, hist, target_date)
        h = num_series(f"{prefix}_range_pct").tail(20)
        sd = h.std()
        if cur_range is None or not np.isfinite(cur_range) or len(h) < 10 or sd == 0 or not np.isfinite(sd):
            return np.nan
        return (cur_range - h.mean()) / sd

    # Recompute sector-vs-benchmark using as-of sector return and latest known
    # benchmark return from the historical sample row.  This avoids rejecting a
    # model just because benchmark realtime collection is not configured yet.
    for suffix in ["ret1", "ret5", "ret20", "ret60"]:
        if feature == f"sector_vs_bench_{suffix}":
            sec = estimate_feature(f"sector_{suffix}", current, hist, target_date)
            b = num_series(f"bench_{suffix}")
            if sec is None or not np.isfinite(sec) or b.empty or not np.isfinite(b.iloc[-1]):
                return np.nan
            return sec - b.iloc[-1]

    return np.nan


def context_feature_prefix(feature: str, current: dict[str, Any]) -> str:
    if feature.startswith("stock_vs_sector_") or feature.startswith("sector_"):
        return "sector"
    prefixes = []
    for key in current:
        if key.endswith("_snapshot_time"):
            prefix = key[: -len("_snapshot_time")]
            if feature == prefix or feature.startswith(f"{prefix}_"):
                prefixes.append(prefix)
    if prefixes:
        return sorted(prefixes, key=len, reverse=True)[0]
    parts = feature.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else feature


def context_feature_meta(feature: str, current: dict[str, Any], value: Any, target_date: pd.Timestamp, cutoff_time: Optional[str]) -> dict[str, Any]:
    prefix = context_feature_prefix(feature, current)
    source_mode = str(current.get(f"{prefix}_source_mode") or "")
    is_missing = value is None or (isinstance(value, float) and not np.isfinite(value))
    if not source_mode:
        source_mode = "snapshot" if feature in current else ("missing" if is_missing else "historical_estimate")
    snapshot_time = str(current.get(f"{prefix}_snapshot_time") or "")
    age_seconds = np.nan
    if snapshot_time and cutoff_time:
        try:
            cutoff_dt = pd.to_datetime(f"{target_date.date()} {cutoff_time}:00")
            snap_dt = pd.to_datetime(snapshot_time)
            age_seconds = float((cutoff_dt - snap_dt).total_seconds())
        except Exception:
            age_seconds = np.nan
    is_fallback = bool(current.get(f"{prefix}_is_daily_fallback", False)) or source_mode == "daily_fallback"
    return {
        "source_mode": source_mode,
        "snapshot_time": snapshot_time,
        "age_seconds": age_seconds,
        "is_daily_fallback": bool(is_fallback),
    }


def build_features(args: argparse.Namespace) -> None:
    od = out_day_dir(args.out_dir, args.date)
    plan_path = od / "realtime_context_plan.csv"
    if not plan_path.exists() or args.refresh_plan:
        write_plan(args)
    plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    snapshots = load_snapshots(od / "context_snapshots.csv", args.cutoff_time)
    target = pd.to_datetime(yyyymmdd_to_iso(args.date))
    rows = []
    for _, row in plan.iterrows():
        feats = split_csv_cell(row.get("required_context_features"))
        sector_symbols = split_csv_cell(row.get("sector_symbols"))
        context_groups = split_csv_cell(row.get("context_groups"))
        hist = load_hist_samples(str(row.get("samples") or ""), str(row.get("stock_code") or ""))
        flat = snapshots_to_flat_current(snapshots, sector_symbols, context_groups, hist=hist, target_date=target)
        out = {
            "stock_code": row.get("stock_code"),
            "artifact_name": row.get("artifact_name"),
            "artifact_dir": row.get("artifact_dir"),
            "samples": row.get("samples"),
            "cutoff_time": args.cutoff_time or "",
            "context_groups": row.get("context_groups", ""),
            "sector_symbols": row.get("sector_symbols", ""),
            "required_context_features": ",".join(feats),
            "context_mode": "estimated_asof_cutoff",
            "context_snapshot_time": str(snapshots["datetime"].max()) if not snapshots.empty else "",
        }
        for k, v in flat.items():
            if k.endswith((
                "_amount_source",
                "_volume_source",
                "_summary_amount_raw",
                "_summary_volume_raw",
                "_summary_amount_yuan",
                "_summary_volume_shares",
            )):
                out[k] = v
        missing = []
        source_modes = []
        stale_features = []
        has_daily_fallback = False
        for feat in feats:
            val = estimate_feature(feat, flat, hist, target)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                missing.append(feat)
            else:
                out[feat] = val
            meta = context_feature_meta(feat, flat, val, target, args.cutoff_time)
            out[f"{feat}_source_mode"] = meta["source_mode"]
            out[f"{feat}_snapshot_time"] = meta["snapshot_time"]
            out[f"{feat}_age_seconds"] = meta["age_seconds"]
            out[f"{feat}_is_daily_fallback"] = meta["is_daily_fallback"]
            if meta["source_mode"]:
                source_modes.append(str(meta["source_mode"]))
            if bool(meta["is_daily_fallback"]):
                has_daily_fallback = True
            try:
                if np.isfinite(float(meta["age_seconds"])) and float(meta["age_seconds"]) > 300:
                    stale_features.append(feat)
            except Exception:
                pass
        status = context_status_for_missing(feats, missing, has_daily_fallback)
        out["context_status"] = status
        out["missing_context_features"] = ",".join(missing)
        out["context_has_daily_fallback"] = bool(has_daily_fallback)
        out["context_stale_features"] = ",".join(stale_features)
        out["context_source_modes"] = ",".join(sorted(set(source_modes)))
        rows.append(out)
    base_cols = [
        "stock_code", "artifact_name", "artifact_dir", "samples", "cutoff_time",
        "context_groups", "sector_symbols", "required_context_features",
        "context_mode", "context_snapshot_time", "context_status", "missing_context_features",
        "context_has_daily_fallback", "context_stale_features", "context_source_modes",
    ]
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(columns=base_cols)
    out_df.to_csv(od / "context_features_asof.csv", index=False, encoding="utf-8-sig")
    summary = {
        "date": args.date,
        "cutoff_time": args.cutoff_time,
        "plan_rows": int(len(plan)),
        "snapshot_rows_before_cutoff": int(len(snapshots)),
        "feature_rows": int(len(out_df)),
        "context_status_counts": out_df["context_status"].value_counts(dropna=False).to_dict() if not out_df.empty and "context_status" in out_df else {},
    }
    (od / "context_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {od / 'context_features_asof.csv'} rows={len(out_df)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect generic realtime sector/external context for 14:55 scoring")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--models-dir", default=str(SAVED_MODELS_DIR))
        sp.add_argument("--watchlist", default="selected_watchlist.txt")
        sp.add_argument("--model-policy", choices=["preferred", "all"], default="all")
        sp.add_argument("--config", default=str(DEFAULT_CONFIG))
        sp.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
        sp.add_argument("--date", default=today_yyyymmdd(), help="YYYYMMDD")
        sp.add_argument("--cutoff-time", default="14:55")
        sp.add_argument("--refresh-plan", action="store_true")
        sp.add_argument("--sector-request-timeout-seconds", type=float, default=5.0,
                        help="Hard timeout for one THS sector summary subprocess request.")
        sp.add_argument("--sector-hedge-workers", type=int, default=2,
                        help="Low-concurrency hedged THS sector summary requests per collection round.")
        sp.add_argument("--sector-hedge-delay-seconds", type=float, default=1.5,
                        help="Delay between hedged THS sector summary requests.")

    sp = sub.add_parser("plan")
    add_common(sp)

    sp = sub.add_parser("collect-once")
    add_common(sp)

    sp = sub.add_parser("collect-loop")
    add_common(sp)
    sp.add_argument("--interval-seconds", type=int, default=60)
    sp.add_argument("--until", help="HH:MM")

    sp = sub.add_parser("build-features")
    add_common(sp)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "plan":
        write_plan(args)
    elif args.cmd == "collect-once":
        collect_once(args)
    elif args.cmd == "collect-loop":
        collect_loop(args)
    elif args.cmd == "build-features":
        build_features(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
