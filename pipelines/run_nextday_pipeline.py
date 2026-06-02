#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""One-click next-day trading model pipeline.

This script glues together the existing project scripts:

1) bootstrap/ashare_xgb_dual_opportunity_...py --mode update_data
2) feature_building/build_nextday_samples_from_baostock.py
3) feature_building/build_fundamental_features.py          [optional]
4) feature_building/build_sector_features.py               [optional]
5) feature_building/build_*_external_features.py           [optional]
6) model_training/search_walk_forward_model_complexity.py for multiple targets
7) model_training/summarize_nextday_search_results.py

Typical usage on Windows PowerShell:

    .\.venv\Scripts\python.exe pipelines\run_nextday_pipeline.py `
      --symbol 600176.SH `
      --sector-symbol 建筑材料 `
      --start-date 2018-01-01 `
      --end-date 2026-05-08 `
      --feature-pipeline fundamental,sector `
      --search-targets hit50,hit80,close_profit

The script is intentionally conservative:
- It runs commands with shell=False.
- It writes a log file and per-stage manifests.
- With --resume, completed outputs are skipped.
- With --dry-run, it only prints commands.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"
DEFAULT_DUAL_SCRIPT = "bootstrap/ashare_xgb_dual_opportunity_regression_baostock_full_v19_compressed_trading_axis.py"
BUILD_SAMPLES_SCRIPT = "feature_building/build_nextday_samples_from_baostock.py"
BUILD_ASOF_SAMPLES_SCRIPT = "feature_building/build_asof1455_training_samples.py"
BUILD_FUNDAMENTAL_SCRIPT = "feature_building/build_fundamental_features.py"
BUILD_SECTOR_SCRIPT = "feature_building/build_sector_features.py"
SEARCH_SCRIPT = "model_training/search_walk_forward_model_complexity.py"
SUMMARIZE_SCRIPT = "model_training/summarize_nextday_search_results.py"
DEFAULT_GROUPS = "reversal_fundamental_regime,reversal_fundamental_regime_sector,reversal_fundamental_regime_sector_external,all_no_ak"
DEFAULT_MODELS = (
    "xgb_d2_200_lr003_mcw5,"
    "xgb_d3_400_lr003_mcw3,"
    "xgb_d3_600_lr002_mcw3,"
    "xgb_d4_500_lr002_mcw5,"
    "lgbm_leaves7_400,"
    "lgbm_leaves15_700,"
    "extra_trees_600_d3,"
    "random_forest_600_d4"
)


@dataclass
class SymbolInfo:
    input_symbol: str
    raw_code: str
    suffix: str
    stock_code: str
    baostock_symbol: str


@dataclass
class StageResult:
    name: str
    status: str
    command: List[str]
    started_at: str
    ended_at: str
    elapsed_seconds: float
    returncode: Optional[int]
    outputs: Dict[str, str]
    error: Optional[str] = None


@dataclass
class SearchTarget:
    token: str
    label_mode: str
    target_hit_bps: float
    dirname: str


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def normalize_symbol(symbol: str) -> SymbolInfo:
    s = str(symbol).strip().upper()
    if not s:
        raise ValueError("empty symbol")

    suffix = ""
    raw = s
    m = re.match(r"^(\d{6})\.(SH|SZ)$", s)
    if m:
        raw, suffix = m.group(1), m.group(2)
    else:
        m = re.match(r"^(SH|SZ)[.:-]?(\d{6})$", s)
        if m:
            suffix, raw = m.group(1), m.group(2)
        else:
            raw = re.sub(r"\D", "", s)
            if not re.fullmatch(r"\d{6}", raw):
                raise ValueError(f"cannot normalize symbol: {symbol!r}")

    if not suffix:
        suffix = "SH" if raw.startswith(("6", "9")) else "SZ"
    return SymbolInfo(
        input_symbol=symbol,
        raw_code=raw,
        suffix=suffix,
        stock_code=f"{raw}.{suffix}",
        baostock_symbol=raw,
    )


def parse_search_target(token: str) -> SearchTarget:
    """Parse hit50 / hit80 / close_profit / close_profit50 tokens."""
    t = token.strip().lower().replace("-", "_")
    if not t:
        raise ValueError("empty search target")
    hit = re.fullmatch(r"hit_?(\d+(?:\.\d+)?)", t)
    if hit:
        bps = float(hit.group(1))
        return SearchTarget(token=token, label_mode="hit", target_hit_bps=bps, dirname=f"search_hit_{int(bps)}bps")
    cp = re.fullmatch(r"close_?profit(?:_?(\d+(?:\.\d+)?))?", t)
    if cp:
        bps = float(cp.group(1) or 50.0)
        # Filename inside search_walk_forward_model_complexity.py still uses target_hit_bps.
        # For close_profit, this bps only controls auxiliary target-hit fields, not the label itself.
        return SearchTarget(token=token, label_mode="close_profit", target_hit_bps=bps, dirname=f"search_close_profit_{int(bps)}bps")
    raise ValueError(f"unsupported search target {token!r}; use hit50, hit80, close_profit, close_profit50, ...")




def parse_entry_policy(token: str) -> str:
    value = str(token or "").strip().lower().replace("-", "_")
    aliases = {
        "default": "vwap_low",
        "candidate": "vwap_low",
        "low_vwap": "vwap_low",
        "below_vwap": "vwap_low",
        "all": "all_days",
        "all_day": "all_days",
        "all_dates": "all_days",
        "full": "all_days",
    }
    value = aliases.get(value, value)
    if value not in {"vwap_low", "all_days"}:
        raise ValueError(f"unsupported entry policy {token!r}; use vwap_low or all_days")
    return value


def quote_cmd(cmd: Sequence[str]) -> str:
    # shlex.join is readable on Windows too, only for logs.
    return shlex.join([str(x) for x in cmd])


def append_log(log_path: Path, text: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def run_stage(
    name: str,
    cmd: List[str],
    outputs: Dict[str, Path],
    log_path: Path,
    dry_run: bool = False,
    resume: bool = False,
    continue_on_error: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> StageResult:
    started = now_iso()
    t0 = time.time()
    output_str = {k: str(v) for k, v in outputs.items()}
    if resume and outputs and all(p.exists() for p in outputs.values()):
        result = StageResult(
            name=name,
            status="skipped",
            command=cmd,
            started_at=started,
            ended_at=now_iso(),
            elapsed_seconds=round(time.time() - t0, 3),
            returncode=None,
            outputs=output_str,
        )
        append_log(log_path, f"\n[{started}] SKIP {name}: outputs exist")
        append_log(log_path, f"  {quote_cmd(cmd)}")
        return result

    append_log(log_path, f"\n[{started}] RUN {name}")
    append_log(log_path, f"  {quote_cmd(cmd)}")

    if dry_run:
        result = StageResult(
            name=name,
            status="dry_run",
            command=cmd,
            started_at=started,
            ended_at=now_iso(),
            elapsed_seconds=round(time.time() - t0, 3),
            returncode=None,
            outputs=output_str,
        )
        return result

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    append_log(log_path, proc.stdout or "")
    ended = now_iso()
    status = "ok" if proc.returncode == 0 else "failed"
    err = None if proc.returncode == 0 else f"stage {name} failed with returncode={proc.returncode}"
    result = StageResult(
        name=name,
        status=status,
        command=cmd,
        started_at=started,
        ended_at=ended,
        elapsed_seconds=round(time.time() - t0, 3),
        returncode=proc.returncode,
        outputs=output_str,
        error=err,
    )
    if proc.returncode != 0 and not continue_on_error:
        raise RuntimeError(err)
    return result


def require_scripts(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not (PROJECT_DIR / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing required project scripts: {missing}")


def choose_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def stage_names_from_args(args: argparse.Namespace) -> Optional[set[str]]:
    only = set(split_csv(args.only_stages)) if args.only_stages else set()
    skip = set(split_csv(args.skip_stages)) if args.skip_stages else set()
    if only and skip:
        raise ValueError("--only-stages and --skip-stages should not be used together")
    if only:
        return {"__only__", *only}
    if skip:
        return {"__skip__", *skip}
    return None


def should_run(stage: str, selector: Optional[set[str]]) -> bool:
    if selector is None:
        return True
    if "__skip__" in selector:
        return stage not in selector
    if "__only__" in selector:
        return stage in selector
    return stage in selector


def should_run_grouped(stage: str, group: str, selector: Optional[set[str]]) -> bool:
    """Run a stage that can also be addressed by a group name, e.g. search."""
    if selector is None:
        return True
    if "__skip__" in selector:
        return stage not in selector and group not in selector
    if "__only__" in selector:
        return stage in selector or group in selector
    return stage in selector or group in selector




def expected_final_samples_path(
    samples_dir: Path,
    fund_dir: Path,
    sector_dir: Path,
    external_base_dir: Path,
    feature_pipeline: List[str],
    external_steps: List[str],
) -> Path:
    """Return the expected most-complete sample file path for this pipeline config."""
    external_outputs = {
        "hog": external_base_dir / "hog" / "training_samples_with_hog_industry.csv",
        "feed": external_base_dir / "feed" / "training_samples_with_feed_external.csv",
        "muyuan_hk": external_base_dir / "muyuan_hk" / "training_samples_with_hk_external.csv",
        "zijin": external_base_dir / "zijin" / "training_samples_with_zijin_external.csv",
        "zijin_external": external_base_dir / "zijin_external" / "training_samples_with_zijin_external.csv",
        "ai_compute": external_base_dir / "ai_compute" / "training_samples_with_ai_compute_external.csv",
        "material_wind_battery": external_base_dir / "material_wind_battery" / "training_samples_with_material_wind_battery_external.csv",
        "power_utility_rate": external_base_dir / "power_utility_rate" / "training_samples_with_power_utility_rate_external.csv",
        "fertilizer": external_base_dir / "fertilizer" / "training_samples_with_fertilizer_external.csv",
        "storage_power": external_base_dir / "storage_power" / "training_samples_with_storage_power_external.csv",
        "aero_nuclear_equipment": external_base_dir / "aero_nuclear_equipment" / "training_samples_with_aero_nuclear_equipment_external.csv",
        "optical_cable_grid": external_base_dir / "optical_cable_grid" / "training_samples_with_optical_cable_grid_external.csv",
    }
    effective_external = [x.lower() for x in external_steps if x.lower() not in {"", "none", "no", "false"}]
    if effective_external:
        last = effective_external[-1]
        if last in external_outputs:
            return external_outputs[last]
    if "sector" in feature_pipeline:
        return sector_dir / "training_samples_with_sector.csv"
    if "fundamental" in feature_pipeline:
        return fund_dir / "training_samples_with_fundamentals.csv"
    return samples_dir / "training_samples.csv"


def resolve_final_samples(
    samples_dir: Path,
    fund_dir: Path,
    sector_dir: Path,
    external_base_dir: Path,
    feature_pipeline: List[str],
    external_steps: List[str],
    current_samples: Path,
    samples_override: Optional[str] = None,
    require_exists: bool = False,
) -> Path:
    """Resolve the final sample file used by search/summarize/save-like stages.

    This prevents `--only-stages search` from silently falling back to
    `01_samples/training_samples.csv` when richer sector/fundamental/external
    samples already exist.
    """
    if samples_override:
        p = Path(samples_override)
        if not p.is_absolute():
            p = PROJECT_DIR / p
        if require_exists and not p.exists():
            raise FileNotFoundError(f"--samples-override not found: {p}")
        return p

    candidates: List[Path] = []
    # External outputs are the richest; check likely outputs in reverse pipeline order.
    external_map = {
        "hog": external_base_dir / "hog" / "training_samples_with_hog_industry.csv",
        "feed": external_base_dir / "feed" / "training_samples_with_feed_external.csv",
        "muyuan_hk": external_base_dir / "muyuan_hk" / "training_samples_with_hk_external.csv",
        "zijin": external_base_dir / "zijin" / "training_samples_with_zijin_external.csv",
        "zijin_external": external_base_dir / "zijin_external" / "training_samples_with_zijin_external.csv",
        "ai_compute": external_base_dir / "ai_compute" / "training_samples_with_ai_compute_external.csv",
        "material_wind_battery": external_base_dir / "material_wind_battery" / "training_samples_with_material_wind_battery_external.csv",
        "power_utility_rate": external_base_dir / "power_utility_rate" / "training_samples_with_power_utility_rate_external.csv",
        "fertilizer": external_base_dir / "fertilizer" / "training_samples_with_fertilizer_external.csv",
        "storage_power": external_base_dir / "storage_power" / "training_samples_with_storage_power_external.csv",
        "aero_nuclear_equipment": external_base_dir / "aero_nuclear_equipment" / "training_samples_with_aero_nuclear_equipment_external.csv",
        "optical_cable_grid": external_base_dir / "optical_cable_grid" / "training_samples_with_optical_cable_grid_external.csv",
    }
    for step in reversed([x.lower() for x in external_steps if x.lower() not in {"", "none", "no", "false"}]):
        if step in external_map:
            candidates.append(external_map[step])

    # Also include both Zijin aliases for backward compatibility.
    candidates.extend([
        external_base_dir / "optical_cable_grid" / "training_samples_with_optical_cable_grid_external.csv",
        external_base_dir / "aero_nuclear_equipment" / "training_samples_with_aero_nuclear_equipment_external.csv",
        external_base_dir / "storage_power" / "training_samples_with_storage_power_external.csv",
        external_base_dir / "fertilizer" / "training_samples_with_fertilizer_external.csv",
        external_base_dir / "power_utility_rate" / "training_samples_with_power_utility_rate_external.csv",
        external_base_dir / "material_wind_battery" / "training_samples_with_material_wind_battery_external.csv",
        external_base_dir / "ai_compute" / "training_samples_with_ai_compute_external.csv",
        external_base_dir / "zijin_external" / "training_samples_with_zijin_external.csv",
        external_base_dir / "zijin" / "training_samples_with_zijin_external.csv",
        external_base_dir / "muyuan_hk" / "training_samples_with_hk_external.csv",
        external_base_dir / "feed" / "training_samples_with_feed_external.csv",
        external_base_dir / "hog" / "training_samples_with_hog_industry.csv",
    ])
    if "sector" in feature_pipeline:
        candidates.append(sector_dir / "training_samples_with_sector.csv")
    if "fundamental" in feature_pipeline:
        candidates.append(fund_dir / "training_samples_with_fundamentals.csv")
    candidates.append(samples_dir / "training_samples.csv")

    seen = set()
    deduped = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    for p in deduped:
        if p.exists():
            return p

    expected = expected_final_samples_path(samples_dir, fund_dir, sector_dir, external_base_dir, feature_pipeline, external_steps)
    if require_exists:
        raise FileNotFoundError(
            "No usable samples file found for search. Expected one of: "
            + ", ".join(str(x) for x in deduped)
            + f". Use --samples-override to specify the intended file. Expected final path: {expected}"
        )
    return expected if expected else current_samples

def sanitize_run_tag(value: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "").strip())
    return tag.strip("_.-")


def build_pipeline(args: argparse.Namespace) -> Tuple[SymbolInfo, Path, Dict[str, object]]:
    info = normalize_symbol(args.symbol)
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        tag = sanitize_run_tag(getattr(args, "run_tag", ""))
        suffix = "_pipeline_out"
        out_root = SAVED_DATA_DIR / f"{info.raw_code}{suffix}"
    if not out_root.is_absolute():
        out_root = PROJECT_DIR / out_root
    ensure_dir(out_root)
    meta = {
        "symbol": asdict(info),
        "out_root": str(out_root),
        "created_at": now_iso(),
        "args": vars(args),
    }
    return info, out_root, meta


def run_pipeline(args: argparse.Namespace) -> int:
    require_scripts([
        args.dual_script,
        BUILD_SAMPLES_SCRIPT,
        BUILD_ASOF_SAMPLES_SCRIPT,
        BUILD_FUNDAMENTAL_SCRIPT,
        BUILD_SECTOR_SCRIPT,
        SEARCH_SCRIPT,
        SUMMARIZE_SCRIPT,
    ])

    info, out_root, meta = build_pipeline(args)
    log_path = out_root / "run.log"
    manifest_path = out_root / "pipeline_summary.json"
    stages: List[StageResult] = []
    selector = stage_names_from_args(args)

    py = args.python or sys.executable
    today = dt.date.today().isoformat()
    start_date = args.start_date
    end_date = args.end_date or today
    intraday_start = args.intraday_start or f"{start_date} 09:30:00"
    intraday_end = args.intraday_end or f"{end_date} 15:00:00"

    base_dir = out_root / "00_base"
    samples_dir = out_root / "01_samples"
    asof_samples_dir = out_root / "01_samples_asof1455"
    fund_dir = out_root / "02_fundamental"
    sector_dir = out_root / "03_sector"
    external_base_dir = out_root / "04_external"
    search_root = out_root / "10_search"
    summary_dir = out_root / "99_summary"

    append_log(log_path, f"=== nextday pipeline start: {now_iso()} ===")
    append_log(log_path, f"symbol={info.stock_code}, out_root={out_root}")

    feature_pipeline = split_csv(args.feature_pipeline)
    external_steps = split_csv(args.external)
    feature_time_mode = str(args.feature_time_mode or "eod").strip().lower()
    if feature_time_mode == "asof":
        feature_time_mode = "asof1455"
    search_targets = [parse_search_target(x) for x in split_csv(args.search_targets)]
    entry_policies = [parse_entry_policy(x) for x in split_csv(args.entry_policies)]

    if args.sector_symbol and "sector" not in feature_pipeline:
        feature_pipeline.append("sector")
    if "fundamental" not in feature_pipeline and not args.no_fundamental:
        feature_pipeline.insert(0, "fundamental")
    if args.no_fundamental:
        feature_pipeline = [x for x in feature_pipeline if x != "fundamental"]

    # 0. update_data
    daily_features = base_dir / "daily_features.csv"
    intraday_bars = base_dir / f"{info.raw_code}_5m.csv"
    if args.intraday_bars_override:
        intraday_bars = Path(args.intraday_bars_override)
        if not intraday_bars.is_absolute():
            intraday_bars = PROJECT_DIR / intraday_bars
    if should_run("update_data", selector):
        cmd = [
            py, args.dual_script,
            "--mode", "update_data",
            "--symbol", info.baostock_symbol,
            "--benchmark_symbol", args.benchmark_symbol,
            "--daily_start", start_date,
            "--daily_end", end_date,
            "--intraday_start", intraday_start,
            "--intraday_end", intraday_end,
            "--output_dir", str(base_dir),
            "--cache_mode", args.cache_mode,
            "--feature_cache_mode", args.feature_cache_mode,
        ]
        if args.force_refresh:
            cmd.append("--force_refresh")
        if args.adjust:
            cmd.extend(["--adjust", args.adjust])
        result = run_stage(
            "update_data", cmd,
            {"daily_features": daily_features, "intraday_bars": intraday_bars},
            log_path,
            dry_run=args.dry_run,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
        stages.append(result)

    # Some older runs may only have raw cache; keep a fallback but prefer exported request CSV.
    intraday_candidates = [
        base_dir / f"{info.raw_code}_5m.csv",
        base_dir / "raw_cache" / f"{info.raw_code}_5m_raw.csv",
    ]
    if args.intraday_bars_override:
        intraday_candidates.insert(0, intraday_bars)
    intraday_bars = choose_existing(intraday_candidates) or (base_dir / f"{info.raw_code}_5m.csv")

    # 1. build next-day samples
    current_samples = samples_dir / "training_samples.csv"
    if args.base_samples_override:
        current_samples = Path(args.base_samples_override)
        if not current_samples.is_absolute():
            current_samples = PROJECT_DIR / current_samples
    if should_run("samples", selector):
        cmd = [
            py, BUILD_SAMPLES_SCRIPT,
            "--daily-features", str(daily_features),
            "--intraday-bars", str(intraday_bars),
            "--out-dir", str(samples_dir),
            "--min-bars", str(args.min_bars),
        ]
        if args.keep_unlabeled_tail:
            cmd.append("--keep-unlabeled-tail")
        result = run_stage(
            "samples", cmd,
            {"samples": current_samples},
            log_path,
            dry_run=args.dry_run,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
        stages.append(result)

    if feature_time_mode == "asof1455":
        out_file = asof_samples_dir / "training_samples_asof1455.csv"
        if should_run("asof_samples", selector):
            cmd = [
                py, BUILD_ASOF_SAMPLES_SCRIPT,
                "--samples", str(current_samples),
                "--intraday-bars", str(intraday_bars),
                "--out-dir", str(asof_samples_dir),
                "--cutoff-time", args.feature_cutoff_time,
                "--min-bars", str(args.min_bars),
            ]
            result = run_stage(
                "asof_samples", cmd,
                {"samples": out_file},
                log_path,
                dry_run=args.dry_run,
                resume=args.resume,
                continue_on_error=args.continue_on_error,
            )
            stages.append(result)
        current_samples = out_file

    # 2. optional features
    if "fundamental" in feature_pipeline and should_run("fundamental", selector):
        out_file = fund_dir / "training_samples_with_fundamentals.csv"
        cmd = [
            py, BUILD_FUNDAMENTAL_SCRIPT,
            "--symbol", info.stock_code,
            "--daily-samples", str(current_samples),
            "--out-dir", str(fund_dir),
            "--start-date", start_date,
            "--end-date", end_date,
            "--fallback-lag-days", str(args.fallback_lag_days),
        ]
        if args.skip_akshare_fund_flow:
            cmd.append("--skip-akshare")
        result = run_stage(
            "fundamental", cmd,
            {"samples": out_file},
            log_path,
            dry_run=args.dry_run,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
        )
        stages.append(result)
        current_samples = out_file

    if "sector" in feature_pipeline:
        if not args.sector_symbol:
            raise ValueError("feature pipeline includes sector, but --sector-symbol is not provided")
        if should_run("sector", selector):
            out_file = sector_dir / "training_samples_with_sector.csv"
            cmd = [
                py, BUILD_SECTOR_SCRIPT,
                "--samples", str(current_samples),
                "--out-dir", str(sector_dir),
                "--sector-symbol", args.sector_symbol,
                "--start-date", start_date,
                "--end-date", end_date,
            ]
            result = run_stage(
                "sector", cmd,
                {"samples": out_file},
                log_path,
                dry_run=args.dry_run,
                resume=args.resume,
                continue_on_error=args.continue_on_error,
            )
            stages.append(result)
            current_samples = out_file

    # 3. optional external feature builders, chained in the order provided.
    external_outputs = {
        "hog": ("feature_building/build_hog_industry_features.py", "training_samples_with_hog_industry.csv", []),
        "feed": ("feature_building/build_haida_feed_external_features.py", "training_samples_with_feed_external.csv", []),
        "muyuan_hk": ("feature_building/build_muyuan_hk_external_features.py", "training_samples_with_hk_external.csv", []),
        "zijin": ("feature_building/build_zijin_external_features.py", "training_samples_with_zijin_external.csv", []),
        "zijin_external": ("feature_building/build_zijin_external_features.py", "training_samples_with_zijin_external.csv", []),
        "ai_compute": ("feature_building/build_stock_external_features.py", "training_samples_with_ai_compute_external.csv", ["--profile", "ai_compute"]),
        "material_wind_battery": ("feature_building/build_stock_external_features.py", "training_samples_with_material_wind_battery_external.csv", ["--profile", "material_wind_battery"]),
        "power_utility_rate": ("feature_building/build_stock_external_features.py", "training_samples_with_power_utility_rate_external.csv", ["--profile", "power_utility_rate"]),
        "fertilizer": ("feature_building/build_stock_external_features.py", "training_samples_with_fertilizer_external.csv", ["--profile", "fertilizer"]),
        "storage_power": ("feature_building/build_stock_external_features.py", "training_samples_with_storage_power_external.csv", ["--profile", "storage_power"]),
        "aero_nuclear_equipment": ("feature_building/build_stock_external_features.py", "training_samples_with_aero_nuclear_equipment_external.csv", ["--profile", "aero_nuclear_equipment"]),
        "optical_cable_grid": ("feature_building/build_stock_external_features.py", "training_samples_with_optical_cable_grid_external.csv", ["--profile", "optical_cable_grid"]),
    }
    for step in external_steps:
        key = step.lower()
        if key in {"none", "no", "false"}:
            continue
        if key not in external_outputs:
            raise ValueError(f"unknown external step={step!r}; supported: {sorted(external_outputs)}")
        script, output_name, extra = external_outputs[key]
        require_scripts([script])
        stage_name = f"external_{key}"
        out_dir = external_base_dir / key
        out_file = out_dir / output_name
        if should_run(stage_name, selector):
            cmd = [
                py, script,
                "--samples", str(current_samples),
                "--out-dir", str(out_dir),
                "--lag-days", str(args.external_lag_days),
                *extra,
            ]
            if script.endswith("build_stock_external_features.py"):
                # New stock external builder supports source-specific lag policies.
                # Remove the legacy --lag-days pair that is still needed by old external builders.
                if "--lag-days" in cmd:
                    i = cmd.index("--lag-days")
                    del cmd[i:i + 2]
                cmd.extend([
                    "--target-symbol", info.stock_code,
                    "--start-date", start_date,
                    "--end-date", end_date,
                    "--adjust", args.adjust if args.adjust in {"qfq", "hfq", ""} else "qfq",
                    "--domestic-lag-days", str(args.stock_external_domestic_lag_days),
                    "--future-lag-days", str(args.stock_external_future_lag_days),
                    "--us-lag-days", str(args.stock_external_us_lag_days),
                ])
                if args.enable_us_yf:
                    cmd.append("--enable-us-yf")
            # Allow speed options for Zijin when requested.
            if key in {"zijin", "zijin_external"}:
                if args.zijin_skip_basis:
                    cmd.append("--skip-basis")
                if args.zijin_skip_sector:
                    cmd.append("--skip-sector")
                if args.zijin_skip_hk:
                    cmd.append("--skip-hk")
            result = run_stage(
                stage_name, cmd,
                {"samples": out_file},
                log_path,
                dry_run=args.dry_run,
                resume=args.resume,
                continue_on_error=args.continue_on_error,
            )
            stages.append(result)
            current_samples = out_file

    search_selected = should_run_grouped("search", "search", selector)
    final_samples = resolve_final_samples(
        samples_dir=samples_dir,
        fund_dir=fund_dir,
        sector_dir=sector_dir,
        external_base_dir=external_base_dir,
        feature_pipeline=feature_pipeline,
        external_steps=external_steps,
        current_samples=current_samples,
        samples_override=args.samples_override,
        require_exists=bool(search_selected and not args.dry_run),
    )
    append_log(log_path, f"[INFO] final samples for search: {final_samples}")
    if search_selected and not args.dry_run and not final_samples.exists():
        raise FileNotFoundError(f"final samples for search not found: {final_samples}")

    # 4. searches
    search_dirs: List[Path] = []
    for entry_policy in entry_policies:
        for target in search_targets:
            # Keep separate output directories so vwap_low and all_days are directly comparable.
            target_suffix = target.dirname.removeprefix("search_")
            stage_name = f"search_{entry_policy}_{target_suffix}"
            out_dir = search_root / stage_name
            search_dirs.append(out_dir)
            if not should_run_grouped(stage_name, "search", selector):
                continue
            expected_summary = out_dir / f"summary_{int(target.target_hit_bps)}bps.csv"
            cmd = [
                py, SEARCH_SCRIPT,
                "--samples", str(final_samples),
                "--intraday-bars", str(intraday_bars),
                "--out-dir", str(out_dir),
                "--round-trip-cost-bps", str(args.round_trip_cost_bps),
                "--target-hit-bps", str(target.target_hit_bps),
                "--label-mode", target.label_mode,
                "--entry-policy", entry_policy,
                "--entry-vwap-premium-bps", str(args.entry_vwap_premium_bps),
                "--feature-time-mode", feature_time_mode,
                "--feature-cutoff-time", args.feature_cutoff_time if feature_time_mode == "asof1455" else "",
                "--max-missing", str(args.max_missing),
                "--groups", args.groups,
                "--models", args.models,
                "--quantiles", args.quantiles,
                "--train-rows", str(args.train_rows),
                "--valid-rows", str(args.valid_rows),
                "--test-rows", str(args.test_rows),
                "--min-valid-trades", str(args.min_valid_trades),
                "--min-train-entries", str(args.min_train_entries),
            ]
            result = run_stage(
                stage_name, cmd,
                {"summary": expected_summary},
                log_path,
                dry_run=args.dry_run,
                resume=args.resume,
                continue_on_error=args.continue_on_error,
            )
            stages.append(result)
            # Write per-search metadata even if the search itself is skipped/dry-run.
            write_json({
                "symbol": asdict(info),
                "label_mode": target.label_mode,
                "target_hit_bps": target.target_hit_bps,
                "entry_policy": entry_policy,
                "entry_vwap_premium_bps": args.entry_vwap_premium_bps,
                "feature_time_mode": feature_time_mode,
                "feature_cutoff_time": args.feature_cutoff_time if feature_time_mode == "asof1455" else "",
                "sample_file": str(final_samples),
                "intraday_bars": str(intraday_bars),
                "feature_pipeline": feature_pipeline,
                "external": external_steps,
                "groups": split_csv(args.groups),
                "models": split_csv(args.models),
                "quantiles": split_csv(args.quantiles),
                "command": cmd,
                "stage_result": asdict(result),
            }, out_dir / "search_run_manifest.json")

    # 5. final summary
    if should_run("summarize", selector):
        cmd = [
            py, SUMMARIZE_SCRIPT,
            "--pipeline-out", str(out_root),
            "--out-dir", str(summary_dir),
        ]
        if args.excel:
            cmd.append("--excel")
        result = run_stage(
            "summarize", cmd,
            {"leaderboard": summary_dir / "final_leaderboard.csv"},
            log_path,
            dry_run=args.dry_run,
            resume=False,
            continue_on_error=args.continue_on_error,
        )
        stages.append(result)

    meta.update({
        "finished_at": now_iso(),
        "status": "ok" if all(s.status in {"ok", "skipped", "dry_run"} for s in stages) else "partial_failed",
        "feature_pipeline_effective": feature_pipeline,
        "external_effective": external_steps,
        "entry_policies_effective": entry_policies,
        "entry_vwap_premium_bps": args.entry_vwap_premium_bps,
        "feature_time_mode": feature_time_mode,
        "feature_cutoff_time": args.feature_cutoff_time if feature_time_mode == "asof1455" else "",
        "final_samples": str(final_samples),
        "intraday_bars": str(intraday_bars),
        "search_dirs": [str(x) for x in search_dirs],
        "stages": [asdict(s) for s in stages],
    })
    write_json(meta, manifest_path)
    append_log(log_path, f"=== nextday pipeline end: {now_iso()} status={meta['status']} ===")
    print(json.dumps({
        "status": meta["status"],
        "symbol": info.stock_code,
        "out_root": str(out_root),
        "final_samples": str(final_samples),
        "leaderboard": str(summary_dir / "final_leaderboard.csv"),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))
    return 0 if meta["status"] == "ok" else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full next-day model pipeline for one stock")
    p.add_argument("--symbol", required=True, help="Stock code, e.g. 600176.SH / 000001.SZ / 600176")
    p.add_argument("--sector-symbol", default=None, help="THS sector name, e.g. 建筑材料 / 养殖业")
    p.add_argument("--out-root", default=None, help="Output root; default: saved_data/<code>_pipeline_out")
    p.add_argument("--run-tag", default="", help="Log/summary tag only; default output root remains saved_data/<code>_pipeline_out")
    p.add_argument("--python", default=None, help="Python executable; default: current interpreter")
    p.add_argument("--dual-script", default=DEFAULT_DUAL_SCRIPT)

    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--end-date", default=dt.date.today().isoformat())
    p.add_argument("--intraday-start", default=None)
    p.add_argument("--intraday-end", default=None)
    p.add_argument("--benchmark-symbol", default="000300")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none", ""])
    p.add_argument("--cache-mode", default="incremental", choices=["incremental", "full"])
    p.add_argument("--feature-cache-mode", default="incremental", choices=["incremental", "full"])
    p.add_argument("--force-refresh", action="store_true")

    p.add_argument("--min-bars", type=int, default=40)
    p.add_argument("--keep-unlabeled-tail", action="store_true", default=True)
    p.add_argument("--no-keep-unlabeled-tail", action="store_false", dest="keep_unlabeled_tail")
    p.add_argument("--feature-time-mode", choices=["eod", "asof", "asof1455"], default="eod",
                   help="Feature timestamp policy. asof/asof1455 adds cutoff-time entry features and labels")
    p.add_argument("--feature-cutoff-time", default="14:55", help="Cutoff time for --feature-time-mode asof/asof1455")

    p.add_argument("--feature-pipeline", default="fundamental,sector", help="Comma list: fundamental,sector")
    p.add_argument("--samples-override", default=None,
                   help="Explicit samples CSV for search/summarize-only workflows; overrides automatic final sample resolution")
    p.add_argument("--base-samples-override", default=None,
                   help="Reuse an existing base training_samples.csv before optional asof/fundamental/sector/external stages")
    p.add_argument("--intraday-bars-override", default=None,
                   help="Reuse an existing 5m bar CSV before optional asof/search stages")
    p.add_argument("--no-fundamental", action="store_true", help="Do not build fundamental features")
    p.add_argument("--fallback-lag-days", type=int, default=120)
    p.add_argument("--skip-akshare-fund-flow", action="store_true")
    p.add_argument("--external", default="", help="Comma list: hog,feed,muyuan_hk,zijin_external,ai_compute,material_wind_battery,power_utility_rate,fertilizer,storage_power,aero_nuclear_equipment,optical_cable_grid")
    p.add_argument("--external-lag-days", type=int, default=1, help="Legacy lag for old external builders such as hog/feed/muyuan_hk/zijin_external")
    p.add_argument("--stock-external-domestic-lag-days", type=int, default=0, help="New stock external builder: A-share/ETF/THS board lag; default 0")
    p.add_argument("--stock-external-future-lag-days", type=int, default=1, help="New stock external builder: domestic futures lag; default 1")
    p.add_argument("--stock-external-us-lag-days", type=int, default=1, help="New stock external builder: U.S. yfinance lag; forced to >=1 inside builder")
    p.add_argument("--enable-us-yf", action="store_true", help="Enable optional yfinance U.S. mappings for stock external profiles, mainly ai_compute")
    p.add_argument("--zijin-skip-basis", action="store_true")
    p.add_argument("--zijin-skip-sector", action="store_true")
    p.add_argument("--zijin-skip-hk", action="store_true")

    p.add_argument("--search-targets", default="hit50,hit80,close_profit", help="Comma list: hit50,hit80,hit100,close_profit")
    p.add_argument("--entry-policies", default="vwap_low,all_days",
                   help="Comma list: vwap_low,all_days. vwap_low keeps the old close<=VWAP*1.005 candidate filter; all_days trains/evaluates on all valid days")
    p.add_argument("--entry-vwap-premium-bps", type=float, default=50.0,
                   help="VWAP premium for vwap_low; 50 means close <= daily_vwap*1.005")
    p.add_argument("--groups", default=DEFAULT_GROUPS)
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--max-missing", type=float, default=0.35)
    p.add_argument("--train-rows", type=int, default=756)
    p.add_argument("--valid-rows", type=int, default=126)
    p.add_argument("--test-rows", type=int, default=63)
    p.add_argument("--min-valid-trades", type=int, default=8)
    p.add_argument("--min-train-entries", type=int, default=80)

    p.add_argument("--resume", action="store_true", help="Skip stages whose expected outputs already exist")
    p.add_argument("--dry-run", action="store_true", help="Only print/write commands, do not execute")
    p.add_argument("--continue-on-error", action="store_true", help="Continue following stages after a failed command")
    p.add_argument("--only-stages", default="", help="Comma list stage names to run, e.g. search,summarize or update_data,samples")
    p.add_argument("--skip-stages", default="", help="Comma list stage names to skip")
    p.add_argument("--excel", action="store_true", help="Also create final_leaderboard.xlsx if openpyxl is installed")
    return p.parse_args()


def main() -> None:
    try:
        raise SystemExit(run_pipeline(parse_args()))
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
