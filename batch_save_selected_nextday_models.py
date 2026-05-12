#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-save selected next-day stock models into saved_models/.

Usage from project root:
    python batch_save_selected_nextday_models.py

Options:
    python batch_save_selected_nextday_models.py --dry-run
    python batch_save_selected_nextday_models.py --overwrite
    python batch_save_selected_nextday_models.py --only 002311.SZ,002714.SZ

This script intentionally excludes:
    600176.SH 中国巨石
    600309.SH 万华化学

It keeps/saves:
    002270.SZ 华明装备
    002311.SZ 海大集团
    002714.SZ 牧原股份
    600276.SH 恒瑞医药
    600312.SH 平高电气
    601899.SH 紫金矿业
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SaveJob:
    stock_code: str
    artifact_name: str
    samples: str
    intraday_bars: str
    feature_group: str
    model_name: str
    label_mode: str
    entry_policy: str
    target_hit_bps: float = 50.0
    entry_vwap_premium_bps: float = 50.0
    note: str = ""


# Best/retained configurations selected from saved_data/*_pipeline_out/99_summary/final_leaderboard.csv.
# Paths are relative to PROJECT_ROOT.
JOBS: list[SaveJob] = [
    SaveJob(
        stock_code="002270.SZ",
        artifact_name="nextday_all_days_close_profit_extra_trees_all_no_ak_v1",
        samples="saved_data/002270_pipeline_out/03_sector/training_samples_with_sector.csv",
        intraday_bars="saved_data/002270_pipeline_out/00_base/002270_5m.csv",
        feature_group="all_no_ak",
        model_name="extra_trees_600_d3",
        label_mode="close_profit",
        entry_policy="all_days",
        target_hit_bps=50.0,
        note="华明装备：保留观察模型；收益尚可但历史回撤偏大。",
    ),
    SaveJob(
        stock_code="002311.SZ",
        artifact_name="nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_sector_hog_v1",
        samples="saved_data/002311_pipeline_out/04_external/hog/training_samples_with_hog_industry.csv",
        intraday_bars="saved_data/002311_pipeline_out/00_base/002311_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="xgb_d3_600_lr002_mcw3",
        label_mode="close_profit",
        entry_policy="vwap_low",
        target_hit_bps=50.0,
        note="海大集团：优先保存主模型。",
    ),
    SaveJob(
        stock_code="002714.SZ",
        artifact_name="nextday_vwap_low_close_profit_random_forest_reversal_fundamental_regime_sector_muyuan_hk_v1",
        samples="saved_data/002714_pipeline_out/04_external/muyuan_hk/training_samples_with_hk_external.csv",
        intraday_bars="saved_data/002714_pipeline_out/00_base/002714_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="random_forest_600_d4",
        label_mode="close_profit",
        entry_policy="vwap_low",
        target_hit_bps=50.0,
        note="牧原股份：按要求保留；该模型稳定性一般，建议实盘降权观察。",
    ),
    SaveJob(
        stock_code="600276.SH",
        artifact_name="nextday_all_days_hit80_extra_trees_reversal_fundamental_regime_sector_v1",
        samples="saved_data/600276_pipeline_out/03_sector/training_samples_with_sector.csv",
        intraday_bars="saved_data/600276_pipeline_out/00_base/600276_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="extra_trees_600_d3",
        label_mode="hit",
        entry_policy="all_days",
        target_hit_bps=80.0,
        note="恒瑞医药：80bps目标命中模型，不是close_profit主模型。",
    ),
    SaveJob(
        stock_code="600312.SH",
        artifact_name="nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_v1",
        samples="saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv",
        intraday_bars="saved_data/600312_pipeline_out/00_base/600312_5m.csv",
        feature_group="reversal_fundamental_regime",
        model_name="xgb_d3_400_lr003_mcw3",
        label_mode="close_profit",
        entry_policy="all_days",
        target_hit_bps=50.0,
        note="平高电气：当前最强主模型；若已存在默认跳过。",
    ),
    SaveJob(
        stock_code="601899.SH",
        artifact_name="nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_zijin_v1",
        samples="saved_data/601899_pipeline_out/04_external/zijin_external/training_samples_with_zijin_external.csv",
        intraday_bars="saved_data/601899_pipeline_out/00_base/601899_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="extra_trees_600_d3",
        label_mode="close_profit",
        entry_policy="vwap_low",
        target_hit_bps=50.0,
        note="紫金矿业：新增vwap_low ExtraTrees稳健版本；可与已有all_days XGB并存。",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch save selected next-day model artifacts")
    p.add_argument("--project-root", default=".", help="Project root containing model_saving/ and saved_data/; default: current directory")
    p.add_argument("--out-dir", default="saved_models", help="Output model directory relative to project root; default: saved_models")
    p.add_argument("--python", default=sys.executable, help="Python executable used to run save_nextday_model.py")
    p.add_argument("--only", default="", help="Comma-separated stock codes to run, e.g. 002311.SZ,601899.SH")
    p.add_argument("--overwrite", action="store_true", help="Overwrite/retrain artifact if output directory already exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    p.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    p.add_argument("--valid-rows", type=int, default=252)
    p.add_argument("--min-train-entries", type=int, default=80)
    p.add_argument("--min-valid-trades", type=int, default=8)
    p.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    return p.parse_args()


def selected_jobs(only: str) -> list[SaveJob]:
    if not only.strip():
        return JOBS
    allow = {x.strip().upper() for x in only.split(",") if x.strip()}
    return [j for j in JOBS if j.stock_code.upper() in allow or j.stock_code.split(".")[0] in allow]


def rel_or_abs(project_root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return project_root / path


def build_cmd(args: argparse.Namespace, project_root: Path, job: SaveJob) -> list[str]:
    save_script = project_root / "model_saving" / "save_nextday_model.py"
    return [
        args.python,
        str(save_script),
        "--stock-code", job.stock_code,
        "--artifact-name", job.artifact_name,
        "--samples", str(rel_or_abs(project_root, job.samples)),
        "--intraday-bars", str(rel_or_abs(project_root, job.intraday_bars)),
        "--out-dir", str(rel_or_abs(project_root, args.out_dir)),
        "--feature-group", job.feature_group,
        "--model-name", job.model_name,
        "--label-mode", job.label_mode,
        "--entry-policy", job.entry_policy,
        "--entry-vwap-premium-bps", str(job.entry_vwap_premium_bps),
        "--target-hit-bps", str(job.target_hit_bps),
        "--round-trip-cost-bps", str(args.round_trip_cost_bps),
        "--valid-rows", str(args.valid_rows),
        "--min-train-entries", str(args.min_train_entries),
        "--min-valid-trades", str(args.min_valid_trades),
        "--quantiles", args.quantiles,
    ]


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    save_script = project_root / "model_saving" / "save_nextday_model.py"
    if not save_script.exists():
        print(f"[FATAL] Cannot find {save_script}", file=sys.stderr)
        return 2

    jobs = selected_jobs(args.only)
    if not jobs:
        print("[FATAL] No jobs selected", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = rel_or_abs(project_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"batch_save_nextday_models_{ts}.log"
    summary_json = out_dir / f"batch_save_nextday_models_{ts}.json"
    summary_csv = out_dir / f"batch_save_nextday_models_{ts}.csv"

    rows: list[dict] = []
    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] jobs = {len(jobs)}")
    print(f"[INFO] log = {log_path}")

    with log_path.open("w", encoding="utf-8") as log:
        for idx, job in enumerate(jobs, 1):
            artifact_dir = out_dir / job.stock_code / job.artifact_name
            sample_path = rel_or_abs(project_root, job.samples)
            intraday_path = rel_or_abs(project_root, job.intraday_bars)
            row = asdict(job) | {
                "index": idx,
                "artifact_dir": str(artifact_dir),
                "sample_path": str(sample_path),
                "intraday_path": str(intraday_path),
                "status": "pending",
                "returncode": None,
            }

            print("\n" + "=" * 100)
            print(f"[{idx}/{len(jobs)}] {job.stock_code} -> {job.artifact_name}")
            print(f"[NOTE] {job.note}")

            if artifact_dir.exists() and not args.overwrite:
                print(f"[SKIP] artifact exists: {artifact_dir}")
                row["status"] = "skipped_exists"
                rows.append(row)
                continue
            if not sample_path.exists():
                print(f"[ERROR] missing samples: {sample_path}")
                row["status"] = "missing_samples"
                rows.append(row)
                continue
            if not intraday_path.exists():
                print(f"[ERROR] missing intraday bars: {intraday_path}")
                row["status"] = "missing_intraday_bars"
                rows.append(row)
                continue

            cmd = build_cmd(args, project_root, job)
            printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
            print("[CMD]", printable)
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"[{idx}/{len(jobs)}] {job.stock_code} {job.artifact_name}\n")
            log.write("[CMD] " + printable + "\n")
            log.flush()

            if args.dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                continue

            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            row["returncode"] = proc.returncode
            log.write(proc.stdout or "")
            log.flush()
            if proc.returncode == 0:
                print(f"[OK] saved: {artifact_dir}")
                row["status"] = "ok"
            else:
                print(f"[FAIL] returncode={proc.returncode}; see log: {log_path}")
                row["status"] = "failed"
            rows.append(row)

    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print(f"[DONE] summary_json = {summary_json}")
    print(f"[DONE] summary_csv  = {summary_csv}")
    print(f"[DONE] log          = {log_path}")
    failed = [r for r in rows if r["status"] in {"failed", "missing_samples", "missing_intraday_bars"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
