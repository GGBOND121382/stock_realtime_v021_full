#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-save v2 artifacts corresponding to retained existing/planned next-day models.

This script intentionally separates v2 artifacts from v1:
  - pipeline samples are read from saved_data/<code>_pipeline_out/...
  - default model output is saved_models_v2/ rather than saved_models/
  - every artifact_name ends with _v2

If you want the v2 artifacts under saved_models/ instead, pass:
  python3 scripts/batch_save_existing_models_v2.py --out-dir saved_models
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


@dataclass(frozen=True)
class SaveJob:
    stock_code: str
    artifact_name: str
    samples_template: str
    intraday_template: str
    feature_group: str
    model_name: str
    label_mode: str
    entry_policy: str
    target_hit_bps: float = 50.0
    entry_vwap_premium_bps: float = 50.0
    note: str = ""


# Retained set: excludes 中国巨石(600176) and 万华化学(600309).
# Includes both the actually saved 601899 all-days XGB v1 counterpart and the later selected vwap_low ExtraTrees candidate.
JOBS: list[SaveJob] = [
    SaveJob(
        stock_code="002270.SZ",
        artifact_name="nextday_all_days_close_profit_extra_trees_all_no_ak_v2",
        samples_template="saved_data/002270_pipeline_out/03_sector/training_samples_with_sector.csv",
        intraday_template="saved_data/002270_pipeline_out/00_base/002270_5m.csv",
        feature_group="all_no_ak",
        model_name="extra_trees_600_d3",
        label_mode="close_profit",
        entry_policy="all_days",
        note="华明装备 v2：对应保留观察模型；独立 pipeline/tag 输出。",
    ),
    SaveJob(
        stock_code="002311.SZ",
        artifact_name="nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_sector_hog_v2",
        samples_template="saved_data/002311_pipeline_out/04_external/hog/training_samples_with_hog_industry.csv",
        intraday_template="saved_data/002311_pipeline_out/00_base/002311_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="xgb_d3_600_lr002_mcw3",
        label_mode="close_profit",
        entry_policy="vwap_low",
        note="海大集团 v2：对应原 hog 样本主模型。",
    ),
    SaveJob(
        stock_code="002714.SZ",
        artifact_name="nextday_vwap_low_close_profit_random_forest_reversal_fundamental_regime_sector_muyuan_hk_v2",
        samples_template="saved_data/002714_pipeline_out/04_external/muyuan_hk/training_samples_with_hk_external.csv",
        intraday_template="saved_data/002714_pipeline_out/00_base/002714_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="random_forest_600_d4",
        label_mode="close_profit",
        entry_policy="vwap_low",
        note="牧原股份 v2：按保留要求重训；建议继续降权观察。",
    ),
    SaveJob(
        stock_code="600276.SH",
        artifact_name="nextday_all_days_hit80_extra_trees_reversal_fundamental_regime_sector_v2",
        samples_template="saved_data/600276_pipeline_out/03_sector/training_samples_with_sector.csv",
        intraday_template="saved_data/600276_pipeline_out/00_base/600276_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="extra_trees_600_d3",
        label_mode="hit",
        entry_policy="all_days",
        target_hit_bps=80.0,
        note="恒瑞医药 v2：80bps hit 模型。",
    ),
    SaveJob(
        stock_code="600312.SH",
        artifact_name="nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_v2",
        samples_template="saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv",
        intraday_template="saved_data/600312_pipeline_out/00_base/600312_5m.csv",
        feature_group="reversal_fundamental_regime",
        model_name="xgb_d3_400_lr003_mcw3",
        label_mode="close_profit",
        entry_policy="all_days",
        note="平高电气 v2：对应当前 saved_models 中的主模型。",
    ),
    SaveJob(
        stock_code="601899.SH",
        artifact_name="nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_sector_v2",
        samples_template="saved_data/601899_pipeline_out/04_external/zijin_external/training_samples_with_zijin_external.csv",
        intraday_template="saved_data/601899_pipeline_out/00_base/601899_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="xgb_d3_400_lr003_mcw3",
        label_mode="close_profit",
        entry_policy="all_days",
        note="紫金矿业 v2：对应当前 saved_models 中实际已有的 all_days XGB v1。",
    ),
    SaveJob(
        stock_code="601899.SH",
        artifact_name="nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_zijin_v2",
        samples_template="saved_data/601899_pipeline_out/04_external/zijin_external/training_samples_with_zijin_external.csv",
        intraday_template="saved_data/601899_pipeline_out/00_base/601899_5m.csv",
        feature_group="reversal_fundamental_regime_sector",
        model_name="extra_trees_600_d3",
        label_mode="close_profit",
        entry_policy="vwap_low",
        note="紫金矿业 v2：对应后续挑出的 vwap_low ExtraTrees 稳健候选。",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch save v2 next-day model artifacts")
    p.add_argument("--project-root", default=".", help="Project root containing model_saving/ and saved_data/")
    p.add_argument("--out-dir", default="saved_models_v2", help="Default isolates v2 from saved_models; pass saved_models to store together")
    p.add_argument("--python", default=sys.executable, help="Python executable used to run save_nextday_model.py")
    p.add_argument("--pipeline-run-tag", default="v2_models", help="Deprecated; paths now always read saved_data/<code>_pipeline_out")
    p.add_argument("--only", default="", help="Comma-separated stock codes or raw codes to run")
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


def rel_or_abs(project_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def materialize(template: str, tag: str) -> str:
    return template.format(tag=tag)


def build_cmd(args: argparse.Namespace, project_root: Path, job: SaveJob) -> list[str]:
    save_script = project_root / "model_saving" / "save_nextday_model.py"
    samples = rel_or_abs(project_root, materialize(job.samples_template, args.pipeline_run_tag))
    intraday = rel_or_abs(project_root, materialize(job.intraday_template, args.pipeline_run_tag))
    return [
        args.python,
        str(save_script),
        "--stock-code", job.stock_code,
        "--artifact-name", job.artifact_name,
        "--samples", str(samples),
        "--intraday-bars", str(intraday),
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
    log_path = out_dir / f"batch_save_existing_models_v2_{ts}.log"
    summary_json = out_dir / f"batch_save_existing_models_v2_{ts}.json"
    summary_csv = out_dir / f"batch_save_existing_models_v2_{ts}.csv"

    rows: list[dict] = []
    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] pipeline_run_tag = {args.pipeline_run_tag}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] jobs = {len(jobs)}")

    with log_path.open("w", encoding="utf-8") as log:
        for idx, job in enumerate(jobs, 1):
            artifact_dir = out_dir / job.stock_code / job.artifact_name
            samples = rel_or_abs(project_root, materialize(job.samples_template, args.pipeline_run_tag))
            intraday = rel_or_abs(project_root, materialize(job.intraday_template, args.pipeline_run_tag))
            row = asdict(job) | {
                "index": idx,
                "pipeline_run_tag": args.pipeline_run_tag,
                "artifact_dir": str(artifact_dir),
                "samples": str(samples),
                "intraday_bars": str(intraday),
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
            if not samples.exists():
                print(f"[ERROR] missing samples: {samples}")
                row["status"] = "missing_samples"
                rows.append(row)
                continue
            if not intraday.exists():
                print(f"[ERROR] missing intraday bars: {intraday}")
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

            proc = subprocess.run(cmd, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(proc.stdout or "")
            log.flush()
            row["returncode"] = proc.returncode
            row["status"] = "ok" if proc.returncode == 0 else "failed"
            rows.append(row)
            print(proc.stdout[-4000:] if proc.stdout else "")
            print(f"[STATUS] {row['status']} returncode={proc.returncode}")

    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] in {"ok", "skipped_exists", "dry_run"})
    print("\n" + "=" * 100)
    print(f"[SUMMARY] ok/skipped/dry_run={ok}/{len(rows)}")
    print(f"[LOG] {log_path}")
    print(f"[JSON] {summary_json}")
    print(f"[CSV] {summary_csv}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
