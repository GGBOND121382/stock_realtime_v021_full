#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain existing saved_models safely from their own metadata.

Default:
  - scans saved_models/*/*/metadata.json
  - retrains every artifact with readable metadata and existing samples/intraday bars
  - dry-run unless wrapper sets APPLY=1
  - does not run pipeline
  - does not delete anything

Two output modes:
  1) default creates new artifact names with a suffix
  2) --replace-existing keeps the same artifact name, first moving old artifact to cleanup_trash
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]


def norm_symbol(s: str) -> str:
    s = str(s or "").strip().upper()
    if "." in s:
        code, mkt = s.split(".", 1)
        return f"{code.zfill(6)}.{mkt}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    if not code:
        return ""
    mkt = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{mkt}"


def resolve_path(raw: Any, stock_code: str = "") -> Optional[Path]:
    if raw is None:
        return None
    text = str(raw).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        return None
    p = Path(text)
    if p.exists():
        return p
    for marker in ["stock_realtime_v021_full/", "stock_realtime/"]:
        if marker in text:
            cand = ROOT / text.split(marker, 1)[1]
            if cand.exists():
                return cand
    if "saved_data/" in text:
        cand = ROOT / text[text.index("saved_data/") :]
        if cand.exists():
            return cand
    name = Path(text).name
    if name:
        code = norm_symbol(stock_code).split(".", 1)[0] if stock_code else ""
        roots = []
        if code:
            roots.extend(sorted((ROOT / "saved_data").glob(f"{code}_pipeline_out*")))
        roots.append(ROOT / "saved_data")
        for r in roots:
            if not r.exists():
                continue
            hits = list(r.rglob(name))
            if hits:
                return hits[0]
    return None


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    for i in range(1, 10000):
        cand = Path(f"{dest}_{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot create unique destination: {dest}")


@dataclass
class JobReport:
    stock_code: str
    source_artifact: str
    target_artifact: str
    status: str
    reason: str
    metadata_path: str
    samples: str = ""
    intraday_bars: str = ""
    log_path: str = ""
    backup_dir: str = ""
    returncode: str = ""


def selected_by_only(stock: str, only: set[str]) -> bool:
    if not only:
        return True
    raw = stock.split(".", 1)[0]
    return stock.upper() in only or raw.upper() in only


def build_jobs(args):
    models_dir = Path(args.models_dir)
    only = {x.strip().upper() for x in args.only.replace(";", ",").split(",") if x.strip()}
    reports: list[JobReport] = []
    jobs = []

    for meta_path in sorted(models_dir.glob("*/*/metadata.json")):
        artifact_dir = meta_path.parent
        source_artifact = artifact_dir.name
        stock = norm_symbol(artifact_dir.parent.name)
        if not selected_by_only(stock, only):
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            reports.append(JobReport(stock, source_artifact, "", "skip", f"metadata_read_error:{type(exc).__name__}:{exc}", str(meta_path)))
            continue

        stock = norm_symbol(meta.get("stock_code") or stock)
        samples = resolve_path(meta.get("samples"), stock)
        intraday = resolve_path(meta.get("intraday_bars"), stock)

        if samples is None or not samples.exists():
            reports.append(JobReport(stock, source_artifact, "", "skip", "samples_missing", str(meta_path), str(meta.get("samples", "")), str(meta.get("intraday_bars", ""))))
            continue
        if intraday is None or not intraday.exists():
            reports.append(JobReport(stock, source_artifact, "", "skip", "intraday_bars_missing", str(meta_path), str(samples), str(meta.get("intraday_bars", ""))))
            continue

        target_artifact = source_artifact if args.replace_existing else f"{source_artifact}_{args.artifact_suffix}"
        jobs.append((meta_path, meta, stock, source_artifact, target_artifact, samples, intraday))

    return jobs, reports


def run_job(job, args, out_dir: Path) -> JobReport:
    meta_path, meta, stock, source_artifact, target_artifact, samples, intraday = job
    models_dir = Path(args.models_dir)
    target_dir = models_dir / stock / target_artifact
    backup_dir = ""

    if target_dir.exists():
        if not args.replace_existing:
            return JobReport(stock, source_artifact, target_artifact, "skip_existing_target", "target_exists", str(meta_path), str(samples), str(intraday))
        backup_base = Path(args.trash_root) / f"model_refresh_backup_{datetime.now():%Y%m%d_%H%M%S}" / stock / target_artifact
        backup_base = unique_dest(backup_base)
        backup_base.parent.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            shutil.move(str(target_dir), str(backup_base))
        backup_dir = str(backup_base)

    log_path = out_dir / f"retrain_{stock.replace('.', '_')}_{target_artifact}.log"
    cmd = [
        sys.executable, "model_saving/save_nextday_model.py",
        "--stock-code", stock,
        "--artifact-name", target_artifact,
        "--samples", str(samples),
        "--intraday-bars", str(intraday),
        "--out-dir", args.models_dir,
        "--feature-group", str(meta.get("feature_group") or ""),
        "--model-name", str(meta.get("model_name") or ""),
        "--label-mode", str(meta.get("label_mode") or ""),
        "--entry-policy", str(meta.get("entry_policy") or ""),
        "--target-hit-bps", str(meta.get("target_hit_bps") or 50),
        "--entry-vwap-premium-bps", str(meta.get("entry_vwap_premium_bps") or 50),
        "--round-trip-cost-bps", str(meta.get("round_trip_cost_bps") or 1.7),
        "--valid-rows", str(args.valid_rows),
        "--min-train-entries", str(args.min_train_entries),
        "--min-valid-trades", str(args.min_valid_trades),
        "--quantiles", args.quantiles,
    ]

    if args.dry_run:
        log_path.write_text("DRY_RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return JobReport(stock, source_artifact, target_artifact, "dry_run", "dry_run", str(meta_path), str(samples), str(intraday), str(log_path), backup_dir, "")

    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    status = "ok" if proc.returncode == 0 else "failed"
    return JobReport(stock, source_artifact, target_artifact, status, status, str(meta_path), str(samples), str(intraday), str(log_path), backup_dir, str(proc.returncode))


def write_reports(out_dir: Path, reports: list[JobReport]):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "retrain_existing_models_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(JobReport("", "", "", "", "", "")).keys()))
        w.writeheader()
        for r in reports:
            w.writerow(asdict(r))
    summary = {"report_csv": str(csv_path), "counts": {}}
    for r in reports:
        summary["counts"][r.status] = summary["counts"].get(r.status, 0) + 1
    (out_dir / "retrain_existing_models_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--trash-root", default="cleanup_trash")
    ap.add_argument("--out-dir", default=f"saved_data/model_update_logs/retrain_existing_{datetime.now():%Y%m%d_%H%M%S}")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace-existing", action="store_true")
    ap.add_argument("--artifact-suffix", default=f"refresh_{datetime.now():%Y%m%d}")
    ap.add_argument("--valid-rows", type=int, default=126)
    ap.add_argument("--min-train-entries", type=int, default=80)
    ap.add_argument("--min-valid-trades", type=int, default=8)
    ap.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, reports = build_jobs(args)
    for job in jobs:
        reports.append(run_job(job, args, out_dir))
    write_reports(out_dir, reports)
    return 0 if not any(r.status == "failed" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
