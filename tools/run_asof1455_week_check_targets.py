#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch/build asof1455 features for selected stocks, then compare live scale.

This script is intentionally *not* a training script.  It only runs the stages
needed to answer one operational question:

    After the asof1455 feature policy changes, do the training samples and the
    14:55 watch/replay path produce the same feature values?

For each configured stock it:

1. Calls `pipelines/run_nextday_pipeline.py` with only data/feature stages:
   `update_data`, `samples`, `asof_samples`, `fundamental`, `sector`, and any
   configured external feature builders.
2. Writes each stock into the canonical per-stock output directory under
   `saved_data/<code>_pipeline_out`.
3. Runs `tools/compare_asof1455_training_vs_live.py` over the generated outputs.
   That comparator builds synthetic 14:55 watch rows from the same 5m bars and
   checks the selected feature group column by column.

The default target list is the 15-stock audit set from the migration work.  Use
`--targets` to restrict to a subset, e.g.:

    python tools/run_asof1455_week_check_targets.py \
      --targets 600584.SH,600919.SH,601818.SH,601899.SH,603308.SH,603986.SH

The script skips model search/retraining by design.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TargetConfig:
    """Static feature configuration for one stock.

    `sector_symbol` and `external` mirror the existing pipeline metadata.  They
    are kept explicit here so this audit can be re-run even when old
    `saved_data/*pipeline_out*` directories only contain summaries.
    """

    symbol: str
    sector_symbol: str
    external: str = ""

    @property
    def raw_code(self) -> str:
        return self.symbol.split(".", 1)[0]


# Default audit universe requested during the asof1455 migration.  Keep this
# list close to the top so future audits can review/update mappings quickly.
DEFAULT_TARGETS: list[TargetConfig] = [
    TargetConfig("002028.SZ", "电网设备", "storage_power"),
    TargetConfig("002128.SZ", "煤炭开采加工", "power_utility_rate"),
    TargetConfig("002261.SZ", "软件开发", "ai_compute"),
    TargetConfig("002311.SZ", "农产品加工", "feed,hog"),
    TargetConfig("002895.SZ", "农化制品", "fertilizer"),
    TargetConfig("600312.SH", "电网设备", ""),
    TargetConfig("600361.SH", "工业金属", "zijin_external"),
    TargetConfig("600487.SH", "通信设备", "optical_cable_grid"),
    TargetConfig("600522.SH", "通信设备", "optical_cable_grid"),
    TargetConfig("600584.SH", "半导体", "ai_compute"),
    TargetConfig("600919.SH", "银行", ""),
    TargetConfig("601818.SH", "银行", ""),
    TargetConfig("601899.SH", "贵金属", "zijin_external"),
    TargetConfig("603308.SH", "通用设备", "aero_nuclear_equipment"),
    TargetConfig("603986.SH", "半导体", "ai_compute"),
]


# run_nextday_pipeline stage names for each supported external profile.
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


def normalize_symbol(value: str) -> str:
    """Normalize user input such as `600584` or `sh.600584` to `600584.SH`."""
    text = str(value).strip().upper().replace("_", ".")
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ"}:
            code, market = right, left
        else:
            code, market = left, right
        return f"{''.join(ch for ch in code if ch.isdigit()).zfill(6)}.{market}"
    code = "".join(ch for ch in text if ch.isdigit()).zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def select_targets(raw_targets: str) -> list[TargetConfig]:
    """Return configured targets, optionally filtered by `--targets`."""
    if not raw_targets.strip():
        return DEFAULT_TARGETS
    wanted = {normalize_symbol(x) for x in raw_targets.replace(";", ",").split(",") if x.strip()}
    known = {t.symbol: t for t in DEFAULT_TARGETS}
    missing = sorted(wanted - set(known))
    if missing:
        raise ValueError(f"targets not configured: {missing}")
    return [known[s] for s in sorted(wanted)]


def external_steps(external: str) -> list[str]:
    """Split an external profile string such as `feed,hog` into clean names."""
    return [x.strip() for x in str(external or "").split(",") if x.strip()]


def feature_stage_names(target: TargetConfig) -> list[str]:
    """Build the minimal stage list needed for feature comparison.

    We deliberately omit `search` and `summarize` because this script should not
    train or rank models.
    """
    stages = ["update_data", "samples", "asof_samples", "fundamental", "sector"]
    for ext in external_steps(target.external):
        if ext not in EXTERNAL_STAGE_BY_NAME:
            raise ValueError(f"unsupported external profile for {target.symbol}: {ext}")
        stages.append(EXTERNAL_STAGE_BY_NAME[ext])
    return stages


def final_feature_file(out_root: Path, target: TargetConfig) -> Path:
    """Return the expected final sample file for quick resume checks."""
    if target.external:
        last_ext = external_steps(target.external)[-1]
        if last_ext == "hog":
            name = "training_samples_with_hog_industry.csv"
        elif last_ext == "feed":
            name = "training_samples_with_feed_external.csv"
        elif last_ext == "zijin_external":
            name = "training_samples_with_zijin_external.csv"
        else:
            name = f"training_samples_with_{last_ext}_external.csv"
        return out_root / target.raw_code / "04_external" / last_ext / name
    return out_root / target.raw_code / "03_sector" / "training_samples_with_sector.csv"


def pipeline_out_dir(base_out: Path, target: TargetConfig) -> Path:
    """Use one isolated directory per stock."""
    return base_out / f"{target.raw_code}_pipeline_out"


def command_for_target(args: argparse.Namespace, target: TargetConfig) -> list[str]:
    """Create the run_nextday_pipeline command for one stock."""
    out_dir = pipeline_out_dir(Path(args.out_root), target)
    cmd = [
        args.python,
        "pipelines/run_nextday_pipeline.py",
        "--symbol",
        target.symbol,
        "--sector-symbol",
        target.sector_symbol,
        "--out-root",
        str(out_dir),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--feature-time-mode",
        "asof1455",
        "--feature-cutoff-time",
        args.cutoff_time,
        "--feature-pipeline",
        "fundamental,sector",
        "--only-stages",
        ",".join(feature_stage_names(target)),
        "--skip-akshare-fund-flow",
        "--continue-on-error",
    ]
    if target.external:
        cmd.extend(["--external", target.external])
    if args.resume:
        # run_nextday_pipeline will skip stages with expected outputs when it
        # can.  We still keep a script-level final-file check for speed.
        cmd.append("--resume")
    return cmd


def run_subprocess(cmd: list[str], cwd: Path, dry_run: bool) -> dict:
    """Run a command and keep enough stdout for diagnostics."""
    if dry_run:
        return {"cmd": cmd, "returncode": 0, "status": "dry_run", "stdout_tail": ""}
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "status": "ok" if proc.returncode == 0 else "failed",
        "stdout_tail": (proc.stdout or "")[-8000:],
    }


def build_features(args: argparse.Namespace, targets: Iterable[TargetConfig]) -> list[dict]:
    """Fetch/build features for all selected targets."""
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for target in targets:
        stock_out = pipeline_out_dir(out_root, target)
        expected = final_feature_file(out_root, target)
        if args.resume and expected.exists():
            result = {
                "cmd": [],
                "returncode": 0,
                "status": "skipped_existing",
                "stdout_tail": "",
                "expected_final_feature_file": str(expected),
            }
        else:
            cmd = command_for_target(args, target)
            result = run_subprocess(cmd, ROOT, args.dry_run)
            result["expected_final_feature_file"] = str(expected)
        result.update({
            "symbol": target.symbol,
            "sector_symbol": target.sector_symbol,
            "external": target.external,
            "pipeline_out": str(stock_out),
        })
        reports.append(result)
        print(f"{target.symbol}: {result['status']} returncode={result['returncode']}", flush=True)
    return reports


def run_comparison(args: argparse.Namespace) -> dict:
    """Run the feature-scale comparator over generated pipeline outputs."""
    cmd = [
        args.python,
        "tools/compare_asof1455_training_vs_live.py",
        "--pipeline-root",
        str(Path(args.out_root)),
        "--dates",
        "auto",
        "--cutoff-time",
        args.cutoff_time,
        "--feature-group",
        args.feature_group,
        "--max-missing",
        str(args.max_missing),
        "--out-dir",
        str(Path(args.report_dir)),
    ]
    return run_subprocess(cmd, ROOT, args.dry_run)


def write_reports(report_dir: Path, build_report: list[dict], compare_report: dict) -> None:
    """Persist machine-readable run reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "build_report.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "compare_report.json").write_text(
        json.dumps(compare_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and compare asof1455 feature scale for selected targets")
    parser.add_argument("--targets", default="", help="Comma list of symbols. Empty means the default 15-stock set.")
    parser.add_argument("--out-root", default=str(ROOT / "saved_data"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports" / "asof1455_target_week_compare"))
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--cutoff-time", default="14:55")
    parser.add_argument("--feature-group", default="all_no_ak")
    parser.add_argument("--max-missing", type=float, default=0.35)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="Skip targets whose expected final feature CSV already exists.")
    parser.add_argument("--skip-compare", action="store_true", help="Only build feature data; do not run the comparator.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = select_targets(args.targets)
    build_report = build_features(args, targets)
    compare_report = {"status": "skipped", "returncode": 0}
    if not args.skip_compare:
        compare_report = run_comparison(args)
        print(f"compare: {compare_report['status']} returncode={compare_report['returncode']}", flush=True)
    write_reports(Path(args.report_dir), build_report, compare_report)
    failed = [r for r in build_report if r.get("returncode") not in {0, None}]
    if failed or compare_report.get("returncode", 0) != 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
