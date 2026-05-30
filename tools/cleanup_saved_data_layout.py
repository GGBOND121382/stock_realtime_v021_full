#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonicalize saved_data layout without deleting data.

This tool is intended for server-side cleanup after the asof1455 migration.
It never deletes files.  Top-level saved_data directories are validated by a
whitelist; anything outside the whitelist is moved into a timestamped recycle
directory under saved_data.

Canonical rules
---------------
- Per-stock data lives at saved_data/<6-digit-code>_pipeline_out.
- Asof1455 samples live at:
  saved_data/<code>_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv
- Default top-level whitelist:
  - saved_data/<6-digit-code>_pipeline_out
  - saved_data/ml4t_asof1455_lgbm_pipeline_out
  - saved_data/_recycle_*
- Retired copies such as 05_ml4t_asof1455 should not be read by default.

Example
-------
Dry-run first:

    python tools/cleanup_saved_data_layout.py --saved-data saved_data

Then execute:

    python tools/cleanup_saved_data_layout.py --saved-data saved_data --execute
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Pattern


CANONICAL_PIPELINE_RE = re.compile(r"^\d{6}_pipeline_out$")
EXPERIMENT_PIPELINE_RE = re.compile(r"^\d{6}_pipeline_out_.+")

DEFAULT_LEGACY_DIR_NAMES = {
    "asof1455_audit_minidata",
    "asof1455_dryrun_check",
    "asof1455_v1",
    "asof1455_week_check",
}

DEFAULT_RETIRED_STAGE_NAMES = {
    "05_ml4t_asof1455",
}

DEFAULT_TOP_DIR_WHITELIST = {
    "ml4t_asof1455_lgbm_pipeline_out",
}

DEFAULT_TOP_DIR_WHITELIST_REGEX = {
    r"^\d{6}_pipeline_out$",
    r"^_recycle_.*",
}


@dataclass
class MovePlan:
    kind: str
    source: str
    destination: str
    reason: str


def resolve_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"refusing path outside saved_data: {resolved}") from exc
    return resolved


def unique_destination(dst: Path) -> Path:
    if not dst.exists():
        return dst
    base = dst
    for i in range(1, 10000):
        candidate = base.with_name(f"{base.name}__{i:04d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot find unused destination for {dst}")


def canonical_pipeline_dirs(saved_data: Path) -> list[Path]:
    if not saved_data.exists():
        return []
    return sorted(
        p for p in saved_data.iterdir()
        if p.is_dir() and CANONICAL_PIPELINE_RE.fullmatch(p.name)
    )


def compile_patterns(values: Iterable[str]) -> list[Pattern[str]]:
    return [re.compile(v) for v in values if str(v).strip()]


def is_whitelisted_top_dir(name: str, exact_names: set[str], regexes: list[Pattern[str]]) -> bool:
    if name in exact_names:
        return True
    return any(p.fullmatch(name) for p in regexes)


def build_move_plan(
    saved_data: Path,
    recycle_dir: Path,
    whitelist_exact: set[str],
    whitelist_regexes: list[Pattern[str]],
    enforce_top_whitelist: bool,
    move_retired_stages: bool,
) -> list[MovePlan]:
    plans: list[MovePlan] = []
    planned_sources: set[Path] = set()

    if enforce_top_whitelist:
        for p in sorted(saved_data.iterdir() if saved_data.exists() else []):
            if not p.is_dir():
                continue
            if p.resolve() == recycle_dir.resolve():
                continue
            if is_whitelisted_top_dir(p.name, whitelist_exact, whitelist_regexes):
                continue
            dst = unique_destination(recycle_dir / p.name)
            reason = "top-level saved_data directory is not in whitelist"
            kind = "top_non_whitelisted_dir"
            if EXPERIMENT_PIPELINE_RE.fullmatch(p.name):
                kind = "top_experiment_pipeline_dir"
                reason = "non-canonical <code>_pipeline_out_* directory"
            elif p.name in DEFAULT_LEGACY_DIR_NAMES:
                kind = "top_legacy_dir"
                reason = "legacy/audit data root outside canonical per-stock pipeline dirs"
            plans.append(MovePlan(
                kind=kind,
                source=str(p),
                destination=str(dst),
                reason=reason,
            ))
            planned_sources.add(p.resolve())

    if move_retired_stages:
        for pipe in canonical_pipeline_dirs(saved_data):
            if pipe.resolve() in planned_sources:
                continue
            for stage_name in sorted(DEFAULT_RETIRED_STAGE_NAMES):
                stage = pipe / stage_name
                if stage.exists() and stage.is_dir():
                    dst = unique_destination(recycle_dir / pipe.name / stage.name)
                    plans.append(MovePlan(
                        kind="retired_stage",
                        source=str(stage),
                        destination=str(dst),
                        reason="retired asof1455 copy stage; canonical stage is 01_samples_asof1455",
                    ))

    return plans


def inspect_layout(saved_data: Path) -> dict:
    canonical = canonical_pipeline_dirs(saved_data)
    missing_asof = []
    present_asof = []
    retired_stage_present = []
    for pipe in canonical:
        sample = pipe / "01_samples_asof1455" / "training_samples_asof1455.csv"
        if sample.exists():
            present_asof.append(str(sample))
        else:
            missing_asof.append(pipe.name[:6])
        for stage_name in sorted(DEFAULT_RETIRED_STAGE_NAMES):
            stage = pipe / stage_name
            if stage.exists():
                retired_stage_present.append(str(stage))

    top_dirs = sorted(p.name for p in saved_data.iterdir() if p.is_dir()) if saved_data.exists() else []
    experiment_dirs = [name for name in top_dirs if EXPERIMENT_PIPELINE_RE.fullmatch(name)]
    legacy_dirs = [name for name in top_dirs if name in DEFAULT_LEGACY_DIR_NAMES]
    whitelist_exact = set(DEFAULT_TOP_DIR_WHITELIST)
    whitelist_regexes = compile_patterns(DEFAULT_TOP_DIR_WHITELIST_REGEX)
    non_whitelisted = [
        name for name in top_dirs
        if not is_whitelisted_top_dir(name, whitelist_exact, whitelist_regexes)
    ]
    return {
        "saved_data": str(saved_data),
        "canonical_pipeline_count": len(canonical),
        "canonical_pipeline_dirs": [p.name for p in canonical],
        "present_canonical_asof_samples": present_asof,
        "missing_canonical_asof_sample_codes": missing_asof,
        "top_experiment_pipeline_dirs": experiment_dirs,
        "top_legacy_asof_dirs": legacy_dirs,
        "top_non_whitelisted_dirs_default_policy": non_whitelisted,
        "retired_stage_dirs": retired_stage_present,
    }


def execute_moves(plans: list[MovePlan], saved_data: Path, recycle_dir: Path) -> None:
    saved_root = saved_data.resolve()
    recycle_dir.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        src = resolve_inside(Path(plan.source), saved_root)
        dst = Path(plan.destination)
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    before: dict,
    after: dict | None,
    plans: list[MovePlan],
) -> None:
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "execute": bool(args.execute),
        "saved_data": str(Path(args.saved_data).resolve()),
        "recycle_dir": str(path.parent.resolve()),
        "args": vars(args),
        "before": before,
        "after": after,
        "moves": [asdict(p) for p in plans],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move non-canonical saved_data directories into a recycle folder.")
    p.add_argument("--saved-data", default="saved_data", help="Path to saved_data. Default: saved_data")
    p.add_argument("--recycle-dir", default="", help="Recycle dir. Default: saved_data/_recycle_data_cleanup_<timestamp>")
    p.add_argument("--execute", action="store_true", help="Actually move files. Omit for dry-run.")
    p.add_argument("--no-top-whitelist", action="store_true", help="Disable top-level whitelist moves; only retired stages are handled.")
    p.add_argument("--keep-top-dir", action="append", default=[], help="Additional exact top-level saved_data dir name to keep; repeatable.")
    p.add_argument("--keep-top-regex", action="append", default=[], help="Additional top-level saved_data dir regex to keep; repeatable.")
    p.add_argument("--keep-retired-stages", action="store_true", help="Do not move 05_ml4t_asof1455 inside canonical dirs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    saved_data = Path(args.saved_data).resolve()
    if not saved_data.exists() or not saved_data.is_dir():
        raise FileNotFoundError(f"saved_data directory not found: {saved_data}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    recycle_dir = Path(args.recycle_dir).resolve() if args.recycle_dir else saved_data / f"_recycle_data_cleanup_{stamp}"
    resolve_inside(recycle_dir, saved_data)

    before = inspect_layout(saved_data)
    whitelist_exact = set(DEFAULT_TOP_DIR_WHITELIST)
    whitelist_exact.update(x.strip() for x in args.keep_top_dir if x.strip())
    whitelist_regexes = compile_patterns([*DEFAULT_TOP_DIR_WHITELIST_REGEX, *args.keep_top_regex])
    plans = build_move_plan(
        saved_data=saved_data,
        recycle_dir=recycle_dir,
        whitelist_exact=whitelist_exact,
        whitelist_regexes=whitelist_regexes,
        enforce_top_whitelist=not args.no_top_whitelist,
        move_retired_stages=not args.keep_retired_stages,
    )

    print(f"[INFO] saved_data={saved_data}")
    print(f"[INFO] recycle_dir={recycle_dir}")
    print(f"[INFO] top_whitelist_exact={sorted(whitelist_exact)}")
    print(f"[INFO] top_whitelist_regex={[p.pattern for p in whitelist_regexes]}")
    print(f"[INFO] canonical_pipeline_count={before['canonical_pipeline_count']}")
    print(f"[INFO] present_canonical_asof_samples={len(before['present_canonical_asof_samples'])}")
    print(f"[INFO] missing_canonical_asof_sample_codes={before['missing_canonical_asof_sample_codes']}")
    print(f"[INFO] planned_moves={len(plans)}")
    for plan in plans:
        print(f"[PLAN] {plan.kind}: {plan.source} -> {plan.destination} ({plan.reason})")

    if args.execute:
        execute_moves(plans, saved_data, recycle_dir)
        after = inspect_layout(saved_data)
        manifest_path = recycle_dir / "move_manifest.json"
        write_manifest(manifest_path, args, before, after, plans)
        print(f"[DONE] moved={len(plans)} manifest={manifest_path}")
        print(f"[DONE] remaining_experiment_dirs={after['top_experiment_pipeline_dirs']}")
        print(f"[DONE] remaining_legacy_asof_dirs={after['top_legacy_asof_dirs']}")
        print(f"[DONE] remaining_non_whitelisted_dirs_default_policy={after['top_non_whitelisted_dirs_default_policy']}")
        print(f"[DONE] remaining_retired_stage_dirs={len(after['retired_stage_dirs'])}")
    else:
        manifest_path = recycle_dir / "dry_run_manifest.json"
        write_manifest(manifest_path, args, before, None, plans)
        print(f"[DRY-RUN] no files moved. manifest={manifest_path}")
        print("[DRY-RUN] rerun with --execute to move planned paths into the recycle dir.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
