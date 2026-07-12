#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and clean AS1455 storage without changing model semantics.

The default is dry-run.  Use ``--apply`` only after reviewing the emitted JSON
manifest.  Every deletion is constrained to ``saved_data/ashare_ml4t`` (or the
explicit ``--base``) and is recorded with its pre-delete byte size.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_artifact_retention import (  # noqa: E402
    prediction_sidecar_candidates,
    update_prediction_manifests,
)
from utils.as1455_model_selection import (  # noqa: E402
    find_summary_file,
    read_csv_auto,
    successful_rows,
)

OBSOLETE_DIRS = (
    "ch17_as1455_sector_rotation_onehot_fold0_check",
    "as1455_sector_audit",
    "as1455_sector_audit_v2",
    "ch17_as1455_weekly_retrain_top5_full_20260625_152927",
    "live_as1455_probe",
    "ch17_as1455_weekly_retrain_empty_2026-05-16_to_2026-26",
    "ch17_as1455_weekly_retrain_empty_2026-05-16_to_2026-06-26_clean_adj",
    "ch17_as1455_smoke",
    "ch17_nn_smoke",
    "ch17_as1455_sector_rotation_onehot_fold0_smoke",
    "ch17_as1455_top5_full_20260625_144220",
    "ch17_as1455_sector_rotation_onehot_fold0_full",
    "ch17_as1455_backtest_grid_v7_smoke",
    "ch17_as1455_first_batch_features_compact_fold0_search",
    "as1455_baostock_audit_20260630",
    "ch17_as1455_sector_onehot_fold_test",
    "ch17_as1455_weekly_retrain_top5_full_20260625_163510",
    "ch17_as1455_close_auction_bt_cv7_v3",
    "ch17_as1455_backtest_grid_v7_models_smoke",
    "ch17_as1455_train_20260622_cv7",
    "ch17_as1455_backtest_grid_v7_models_20260625",
    "ch17_as1455_train_latest_cv7",
)

LIVE_TRANSIENT_FILES = (
    "04_history_tail_raw.parquet",
    "04_history_tail_raw.csv",
    "05_history_tail_qfq_livebase.parquet",
    "05_history_tail_qfq_livebase.csv",
    "10_live_feature_panel_tail.parquet",
    "10_live_feature_panel_tail.csv",
)

FORWARD_REDUNDANT_ARTIFACTS = (
    "as1455_ohlcv_raw.h5",
    "as1455_ohlcv_adj.h5",
    "as1455_execution_metadata.h5",
)

RANK_METRICS = (
    ("sharpe", False),
    ("calmar", False),
    ("total_return", False),
    ("annual_return", False),
    ("max_drawdown", False),
    ("avg_turnover", True),
    ("fee_to_initial_cash", True),
)


def path_size(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return int(path.lstat().st_size)
        except FileNotFoundError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            candidate = Path(root) / name
            try:
                total += int(candidate.lstat().st_size)
            except FileNotFoundError:
                pass
    return total


def active_as1455_processes() -> list[dict[str, Any]]:
    current = os.getpid()
    rows: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.exists():
        return rows
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == current:
            continue
        try:
            raw = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = raw.lower()
        if "as1455" in lowered and "cleanup_as1455_storage.py" not in lowered:
            rows.append({"pid": int(item.name), "cmdline": raw.strip()})
    return rows


def validate_forward_model_data(forward_dir: Path) -> dict[str, Any]:
    path = forward_dir / "model_data_as1455.h5"
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"forward model_data missing or empty: {path}")
    with pd.HDFStore(path, mode="r") as store:
        if "/model_data" not in store.keys():
            raise RuntimeError(f"missing /model_data key: {path}")
        sample = store.select("model_data", start=0, stop=10)
        storer = store.get_storer("model_data")
        rows = int(storer.nrows or 0)
    if list(sample.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"bad forward model_data index: {sample.index.names}")
    if sample.shape[1] != 34:
        raise RuntimeError(f"bad forward model_data width: {sample.shape[1]}")
    return {"path": str(path), "rows": rows, "columns": int(sample.shape[1])}


class Cleaner:
    def __init__(self, base: Path, apply: bool) -> None:
        self.base = base.expanduser().resolve()
        self.apply = apply
        self.actions: list[dict[str, Any]] = []

    def _assert_inside(self, path: Path) -> Path:
        resolved = path.expanduser().resolve(strict=False)
        if resolved == self.base or self.base not in resolved.parents:
            raise RuntimeError(f"refusing path outside cleanup base: {path}")
        return resolved

    def remove(self, path: Path, reason: str) -> None:
        resolved = self._assert_inside(path)
        if not path.exists() and not path.is_symlink():
            return
        size = path_size(path)
        self.actions.append(
            {"action": "delete", "path": str(resolved), "bytes": size, "reason": reason}
        )
        print(f"[{'DELETE' if self.apply else 'DRY DELETE'}] {size:>12} {resolved} :: {reason}")
        if not self.apply:
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def gzip_file(self, path: Path, reason: str) -> None:
        resolved = self._assert_inside(path)
        if not path.exists() or not path.is_file() or path.suffix == ".gz":
            return
        size = path_size(path)
        target = path.with_suffix(path.suffix + ".gz")
        self.actions.append(
            {
                "action": "gzip",
                "path": str(resolved),
                "target": str(target.resolve(strict=False)),
                "bytes": size,
                "reason": reason,
            }
        )
        print(f"[{'GZIP' if self.apply else 'DRY GZIP'}] {size:>12} {resolved} :: {reason}")
        if not self.apply:
            return
        with path.open("rb") as source, gzip.open(target, "wb", compresslevel=9) as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        path.unlink()


def cleanup_forward(cleaner: Cleaner) -> dict[str, Any] | None:
    forward = cleaner.base / "ch12_as1455_forward_latest"
    if not forward.exists():
        return None
    validation = validate_forward_model_data(forward)
    for name in FORWARD_REDUNDANT_ARTIFACTS:
        cleaner.remove(forward / name, "duplicate forward build intermediate; source caches are shared")
    return validation


def cleanup_live(cleaner: Cleaner, keep_dates: int) -> dict[str, Any]:
    live_root = cleaner.base / "live_as1455"
    if not live_root.exists():
        return {"live_dates": [], "kept": []}
    dates = sorted(
        path for path in live_root.iterdir()
        if path.is_dir() and len(path.name) == 8 and path.name.isdigit()
    )
    keep_count = max(1, int(keep_dates))
    kept = dates[-keep_count:]
    for path in dates[:-keep_count]:
        cleaner.remove(path, f"live retention keeps latest {keep_count} date directories")
    latest = kept[-1] if kept else None
    for path in kept:
        if path == latest:
            continue
        for name in LIVE_TRANSIENT_FILES:
            cleaner.remove(path / name, "old retained live date: remove reproducible history tail")
    return {"live_dates": [p.name for p in dates], "kept": [p.name for p in kept]}


def cleanup_prediction_csv(cleaner: Cleaner) -> int:
    removed = 0
    for prediction_dir in cleaner.base.glob("**/00_predictions"):
        if not prediction_dir.is_dir():
            continue
        candidates = prediction_sidecar_candidates(prediction_dir)
        for path in candidates:
            cleaner.remove(path, "duplicate prediction CSV; HDF is canonical")
            removed += 1
        if cleaner.apply and candidates:
            update_prediction_manifests(prediction_dir, candidates)
    return removed


def selected_run_names(root: Path) -> set[str]:
    summary_file, _grid_dir = find_summary_file(root)
    frame = successful_rows(read_csv_auto(summary_file))
    keep: set[str] = set()
    for metric, ascending in RANK_METRICS:
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        valid = frame.loc[values.notna()].copy()
        if valid.empty:
            continue
        valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
        keep.add(str(valid.sort_values(metric, ascending=ascending).iloc[0]["run_name"]))
        if "signal_name" in valid.columns:
            for _signal, group in valid.groupby("signal_name"):
                keep.add(str(group.sort_values(metric, ascending=ascending).iloc[0]["run_name"]))

    path = root / "materialized_best_run.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            run_name = (obj.get("selection") or {}).get("run_name")
            if run_name:
                keep.add(str(run_name))
        except Exception:
            pass
    for path in root.glob("**/strict_oos_manifest.json"):
        try:
            run_name = json.loads(path.read_text(encoding="utf-8")).get("retained_run_name")
            if run_name:
                keep.add(str(run_name))
        except Exception:
            pass
    return keep


def grid_roots(base: Path) -> list[Path]:
    roots: list[Path] = []
    for parent_name in (
        "ch17_as1455_target_backtest",
        "ch17_as1455_fold0_forward_backtest",
    ):
        parent = base / parent_name
        if parent.exists():
            roots.extend(path for path in parent.iterdir() if path.is_dir())
    roots.extend(base.glob("ch17_as1455_rotation_*_one_lag_daily_backtest_*"))
    return sorted(set(roots))


def prune_grid_runs(cleaner: Cleaner) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for root in grid_roots(cleaner.base):
        try:
            _summary, grid_dir = find_summary_file(root)
            keep = selected_run_names(root)
        except (FileNotFoundError, RuntimeError, pd.errors.EmptyDataError) as exc:
            reports.append({"root": str(root), "status": "skipped", "reason": str(exc)})
            continue
        runs_root = grid_dir / "01_runs"
        if not runs_root.exists():
            continue
        deleted = 0
        for path in runs_root.iterdir():
            if path.is_dir() and path.name not in keep:
                cleaner.remove(path, "grid run not selected by retained audit metrics")
                deleted += 1
        reports.append(
            {"root": str(root), "status": "ok", "kept": sorted(keep), "deleted": deleted}
        )
    return {"roots": reports}


def compress_reports(cleaner: Cleaner, min_size_mb: float) -> int:
    threshold = int(min_size_mb * 1024 * 1024)
    count = 0
    for path in cleaner.base.glob("**/reports/*.csv"):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size >= threshold:
            cleaner.gzip_file(path, f"large audit CSV >= {min_size_mb:g} MiB")
            count += 1
    return count


def disk_snapshot(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    gib = 1024 ** 3
    return {
        "total_gb": round(usage.total / gib, 3),
        "used_gb": round(usage.used / gib, 3),
        "free_gb": round(usage.free / gib, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and clean AS1455 storage")
    parser.add_argument("--base", default="saved_data/ashare_ml4t")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-live-dates", type=int, default=3)
    parser.add_argument("--skip-forward-artifacts", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--skip-prediction-csv", action="store_true")
    parser.add_argument("--include-obsolete", action="store_true")
    parser.add_argument("--prune-grid-runs", action="store_true")
    parser.add_argument("--compress-reports", action="store_true")
    parser.add_argument("--compress-min-mb", type=float, default=20.0)
    parser.add_argument("--allow-active-processes", action="store_true")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"cleanup base does not exist: {base}")
    active = active_as1455_processes()
    if args.apply and active and not args.allow_active_processes:
        print(json.dumps({"active_as1455_processes": active}, ensure_ascii=False, indent=2))
        raise SystemExit(
            "refusing --apply while AS1455 processes are active; stop them or use "
            "--allow-active-processes after manual verification"
        )

    cleaner = Cleaner(base, args.apply)
    before = disk_snapshot(base)
    details: dict[str, Any] = {"active_as1455_processes": active}

    if not args.skip_forward_artifacts:
        details["forward_validation"] = cleanup_forward(cleaner)
    if not args.skip_live:
        details["live"] = cleanup_live(cleaner, args.keep_live_dates)
    if not args.skip_prediction_csv:
        details["prediction_csv_candidates"] = cleanup_prediction_csv(cleaner)
    if args.include_obsolete:
        for name in OBSOLETE_DIRS:
            cleaner.remove(base / name, "explicitly classified obsolete AS1455 run")
    if args.prune_grid_runs:
        details["grid_pruning"] = prune_grid_runs(cleaner)
    if args.compress_reports:
        details["report_gzip_candidates"] = compress_reports(
            cleaner, args.compress_min_mb
        )

    after = disk_snapshot(base)
    estimated = sum(
        int(row.get("bytes", 0))
        for row in cleaner.actions
        if row.get("action") == "delete"
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry_run",
        "base": str(base),
        "disk_before": before,
        "disk_after": after,
        "estimated_delete_bytes": estimated,
        "estimated_delete_gb": round(estimated / 1024 ** 3, 3),
        "actions": cleaner.actions,
        "details": details,
    }
    manifest = (
        Path(args.manifest)
        if args.manifest
        else base / f"cleanup_audit_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[MANIFEST] {manifest}")
    print(
        f"[SUMMARY] mode={payload['mode']} actions={len(cleaner.actions)} "
        f"estimated_delete_gb={payload['estimated_delete_gb']} "
        f"free_before_gb={before['free_gb']} free_after_gb={after['free_gb']}"
    )


if __name__ == "__main__":
    main()
