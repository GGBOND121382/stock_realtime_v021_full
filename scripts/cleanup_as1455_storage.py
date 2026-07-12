#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and clean AS1455 storage without changing model semantics.

Dry-run is the default.  Every candidate action is constrained to the selected
base directory and recorded in a JSON manifest before the user applies it.
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
    "ch17_as1455_weekly_retrain_empty_2026-05-16_to_2026-06-26",
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

FORWARD_REDUNDANT = (
    "as1455_ohlcv_raw.h5",
    "as1455_ohlcv_adj.h5",
    "as1455_execution_metadata.h5",
)

LIVE_TRANSIENT = (
    "04_history_tail_raw.parquet",
    "04_history_tail_raw.csv",
    "05_history_tail_qfq_livebase.parquet",
    "05_history_tail_qfq_livebase.csv",
    "10_live_feature_panel_tail.parquet",
    "10_live_feature_panel_tail.csv",
)

KEEP_METRICS = (
    ("sharpe", False),
    ("calmar", False),
    ("total_return", False),
    ("annual_return", False),
    ("max_drawdown", False),
    ("avg_turnover", True),
    ("fee_to_initial_cash", True),
)


def bytes_on_disk(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        return int(path.lstat().st_size)
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for filename in files:
            candidate = Path(root) / filename
            try:
                total += int(candidate.lstat().st_size)
            except FileNotFoundError:
                pass
    return total


def disk_snapshot(path: Path) -> dict[str, float]:
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    gib = 1024 ** 3
    return {
        "total_gb": round(usage.total / gib, 3),
        "used_gb": round(usage.used / gib, 3),
        "free_gb": round(usage.free / gib, 3),
    }


def active_as1455_processes() -> list[dict[str, Any]]:
    current = os.getpid()
    rows: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.exists():
        return rows
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current:
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = command.lower()
        if "as1455" in lowered and "cleanup_as1455_storage.py" not in lowered:
            rows.append({"pid": int(entry.name), "cmdline": command})
    return rows


class Cleaner:
    def __init__(self, base: Path, apply: bool) -> None:
        self.base = base.expanduser().resolve()
        self.apply = apply
        self.actions: list[dict[str, Any]] = []

    def inside(self, path: Path) -> Path:
        resolved = path.expanduser().resolve(strict=False)
        if resolved == self.base or self.base not in resolved.parents:
            raise RuntimeError(f"refusing path outside cleanup base: {path}")
        return resolved

    def remove(self, path: Path, reason: str) -> None:
        resolved = self.inside(path)
        if not path.exists() and not path.is_symlink():
            return
        size = bytes_on_disk(path)
        self.actions.append(
            {"action": "delete", "path": str(resolved), "bytes": size, "reason": reason}
        )
        prefix = "DELETE" if self.apply else "DRY DELETE"
        print(f"[{prefix}] {size:>12} {resolved} :: {reason}")
        if not self.apply:
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def gzip_csv(self, path: Path, reason: str) -> None:
        resolved = self.inside(path)
        if not path.exists() or not path.is_file() or path.suffix == ".gz":
            return
        target = path.with_suffix(path.suffix + ".gz")
        size = bytes_on_disk(path)
        self.actions.append(
            {
                "action": "gzip",
                "path": str(resolved),
                "target": str(target.resolve(strict=False)),
                "bytes": size,
                "reason": reason,
            }
        )
        prefix = "GZIP" if self.apply else "DRY GZIP"
        print(f"[{prefix}] {size:>12} {resolved} :: {reason}")
        if not self.apply:
            return
        with path.open("rb") as source, gzip.open(target, "wb", compresslevel=9) as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        path.unlink()


def validate_forward_model_data(forward_dir: Path) -> dict[str, Any]:
    path = forward_dir / "model_data_as1455.h5"
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"forward model_data missing or empty: {path}")
    with pd.HDFStore(path, mode="r") as store:
        if "/model_data" not in store.keys():
            raise RuntimeError(f"missing /model_data key: {path}")
        sample = store.select("model_data", start=0, stop=10)
        rows = int(store.get_storer("model_data").nrows or 0)
    if list(sample.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"bad forward model_data index: {sample.index.names}")
    if sample.shape[1] != 34:
        raise RuntimeError(f"bad forward model_data width: {sample.shape[1]}")
    return {"path": str(path), "rows": rows, "columns": int(sample.shape[1])}


def cleanup_forward(cleaner: Cleaner) -> dict[str, Any] | None:
    root = cleaner.base / "ch12_as1455_forward_latest"
    if not root.exists():
        return None
    report = validate_forward_model_data(root)
    for filename in FORWARD_REDUNDANT:
        cleaner.remove(root / filename, "duplicate forward build intermediate")
    return report


def cleanup_live(cleaner: Cleaner, keep_dates: int) -> dict[str, Any]:
    root = cleaner.base / "live_as1455"
    if not root.exists():
        return {"all_dates": [], "kept_dates": []}
    dates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and len(path.name) == 8 and path.name.isdigit()
    )
    keep_count = max(1, int(keep_dates))
    kept = dates[-keep_count:]
    for path in dates[:-keep_count]:
        cleaner.remove(path, f"live retention keeps latest {keep_count} dates")
    latest = kept[-1] if kept else None
    for path in kept:
        if path == latest:
            continue
        for filename in LIVE_TRANSIENT:
            cleaner.remove(path / filename, "reproducible tail from an older retained live date")
    return {
        "all_dates": [path.name for path in dates],
        "kept_dates": [path.name for path in kept],
    }


def cleanup_prediction_sidecars(cleaner: Cleaner) -> int:
    count = 0
    for prediction_dir in cleaner.base.glob("**/00_predictions"):
        if not prediction_dir.is_dir():
            continue
        candidates = prediction_sidecar_candidates(prediction_dir)
        for path in candidates:
            cleaner.remove(path, "duplicate prediction CSV; HDF is canonical")
            count += 1
        if cleaner.apply and candidates:
            update_prediction_manifests(prediction_dir, candidates)
    return count


def selected_run_names(root: Path) -> set[str]:
    summary_path, _grid_dir = find_summary_file(root)
    frame = successful_rows(read_csv_auto(summary_path))
    keep: set[str] = set()
    for metric, ascending in KEEP_METRICS:
        if metric not in frame.columns:
            continue
        ranked = frame.copy()
        ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=[metric])
        if ranked.empty:
            continue
        ranked = ranked.sort_values(metric, ascending=ascending)
        keep.add(str(ranked.iloc[0]["run_name"]))
        if "signal_name" in ranked.columns:
            for _signal, group in ranked.groupby("signal_name"):
                keep.add(str(group.iloc[0]["run_name"]))

    materialized = root / "materialized_best_run.json"
    if materialized.exists():
        try:
            run_name = (
                json.loads(materialized.read_text(encoding="utf-8"))
                .get("selection", {})
                .get("run_name")
            )
            if run_name:
                keep.add(str(run_name))
        except Exception:
            pass
    for manifest in root.glob("**/strict_oos_manifest.json"):
        try:
            run_name = json.loads(manifest.read_text(encoding="utf-8")).get(
                "retained_run_name"
            )
            if run_name:
                keep.add(str(run_name))
        except Exception:
            pass
    return keep


def candidate_grid_roots(base: Path) -> list[Path]:
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


def prune_grid_runs(cleaner: Cleaner) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for root in candidate_grid_roots(cleaner.base):
        try:
            _summary_path, grid_dir = find_summary_file(root)
            keep = selected_run_names(root)
        except (FileNotFoundError, RuntimeError, pd.errors.EmptyDataError) as exc:
            reports.append({"root": str(root), "status": "skipped", "reason": str(exc)})
            continue
        runs_dir = grid_dir / "01_runs"
        deleted = 0
        if runs_dir.exists():
            for path in runs_dir.iterdir():
                if path.is_dir() and path.name not in keep:
                    cleaner.remove(path, "grid run not selected by retained audit metrics")
                    deleted += 1
        reports.append(
            {"root": str(root), "status": "ok", "kept": sorted(keep), "deleted": deleted}
        )
    return reports


def compress_large_reports(cleaner: Cleaner, minimum_mb: float) -> int:
    threshold = int(float(minimum_mb) * 1024 * 1024)
    count = 0
    for path in cleaner.base.glob("**/reports/*.csv"):
        try:
            eligible = path.stat().st_size >= threshold
        except FileNotFoundError:
            eligible = False
        if eligible:
            cleaner.gzip_csv(path, f"large audit CSV >= {minimum_mb:g} MiB")
            count += 1
    return count


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
        raise SystemExit("refusing --apply while AS1455 processes are active")

    cleaner = Cleaner(base, args.apply)
    before = disk_snapshot(base)
    details: dict[str, Any] = {"active_as1455_processes": active}

    if not args.skip_forward_artifacts:
        details["forward_validation"] = cleanup_forward(cleaner)
    if not args.skip_live:
        details["live"] = cleanup_live(cleaner, args.keep_live_dates)
    if not args.skip_prediction_csv:
        details["prediction_sidecar_candidates"] = cleanup_prediction_sidecars(cleaner)
    if args.include_obsolete:
        for dirname in OBSOLETE_DIRS:
            cleaner.remove(base / dirname, "explicitly classified obsolete AS1455 run")
    if args.prune_grid_runs:
        details["grid_pruning"] = prune_grid_runs(cleaner)
    if args.compress_reports:
        details["large_report_candidates"] = compress_large_reports(
            cleaner, args.compress_min_mb
        )

    after = disk_snapshot(base)
    estimated_delete = sum(
        int(action.get("bytes", 0))
        for action in cleaner.actions
        if action.get("action") == "delete"
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry_run",
        "base": str(base),
        "disk_before": before,
        "disk_after": after,
        "estimated_delete_bytes": estimated_delete,
        "estimated_delete_gb": round(estimated_delete / 1024 ** 3, 3),
        "actions": cleaner.actions,
        "details": details,
    }
    manifest = (
        Path(args.manifest)
        if args.manifest
        else base / f"cleanup_audit_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MANIFEST] {manifest}")
    print(
        f"[SUMMARY] mode={payload['mode']} actions={len(cleaner.actions)} "
        f"estimated_delete_gb={payload['estimated_delete_gb']} "
        f"free_before_gb={before['free_gb']} free_after_gb={after['free_gb']}"
    )


if __name__ == "__main__":
    main()
