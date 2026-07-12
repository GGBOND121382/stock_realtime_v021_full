#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retention helpers for duplicate AS1455 prediction sidecars."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def prediction_sidecar_candidates(prediction_dir: Path) -> list[Path]:
    prediction_dir = prediction_dir.expanduser().resolve()
    hdfs = [path for path in prediction_dir.glob("*.h5") if path.stat().st_size > 0]
    if not hdfs:
        return []
    candidates: list[Path] = []
    for csv_path in prediction_dir.glob("*.csv"):
        if csv_path.name.startswith("selected_"):
            continue
        if csv_path.name.startswith("actual_"):
            candidates.append(csv_path)
            continue
        hdf = csv_path.with_suffix(".h5")
        if hdf.exists() and hdf.stat().st_size > 0:
            candidates.append(csv_path)
    return sorted(set(candidates))


def update_prediction_manifests(
    prediction_dir: Path,
    removed_paths: list[Path],
) -> dict[str, Any]:
    prediction_dir = prediction_dir.expanduser().resolve()
    removed_resolved = {str(path.expanduser().resolve()) for path in removed_paths}
    removed_names = {path.name for path in removed_paths}
    updated: list[str] = []

    for manifest_path in prediction_dir.glob("*.json"):
        if manifest_path.name == "prediction_artifact_retention.json":
            continue
        try:
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False
        for key in ("prediction_csv", "actual_file"):
            value = obj.get(key)
            if not value:
                continue
            value_path = Path(str(value))
            if not value_path.is_absolute():
                value_path = (Path.cwd() / value_path).resolve()
            else:
                value_path = value_path.resolve()
            if str(value_path) in removed_resolved or value_path.name in removed_names:
                obj[f"{key}_removed_path"] = str(value)
                obj[f"{key}_retained"] = False
                obj[key] = None
                changed = True
        if changed:
            obj["artifact_retention_updated_at"] = datetime.now().isoformat(
                timespec="seconds"
            )
            manifest_path.write_text(
                json.dumps(obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            updated.append(str(manifest_path))

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_dir": str(prediction_dir),
        "canonical_format": "hdf",
        "removed_sidecars": [str(path) for path in removed_paths],
        "updated_manifests": updated,
    }
    (prediction_dir / "prediction_artifact_retention.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def compact_prediction_dir(prediction_dir: Path, apply: bool = True) -> dict[str, Any]:
    candidates = prediction_sidecar_candidates(prediction_dir)
    sizes = {str(path): int(path.stat().st_size) for path in candidates if path.exists()}
    if apply:
        for path in candidates:
            path.unlink(missing_ok=True)
        retention = update_prediction_manifests(prediction_dir, candidates)
    else:
        retention = {
            "prediction_dir": str(prediction_dir),
            "canonical_format": "hdf",
            "removed_sidecars": [str(path) for path in candidates],
            "updated_manifests": [],
        }
    retention["candidate_sizes"] = sizes
    retention["candidate_bytes"] = int(sum(sizes.values()))
    retention["applied"] = bool(apply)
    return retention
