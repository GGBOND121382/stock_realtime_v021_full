#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production model registry for rolling AS1455 live inference.

Historical ``fold0..fold6`` keep their original CV meaning. Production models
are addressed by monotonically increasing ``genNNN`` generation ids instead.
``gen000`` is a compatibility view over the existing fold0 model directories;
it does not move or copy any historical artifacts.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils import as1455_ch17_common as common

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_ROOT = (
    PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "ch17_as1455_model_registry"
)
REGISTRY_SCHEMA_VERSION = 1
LEGACY_GENERATION = "gen000"
DEFAULT_PERIOD_LENGTH = 63
DEFAULT_TRAIN_LENGTH = 21 * 12 * 4
GEN_RE = re.compile(r"^gen(?P<index>\d{3,})$")
PERIOD_RE = re.compile(r"^period(?P<index>\d{3,})$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def generation_index(generation_id: str) -> int:
    match = GEN_RE.fullmatch(str(generation_id))
    if match is None:
        raise RuntimeError(f"invalid model generation id: {generation_id!r}")
    return int(match.group("index"))


def generation_id(index: int) -> str:
    if int(index) < 0:
        raise ValueError("generation index must be non-negative")
    return f"gen{int(index):03d}"


def period_id(index: int) -> str:
    if int(index) < 0:
        raise ValueError("period index must be non-negative")
    return f"period{int(index):03d}"


def _legacy_fold_reference_date(model_dir: Path) -> str | None:
    """Return fold0's historical validation boundary, not production use date."""
    manifest_path = model_dir / "preprocess" / "feature_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return None
    value = manifest.get("test_end") or manifest.get("train_end")
    if not value:
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def legacy_target_entry(feature_preset: str, target_col: str) -> dict[str, Any]:
    model_dir = common.default_fold0_dir(feature_preset, target_col).resolve()
    return {
        "generation_id": LEGACY_GENERATION,
        "source_type": "legacy_cv_fold",
        "source_fold": 0,
        "model_dir": str(model_dir),
        # The production activation date is filled by the one-time legacy
        # forward reconciliation. Never use fold0 test_end as a fake update date.
        "model_updated_date": None,
        "effective_from": None,
        "legacy_fold_reference_date": _legacy_fold_reference_date(model_dir),
        "trained_at": None,
        "train_start": None,
        "train_end": None,
        "label_valid_end": None,
    }


def virtual_legacy_registry(feature_preset: str) -> dict[str, Any]:
    targets = {
        target_col: legacy_target_entry(feature_preset, target_col)
        for target_col in common.TARGET_SPECS
    }
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "feature_preset": feature_preset,
        "period_length": DEFAULT_PERIOD_LENGTH,
        "train_length": DEFAULT_TRAIN_LENGTH,
        "created_at": None,
        "updated_at": None,
        "virtual_legacy": True,
        "active_generation": LEGACY_GENERATION,
        "active_models": targets,
        "generations": [
            {
                "generation_id": LEGACY_GENERATION,
                "generation_index": 0,
                "source_type": "legacy_cv_fold",
                "source_fold": 0,
                "model_updated_date": None,
                "targets": targets,
            }
        ],
        "current_period": {
            "period_id": period_id(0),
            "period_index": 0,
            "generation_id": LEGACY_GENERATION,
            "start_date": None,
            "observed_dates": [],
            "observed_days": 0,
            "required_days": DEFAULT_PERIOD_LENGTH,
            "legacy_cache_initialized": False,
        },
    }


def validate_registry(registry: dict[str, Any]) -> None:
    if int(registry.get("schema_version", 0) or 0) != REGISTRY_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported model registry schema: {registry.get('schema_version')}"
        )
    active = registry.get("active_models")
    if not isinstance(active, dict):
        raise RuntimeError("model registry active_models must be an object")
    missing = [target for target in common.TARGET_SPECS if target not in active]
    if missing:
        raise RuntimeError(f"model registry misses active targets: {missing}")
    for target_col, entry in active.items():
        if target_col not in common.TARGET_SPECS:
            continue
        if not isinstance(entry, dict):
            raise RuntimeError(f"active model entry must be an object: {target_col}")
        generation_index(str(entry.get("generation_id")))
        if not entry.get("model_dir"):
            raise RuntimeError(f"active model entry lacks model_dir: {target_col}")


def load_registry(
    registry_root: Path | str | None = None,
    *,
    feature_preset: str = "rotation_addon_onehot",
    allow_virtual_legacy: bool = True,
) -> dict[str, Any]:
    root = Path(registry_root or DEFAULT_REGISTRY_ROOT).expanduser().resolve()
    path = root / "registry.json"
    if not path.is_file():
        if not allow_virtual_legacy:
            raise FileNotFoundError(path)
        return virtual_legacy_registry(feature_preset)
    registry = read_json(path)
    validate_registry(registry)
    return registry


def bootstrap_registry(
    registry_root: Path | str | None = None,
    *,
    feature_preset: str = "rotation_addon_onehot",
) -> dict[str, Any]:
    root = Path(registry_root or DEFAULT_REGISTRY_ROOT).expanduser().resolve()
    path = root / "registry.json"
    if path.is_file():
        registry = read_json(path)
        validate_registry(registry)
        return registry
    registry = virtual_legacy_registry(feature_preset)
    now = _now()
    registry["created_at"] = now
    registry["updated_at"] = now
    registry["virtual_legacy"] = False
    atomic_write_json(path, registry)
    return registry


def resolve_active_model(
    registry: dict[str, Any],
    target_col: str,
) -> dict[str, Any]:
    validate_registry(registry)
    if target_col not in common.TARGET_SPECS:
        raise RuntimeError(f"unsupported target_col={target_col!r}")
    entry = dict(registry["active_models"][target_col])
    model_dir = Path(str(entry["model_dir"])).expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    entry["model_dir"] = str(model_dir.resolve())
    entry["target_col"] = target_col
    return entry


def build_active_snapshot(
    registry: dict[str, Any],
    *,
    trade_date: str,
) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {}
    for target_col in common.TARGET_SPECS:
        entry = resolve_active_model(registry, target_col)
        model_dir = Path(entry["model_dir"])
        required = [
            model_dir / "preprocess" / "scaler.pkl",
            model_dir / "preprocess" / "feature_manifest.json",
            model_dir / "search_best_checkpoints.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"active model bundle incomplete for {target_col}: {missing}"
            )
        targets[target_col] = entry
    return {
        "schema_version": 1,
        "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
        "created_at": _now(),
        "registry_schema_version": registry.get("schema_version"),
        "registry_updated_at": registry.get("updated_at"),
        "active_generation": registry.get("active_generation"),
        "targets": targets,
    }


def write_active_snapshot(
    out_file: Path | str,
    registry: dict[str, Any],
    *,
    trade_date: str,
) -> dict[str, Any]:
    snapshot = build_active_snapshot(registry, trade_date=trade_date)
    atomic_write_json(Path(out_file), snapshot)
    return snapshot


def load_snapshot(path: Path | str) -> dict[str, Any]:
    snapshot = read_json(Path(path))
    targets = snapshot.get("targets")
    if not isinstance(targets, dict):
        raise RuntimeError(f"model snapshot lacks targets: {path}")
    missing = [target for target in common.TARGET_SPECS if target not in targets]
    if missing:
        raise RuntimeError(f"model snapshot misses targets: {missing}")
    return snapshot


def snapshot_model(snapshot: dict[str, Any], target_col: str) -> dict[str, Any]:
    targets = snapshot.get("targets", {})
    if target_col not in targets:
        raise RuntimeError(f"model snapshot has no target {target_col}")
    entry = dict(targets[target_col])
    if not entry.get("model_dir"):
        raise RuntimeError(f"model snapshot target lacks model_dir: {target_col}")
    return entry


def target_col_from_experiment(experiment: str) -> str | None:
    prefix = str(experiment).split("_", 1)[0]
    target_col = f"{prefix}_fwd"
    return target_col if target_col in common.TARGET_SPECS else None


def model_display_for_experiment(
    registry: dict[str, Any], experiment: str
) -> dict[str, Any]:
    target_col = target_col_from_experiment(experiment)
    if target_col is None:
        return {
            "model_generation": None,
            "model_updated_date": None,
            "model_source_type": None,
        }
    entry = resolve_active_model(registry, target_col)
    return {
        "model_generation": entry.get("generation_id"),
        "model_updated_date": entry.get("model_updated_date")
        or entry.get("effective_from"),
        "model_source_type": entry.get("source_type"),
        "model_train_start": entry.get("train_start"),
        "model_train_end": entry.get("train_end"),
        "model_trained_at": entry.get("trained_at"),
        "model_target_col": target_col,
        "legacy_fold_reference_date": entry.get("legacy_fold_reference_date"),
    }
