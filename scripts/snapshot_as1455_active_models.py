#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze one immutable model-generation snapshot for an AS1455 live day."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.as1455_model_registry import (  # noqa: E402
    DEFAULT_REGISTRY_ROOT,
    bootstrap_registry,
    write_active_snapshot,
)
from utils.as1455_model_roll import ensure_legacy_period_initialized  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    parser.add_argument("--out-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_root = Path(args.registry_root).expanduser().resolve()
    bootstrap_registry(registry_root, feature_preset=args.feature_preset)
    registry = ensure_legacy_period_initialized(
        registry_root, feature_preset=args.feature_preset
    )

    # A newly activated generation does not know its exact first trading date
    # until the next live run starts.  Put that date into the immutable day
    # snapshot provisionally, but do not mutate registry progress here.  The
    # pipeline records the date only after all nine plans finish successfully.
    snapshot_registry = copy.deepcopy(registry)
    date_text = str(args.trade_date)
    for entry in snapshot_registry.get("active_models", {}).values():
        if not entry.get("model_updated_date"):
            entry["model_updated_date"] = date_text
            entry["effective_from"] = date_text
    for generation in snapshot_registry.get("generations", []):
        if generation.get("generation_id") != snapshot_registry.get("active_generation"):
            continue
        if not generation.get("model_updated_date"):
            generation["model_updated_date"] = date_text
        for entry in (generation.get("targets") or {}).values():
            if not entry.get("model_updated_date"):
                entry["model_updated_date"] = date_text
                entry["effective_from"] = date_text

    snapshot = write_active_snapshot(
        Path(args.out_file), snapshot_registry, trade_date=args.trade_date
    )
    current_period = registry.get("current_period") or {}
    print(
        json.dumps(
            {
                "status": "ok",
                "trade_date": snapshot["trade_date"],
                "active_generation": snapshot.get("active_generation"),
                "period_id": current_period.get("period_id"),
                "period_progress_before_today": {
                    "observed_days": current_period.get("observed_days"),
                    "required_days": current_period.get("required_days"),
                },
                "targets": {
                    key: {
                        "generation_id": value.get("generation_id"),
                        "model_updated_date": value.get("model_updated_date"),
                        "model_dir": value.get("model_dir"),
                    }
                    for key, value in snapshot["targets"].items()
                },
                "out_file": str(Path(args.out_file).expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
