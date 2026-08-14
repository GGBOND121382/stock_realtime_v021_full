#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze one immutable model-generation snapshot for an AS1455 live day."""
from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    parser.add_argument("--out-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = bootstrap_registry(
        Path(args.registry_root), feature_preset=args.feature_preset
    )
    snapshot = write_active_snapshot(
        Path(args.out_file), registry, trade_date=args.trade_date
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "trade_date": snapshot["trade_date"],
                "active_generation": snapshot.get("active_generation"),
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
