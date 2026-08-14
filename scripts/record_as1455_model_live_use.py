#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Commit one successful live day to the active AS1455 model period."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.as1455_model_registry import DEFAULT_REGISTRY_ROOT  # noqa: E402
from utils.as1455_model_roll import record_live_generation_use  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = record_live_generation_use(
        Path(args.registry_root),
        trade_date=args.trade_date,
        feature_preset=args.feature_preset,
    )
    period = registry.get("current_period") or {}
    print(
        json.dumps(
            {
                "status": "ok",
                "trade_date": args.trade_date,
                "active_generation": registry.get("active_generation"),
                "period_id": period.get("period_id"),
                "observed_days": period.get("observed_days"),
                "required_days": period.get("required_days"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
