#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report whether the active AS1455 production period reached 63 trading days."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.as1455_model_registry import DEFAULT_REGISTRY_ROOT  # noqa: E402
from utils.as1455_model_roll import rollover_status  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    args = parser.parse_args()
    status = rollover_status(
        Path(args.registry_root), feature_preset=args.feature_preset
    )
    # Normally the nightly checker runs on day 63, so these two dates are equal.
    # If generation management is introduced after an old forward period has
    # already exceeded 63 days, catch up at the latest actually observed day
    # rather than pretending the new model was trained at a historical boundary.
    if status.get("due") and status.get("period_last_observed"):
        status["rollover_boundary"] = status["period_last_observed"]
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
