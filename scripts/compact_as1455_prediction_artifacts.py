#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove duplicate prediction CSV sidecars and keep manifests truthful."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_artifact_retention import compact_prediction_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact AS1455 prediction artifacts")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = compact_prediction_dir(
        Path(args.prediction_dir),
        apply=not args.dry_run,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
