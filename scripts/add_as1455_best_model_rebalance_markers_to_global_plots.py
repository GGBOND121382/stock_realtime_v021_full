#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add rebalance markers to the global best-model experiment plots."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import add_as1455_rebalance_markers_to_global_plots as shared  # noqa: E402

FIXED_SIGNAL_SPEC = "model_0:0:single"


def main() -> None:
    shared.FIXED_SIGNAL_SPEC = FIXED_SIGNAL_SPEC
    shared.main()


if __name__ == "__main__":
    main()
