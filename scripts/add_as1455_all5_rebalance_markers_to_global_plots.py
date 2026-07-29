#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add rebalance markers to fixed top-five ensemble global-fold plots."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import add_as1455_rebalance_markers_to_global_plots as base  # noqa: E402


def main() -> None:
    base.FIXED_SIGNAL_SPEC = "ensemble_all5_mean:0,1,2,3,4:mean"
    base.main()


if __name__ == "__main__":
    main()
