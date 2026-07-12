#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for the shared AS1455 in-process grid runner.

Grid orchestration and rank caching live in ``utils.as1455_grid_runner``.
Portfolio simulation remains exclusively in the v7 backtest engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_grid_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
