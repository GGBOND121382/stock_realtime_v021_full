#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical project paths for AS1455 Chapter-17 workflows."""
from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t"
CH12_AS1455_DIR = SAVED_DATA_DIR / "ch12_as1455"
DEFAULT_MODEL_DATA = CH12_AS1455_DIR / "model_data_as1455.h5"
DEFAULT_RAW_DAILY_CACHE_DIR = CH12_AS1455_DIR / "baostock_raw_daily_cache"
DEFAULT_GRID_SCRIPT = (
    PROJECT_DIR
    / "code"
    / "backtest"
    / "run_as1455_close_auction_grid_inprocess.py"
)
TARGET_SEARCH_ROOT = SAVED_DATA_DIR / "ch17_as1455_target_search"
TARGET_BACKTEST_ROOT = SAVED_DATA_DIR / "ch17_as1455_target_backtest"
FOLD0_FORWARD_ROOT = SAVED_DATA_DIR / "ch17_as1455_fold0_forward_backtest"
PLOT_ROOT = SAVED_DATA_DIR / "ch17_as1455_backtest_plots"
FORWARD_MODEL_DIR = SAVED_DATA_DIR / "ch12_as1455_forward_latest"
FORWARD_MODEL_DATA = FORWARD_MODEL_DIR / "model_data_as1455.h5"
