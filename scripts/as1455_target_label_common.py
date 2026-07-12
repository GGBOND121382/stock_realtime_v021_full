#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for AS1455 target-aware helpers.

New code should import :mod:`utils.as1455_ch17_common`. This module preserves
the historical import path while keeping one implementation of target specs,
label filtering, and fold construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_ch17_common import (  # noqa: E402,F401
    TARGET_SPECS,
    get_fold_target,
    load_xy_target,
    target_lookahead,
)

TARGET_LOOKAHEAD = {
    name: spec.lookahead for name, spec in TARGET_SPECS.items()
}

__all__ = [
    "TARGET_LOOKAHEAD",
    "get_fold_target",
    "load_xy_target",
    "target_lookahead",
]
