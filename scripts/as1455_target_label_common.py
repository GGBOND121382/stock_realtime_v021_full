#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for AS1455 target-aware helpers.

New code should import :mod:`utils.as1455_ch17_common`. This module preserves
the historical import path while keeping one implementation of target specs,
label filtering, and fold construction.
"""
from __future__ import annotations

from utils.as1455_ch17_common import (  # noqa: F401
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
