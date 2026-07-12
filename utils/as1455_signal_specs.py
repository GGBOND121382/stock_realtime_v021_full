#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared signal specifications derived from checkpoint count."""
from __future__ import annotations


def signal_specs_for_top_n(top_n: int) -> list[str]:
    """Return grid ``--signal-spec`` values valid for the prediction columns.

    With the standard five checkpoints this preserves the historical seven
    signals.  With ``top_n=1`` it correctly runs only ``model_0`` instead of
    asking the grid to load nonexistent model_1..model_4 columns.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    specs = [f"model_{index}:{index}:single" for index in range(top_n)]
    if top_n >= 3:
        specs.append("ensemble_first3_mean:0,1,2:mean")
    if top_n >= 2:
        columns = ",".join(str(index) for index in range(top_n))
        name = "ensemble_all5_mean" if top_n == 5 else f"ensemble_all{top_n}_mean"
        specs.append(f"{name}:{columns}:mean")
    return specs


def append_signal_specs(command: list[str], top_n: int) -> list[str]:
    out = list(command)
    for spec in signal_specs_for_top_n(top_n):
        out.extend(["--signal-spec", spec])
    return out
