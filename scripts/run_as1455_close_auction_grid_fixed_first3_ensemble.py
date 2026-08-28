#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Force the AS1455 close-auction grid to use the top-three mean ensemble.

This adapter keeps the standard grid engine and all execution assumptions intact,
but replaces every incoming ``--signal-spec`` with exactly one fixed signal:
``ensemble_first3_mean:0,1,2:mean``. It is intended for nested validation
experiments where model selection is fixed and only trading parameters are tuned.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_GRID_SCRIPT = (
    PROJECT_DIR / "code" / "backtest" / "run_as1455_close_auction_grid_inprocess.py"
)
FIXED_SIGNAL_SPEC = "ensemble_first3_mean:0,1,2:mean"


def replace_signal_specs(argv: list[str]) -> list[str]:
    """Remove all supplied signal specs and append the fixed first-three ensemble."""
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--signal-spec":
            if index + 1 >= len(argv):
                raise SystemExit("--signal-spec requires a value")
            index += 2
            continue
        if token.startswith("--signal-spec="):
            index += 1
            continue
        filtered.append(token)
        index += 1
    filtered.extend(["--signal-spec", FIXED_SIGNAL_SPEC])
    return filtered


def main() -> None:
    if not BASE_GRID_SCRIPT.is_file():
        raise FileNotFoundError(BASE_GRID_SCRIPT)
    command = [sys.executable, str(BASE_GRID_SCRIPT), *replace_signal_specs(sys.argv[1:])]
    print("[FIXED_SIGNAL] ensemble_first3_mean", flush=True)
    print("[CMD] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
