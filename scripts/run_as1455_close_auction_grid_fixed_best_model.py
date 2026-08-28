#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Force the AS1455 close-auction grid to use each fold's best checkpoint.

Prediction column 0 is the highest-ranked saved checkpoint for the corresponding
source fold. This adapter removes every incoming ``--signal-spec`` and supplies
exactly ``model_0:0:single`` so the grid tunes trading parameters only.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_GRID_SCRIPT = (
    PROJECT_DIR / "code" / "backtest" / "run_as1455_close_auction_grid_inprocess.py"
)
FIXED_SIGNAL_SPEC = "model_0:0:single"


def replace_signal_specs(argv: list[str]) -> list[str]:
    """Remove all supplied signal specs and append the fixed best model."""
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
    print("[FIXED_SIGNAL] model_0", flush=True)
    print("[CMD] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
