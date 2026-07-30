#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add rebalance markers for a dynamic-fold fixed-signal experiment."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import add_as1455_rebalance_markers_to_global_plots as shared  # noqa: E402

SIGNALS = {
    "all5": "ensemble_all5_mean:0,1,2,3,4:mean",
    "first3": "ensemble_first3_mean:0,1,2:mean",
    "best": "model_0:0:single",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark rebalance days for a dynamic fixed-signal global result"
    )
    parser.add_argument("--signal-kind", choices=sorted(SIGNALS), required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--historical-root", required=True)
    parser.add_argument("--plots-dir", default=None)
    return parser.parse_args()


def ensure_legacy_history_link(out_root: Path, historical_root: Path) -> None:
    legacy = out_root / "historical_fold0_to_fold5_selection"
    historical_root = historical_root.expanduser().resolve()
    if not historical_root.is_dir():
        raise FileNotFoundError(historical_root)
    if legacy.resolve() == historical_root:
        return
    if legacy.is_symlink():
        if legacy.resolve() == historical_root:
            return
        legacy.unlink()
    elif legacy.exists():
        raise RuntimeError(
            f"legacy historical path exists and differs from requested root: {legacy}"
        )
    relative = os.path.relpath(historical_root, start=legacy.parent)
    legacy.symlink_to(relative, target_is_directory=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    historical_root = Path(args.historical_root).expanduser().resolve()
    ensure_legacy_history_link(out_root, historical_root)

    shared.FIXED_SIGNAL_SPEC = SIGNALS[args.signal_kind]
    forwarded = [
        "add_as1455_rebalance_markers_to_global_plots.py",
        "--out-root",
        str(out_root),
    ]
    if args.plots_dir:
        forwarded.extend(["--plots-dir", args.plots_dir])
    sys.argv = forwarded
    shared.main()


if __name__ == "__main__":
    main()
