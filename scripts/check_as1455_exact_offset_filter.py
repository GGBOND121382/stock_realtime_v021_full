#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic check for exact rebalance-offset filtering."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_grid_runner as runner  # noqa: E402


def main() -> None:
    args = SimpleNamespace(
        signal_specs=[runner.legacy.parse_signal_spec("model_0:0:single")],
        smoke=False,
        max_positions_list=[20],
        sell_rank_list=[150],
        rebalance_every_list=[5],
        offset_mode="full",
        rebalance_offset_list=[3],
    )
    configs = runner.build_configs(args)
    assert len(configs) == 1, configs
    spec, max_positions, sell_rank, rebalance_every, offset = configs[0]
    assert spec["signal_name"] == "model_0"
    assert (max_positions, sell_rank, rebalance_every, offset) == (20, 150, 5, 3)
    print("[PASS] AS1455 exact rebalance-offset filter")


if __name__ == "__main__":
    main()
