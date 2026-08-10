#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for the nine-strategy live planner.

The canonical v7 backtest correctly initializes ``last_nav`` as cash plus the
marked value of ``initial_positions``.  Its generic summary helper, however,
assumes ``cfg.initial_cash`` is the complete starting NAV.  That assumption is
valid for ordinary historical runs that start flat, but not for the live
planner, where ``cfg.initial_cash`` is only the T-1 residual cash and positions
are supplied separately.

For live planning we therefore adapt the summary-only config to the actual
starting portfolio NAV recovered from the first NAV row and its daily return.
Trading still uses the original cash balance; only performance summarization is
corrected.  This keeps the canonical trading engine and order generation
unchanged.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# When a Python file under scripts/ is executed directly, sys.path[0] is the
# scripts directory rather than the repository root. Bootstrap the project root
# before importing the scripts package so the documented direct command works
# without requiring callers to set PYTHONPATH manually.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_nine_strategy_planner as planner  # noqa: E402


def _starting_portfolio_nav(nav: pd.DataFrame) -> float | None:
    """Recover the NAV used as ``last_nav`` before the first simulated day."""
    if nav.empty or "nav" not in nav.columns or "daily_return" not in nav.columns:
        return None
    first_nav = pd.to_numeric(pd.Series([nav.iloc[0]["nav"]]), errors="coerce").iloc[0]
    first_ret = pd.to_numeric(
        pd.Series([nav.iloc[0]["daily_return"]]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(first_nav) or not np.isfinite(first_ret):
        return None
    gross = 1.0 + float(first_ret)
    if gross <= 0.0:
        return None
    starting_nav = float(first_nav) / gross
    if not np.isfinite(starting_nav) or starting_nav <= 0.0:
        return None
    return starting_nav


def install_live_summary_adapter() -> None:
    """Patch the v7 module returned to this live planner only."""
    original_loader = planner.live.load_v7_module

    def load_v7_with_live_summary():
        v7 = original_loader()
        original_summarize = v7.summarize_nav

        def summarize_nav_live(
            nav: pd.DataFrame,
            orders: pd.DataFrame,
            rejects: pd.DataFrame,
            cfg: Any,
            actions: pd.DataFrame | None = None,
            round_trips: pd.DataFrame | None = None,
            daily_drawdown: pd.DataFrame | None = None,
        ) -> dict:
            starting_nav = _starting_portfolio_nav(nav)
            summary_cfg = cfg
            if starting_nav is not None:
                # This replacement is used only inside summarize_nav().  The
                # actual backtest has already traded with cfg.initial_cash, so
                # no artificial buying power is introduced.
                summary_cfg = replace(cfg, initial_cash=starting_nav)
            return original_summarize(
                nav,
                orders,
                rejects,
                summary_cfg,
                actions,
                round_trips,
                daily_drawdown,
            )

        v7.summarize_nav = summarize_nav_live
        return v7

    planner.live.load_v7_module = load_v7_with_live_summary


def main() -> None:
    install_live_summary_adapter()
    planner.main()


if __name__ == "__main__":
    main()
