#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static contract checks for the nested AS1455 fold-selection workflow."""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "run_as1455_nested_fold_protocol.py"
WRAPPER = PROJECT_DIR / "scripts" / "run_as1455_r05_addon_fold_comparison.sh"


def require(text: str, token: str, label: str) -> None:
    if token not in text:
        raise AssertionError(f"missing {label}: {token}")


def main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT))
    wrapper = WRAPPER.read_text(encoding="utf-8")

    require(source, "for source_fold in range(6, -1, -1):", "seven source folds")
    require(source, "source_to_target(source_fold)", "source-to-target mapping")
    require(source, "source_fold_validation_grid", "validation-only grid protocol")
    require(source, "run_validation_grid(", "per-source validation grid")
    require(source, "run_frozen_target(", "frozen next-window evaluation")
    require(source, '"global_concatenated_target_grid": False', "global-grid prohibition")
    require(source, '"target_results_used_for_selection": False', "target selection guard")
    require(source, '"forward_results_used_for_selection": False', "forward selection guard")
    require(source, "--rebalance-phase-history-offset", "phase-aligned target application")
    require(source, "initial_positions=state[\"positions\"]", "continuous account state")

    if "resolve_as1455_existing_result_pairs.py" in wrapper:
        raise AssertionError("wrapper still resolves the invalid global-grid artifact pair")
    if "run_as1455_r05_addon_fold_comparison_v2.py" in wrapper:
        raise AssertionError("wrapper still invokes the invalid global-grid comparison")
    require(wrapper, "run_as1455_nested_fold_protocol.py", "nested runner entry")
    print("[PASS] nested fold protocol static contract")


if __name__ == "__main__":
    main()
