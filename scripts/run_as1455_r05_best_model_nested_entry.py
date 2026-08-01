#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the r05 nested protocol with each source fold's best checkpoint fixed.

The historical validation and target-fold windows are built from the original
historical model-data file.  Only the fold0 strict-forward feature matrix is
built from the separately refreshed forward model-data file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_nested_fold_protocol as base  # noqa: E402

TARGET_COL = "r05_fwd"
FIXED_SIGNAL_SPEC = "model_0:0:single"


def pop_option(name: str) -> str:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[index + 1]
            del args[index : index + 2]
            sys.argv[1:] = args
            return value
        prefix = name + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            del args[index]
            sys.argv[1:] = args
            return value
    raise SystemExit(f"{name} is required")


def option_value(name: str) -> str | None:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def main() -> None:
    forward_model_data = Path(pop_option("--forward-model-data")).expanduser().resolve()
    if not forward_model_data.is_file():
        raise FileNotFoundError(forward_model_data)

    target_col = option_value("--target-col")
    if target_col != TARGET_COL:
        raise SystemExit(
            f"this entry requires --target-col {TARGET_COL}; got {target_col!r}"
        )

    original_builder = base.build_inference_features

    def build_latest_forward_features(
        _historical_model_data: Path,
        train_end: str | None,
        target_col_arg: str,
        feature_preset: str,
        sector_encoding: str,
    ) -> Any:
        if target_col_arg != TARGET_COL:
            raise RuntimeError(
                f"unexpected forward target: {target_col_arg!r}; expected {TARGET_COL!r}"
            )
        print(f"[FORWARD MODEL DATA] {forward_model_data}")
        return original_builder(
            forward_model_data,
            train_end,
            target_col_arg,
            feature_preset,
            sector_encoding,
        )

    # The generic nested runner uses build_inference_features only for the
    # source_fold0 -> strict-forward branch. Historical validation/target
    # features continue to use --model-data through common.build_target_features.
    base.build_inference_features = build_latest_forward_features
    base.main()


if __name__ == "__main__":
    main()
