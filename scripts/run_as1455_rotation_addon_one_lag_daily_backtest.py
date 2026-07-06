#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-fold-lag AS1455 full-rotation + compact-add-on + one-hot daily backtest.

This is the compact-add-on counterpart of:
    scripts/run_as1455_rotation_one_lag_daily_backtest.py

It keeps the same one-fold-lag checkpoint-transfer protocol and the same
close-auction daily grid, but replaces the feature builder with the exact
feature pipeline used by:
    scripts/run_as1455_first_batch_features_fold0_param_search.py

Feature preset:
    original 31 features
    + full sector rotation features
    + compact add-on features
    + sector one-hot
    + dropna_mode=r01_only by default

Fold convention:
    fold0 = newest, fold6 = oldest

One-fold-lag mapping:
    source fold6 -> target fold5
    source fold5 -> target fold4
    source fold4 -> target fold3
    source fold3 -> target fold2
    source fold2 -> target fold1
    source fold1 -> target fold0

No model is retrained here. No target-fold IC, score, or backtest metric is
used for checkpoint selection.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in [PROJECT_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_as1455_rotation_one_lag_daily_backtest as core  # noqa: E402
import run_as1455_first_batch_features_fold0_param_search as addon_mod  # noqa: E402


DEFAULT_ADDON_FOLD_DIR_TEMPLATE = str(
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_full_rotation_plus_first_batch_compact_fold{fold}_search"
)
DEFAULT_ADDON_OUT_ROOT = str(
    PROJECT_DIR
    / "saved_data"
    / "ashare_ml4t"
    / f"ch17_as1455_rotation_addon_one_lag_daily_backtest_{datetime.now():%Y%m%d}"
)
DEFAULT_ADDON_MODEL_FAMILY = "AS1455 full-rotation compact-add-on one-lag NN"
DEFAULT_ADDON_MODEL_RUN = "AS1455 full rotation + compact add-on one-fold-lag search checkpoints"


def build_feature_matrix(model_data: Path, train_end: str | None, dropna_mode: str, sector_encoding: str):
    """Build the exact feature matrix used by compact-add-on model search."""
    X_base, y, meta = addon_mod.base.load_xy(model_data, train_end, dropna_mode)
    X_rot, rotation_cols = addon_mod.base.add_sector_rotation_features(X_base)
    X_ctx, addon_cols, feature_groups = addon_mod.add_compact_addon_features(X_rot)
    X_final, no_scale_cols, sector_onehot_cols = addon_mod.base.apply_sector_encoding(X_ctx, sector_encoding)
    feature_meta: dict[str, Any] = {
        **meta,
        "model_data": str(model_data.resolve()),
        "feature_preset": "full_rotation_plus_compact_addons_onehot",
        "base_feature_count": int(X_base.shape[1]),
        "rotation_feature_count": int(len(rotation_cols)),
        "addon_feature_count": int(len(addon_cols)),
        "final_feature_count": int(X_final.shape[1]),
        "sector_encoding": sector_encoding,
        "dropna_mode": dropna_mode,
        "sector_onehot_count": int(len(sector_onehot_cols)),
        "addon_feature_cols": list(addon_cols),
        "addon_feature_groups": feature_groups,
    }
    return X_final, y, feature_meta


def run_daily_grid(args, prediction_file: Path) -> None:
    """Call the shared grid runner but record the compact-add-on model_run label."""
    grid_out = Path(args.grid_out_root) if args.grid_out_root else Path(args.out_root) / "01_close_auction_daily_grid"
    grid_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(Path(args.grid_script)),
        "--out-root", str(grid_out),
        "--predictions", str(prediction_file),
        "--prediction-key", "predictions",
        "--raw-daily-cache-dir", str(Path(args.raw_daily_cache_dir)),
        "--profile", args.profile,
        "--capacity-mode", args.capacity_mode,
        "--run-output-mode", args.output_mode,
        "--offset-mode", "zero",
        "--rebalance-every-list", "1",
        "--max-positions-list", args.max_positions_list,
        "--sell-rank-list", args.sell_rank_list,
        "--model-family", args.model_family,
        "--model-run", DEFAULT_ADDON_MODEL_RUN,
    ]
    if args.force_grid:
        cmd.append("--force")
    if args.smoke:
        cmd.append("--smoke")
    print("[GRID CMD] " + " ".join(cmd))
    if args.dry_run:
        return
    import subprocess

    subprocess.run(cmd, check=True)


def main() -> None:
    core.build_feature_matrix = build_feature_matrix
    core.run_daily_grid = run_daily_grid

    args = core.parse_args()
    argv = sys.argv[1:]
    if "--fold-dir-template" not in argv:
        args.fold_dir_template = DEFAULT_ADDON_FOLD_DIR_TEMPLATE
    if "--out-root" not in argv:
        args.out_root = DEFAULT_ADDON_OUT_ROOT
    if "--model-family" not in argv:
        args.model_family = DEFAULT_ADDON_MODEL_FAMILY

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.skip_predictions:
        if not args.prediction_file:
            raise SystemExit("--skip-predictions requires --prediction-file")
        prediction_file = Path(args.prediction_file)
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)
    else:
        prediction_file = core.make_one_lag_predictions(args)

    if not args.skip_grid:
        run_daily_grid(args, prediction_file)

    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
