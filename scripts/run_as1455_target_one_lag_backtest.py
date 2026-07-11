#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-aware one-fold-lag AS1455 close-auction backtest.

This script predicts target folds from previous-fold search-time checkpoints and
then runs the existing v7 close-auction grid.  It is intended for natural
horizon/frequency tests, e.g. r05_fwd with rebalance_every=5.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in [PROJECT_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_as1455_sector_rotation_fold0_param_search as base  # noqa: E402
import run_as1455_first_batch_features_fold0_param_search as addon  # noqa: E402
import run_as1455_rotation_one_lag_daily_backtest as core  # noqa: E402
import as1455_target_label_common as common  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"empty integer list: {value!r}")
    return out


def default_fold_template(feature_preset: str, target_col: str) -> str:
    return str(PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_target_search" / feature_preset / target_col / "fold{fold}_search")


def default_out_root(feature_preset: str, target_col: str, rebalance_every: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    return str(PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_target_backtest" / f"{feature_preset}_{target_col}_reb{rebalance_every}_{stamp}")


def build_feature_matrix_for_args(args: argparse.Namespace):
    def _builder(model_data: Path, train_end: str | None, dropna_mode: str, sector_encoding: str):
        X_base, y, meta = common.load_xy_target(model_data, train_end, dropna_mode, args.target_col)
        X_rot, rotation_cols = base.add_sector_rotation_features(X_base)
        addon_cols = []
        feature_groups: dict[str, Any] = {}
        if args.feature_preset == "rotation_onehot":
            X_ctx = X_rot
        elif args.feature_preset == "rotation_addon_onehot":
            X_ctx, addon_cols, feature_groups = addon.add_compact_addon_features(X_rot)
        else:
            raise RuntimeError(f"bad feature_preset: {args.feature_preset}")
        X_final, no_scale_cols, sector_onehot_cols = base.apply_sector_encoding(X_ctx, sector_encoding)
        feature_meta = {
            **meta,
            "feature_preset": args.feature_preset,
            "model_data": str(model_data.resolve()),
            "base_feature_count": int(X_base.shape[1]),
            "rotation_feature_count": int(len(rotation_cols)),
            "addon_feature_count": int(len(addon_cols)),
            "final_feature_count": int(X_final.shape[1]),
            "sector_encoding": sector_encoding,
            "sector_onehot_count": int(len(sector_onehot_cols)),
            "addon_feature_cols": list(addon_cols),
            "addon_feature_groups": feature_groups,
        }
        return X_final, y, feature_meta
    return _builder


def patch_core_for_target(args: argparse.Namespace) -> None:
    core.build_feature_matrix = build_feature_matrix_for_args(args)

    def _get_fold(X, fold_index: int):
        return common.get_fold_target(X, fold_index, args.target_col)

    core.train_mod.get_fold = _get_fold


def rewrite_actual_manifest(out_root: Path, target_col: str) -> None:
    pred_dir = out_root / "00_predictions"
    old_actual = pred_dir / "actual_r01_fwd.csv"
    new_actual = pred_dir / f"actual_{target_col}.csv"
    if old_actual.exists() and old_actual != new_actual:
        old_actual.replace(new_actual)
    manifest_path = pred_dir / "one_lag_prediction_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["target_col"] = target_col
        payload["actual_file"] = str(new_actual)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_grid(args: argparse.Namespace, prediction_file: Path) -> None:
    grid_out = Path(args.grid_out_root) if args.grid_out_root else Path(args.out_root) / "01_close_auction_grid"
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
        "--offset-mode", args.offset_mode,
        "--rebalance-every-list", str(args.rebalance_every),
        "--max-positions-list", args.max_positions_list,
        "--sell-rank-list", args.sell_rank_list,
        "--model-family", f"AS1455 {args.feature_preset} {args.target_col}",
        "--model-run", f"{args.feature_preset} {args.target_col} one-fold-lag checkpoints rebalance_every={args.rebalance_every}",
    ]
    if args.force_grid:
        cmd.append("--force")
    if args.smoke:
        cmd.append("--smoke")
    print("[GRID CMD] " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Target-aware one-fold-lag AS1455 backtest")
    p.add_argument("--feature-preset", choices=["rotation_onehot", "rotation_addon_onehot"], required=True)
    p.add_argument("--target-col", choices=list(common.TARGET_LOOKAHEAD), default="r05_fwd")
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--offset-mode", choices=["zero", "full"], default="full")
    p.add_argument("--model-data", default=str(base.DEFAULT_MODEL_DATA))
    p.add_argument("--train-end", default=None)
    p.add_argument("--dropna-mode", choices=["target_only", "strict_original"], default="target_only")
    p.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    p.add_argument("--fold-dir-template", default=None)
    p.add_argument("--target-folds", default="0,1,2,3,4,5")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-root", default=None)
    p.add_argument("--prediction-file", default=None)
    p.add_argument("--skip-predictions", action="store_true")
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument(
        "--grid-script",
        default=str(PROJECT_DIR / "code" / "backtest" / "run_as1455_close_auction_grid_inprocess.py"),
    )
    p.add_argument("--grid-out-root", default=None)
    p.add_argument("--raw-daily-cache-dir", default=str(core.DEFAULT_RAW_DAILY_CACHE_DIR))
    p.add_argument("--profile", default="close_auction_skip_limit")
    p.add_argument("--capacity-mode", default="none", choices=["none", "last5_amount", "last5_volume", "last5_both"])
    p.add_argument("--output-mode", default="compact", choices=["summary", "compact", "full"])
    p.add_argument("--max-positions-list", default="5,10,15,20,25")
    p.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    p.add_argument("--python-bin", default=sys.executable or "python3")
    p.add_argument("--force-grid", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.target_folds_list = parse_int_list(args.target_folds)
    if args.fold_dir_template is None:
        args.fold_dir_template = default_fold_template(args.feature_preset, args.target_col)
    if args.out_root is None:
        args.out_root = default_out_root(args.feature_preset, args.target_col, args.rebalance_every)
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    for f in args.target_folds_list:
        if f < 0 or f > 5:
            raise SystemExit("target folds must be in 0..5 for one-fold-lag mapping")
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    patch_core_for_target(args)

    if args.skip_predictions:
        if not args.prediction_file:
            raise SystemExit("--skip-predictions requires --prediction-file")
        prediction_file = Path(args.prediction_file)
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)
    else:
        prediction_file = core.make_one_lag_predictions(args)
        rewrite_actual_manifest(out_root, args.target_col)

    if not args.skip_grid:
        run_grid(args, prediction_file)
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
