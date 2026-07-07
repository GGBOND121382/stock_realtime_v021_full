#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-aware AS1455 NN parameter search.

Supports r01_fwd/r05_fwd/r21_fwd without changing the original search grid.
Use this for horizon-specific searches before one-fold-lag backtests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in [PROJECT_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_as1455_sector_rotation_fold0_param_search as base  # noqa: E402
import run_as1455_first_batch_features_fold0_param_search as addon  # noqa: E402
import as1455_target_label_common as common  # noqa: E402


def default_out_dir(feature_preset: str, target_col: str, fold_index: int) -> Path:
    return (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_target_search"
        / feature_preset
        / target_col
        / f"fold{fold_index}_search"
    )


def build_features(model_data: Path, train_end: str | None, dropna_mode: str, target_col: str, feature_preset: str, sector_encoding: str):
    X_base, y, meta = common.load_xy_target(model_data, train_end, dropna_mode, target_col)
    X_rot, rotation_cols = base.add_sector_rotation_features(X_base)
    addon_cols = []
    feature_groups = {}
    if feature_preset == "rotation_onehot":
        X_ctx = X_rot
    elif feature_preset == "rotation_addon_onehot":
        X_ctx, addon_cols, feature_groups = addon.add_compact_addon_features(X_rot)
    else:
        raise RuntimeError(f"bad feature_preset: {feature_preset}")
    X_final, no_scale_cols, sector_onehot_cols = base.apply_sector_encoding(X_ctx, sector_encoding)
    report = {
        **meta,
        "feature_preset": feature_preset,
        "model_data": str(model_data.resolve()),
        "base_feature_count": int(X_base.shape[1]),
        "rotation_feature_count": int(len(rotation_cols)),
        "addon_feature_count": int(len(addon_cols)),
        "final_feature_count": int(X_final.shape[1]),
        "sector_encoding": sector_encoding,
        "sector_onehot_count": int(len(sector_onehot_cols)),
    }
    return X_final, y, no_scale_cols, rotation_cols, addon_cols, feature_groups, sector_onehot_cols, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Target-aware AS1455 NN parameter search")
    p.add_argument("--model-data", default=str(base.DEFAULT_MODEL_DATA))
    p.add_argument("--feature-preset", choices=["rotation_onehot", "rotation_addon_onehot"], required=True)
    p.add_argument("--target-col", choices=list(common.TARGET_LOOKAHEAD), default="r05_fwd")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--train-end", default=None)
    p.add_argument("--fold-index", type=int, required=True, help="0=newest fold, 6=oldest fold")
    p.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    p.add_argument("--dropna-mode", choices=["target_only", "strict_original"], default="target_only")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--best-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input-check-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retrain-best", action="store_true", help="Optional diagnostic retrain; backtests use search-time checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.feature_preset, args.target_col, args.fold_index)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir already has files; pass --force or choose another --out-dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_final, y, no_scale_cols, rotation_cols, addon_cols, feature_groups, sector_onehot_cols, report = build_features(
        Path(args.model_data), args.train_end, args.dropna_mode, args.target_col, args.feature_preset, args.sector_encoding
    )
    train_idx, test_idx, fold = common.get_fold_target(X_final, args.fold_index, args.target_col)
    grid = base.param_grid(args.smoke)

    base.write_json(out_dir / "run_summary.json", {**report, "out_dir": str(out_dir.resolve()), "fold_index": args.fold_index, "epochs": args.epochs, "param_grid_size": len(grid), "best_n": args.best_n})
    base.write_json(out_dir / "fold_report.json", fold)
    pd.DataFrame([fold]).to_csv(out_dir / "fold_report.csv", index=False, encoding="utf-8-sig")
    base.write_json(out_dir / "rotation_feature_cols.json", rotation_cols)
    base.write_json(out_dir / "addon_feature_cols.json", addon_cols)
    base.write_json(out_dir / "feature_group_cols.json", feature_groups)
    base.write_json(out_dir / "feature_cols_final.json", list(X_final.columns))
    base.write_json(out_dir / "sector_onehot_cols.json", sector_onehot_cols)
    pd.DataFrame(grid).to_csv(out_dir / "param_grid.csv", index=False, encoding="utf-8-sig")

    print(f"[DATA] preset={args.feature_preset} target={args.target_col} final_features={X_final.shape[1]} rows={len(X_final)}")
    print(f"[FOLD] {fold}")
    if args.input_check_only:
        print(f"[OK] input reports written to {out_dir}")
        return

    base.require_deps()
    summary = base.train_search(X_final, y, train_idx, test_idx, no_scale_cols, grid, args.epochs, args.seed, out_dir, checkpoint_top_n=args.best_n)
    best = summary.head(args.best_n).copy()
    best.to_csv(out_dir / "best_params.csv", index=False, encoding="utf-8-sig")
    print("[BEST]")
    print(best[base.PARAM_COLS + ["pooled_spearman", "daily_ic_mean", "daily_ic_median", "daily_ic_positive_rate"]].to_string(index=False))

    if args.retrain_best:
        base.retrain_best(X_final, y, train_idx, test_idx, no_scale_cols, best[base.PARAM_COLS], args.seed, out_dir)
        actual_path = out_dir / "best_actual.csv"
        if actual_path.exists():
            actual = pd.read_csv(actual_path, index_col=[0, 1])
            actual.columns = [args.target_col]
            actual.to_csv(actual_path, encoding="utf-8-sig")
    print(f"[OK] written to {out_dir}")


if __name__ == "__main__":
    main()
