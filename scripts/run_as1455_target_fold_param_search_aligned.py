#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-aware NN search on the shared, target-independent fold calendar."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils import as1455_ch17_common as common  # noqa: E402
from utils import as1455_paths  # noqa: E402
from utils.as1455_fold_calendar import aligned_fold_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-aware AS1455 NN search with aligned fold dates"
    )
    parser.add_argument("--model-data", default=str(as1455_paths.DEFAULT_MODEL_DATA))
    parser.add_argument("--feature-preset", choices=list(common.FEATURE_PRESETS), required=True)
    parser.add_argument("--target-col", choices=list(common.TARGET_SPECS), default="r05_fwd")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--fold-index", type=int, required=True, help="0=newest fold, 6=oldest fold")
    parser.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    parser.add_argument("--dropna-mode", choices=["target_only", "strict_original"], default="target_only")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--best-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input-check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else common.fold_dir_from_template(
            common.default_fold_dir_template(args.feature_preset, args.target_col),
            args.fold_index,
        )
    )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir already has files; pass --force: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_data = Path(args.model_data)
    features = common.build_target_features(
        model_data,
        args.train_end,
        args.dropna_mode,
        args.target_col,
        args.feature_preset,
        args.sector_encoding,
    )
    train_idx, fold_idx, fold = aligned_fold_split(
        features.X,
        model_data,
        args.fold_index,
        args.target_col,
        args.train_end,
    )
    grid = common.base.param_grid(args.smoke)

    common.write_json(
        out_dir / "run_summary.json",
        {
            **features.report,
            **{f"aligned_{key}": value for key, value in fold.items()},
            "out_dir": str(out_dir.resolve()),
            "fold_index": args.fold_index,
            "epochs": args.epochs,
            "param_grid_size": len(grid),
            "best_n": args.best_n,
        },
    )
    common.write_json(out_dir / "fold_report.json", fold)
    pd.DataFrame([fold]).to_csv(out_dir / "fold_report.csv", index=False, encoding="utf-8-sig")
    common.write_json(out_dir / "rotation_feature_cols.json", features.rotation_cols)
    common.write_json(out_dir / "addon_feature_cols.json", features.addon_cols)
    common.write_json(out_dir / "feature_group_cols.json", features.feature_groups)
    common.write_json(out_dir / "feature_cols_final.json", list(features.X.columns))
    common.write_json(out_dir / "sector_onehot_cols.json", features.sector_onehot_cols)
    pd.DataFrame(grid).to_csv(out_dir / "param_grid.csv", index=False, encoding="utf-8-sig")

    print(
        f"[DATA] preset={args.feature_preset} target={args.target_col} "
        f"final_features={features.X.shape[1]} rows={len(features.X)}"
    )
    print(f"[ALIGNED FOLD] {fold}")
    if args.input_check_only:
        print(f"[OK] aligned input reports written to {out_dir}")
        return

    common.base.require_deps()
    summary = common.base.train_search(
        features.X,
        features.y,
        train_idx,
        fold_idx,
        features.no_scale_cols,
        grid,
        args.epochs,
        args.seed,
        out_dir,
        checkpoint_top_n=args.best_n,
    )
    best = summary.head(args.best_n).copy()
    best.to_csv(out_dir / "best_params.csv", index=False, encoding="utf-8-sig")
    print("[BEST]")
    print(
        best[
            common.base.PARAM_COLS
            + ["pooled_spearman", "daily_ic_mean", "daily_ic_median", "daily_ic_positive_rate"]
        ].to_string(index=False)
    )
    if args.retrain_best:
        common.base.retrain_best(
            features.X,
            features.y,
            train_idx,
            fold_idx,
            features.no_scale_cols,
            best[common.base.PARAM_COLS],
            args.seed,
            out_dir,
        )
    print(f"[OK] written to {out_dir}")


if __name__ == "__main__":
    main()
