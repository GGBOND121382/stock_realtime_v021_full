#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use fold0 search-time checkpoints on dates strictly after fold0 test_end.

The workflow is deliberately separate from the one-fold-lag evaluation:

1. Rebuild the exact feature matrix for one target and feature preset.
2. Load fold0 search-time top-N checkpoints and the fold0 scaler/manifest.
3. Predict only rows whose date is strictly later than fold0 test_end.
4. Start every portfolio configuration from initial cash and no positions.
5. Run the shared-ranking in-process close-auction grid.

This is a forward holdout/deployment-style test. It does not retrain the model and
it does not carry positions from the fold0 test window into the forward window.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in [PROJECT_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import as1455_target_label_common as common  # noqa: E402
import run_as1455_rotation_one_lag_daily_backtest as core  # noqa: E402
import run_as1455_target_one_lag_backtest as target_runner  # noqa: E402


def default_fold0_dir(feature_preset: str, target_col: str) -> Path:
    if target_col == "r01_fwd":
        if feature_preset == "rotation_onehot":
            return PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_sector_rotation_onehot_fold0_search"
        return PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_full_rotation_plus_first_batch_compact_fold0_search"
    return (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_target_search"
        / feature_preset
        / target_col
        / "fold0_search"
    )


def default_out_root(feature_preset: str, target_col: str, rebalance_every: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_fold0_forward_backtest"
        / f"{feature_preset}_{target_col}_reb{rebalance_every}_{stamp}"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_fold0_test_end(fold0_dir: Path) -> pd.Timestamp:
    candidates = [
        fold0_dir / "fold_report.json",
        fold0_dir / "preprocess" / "feature_manifest.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path)
        value = payload.get("test_end")
        if value:
            return pd.Timestamp(value).normalize()
    raise FileNotFoundError(
        "cannot resolve fold0 test_end; expected fold_report.json or "
        f"preprocess/feature_manifest.json under {fold0_dir}"
    )


def build_forward_predictions(args: argparse.Namespace) -> Path:
    from tensorflow.keras.backend import clear_session
    from tensorflow.keras.models import load_model

    fold0_dir = Path(args.fold0_dir).expanduser().resolve()
    if not fold0_dir.exists():
        raise FileNotFoundError(fold0_dir)

    builder = target_runner.build_feature_matrix_for_args(args)
    X_final, y, feature_meta = builder(
        Path(args.model_data),
        args.train_end,
        args.dropna_mode,
        args.sector_encoding,
    )

    fold0_test_end = resolve_fold0_test_end(fold0_dir)
    dates = pd.DatetimeIndex(X_final.index.get_level_values("date"))
    mask = dates > fold0_test_end
    if args.start_date:
        mask &= dates >= pd.Timestamp(args.start_date)
    if args.end_date:
        mask &= dates <= pd.Timestamp(args.end_date)
    forward_idx = np.flatnonzero(mask)
    if len(forward_idx) == 0:
        available_max = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
        raise RuntimeError(
            "no forward rows after fold0 test_end; "
            f"fold0_test_end={fold0_test_end:%Y-%m-%d} "
            f"available_max={available_max} start_date={args.start_date} "
            f"end_date={args.end_date}"
        )

    bundle, manifest = core.load_preprocess(fold0_dir)
    x_model = core.transform_for_source_model(X_final, forward_idx, bundle, manifest)
    top = core.read_top_checkpoints(fold0_dir, args.top_n)
    forward_index = X_final.iloc[forward_idx].index

    pred_cols: list[pd.Series] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for model_rank, row in top.iterrows():
        keras_path = core.resolve_checkpoint_path(
            fold0_dir,
            row.get("keras_model"),
            row.get("checkpoint_name"),
        )
        model = load_model(str(keras_path), compile=False)
        pred = model.predict(x_model, verbose=0).reshape(-1)
        if len(pred) != len(forward_index):
            raise RuntimeError(
                f"prediction length mismatch for {keras_path}: "
                f"got {len(pred)}, expected {len(forward_index)}"
            )
        pred_cols.append(
            pd.Series(pred, index=forward_index, name=int(model_rank))
        )
        meta = row.to_dict()
        meta.update(
            {
                "source_fold": 0,
                "model_rank": int(model_rank),
                "resolved_keras_model": str(keras_path),
            }
        )
        checkpoint_rows.append(meta)
        del model
        clear_session()
        gc.collect()

    preds = pd.concat(pred_cols, axis=1).sort_index()
    preds.columns = list(range(len(pred_cols)))
    if preds.index.duplicated().any():
        raise RuntimeError("duplicate symbol/date rows in forward predictions")

    pred_dir = Path(args.out_root) / "00_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = (
        Path(args.prediction_file)
        if args.prediction_file
        else pred_dir / "fold0_forward_preds.h5"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_hdf(pred_path, key="predictions", mode="w")
    preds.to_csv(pred_path.with_suffix(".csv"), encoding="utf-8-sig")
    y.reindex(preds.index).rename(args.target_col).to_csv(
        pred_dir / f"actual_{args.target_col}.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(checkpoint_rows).to_csv(
        pred_dir / "selected_fold0_checkpoints.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_dates = pd.DatetimeIndex(preds.index.get_level_values("date"))
    payload = {
        "protocol": "fold0_search_checkpoint_forward_test",
        "feature_preset": args.feature_preset,
        "target_col": args.target_col,
        "target_lookahead": common.target_lookahead(args.target_col),
        "fold0_dir": str(fold0_dir),
        "fold0_test_end": fold0_test_end.strftime("%Y-%m-%d"),
        "forward_start": pd.Timestamp(pred_dates.min()).strftime("%Y-%m-%d"),
        "forward_end": pd.Timestamp(pred_dates.max()).strftime("%Y-%m-%d"),
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "n_rows": int(len(preds)),
        "n_dates": int(pred_dates.nunique()),
        "n_symbols": int(preds.index.get_level_values("symbol").nunique()),
        "top_n": int(args.top_n),
        "prediction_file": str(pred_path),
        "portfolio_initial_state": "empty_positions_and_initial_cash",
        "feature_meta": feature_meta,
    }
    (pred_dir / "fold0_forward_prediction_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"[PRED] fold0 test_end={payload['fold0_test_end']} "
        f"forward={payload['forward_start']}..{payload['forward_end']} "
        f"dates={payload['n_dates']} rows={payload['n_rows']}"
    )
    print(f"[OK] predictions: {pred_path}")
    return pred_path


def run_grid(args: argparse.Namespace, prediction_file: Path) -> None:
    grid_out = (
        Path(args.grid_out_root)
        if args.grid_out_root
        else Path(args.out_root) / "01_close_auction_grid"
    )
    cmd = [
        args.python_bin,
        str(Path(args.grid_script)),
        "--out-root",
        str(grid_out),
        "--predictions",
        str(prediction_file),
        "--prediction-key",
        "predictions",
        "--raw-daily-cache-dir",
        str(Path(args.raw_daily_cache_dir)),
        "--profile",
        args.profile,
        "--capacity-mode",
        args.capacity_mode,
        "--run-output-mode",
        args.output_mode,
        "--offset-mode",
        args.offset_mode,
        "--rebalance-every-list",
        str(args.rebalance_every),
        "--max-positions-list",
        args.max_positions_list,
        "--sell-rank-list",
        args.sell_rank_list,
        "--model-family",
        f"AS1455 fold0 forward {args.feature_preset} {args.target_col}",
        "--model-run",
        (
            f"fold0 search-time checkpoints; dates after fold0 test_end; "
            f"rebalance_every={args.rebalance_every}; empty start"
        ),
    ]
    if args.force_grid:
        cmd.append("--force")
    if args.smoke:
        cmd.append("--smoke")
    if args.parity_check_only:
        cmd.append("--parity-check-only")
    print("[GRID CMD] " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest fold0 top checkpoints on dates after fold0 test_end"
    )
    p.add_argument(
        "--feature-preset",
        choices=["rotation_onehot", "rotation_addon_onehot"],
        required=True,
    )
    p.add_argument(
        "--target-col",
        choices=list(common.TARGET_LOOKAHEAD),
        required=True,
    )
    p.add_argument("--rebalance-every", type=int, default=None)
    p.add_argument("--offset-mode", choices=["zero", "full"], default=None)
    p.add_argument("--model-data", default=str(core.DEFAULT_MODEL_DATA))
    p.add_argument("--fold0-dir", default=None)
    p.add_argument("--train-end", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--dropna-mode",
        choices=["target_only", "strict_original"],
        default="target_only",
    )
    p.add_argument("--sector-encoding", choices=["onehot"], default="onehot")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-root", default=None)
    p.add_argument("--prediction-file", default=None)
    p.add_argument("--skip-predictions", action="store_true")
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument(
        "--grid-script",
        default=str(
            PROJECT_DIR
            / "code"
            / "backtest"
            / "run_as1455_close_auction_grid_inprocess.py"
        ),
    )
    p.add_argument("--grid-out-root", default=None)
    p.add_argument(
        "--raw-daily-cache-dir",
        default=str(core.DEFAULT_RAW_DAILY_CACHE_DIR),
    )
    p.add_argument("--profile", default="close_auction_skip_limit")
    p.add_argument(
        "--capacity-mode",
        default="none",
        choices=["none", "last5_amount", "last5_volume", "last5_both"],
    )
    p.add_argument(
        "--output-mode",
        default="full",
        choices=["summary", "compact", "full"],
    )
    p.add_argument("--max-positions-list", default="5,10,15,20,25")
    p.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    p.add_argument("--python-bin", default=sys.executable or "python3")
    p.add_argument("--force-grid", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--parity-check-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    natural = common.target_lookahead(args.target_col)
    if args.rebalance_every is None:
        args.rebalance_every = natural
    if args.offset_mode is None:
        args.offset_mode = "zero" if args.rebalance_every == 1 else "full"
    if args.fold0_dir is None:
        args.fold0_dir = str(
            default_fold0_dir(args.feature_preset, args.target_col)
        )
    if args.out_root is None:
        args.out_root = str(
            default_out_root(
                args.feature_preset,
                args.target_col,
                args.rebalance_every,
            )
        )
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.skip_predictions:
        if not args.prediction_file:
            raise SystemExit("--skip-predictions requires --prediction-file")
        prediction_file = Path(args.prediction_file)
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)
    else:
        prediction_file = build_forward_predictions(args)

    if not args.skip_grid:
        run_grid(args, prediction_file)
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
