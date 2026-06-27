#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create an AS1455 checkpoint-based live inference bundle.

This is compatible with scripts/run_ashare_ch17_nn_reproduce.py artifacts:
  results/best_params.csv
  results/cv_split_report.csv
  results/logs/{dense_layers}/{activation}/{dropout}/{batch_size}/ckpt_{fold}_{epoch}.weights.h5

It does not require .keras model files.  A model_i is represented by one row
from best_params.csv plus one checkpoint per selected fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from ast import literal_eval
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = ['dollar_vol', 'dollar_vol_rank', 'rsi', 'bb_high', 'bb_low', 'NATR', 'ATR', 'PPO', 'MACD', 'sector', 'r01', 'r05', 'r10', 'r21', 'r42', 'r63', 'r01dec', 'r05dec', 'r10dec', 'r21dec', 'r42dec', 'r63dec', 'r01q_sector', 'r05q_sector', 'r10q_sector', 'r21q_sector', 'r42q_sector', 'r63q_sector', 'year', 'month', 'weekday']


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_ints(value: str) -> list[int]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"empty int list: {value!r}")
    return out


def dense_layers_to_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    parsed = literal_eval(str(value))
    if isinstance(parsed, int):
        return (int(parsed),)
    return tuple(int(x) for x in parsed)


def dropout_dir_text(value: object) -> str:
    f = float(value)
    if abs(f - int(f)) < 1e-12:
        return str(int(f))
    return str(f)


def find_checkpoint(logs_dir: Path, dense_layers: str, activation: str, dropout: object, batch_size: int, fold: int, epoch: int) -> Path:
    # The original training code writes str(dropout), but prediction has a fallback
    # between "0" and "0.0". Keep both for compatibility.
    candidates = []
    for dtext in [dropout_dir_text(dropout), str(float(dropout)), str(dropout)]:
        p = logs_dir / str(dense_layers) / str(activation) / dtext / str(int(batch_size)) / f"ckpt_{fold}_{epoch}.weights.h5"
        if p not in candidates:
            candidates.append(p)
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Create AS1455 checkpoint ensemble deploy bundle")
    p.add_argument("--train-run-dir", required=True, help="e.g. saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7")
    p.add_argument("--out-dir", required=True, help="deploy bundle output dir")
    p.add_argument("--model-data", default=None, help="model_data_as1455.h5 used to fit live scalers")
    p.add_argument("--best-params", default=None, help="default: TRAIN_RUN/results/best_params.csv")
    p.add_argument("--cv-split-report", default=None, help="default: TRAIN_RUN/results/cv_split_report.csv")
    p.add_argument("--model-rows", type=parse_csv_ints, default=parse_csv_ints("0,1,2,3,4"), help="best_params row numbers to deploy")
    p.add_argument("--folds", type=parse_csv_ints, default=parse_csv_ints("0,1,2,3,4,5,6"), help="checkpoint folds to use")
    p.add_argument("--fold-mode", default="mean_all_folds", choices=["mean_all_folds", "single_fold"], help="live prediction fold aggregation")
    p.add_argument("--single-fold", type=int, default=0, help="used only when --fold-mode single_fold")
    p.add_argument("--bundle-id", default="sharpe1_checkpoint_ensemble_all5_v1")
    p.add_argument("--signal-name", default="ensemble_all5_mean")
    p.add_argument("--copy-checkpoints", action="store_true", help="copy selected weight files into the bundle")
    p.add_argument("--force", action="store_true")
    p.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()

    train_run_dir = Path(args.train_run_dir)
    results_dir = train_run_dir / "results"
    logs_dir = results_dir / "logs"
    out_dir = Path(args.out_dir)

    best_params_path = Path(args.best_params) if args.best_params else results_dir / "best_params.csv"
    cv_path = Path(args.cv_split_report) if args.cv_split_report else results_dir / "cv_split_report.csv"
    train_summary_path = results_dir / "train_data_summary.json"

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not best_params_path.exists():
        raise FileNotFoundError(best_params_path)
    if not cv_path.exists():
        raise FileNotFoundError(cv_path)
    if not logs_dir.exists():
        raise FileNotFoundError(logs_dir)

    best = pd.read_csv(best_params_path)
    cv = pd.read_csv(cv_path)
    selected_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    copied_files: list[str] = []

    active_folds = args.folds
    if args.fold_mode == "single_fold":
        if args.single_fold not in active_folds:
            active_folds = [args.single_fold]
        else:
            active_folds = [args.single_fold]

    for model_idx, row_no in enumerate(args.model_rows):
        if row_no < 0 or row_no >= len(best):
            raise ValueError(f"model row {row_no} out of range for {best_params_path} with {len(best)} rows")
        row = best.iloc[row_no].to_dict()
        dense_layers_text = str(row["dense_layers"])
        activation = str(row["activation"])
        dropout = float(row["dropout"])
        batch_size = int(row["batch_size"])
        epoch = int(row["epoch"])
        checkpoints: dict[str, dict[str, Any]] = {}
        for fold in active_folds:
            cp = find_checkpoint(logs_dir, dense_layers_text, activation, dropout, batch_size, int(fold), epoch)
            cp_info: dict[str, Any] = {
                "fold": int(fold),
                "source_path": str(cp),
                "exists": bool(cp.exists()),
            }
            if cp.exists():
                cp_info["source_sha256"] = sha256_file(cp)
                if args.copy_checkpoints:
                    dst = out_dir / "checkpoints" / f"model_{model_idx}" / f"fold_{fold}" / cp.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cp, dst)
                    cp_info["bundle_path"] = dst.relative_to(out_dir).as_posix()
                    cp_info["bundle_sha256"] = sha256_file(dst)
                    copied_files.append(str(dst))
            else:
                missing.append(str(cp))
            checkpoints[str(fold)] = cp_info

        selected_rows.append({
            "model_name": f"model_{model_idx}",
            "best_params_row": int(row_no),
            "dense_layers_text": dense_layers_text,
            "dense_layers": list(dense_layers_to_tuple(dense_layers_text)),
            "activation": activation,
            "dropout": dropout,
            "batch_size": batch_size,
            "epoch": epoch,
            "checkpoints": checkpoints,
        })

    if missing and args.strict:
        raise FileNotFoundError("missing selected checkpoints:\n" + "\n".join(missing[:100]))

    train_summary: dict[str, Any] = {}
    if train_summary_path.exists():
        try:
            train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            train_summary = {"read_error": str(exc)}

    model_data_path = args.model_data or train_summary.get("model_data_path")
    manifest = {
        "bundle_id": args.bundle_id,
        "bundle_version": 1,
        "bundle_type": "as1455_ch17_checkpoint_ensemble",
        "created_from": {
            "train_run_dir": str(train_run_dir),
            "results_dir": str(results_dir),
            "logs_dir": str(logs_dir),
            "best_params_csv": str(best_params_path),
            "cv_split_report_csv": str(cv_path),
            "train_data_summary_json": str(train_summary_path) if train_summary_path.exists() else None,
        },
        "model_data_path": str(model_data_path) if model_data_path else None,
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "model_rows": args.model_rows,
        "models": selected_rows,
        "ensemble": {
            "signal_name": args.signal_name,
            "model_names": [m["model_name"] for m in selected_rows],
            "method": "mean",
            "fold_mode": args.fold_mode,
            "folds": active_folds,
        },
        "strategy": {
            "max_positions": 15,
            "sell_rank": 300,
            "buy_candidate_rank": 300,
            "rebalance_every": 3,
            "rebalance_offset": 0,
            "anchor_date": "2024-07-17",
            "portfolio_mode": "long_only",
        },
        "missing_checkpoints": missing,
        "copied_checkpoints": copied_files,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "model_name": m["model_name"],
        "best_params_row": m["best_params_row"],
        "dense_layers": m["dense_layers_text"],
        "activation": m["activation"],
        "dropout": m["dropout"],
        "batch_size": m["batch_size"],
        "epoch": m["epoch"],
    } for m in selected_rows]).to_csv(out_dir / "best_params_selected.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(out_dir / "cv_split_report.csv", index=False, encoding="utf-8-sig")
    (out_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "passed": not missing,
        "bundle_dir": str(out_dir.resolve()),
        "bundle_id": args.bundle_id,
        "n_models": len(selected_rows),
        "n_folds": len(active_folds),
        "n_missing_checkpoints": len(missing),
        "copied_checkpoints": len(copied_files),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
