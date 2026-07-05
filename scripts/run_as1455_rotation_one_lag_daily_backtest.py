#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-fold-lag AS1455 rotation+one-hot checkpoint prediction and daily close-auction backtest.

This script implements the current research/backtest protocol:

1. Rebuild the same AS1455 rotation + sector one-hot feature matrix used by
   scripts/run_as1455_sector_rotation_fold0_param_search.py.
2. For each target fold, load the previous chronological fold's saved search-time
   top-N checkpoints, scaler, and feature manifest.
3. Use those source-fold artifacts to predict the target fold test window.
4. Write ML4T-style predictions to HDF key /predictions with numeric columns
   0..N-1, so the existing v7 close-auction grid can compare model_0..model_4,
   ensemble_first3_mean, and ensemble_all5_mean.
5. Optionally run the existing close-auction grid with daily rebalance only:
   rebalance_every=1, rebalance_offset=0.

Fold convention inherited from the training script:
    fold0 = newest, fold6 = oldest

One-fold-lag mapping used here:
    source fold6 -> target fold5
    source fold5 -> target fold4
    source fold4 -> target fold3
    source fold3 -> target fold2
    source fold2 -> target fold1
    source fold1 -> target fold0

No model is retrained here. No target-fold score/IC/backtest result is used for
checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_sector_rotation_fold0_param_search as train_mod  # noqa: E402


DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "model_data_as1455.h5"
DEFAULT_FOLD_DIR_TEMPLATE = str(
    PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_sector_rotation_onehot_fold{fold}_search"
)
DEFAULT_RAW_DAILY_CACHE_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455" / "baostock_raw_daily_cache"
DEFAULT_GRID_SCRIPT = PROJECT_DIR / "code" / "backtest" / "run_as1455_close_auction_grid_v1.py"


ONE_LAG_PAIRS = [
    {"source_fold": 6, "target_fold": 5},
    {"source_fold": 5, "target_fold": 4},
    {"source_fold": 4, "target_fold": 3},
    {"source_fold": 3, "target_fold": 2},
    {"source_fold": 2, "target_fold": 1},
    {"source_fold": 1, "target_fold": 0},
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_int_list(value: str) -> list[int]:
    out = []
    for part in str(value).split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"empty integer list: {value!r}")
    return out


def fold_dir_from_template(template: str, fold: int) -> Path:
    return Path(template.format(fold=fold)).expanduser().resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_path(source_dir: Path, value: Any, checkpoint_name: Any | None = None) -> Path:
    candidates: list[Path] = []
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        p = Path(str(value))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append((PROJECT_DIR / p).resolve())
            candidates.append((source_dir / p).resolve())
            candidates.append((source_dir / "search_checkpoints" / p.name).resolve())
        else:
            candidates.append((source_dir / "search_checkpoints" / p.name).resolve())
    if checkpoint_name is not None and not (isinstance(checkpoint_name, float) and np.isnan(checkpoint_name)):
        name = str(checkpoint_name)
        candidates.append((source_dir / "search_checkpoints" / f"{name}.keras").resolve())
        candidates.append((source_dir / "search_checkpoints" / name).resolve())
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(
        "cannot resolve checkpoint path; "
        f"source_dir={source_dir} value={value!r} checkpoint_name={checkpoint_name!r} "
        f"candidates={[str(c) for c in candidates]}"
    )


def read_top_checkpoints(source_dir: Path, top_n: int) -> pd.DataFrame:
    path = source_dir / "search_best_checkpoints.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"empty checkpoint table: {path}")
    if "checkpoint_saved" in df.columns:
        saved = df["checkpoint_saved"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[saved].copy()
    if "daily_ic_median" in df.columns:
        df["daily_ic_median"] = pd.to_numeric(df["daily_ic_median"], errors="coerce")
        df = df.sort_values("daily_ic_median", ascending=False)
    df = df.head(top_n).reset_index(drop=True)
    if len(df) < top_n:
        raise RuntimeError(f"not enough saved checkpoints in {path}: need {top_n}, got {len(df)}")
    if "keras_model" not in df.columns:
        raise RuntimeError(f"checkpoint table missing keras_model column: {path}")
    return df


def load_preprocess(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scaler_path = source_dir / "preprocess" / "scaler.pkl"
    manifest_path = source_dir / "preprocess" / "feature_manifest.json"
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with scaler_path.open("rb") as f:
        bundle = pickle.load(f)
    manifest = load_json(manifest_path)
    for key in ["scaler", "scale_cols", "no_scale_cols", "model_input_cols"]:
        if key not in bundle:
            raise RuntimeError(f"{scaler_path} missing key {key!r}")
    for key in ["feature_cols_final", "model_input_cols"]:
        if key not in manifest:
            raise RuntimeError(f"{manifest_path} missing key {key!r}")
    if list(bundle["model_input_cols"]) != list(manifest["model_input_cols"]):
        raise RuntimeError(f"model_input_cols mismatch between scaler.pkl and feature_manifest.json in {source_dir}")
    return bundle, manifest


def transform_for_source_model(X_final: pd.DataFrame, test_idx: np.ndarray, bundle: dict[str, Any], manifest: dict[str, Any]) -> np.ndarray:
    feature_cols = list(manifest["feature_cols_final"])
    missing = [c for c in feature_cols if c not in X_final.columns]
    if missing:
        raise RuntimeError(f"target feature matrix missing columns required by source model: {missing[:20]}")
    X_test = X_final.iloc[test_idx][feature_cols]
    scale_cols = list(bundle["scale_cols"])
    no_scale_cols = list(bundle["no_scale_cols"])
    model_input_cols = list(bundle["model_input_cols"])
    if model_input_cols != scale_cols + no_scale_cols:
        raise RuntimeError("model_input_cols must equal scale_cols + no_scale_cols")
    scaler = bundle["scaler"]
    x_scaled = scaler.transform(X_test[scale_cols]).astype(np.float32)
    if no_scale_cols:
        x_raw = X_test[no_scale_cols].to_numpy(dtype=np.float32)
        x_model = np.concatenate([x_scaled, x_raw], axis=1)
    else:
        x_model = x_scaled
    if x_model.shape[1] != len(model_input_cols):
        raise RuntimeError(f"bad model input width: got {x_model.shape[1]}, expected {len(model_input_cols)}")
    return x_model


def build_feature_matrix(model_data: Path, train_end: str | None, dropna_mode: str, sector_encoding: str):
    X_base, y, meta = train_mod.load_xy(model_data, train_end, dropna_mode)
    X_rot, rotation_cols = train_mod.add_sector_rotation_features(X_base)
    X_final, no_scale_cols, sector_onehot_cols = train_mod.apply_sector_encoding(X_rot, sector_encoding)
    feature_meta = {
        **meta,
        "model_data": str(model_data.resolve()),
        "base_feature_count": int(X_base.shape[1]),
        "rotation_feature_count": int(len(rotation_cols)),
        "final_feature_count": int(X_final.shape[1]),
        "sector_encoding": sector_encoding,
        "dropna_mode": dropna_mode,
        "sector_onehot_count": int(len(sector_onehot_cols)),
    }
    return X_final, y, feature_meta


def make_one_lag_predictions(args: argparse.Namespace) -> Path:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.backend import clear_session

    pred_dir = Path(args.out_root) / "00_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.prediction_file) if args.prediction_file else pred_dir / "test_preds.h5"
    csv_path = pred_path.with_suffix(".csv")

    X_final, y, feature_meta = build_feature_matrix(Path(args.model_data), args.train_end, args.dropna_mode, args.sector_encoding)

    all_fold_preds: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    target_folds = set(args.target_folds_list)
    for pair in ONE_LAG_PAIRS:
        source_fold = pair["source_fold"]
        target_fold = pair["target_fold"]
        if target_fold not in target_folds:
            continue
        source_dir = fold_dir_from_template(args.fold_dir_template, source_fold)
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        train_idx, test_idx, fold_report = train_mod.get_fold(X_final, target_fold)
        bundle, source_manifest = load_preprocess(source_dir)
        x_model = transform_for_source_model(X_final, test_idx, bundle, source_manifest)
        top = read_top_checkpoints(source_dir, args.top_n)
        fold_index = X_final.iloc[test_idx].index
        pred_cols: list[pd.Series] = []
        for model_rank, row in top.iterrows():
            keras_path = resolve_checkpoint_path(source_dir, row.get("keras_model"), row.get("checkpoint_name"))
            model = load_model(str(keras_path), compile=False)
            pred = model.predict(x_model, verbose=0).reshape(-1)
            if len(pred) != len(fold_index):
                raise RuntimeError(f"prediction length mismatch for {keras_path}: got {len(pred)}, expected {len(fold_index)}")
            pred_cols.append(pd.Series(pred, index=fold_index, name=int(model_rank)))
            checkpoint_meta = row.to_dict()
            checkpoint_meta.update({
                "source_fold": int(source_fold),
                "target_fold": int(target_fold),
                "model_rank": int(model_rank),
                "resolved_keras_model": str(keras_path),
            })
            checkpoint_rows.append(checkpoint_meta)
            del model
            clear_session()
            gc.collect()
        fold_preds = pd.concat(pred_cols, axis=1)
        fold_preds.columns = list(range(len(pred_cols)))
        all_fold_preds.append(fold_preds)
        manifest_rows.append({
            "source_fold": int(source_fold),
            "target_fold": int(target_fold),
            "source_dir": str(source_dir),
            "target_test_start": fold_report["test_start"],
            "target_test_end": fold_report["test_end"],
            "n_target_rows": int(len(test_idx)),
            "n_models": int(len(pred_cols)),
            "source_train_start": source_manifest.get("train_start"),
            "source_train_end": source_manifest.get("train_end"),
            "source_test_start": source_manifest.get("test_start"),
            "source_test_end": source_manifest.get("test_end"),
            "n_model_input_features": int(len(bundle["model_input_cols"])),
        })
        print(f"[PRED] source fold{source_fold} -> target fold{target_fold}: rows={len(fold_preds)} models={len(pred_cols)}")

    if not all_fold_preds:
        raise RuntimeError("no fold predictions generated; check --target-folds")

    preds = pd.concat(all_fold_preds, axis=0).sort_index()
    preds = preds[~preds.index.duplicated(keep="last")]
    preds.to_hdf(pred_path, key="predictions", mode="w")
    preds.to_csv(csv_path, encoding="utf-8-sig")

    actual = y.reindex(preds.index).rename("r01_fwd")
    actual.to_csv(pred_dir / "actual_r01_fwd.csv", encoding="utf-8-sig")

    write_json(pred_dir / "one_lag_prediction_manifest.json", {
        "created_at_utc": utc_now(),
        "protocol": "one_fold_lag_search_checkpoint_transfer",
        "fold_mapping": manifest_rows,
        "feature_meta": feature_meta,
        "top_n": int(args.top_n),
        "prediction_file": str(pred_path),
        "prediction_csv": str(csv_path),
        "actual_file": str(pred_dir / "actual_r01_fwd.csv"),
        "prediction_columns": [str(c) for c in preds.columns],
        "n_rows": int(len(preds)),
        "n_dates": int(preds.index.get_level_values("date").nunique()),
        "n_symbols": int(preds.index.get_level_values("symbol").nunique()),
    })
    pd.DataFrame(checkpoint_rows).to_csv(pred_dir / "selected_checkpoints.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] predictions: {pred_path}")
    print(f"[OK] checkpoint manifest: {pred_dir / 'selected_checkpoints.csv'}")
    return pred_path


def run_daily_grid(args: argparse.Namespace, prediction_file: Path) -> None:
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
        "--model-run", "AS1455 rotation one-fold-lag search checkpoints",
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
    p = argparse.ArgumentParser(description="AS1455 rotation+one-hot one-fold-lag daily close-auction backtest")
    p.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    p.add_argument("--train-end", default=None)
    p.add_argument("--dropna-mode", choices=["strict_original", "r01_only"], default="r01_only")
    p.add_argument("--sector-encoding", choices=["numeric", "onehot"], default="onehot")
    p.add_argument("--fold-dir-template", default=DEFAULT_FOLD_DIR_TEMPLATE,
                   help="Template containing {fold}; default points to ch17_as1455_sector_rotation_onehot_fold{fold}_search")
    p.add_argument("--target-folds", default="0,1,2,3,4,5",
                   help="Target folds to predict. One-lag source fold is target+1, so fold6 cannot be a target by default.")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-root", default=str(PROJECT_DIR / "saved_data" / "ashare_ml4t" / f"ch17_as1455_rotation_one_lag_daily_backtest_{datetime.now():%Y%m%d}"))
    p.add_argument("--prediction-file", default=None)
    p.add_argument("--skip-predictions", action="store_true", help="Use --prediction-file directly and skip checkpoint prediction generation")
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--grid-script", default=str(DEFAULT_GRID_SCRIPT))
    p.add_argument("--grid-out-root", default=None)
    p.add_argument("--raw-daily-cache-dir", default=str(DEFAULT_RAW_DAILY_CACHE_DIR))
    p.add_argument("--profile", default="close_auction_skip_limit")
    p.add_argument("--capacity-mode", default="none", choices=["none", "last5_amount", "last5_volume", "last5_both"])
    p.add_argument("--output-mode", default="compact", choices=["summary", "compact", "full"])
    p.add_argument("--max-positions-list", default="5,10,15,20,25")
    p.add_argument("--sell-rank-list", default="75,100,150,200,250,300")
    p.add_argument("--python-bin", default=sys.executable or "python3")
    p.add_argument("--model-family", default="AS1455 rotation one-lag NN")
    p.add_argument("--force-grid", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.target_folds_list = parse_int_list(args.target_folds)
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
    if args.skip_predictions:
        if not args.prediction_file:
            raise SystemExit("--skip-predictions requires --prediction-file")
        prediction_file = Path(args.prediction_file)
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)
    else:
        prediction_file = make_one_lag_predictions(args)
    if not args.skip_grid:
        run_daily_grid(args, prediction_file)
    print(f"[DONE] out_root={out_root}")


if __name__ == "__main__":
    main()
