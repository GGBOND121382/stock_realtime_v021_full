#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refit one AS1455 production model generation without rerunning strategy Grid.

For each r01/r05/r21 target this script:
- keeps the active generation's ordered Top-5 model recipes;
- rebuilds features with the existing Chapter-17 contract;
- uses the latest ``train_length`` trading dates whose target label is mature;
- refits the scaler and the five neural networks from scratch;
- publishes all three targets as one generation only after every bundle validates.

Historical fold0..fold6 artifacts are never modified.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import sys
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_model_registry import (  # noqa: E402
    DEFAULT_REGISTRY_ROOT,
    atomic_write_json,
    bootstrap_registry,
    resolve_active_model,
)
from utils.as1455_model_roll import activate_generation, next_generation  # noqa: E402

DEFAULT_MODEL_DATA = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch12_as1455_forward_latest"
    / "model_data_as1455.h5"
)


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    parser.add_argument("--registry-root", default=str(DEFAULT_REGISTRY_ROOT))
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument("--period-end", required=True)
    parser.add_argument("--train-length", type=int, default=21 * 12 * 4)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def recipe_epochs(row: pd.Series) -> int:
    for key in ("epoch_1based", "epochs", "epochs_done"):
        if key in row.index and pd.notna(row[key]):
            value = int(float(row[key]))
            if value > 0:
                return value
    if "epoch" in row.index and pd.notna(row["epoch"]):
        value = int(float(row["epoch"])) + 1
        if value > 0:
            return value
    raise RuntimeError("Top-5 recipe lacks a positive epoch count")


def recipe_value(row: pd.Series, key: str) -> Any:
    if key not in row.index or pd.isna(row[key]):
        raise RuntimeError(f"Top-5 recipe lacks {key!r}")
    return row[key]


def make_preprocess(
    target_dir: Path,
    features: common.FeatureBuildResult,
    train_idx: np.ndarray,
    *,
    generation_id: str,
    target_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    X_train = features.X.iloc[train_idx]
    scale_cols = [column for column in features.X.columns if column not in features.no_scale_cols]
    no_scale_cols = list(features.no_scale_cols)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train[scale_cols]).astype(np.float32)
    if no_scale_cols:
        raw = X_train[no_scale_cols].to_numpy(dtype=np.float32)
        model_input = np.concatenate([scaled, raw], axis=1)
    else:
        model_input = scaled
    model_input_cols = scale_cols + no_scale_cols

    preprocess_dir = target_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    with (preprocess_dir / "scaler.pkl").open("wb") as stream:
        pickle.dump(
            {
                "scaler": scaler,
                "scale_cols": scale_cols,
                "no_scale_cols": no_scale_cols,
                "model_input_cols": model_input_cols,
            },
            stream,
        )
    dates = pd.DatetimeIndex(X_train.index.get_level_values("date")).normalize()
    manifest = {
        "created_at": now_text(),
        "protocol": "as1455_rolling_generation_fixed_recipe_refit_v1",
        "generation_id": generation_id,
        "target_col": target_col,
        "feature_preset": features.report.get("feature_preset"),
        "feature_cols_final": list(features.X.columns),
        "scale_cols": scale_cols,
        "no_scale_cols": no_scale_cols,
        "model_input_cols": model_input_cols,
        "n_features_before_transform": int(features.X.shape[1]),
        "n_model_input_features": int(len(model_input_cols)),
        "train_start": dates.min().strftime("%Y-%m-%d"),
        "train_end": dates.max().strftime("%Y-%m-%d"),
        "test_start": None,
        "test_end": None,
        "selection_role": "production_refit_no_new_grid",
    }
    atomic_write_json(preprocess_dir / "feature_manifest.json", manifest)
    return model_input, manifest


def train_target(
    *,
    model_data: Path,
    source_dir: Path,
    target_dir: Path,
    generation_id: str,
    target_col: str,
    feature_preset: str,
    train_length: int,
    top_n: int,
) -> dict[str, Any]:
    features = common.build_target_features(
        model_data,
        None,
        "target_only",
        target_col,
        feature_preset,
        "onehot",
    )
    dates = (
        pd.DatetimeIndex(features.X.index.get_level_values("date"))
        .normalize()
        .unique()
        .sort_values()
    )
    if len(dates) < train_length:
        raise RuntimeError(
            f"not enough mature training dates for {target_col}: "
            f"need={train_length} got={len(dates)}"
        )
    train_dates = dates[-train_length:]
    date_values = pd.DatetimeIndex(features.X.index.get_level_values("date")).normalize()
    train_idx = np.flatnonzero(date_values.isin(train_dates))
    if not len(train_idx):
        raise RuntimeError(f"empty rolling train rows for {target_col}")

    target_dir.mkdir(parents=True, exist_ok=True)
    x_train, preprocess_manifest = make_preprocess(
        target_dir,
        features,
        train_idx,
        generation_id=generation_id,
        target_col=target_col,
    )
    y_train = features.y.iloc[train_idx].to_numpy(dtype=np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise RuntimeError(f"non-finite rolling training matrix for {target_col}")

    recipes = common.read_top_checkpoints(source_dir, top_n)
    checkpoint_dir = target_dir / "search_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    trained_at = now_text()

    common.base.require_deps()
    for rank, source_row in recipes.iterrows():
        dense_layers = literal_eval(str(recipe_value(source_row, "dense_layers")))
        activation = str(recipe_value(source_row, "activation"))
        dropout = float(recipe_value(source_row, "dropout"))
        batch_size = int(float(recipe_value(source_row, "batch_size")))
        epochs = recipe_epochs(source_row)
        seed = (
            int(float(source_row["seed"]))
            if "seed" in source_row.index and pd.notna(source_row["seed"])
            else 42 + int(rank)
        )
        common.base.set_seed(seed)
        model = common.base.make_model(
            x_train.shape[1], tuple(dense_layers), activation, dropout
        )
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            shuffle=True,
        )
        checkpoint_name = f"rolling_rank{int(rank):02d}"
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.keras"
        model.save(str(checkpoint_path))

        row = source_row.to_dict()
        row.update(
            {
                "model_rank": int(rank),
                "checkpoint_name": checkpoint_name,
                "checkpoint_saved": True,
                "keras_model": str(Path("search_checkpoints") / checkpoint_path.name),
                "generation_id": generation_id,
                "target_col": target_col,
                "rolling_refit": True,
                "rolling_train_start": preprocess_manifest["train_start"],
                "rolling_train_end": preprocess_manifest["train_end"],
                "rolling_train_days": int(train_length),
                "rolling_train_rows": int(len(train_idx)),
                "rolling_epochs": int(epochs),
                "rolling_seed": int(seed),
                "recipe_source_dir": str(source_dir),
                "recipe_rank_inherited": True,
                "trained_at": trained_at,
            }
        )
        if "daily_ic_median" in source_row.index:
            row["recipe_source_daily_ic_median"] = source_row.get("daily_ic_median")
        output_rows.append(row)
        del model
        try:
            from tensorflow.keras.backend import clear_session

            clear_session()
        finally:
            gc.collect()

    pd.DataFrame(output_rows).to_csv(
        target_dir / "search_best_checkpoints.csv",
        index=False,
        encoding="utf-8-sig",
    )
    atomic_write_json(
        target_dir / "rolling_refit_manifest.json",
        {
            "status": "ok",
            "protocol": "as1455_rolling_generation_fixed_recipe_refit_v1",
            "generation_id": generation_id,
            "target_col": target_col,
            "source_model_dir": str(source_dir),
            "top_n": top_n,
            "train_length": train_length,
            "train_rows": int(len(train_idx)),
            "train_start": preprocess_manifest["train_start"],
            "train_end": preprocess_manifest["train_end"],
            "label_valid_end": preprocess_manifest["train_end"],
            "target_lookahead": common.target_lookahead(target_col),
            "new_grid_search": False,
            "recipe_rank_inherited": True,
            "trained_at": trained_at,
        },
    )

    # Validate using the exact production readers before the generation can move
    # out of staging.
    common.load_preprocess(target_dir)
    loaded = common.read_top_checkpoints(target_dir, top_n)
    if len(loaded) != top_n:
        raise RuntimeError(
            f"rolling target validation returned {len(loaded)} checkpoints, expected {top_n}"
        )
    del features, x_train, y_train
    gc.collect()
    return {
        "generation_id": generation_id,
        "source_type": "rolling_refit",
        "source_generation": None,
        "source_fold": None,
        "model_dir": str(target_dir),
        "model_updated_date": None,
        "effective_from": None,
        "trained_at": trained_at,
        "train_start": preprocess_manifest["train_start"],
        "train_end": preprocess_manifest["train_end"],
        "label_valid_end": preprocess_manifest["train_end"],
        "target_lookahead": common.target_lookahead(target_col),
        "train_length": train_length,
        "top_n": top_n,
    }


def main() -> None:
    args = parse_args()
    if args.train_length < 1 or args.top_n < 1:
        raise SystemExit("--train-length and --top-n must be positive")
    model_data = Path(args.model_data).expanduser().resolve()
    if not model_data.is_file():
        raise FileNotFoundError(model_data)
    registry_root = Path(args.registry_root).expanduser().resolve()
    registry = bootstrap_registry(
        registry_root, feature_preset=args.feature_preset
    )
    generation_id = next_generation(registry)
    generation_root = registry_root / "generations" / generation_id
    staging_root = registry_root / "generations" / f".{generation_id}.staging"
    if generation_root.exists():
        raise RuntimeError(f"generation output already exists: {generation_root}")
    if staging_root.exists():
        if not args.force:
            raise RuntimeError(
                f"stale generation staging exists: {staging_root}; pass --force to rebuild"
            )
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=False)

    started_at = now_text()
    targets: dict[str, dict[str, Any]] = {}
    try:
        for target_col in common.TARGET_SPECS:
            source = resolve_active_model(registry, target_col)
            target_dir = staging_root / target_col
            print(
                f"[TRAIN] generation={generation_id} target={target_col} "
                f"source={source['generation_id']} model_dir={source['model_dir']}",
                flush=True,
            )
            entry = train_target(
                model_data=model_data,
                source_dir=Path(source["model_dir"]),
                target_dir=target_dir,
                generation_id=generation_id,
                target_col=target_col,
                feature_preset=args.feature_preset,
                train_length=args.train_length,
                top_n=args.top_n,
            )
            entry["source_generation"] = source.get("generation_id")
            targets[target_col] = entry

        trained_at = now_text()
        generation_manifest = {
            "status": "candidate_ready",
            "protocol": "as1455_rolling_generation_fixed_recipe_refit_v1",
            "generation_id": generation_id,
            "generation_index": int(generation_id.replace("gen", "")),
            "source_type": "rolling_refit",
            "source_generation": registry.get("active_generation"),
            "source_period": (registry.get("current_period") or {}).get("period_id"),
            "period_end": pd.Timestamp(args.period_end).strftime("%Y-%m-%d"),
            "feature_preset": args.feature_preset,
            "train_length": int(args.train_length),
            "top_n": int(args.top_n),
            "new_grid_search": False,
            "strategy_parameters_changed": False,
            "started_at": started_at,
            "trained_at": trained_at,
            "targets": targets,
        }
        atomic_write_json(staging_root / "generation_manifest.json", generation_manifest)
        staging_root.replace(generation_root)

        # Paths stored in target records must point at the final generation dir,
        # not the staging name that was just atomically renamed.
        for target_col, entry in targets.items():
            entry["model_dir"] = str((generation_root / target_col).resolve())
        generation_manifest["targets"] = targets
        generation_manifest["status"] = "ready"
        atomic_write_json(generation_root / "generation_manifest.json", generation_manifest)

        if args.activate:
            registry = activate_generation(
                registry_root,
                generation=generation_manifest,
                period_end=args.period_end,
                feature_preset=args.feature_preset,
            )
            generation_manifest["status"] = "active"
            atomic_write_json(generation_root / "generation_manifest.json", generation_manifest)

        print(
            json.dumps(
                {
                    "status": generation_manifest["status"],
                    "generation_id": generation_id,
                    "source_generation": generation_manifest["source_generation"],
                    "period_end": generation_manifest["period_end"],
                    "targets": {
                        target: {
                            "train_start": entry["train_start"],
                            "train_end": entry["train_end"],
                            "label_valid_end": entry["label_valid_end"],
                            "model_dir": entry["model_dir"],
                        }
                        for target, entry in targets.items()
                    },
                    "activated": bool(args.activate),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception:
        # Keep failed staging artifacts for diagnosis unless they are empty; they
        # are never referenced by registry and therefore cannot affect live use.
        raise


if __name__ == "__main__":
    main()
