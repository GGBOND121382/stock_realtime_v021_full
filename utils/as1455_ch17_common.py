#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared AS1455 Chapter-17 utilities.

This module is the common implementation layer used by training, historical
one-fold-lag prediction, fold0-forward prediction, and backtest wrappers.
Command-line scripts should only select a protocol and supply defaults; they
must not copy feature construction, checkpoint loading, prediction loops, or
backtest command construction.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for _path in (PROJECT_DIR, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_as1455_sector_rotation_fold0_param_search as base  # noqa: E402
import run_as1455_first_batch_features_fold0_param_search as addon  # noqa: E402


@dataclass(frozen=True)
class TargetSpec:
    target_col: str
    lookahead: int
    rebalance_every: int
    offset_mode: str


TARGET_SPECS: dict[str, TargetSpec] = {
    "r01_fwd": TargetSpec("r01_fwd", 1, 1, "zero"),
    "r05_fwd": TargetSpec("r05_fwd", 5, 5, "full"),
    "r21_fwd": TargetSpec("r21_fwd", 21, 21, "full"),
}
FEATURE_PRESETS = ("rotation_onehot", "rotation_addon_onehot")


@dataclass
class FeatureBuildResult:
    X: pd.DataFrame
    y: pd.Series
    no_scale_cols: list[str]
    rotation_cols: list[str]
    addon_cols: list[str]
    feature_groups: dict[str, list[str]]
    sector_onehot_cols: list[str]
    report: dict[str, Any]


def target_spec(target_col: str) -> TargetSpec:
    try:
        return TARGET_SPECS[target_col]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported target_col={target_col!r}; expected {sorted(TARGET_SPECS)}"
        ) from exc


def target_lookahead(target_col: str) -> int:
    return target_spec(target_col).lookahead


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    out: list[int] = []
    for part in parts:
        text = str(part).strip()
        if text:
            out.append(int(text))
    if not out:
        raise argparse.ArgumentTypeError(f"empty integer list: {value!r}")
    return out


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_search_dir(
    feature_preset: str,
    target_col: str,
    fold_index: int,
) -> Path:
    return (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_target_search"
        / feature_preset
        / target_col
        / f"fold{fold_index}_search"
    )


def default_fold_dir_template(feature_preset: str, target_col: str) -> str:
    if target_col == "r01_fwd":
        if feature_preset == "rotation_onehot":
            return str(
                PROJECT_DIR
                / "saved_data"
                / "ashare_ml4t"
                / "ch17_as1455_sector_rotation_onehot_fold{fold}_search"
            )
        return str(
            PROJECT_DIR
            / "saved_data"
            / "ashare_ml4t"
            / "ch17_as1455_full_rotation_plus_first_batch_compact_fold{fold}_search"
        )
    return str(
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_target_search"
        / feature_preset
        / target_col
        / "fold{fold}_search"
    )


def fold_dir_from_template(template: str, fold: int) -> Path:
    return Path(template.format(fold=fold)).expanduser().resolve()


def default_one_lag_out_root(
    feature_preset: str,
    target_col: str,
    rebalance_every: int,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    if target_col == "r01_fwd":
        name = (
            "ch17_as1455_rotation_one_lag_daily_backtest_"
            if feature_preset == "rotation_onehot"
            else "ch17_as1455_rotation_addon_one_lag_daily_backtest_"
        )
        return PROJECT_DIR / "saved_data" / "ashare_ml4t" / f"{name}{stamp}"
    return (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_target_backtest"
        / f"{feature_preset}_{target_col}_reb{rebalance_every}_{stamp}"
    )


def default_fold0_dir(feature_preset: str, target_col: str) -> Path:
    return fold_dir_from_template(
        default_fold_dir_template(feature_preset, target_col), 0
    )


def default_fold0_forward_out_root(
    feature_preset: str,
    target_col: str,
    rebalance_every: int,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return (
        PROJECT_DIR
        / "saved_data"
        / "ashare_ml4t"
        / "ch17_as1455_fold0_forward_backtest"
        / f"{feature_preset}_{target_col}_reb{rebalance_every}_{stamp}"
    )


def load_xy_target(
    path: Path,
    train_end: str | None,
    dropna_mode: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    spec = target_spec(target_col)
    data = pd.read_hdf(path, "model_data")
    n_before = int(len(data))
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    if list(data.columns) != base.EXPECTED_MODEL_COLUMNS:
        raise RuntimeError(f"unexpected model_data columns: {list(data.columns)}")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != base.EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")

    end_ts = base.parse_train_end(train_end)
    dates = data.index.get_level_values("date")
    effective_end = pd.Timestamp(dates.max()) if end_ts is None else end_ts
    if end_ts is not None:
        data = data.loc[dates <= end_ts]

    if dropna_mode == "strict_original":
        data = data.dropna()
    elif dropna_mode in {"target_only", "r01_only"}:
        required = [
            column
            for column in data.columns
            if column not in base.EXPECTED_OUTCOMES or column == target_col
        ]
        data = data.dropna(subset=required)
    else:
        raise RuntimeError(f"bad dropna_mode: {dropna_mode}")

    data = data.sort_index()
    y = data[target_col].copy()
    X = data.drop(base.EXPECTED_OUTCOMES, axis=1)
    if X.shape[1] != 31 or any("fwd" in column for column in X.columns):
        raise RuntimeError(f"bad X shape/columns: {X.shape}")
    report = {
        "rows_before_dropna": n_before,
        "rows_after_dropna": int(len(data)),
        "train_end_effective": effective_end.strftime("%Y-%m-%d"),
        "target_col": target_col,
        "target_lookahead": spec.lookahead,
        "dropna_mode": "target_only" if dropna_mode == "r01_only" else dropna_mode,
    }
    return X, y, report


def build_target_features(
    model_data: Path,
    train_end: str | None,
    dropna_mode: str,
    target_col: str,
    feature_preset: str,
    sector_encoding: str = "onehot",
) -> FeatureBuildResult:
    if feature_preset not in FEATURE_PRESETS:
        raise RuntimeError(
            f"bad feature_preset={feature_preset!r}; expected {FEATURE_PRESETS}"
        )
    X_base, y, meta = load_xy_target(
        model_data, train_end, dropna_mode, target_col
    )
    X_rot, rotation_cols = base.add_sector_rotation_features(X_base)
    addon_cols: list[str] = []
    feature_groups: dict[str, list[str]] = {}
    if feature_preset == "rotation_onehot":
        X_context = X_rot
    else:
        X_context, addon_cols, feature_groups = addon.add_compact_addon_features(
            X_rot
        )
    X_final, no_scale_cols, sector_onehot_cols = base.apply_sector_encoding(
        X_context, sector_encoding
    )
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
        "addon_feature_cols": list(addon_cols),
        "addon_feature_groups": feature_groups,
    }
    return FeatureBuildResult(
        X=X_final,
        y=y,
        no_scale_cols=list(no_scale_cols),
        rotation_cols=list(rotation_cols),
        addon_cols=list(addon_cols),
        feature_groups=feature_groups,
        sector_onehot_cols=list(sector_onehot_cols),
        report=report,
    )


def get_fold_target(
    X: pd.DataFrame,
    fold_index: int,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lookahead = target_lookahead(target_col)
    cv = base.MultipleTimeSeriesCV(
        base.N_SPLITS,
        base.TRAIN_PERIOD_LENGTH,
        base.TEST_PERIOD_LENGTH,
        lookahead,
    )
    for index, (train_idx, test_idx) in enumerate(cv.split(X)):
        if index != fold_index:
            continue
        train_index = X.iloc[train_idx].index
        test_index = X.iloc[test_idx].index
        report = {
            "fold_index": index,
            "target_col": target_col,
            "lookahead": lookahead,
            "train_start": pd.Timestamp(
                train_index.get_level_values("date").min()
            ).strftime("%Y-%m-%d"),
            "train_end": pd.Timestamp(
                train_index.get_level_values("date").max()
            ).strftime("%Y-%m-%d"),
            "test_start": pd.Timestamp(
                test_index.get_level_values("date").min()
            ).strftime("%Y-%m-%d"),
            "test_end": pd.Timestamp(
                test_index.get_level_values("date").max()
            ).strftime("%Y-%m-%d"),
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
        }
        return train_idx, test_idx, report
    raise RuntimeError(
        f"fold_index must be 0..{base.N_SPLITS - 1}, got {fold_index}"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_path(
    source_dir: Path,
    value: Any,
    checkpoint_name: Any | None = None,
) -> Path:
    candidates: list[Path] = []
    if value is not None and not (
        isinstance(value, float) and np.isnan(value)
    ):
        path = Path(str(value))
        candidates.append(path)
        if not path.is_absolute():
            candidates.extend(
                [
                    (PROJECT_DIR / path).resolve(),
                    (source_dir / path).resolve(),
                    (source_dir / "search_checkpoints" / path.name).resolve(),
                ]
            )
        else:
            candidates.append(
                (source_dir / "search_checkpoints" / path.name).resolve()
            )
    if checkpoint_name is not None and not (
        isinstance(checkpoint_name, float) and np.isnan(checkpoint_name)
    ):
        name = str(checkpoint_name)
        candidates.extend(
            [
                (source_dir / "search_checkpoints" / f"{name}.keras").resolve(),
                (source_dir / "search_checkpoints" / name).resolve(),
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "cannot resolve checkpoint path; "
        f"source_dir={source_dir} value={value!r} "
        f"checkpoint_name={checkpoint_name!r} "
        f"candidates={[str(candidate) for candidate in candidates]}"
    )


def read_top_checkpoints(source_dir: Path, top_n: int) -> pd.DataFrame:
    path = source_dir / "search_best_checkpoints.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if table.empty:
        raise RuntimeError(f"empty checkpoint table: {path}")
    if "checkpoint_saved" in table.columns:
        saved = (
            table["checkpoint_saved"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        table = table.loc[saved].copy()
    if "daily_ic_median" in table.columns:
        table["daily_ic_median"] = pd.to_numeric(
            table["daily_ic_median"], errors="coerce"
        )
        table = table.sort_values("daily_ic_median", ascending=False)
    table = table.head(top_n).reset_index(drop=True)
    if len(table) < top_n:
        raise RuntimeError(
            f"not enough saved checkpoints in {path}: "
            f"need {top_n}, got {len(table)}"
        )
    if "keras_model" not in table.columns:
        raise RuntimeError(f"checkpoint table missing keras_model column: {path}")
    return table


def load_preprocess(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scaler_path = source_dir / "preprocess" / "scaler.pkl"
    manifest_path = source_dir / "preprocess" / "feature_manifest.json"
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with scaler_path.open("rb") as stream:
        bundle = pickle.load(stream)
    manifest = read_json(manifest_path)
    for key in ("scaler", "scale_cols", "no_scale_cols", "model_input_cols"):
        if key not in bundle:
            raise RuntimeError(f"{scaler_path} missing key {key!r}")
    for key in ("feature_cols_final", "model_input_cols"):
        if key not in manifest:
            raise RuntimeError(f"{manifest_path} missing key {key!r}")
    if list(bundle["model_input_cols"]) != list(manifest["model_input_cols"]):
        raise RuntimeError(
            "model_input_cols mismatch between scaler.pkl and "
            f"feature_manifest.json in {source_dir}"
        )
    return bundle, manifest


def transform_for_source_model(
    X_final: pd.DataFrame,
    row_indices: np.ndarray,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
) -> np.ndarray:
    feature_cols = list(manifest["feature_cols_final"])
    missing = [column for column in feature_cols if column not in X_final.columns]
    if missing:
        raise RuntimeError(
            "target feature matrix missing columns required by source model: "
            f"{missing[:20]}"
        )
    X_selected = X_final.iloc[row_indices][feature_cols]
    scale_cols = list(bundle["scale_cols"])
    no_scale_cols = list(bundle["no_scale_cols"])
    model_input_cols = list(bundle["model_input_cols"])
    if model_input_cols != scale_cols + no_scale_cols:
        raise RuntimeError("model_input_cols must equal scale_cols + no_scale_cols")
    scaled = bundle["scaler"].transform(X_selected[scale_cols]).astype(np.float32)
    if no_scale_cols:
        raw = X_selected[no_scale_cols].to_numpy(dtype=np.float32)
        model_input = np.concatenate([scaled, raw], axis=1)
    else:
        model_input = scaled
    if model_input.shape[1] != len(model_input_cols):
        raise RuntimeError(
            f"bad model input width: got {model_input.shape[1]}, "
            f"expected {len(model_input_cols)}"
        )
    return model_input


def predict_checkpoint_set(
    X_final: pd.DataFrame,
    row_indices: np.ndarray,
    source_dir: Path,
    top_n: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from tensorflow.keras.backend import clear_session
    from tensorflow.keras.models import load_model

    bundle, manifest = load_preprocess(source_dir)
    model_input = transform_for_source_model(
        X_final, row_indices, bundle, manifest
    )
    top = read_top_checkpoints(source_dir, top_n)
    selected_index = X_final.iloc[row_indices].index
    prediction_columns: list[pd.Series] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for model_rank, row in top.iterrows():
        checkpoint_path = resolve_checkpoint_path(
            source_dir,
            row.get("keras_model"),
            row.get("checkpoint_name"),
        )
        model = load_model(str(checkpoint_path), compile=False)
        prediction = model.predict(model_input, verbose=0).reshape(-1)
        if len(prediction) != len(selected_index):
            raise RuntimeError(
                f"prediction length mismatch for {checkpoint_path}: "
                f"got {len(prediction)}, expected {len(selected_index)}"
            )
        prediction_columns.append(
            pd.Series(prediction, index=selected_index, name=int(model_rank))
        )
        checkpoint_meta = row.to_dict()
        checkpoint_meta.update(metadata or {})
        checkpoint_meta.update(
            {
                "model_rank": int(model_rank),
                "resolved_keras_model": str(checkpoint_path),
            }
        )
        checkpoint_rows.append(checkpoint_meta)
        del model
        clear_session()
        gc.collect()

    predictions = pd.concat(prediction_columns, axis=1)
    predictions.columns = list(range(len(prediction_columns)))
    return predictions, checkpoint_rows, manifest


def write_prediction_artifacts(
    *,
    out_root: Path,
    predictions: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    prediction_filename: str,
    manifest_filename: str,
    checkpoint_filename: str,
    manifest: dict[str, Any],
    checkpoint_rows: Sequence[dict[str, Any]],
    prediction_file: Path | None = None,
) -> Path:
    prediction_dir = out_root / "00_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_file or prediction_dir / prediction_filename
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = predictions.sort_index()
    if ordered.index.duplicated().any():
        duplicates = int(ordered.index.duplicated().sum())
        raise RuntimeError(f"duplicate symbol/date prediction rows: {duplicates}")
    ordered.to_hdf(path, key="predictions", mode="w")
    ordered.to_csv(path.with_suffix(".csv"), encoding="utf-8-sig")
    actual_path = prediction_dir / f"actual_{target_col}.csv"
    y.reindex(ordered.index).rename(target_col).to_csv(
        actual_path, encoding="utf-8-sig"
    )
    pd.DataFrame(checkpoint_rows).to_csv(
        prediction_dir / checkpoint_filename,
        index=False,
        encoding="utf-8-sig",
    )

    dates = pd.DatetimeIndex(ordered.index.get_level_values("date"))
    payload = {
        **manifest,
        "created_at_utc": utc_now(),
        "target_col": target_col,
        "target_lookahead": target_lookahead(target_col),
        "prediction_file": str(path),
        "prediction_csv": str(path.with_suffix(".csv")),
        "actual_file": str(actual_path),
        "checkpoint_file": str(prediction_dir / checkpoint_filename),
        "prediction_columns": [str(column) for column in ordered.columns],
        "n_rows": int(len(ordered)),
        "n_dates": int(dates.nunique()),
        "n_symbols": int(ordered.index.get_level_values("symbol").nunique()),
        "prediction_start": pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        "prediction_end": pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
    }
    write_json(prediction_dir / manifest_filename, payload)
    print(f"[OK] predictions: {path}")
    print(f"[OK] prediction manifest: {prediction_dir / manifest_filename}")
    return path


def resolve_fold_test_end(source_dir: Path) -> pd.Timestamp:
    candidates = [
        source_dir / "fold_report.json",
        source_dir / "preprocess" / "feature_manifest.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        value = read_json(path).get("test_end")
        if value:
            return pd.Timestamp(value).normalize()
    raise FileNotFoundError(
        "cannot resolve fold test_end; expected fold_report.json or "
        f"preprocess/feature_manifest.json under {source_dir}"
    )


def build_grid_command(
    *,
    python_bin: str,
    grid_script: Path,
    grid_out: Path,
    prediction_file: Path,
    raw_daily_cache_dir: Path,
    profile: str,
    capacity_mode: str,
    output_mode: str,
    offset_mode: str,
    rebalance_every: int,
    max_positions_list: str,
    sell_rank_list: str,
    model_family: str,
    model_run: str,
    force_grid: bool = False,
    smoke: bool = False,
    parity_check_only: bool = False,
) -> list[str]:
    command = [
        python_bin,
        str(grid_script),
        "--out-root",
        str(grid_out),
        "--predictions",
        str(prediction_file),
        "--prediction-key",
        "predictions",
        "--raw-daily-cache-dir",
        str(raw_daily_cache_dir),
        "--profile",
        profile,
        "--capacity-mode",
        capacity_mode,
        "--run-output-mode",
        output_mode,
        "--offset-mode",
        offset_mode,
        "--rebalance-every-list",
        str(rebalance_every),
        "--max-positions-list",
        max_positions_list,
        "--sell-rank-list",
        sell_rank_list,
        "--model-family",
        model_family,
        "--model-run",
        model_run,
    ]
    if force_grid:
        command.append("--force")
    if smoke:
        command.append("--smoke")
    if parity_check_only:
        command.append("--parity-check-only")
    return command


def run_command(command: Sequence[str], dry_run: bool = False) -> None:
    print("[CMD] " + " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(list(command), check=True)
