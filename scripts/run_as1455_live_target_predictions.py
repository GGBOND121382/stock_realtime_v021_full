#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one shared Top-5 fold0 prediction panel for a live AS1455 target.

The preferred production path reads a prepared current-day Chapter-17 feature
matrix. This keeps TensorFlow inference independent from the multi-year HDF
history and lets r01/r05/r21 share one low-memory feature-preparation pass.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_strict_oos_monitor as live  # noqa: E402
from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_live_inference import build_inference_features_from_frame  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=live.json_default),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--target-col", choices=list(common.TARGET_SPECS), required=True)
    parser.add_argument("--feature-preset", default="rotation_addon_onehot")
    parser.add_argument(
        "--model-data",
        default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5",
    )
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--prepared-feature-file", default=None)
    parser.add_argument("--prepared-feature-report", default=None)
    parser.add_argument("--fold0-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def _prepared_features(
    path: Path,
    trade_date: pd.Timestamp,
    required_columns: list[str],
) -> pd.DataFrame:
    X = pd.read_pickle(path)
    if not isinstance(X, pd.DataFrame):
        raise RuntimeError(f"prepared live feature payload is not a DataFrame: {path}")
    if list(X.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected prepared feature index: {X.index.names}")
    X = X.copy()
    X.index = pd.MultiIndex.from_arrays(
        [
            X.index.get_level_values("symbol").astype(str),
            pd.DatetimeIndex(pd.to_datetime(X.index.get_level_values("date"))).normalize(),
        ],
        names=["symbol", "date"],
    )
    dates = pd.DatetimeIndex(X.index.get_level_values("date"))
    X = X.loc[dates == trade_date]
    if X.empty:
        raise RuntimeError(f"prepared feature file has no rows for {trade_date:%Y-%m-%d}: {path}")
    missing = [column for column in required_columns if column not in X.columns]
    if missing:
        raise RuntimeError(f"prepared feature matrix misses model columns: {missing[:20]}")
    return X


def main() -> None:
    args = parse_args()
    trade_date = live.parse_trade_date(args.trade_date)
    feature_file = Path(args.feature_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fold0_dir = (
        Path(args.fold0_dir).expanduser().resolve()
        if args.fold0_dir
        else common.default_fold0_dir(args.feature_preset, args.target_col)
    )
    _, preprocess_manifest = common.load_preprocess(fold0_dir)
    required_columns = list(preprocess_manifest["feature_cols_final"])

    if args.prepared_feature_file:
        prepared = Path(args.prepared_feature_file).expanduser().resolve()
        X_final = _prepared_features(prepared, trade_date, required_columns)
        row_indices = np.arange(len(X_final), dtype=int)
        feature_report: dict = {
            "row_mode": "prepared_low_memory_current_day",
            "prepared_feature_file": str(prepared),
            "rows": int(len(X_final)),
            "columns": int(X_final.shape[1]),
        }
        if args.prepared_feature_report:
            report_path = Path(args.prepared_feature_report).expanduser().resolve()
            if report_path.is_file():
                feature_report.update(json.loads(report_path.read_text(encoding="utf-8")))
    else:
        model_data_path = Path(args.model_data).expanduser().resolve()
        historical = pd.read_hdf(model_data_path, "model_data")
        if list(historical.index.names) != ["symbol", "date"]:
            raise RuntimeError(f"unexpected model_data index: {historical.index.names}")
        historical.index = pd.MultiIndex.from_arrays(
            [
                historical.index.get_level_values("symbol").astype(str),
                pd.to_datetime(historical.index.get_level_values("date")).normalize(),
            ],
            names=["symbol", "date"],
        )
        live_base = live.load_live_base_features(
            feature_file,
            historical.index.get_level_values("symbol").astype(str),
            trade_date,
        )
        combined = pd.concat([historical, live_base], axis=0)
        del historical, live_base
        gc.collect()
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        feature_result = build_inference_features_from_frame(
            combined,
            args.target_col,
            args.feature_preset,
            "onehot",
            source_label=f"{model_data_path}+{feature_file}",
        )
        del combined
        gc.collect()
        feature_dates = pd.DatetimeIndex(feature_result.X.index.get_level_values("date")).normalize()
        row_indices = np.flatnonzero(feature_dates == trade_date)
        if not len(row_indices):
            raise RuntimeError(
                f"no current-date inference rows after feature construction: {trade_date:%Y-%m-%d}"
            )
        X_final = feature_result.X
        feature_report = feature_result.report

    predictions, checkpoints, source_manifest = common.predict_checkpoint_set(
        X_final,
        row_indices,
        fold0_dir,
        args.top_n,
        metadata={
            "source_model_fold": 0,
            "live_trade_date": trade_date.strftime("%Y-%m-%d"),
            "shared_across_fixed_signals": True,
            "prepared_feature_path_used": bool(args.prepared_feature_file),
        },
    )
    if len(predictions.columns) < args.top_n:
        raise RuntimeError(
            f"expected at least {args.top_n} prediction columns, got {list(predictions.columns)}"
        )
    pred_csv = predictions.copy().reset_index()
    pred_csv["symbol"] = pred_csv["symbol"].map(live.exchange_symbol)
    prediction_file = out_dir / "top5_live_predictions.csv"
    pred_csv.to_csv(prediction_file, index=False, encoding="utf-8-sig")
    checkpoints_file = out_dir / "top5_live_checkpoints.csv"
    pd.DataFrame(checkpoints).to_csv(checkpoints_file, index=False, encoding="utf-8-sig")

    manifest = {
        "status": "ok",
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "target_col": args.target_col,
        "feature_preset": args.feature_preset,
        "feature_file": str(feature_file),
        "prepared_feature_file": str(Path(args.prepared_feature_file).expanduser().resolve()) if args.prepared_feature_file else None,
        "fold0_dir": str(fold0_dir),
        "top_n": args.top_n,
        "prediction_rows": int(len(predictions)),
        "prediction_columns": [str(column) for column in predictions.columns],
        "prediction_file": str(prediction_file),
        "checkpoints_file": str(checkpoints_file),
        "source_model_manifest": source_manifest,
        "feature_report": feature_report,
        "shared_signal_semantics": {
            "all5": "mean prediction columns 0,1,2,3,4",
            "first3": "mean prediction columns 0,1,2",
            "best": "single prediction column 0",
        },
    }
    write_json(out_dir / "top5_live_prediction_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "target_col": args.target_col,
                "trade_date": manifest["trade_date"],
                "prediction_rows": manifest["prediction_rows"],
                "prediction_file": str(prediction_file),
                "prepared_feature_path_used": bool(args.prepared_feature_file),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
