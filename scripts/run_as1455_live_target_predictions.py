#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one shared Top-5 fold0 prediction panel for a live AS1455 target.

The nine-strategy monitor needs three target-specific inference passes (r01/r05/r21),
not nine model passes.  all5/first3/best are derived later from the same five
prediction columns.
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--fold0-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = live.parse_trade_date(args.trade_date)
    model_data_path = Path(args.model_data).expanduser().resolve()
    feature_file = Path(args.feature_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    historical = pd.read_hdf(model_data_path, "model_data")
    if list(historical.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected model_data index: {historical.index.names}")
    historical = historical.copy()
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
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    feature_result = build_inference_features_from_frame(
        combined,
        args.target_col,
        args.feature_preset,
        "onehot",
        source_label=f"{model_data_path}+{feature_file}",
    )
    feature_dates = pd.DatetimeIndex(
        feature_result.X.index.get_level_values("date")
    ).normalize()
    row_indices = np.flatnonzero(feature_dates == trade_date)
    if not len(row_indices):
        raise RuntimeError(
            f"no current-date inference rows after feature construction: {trade_date:%Y-%m-%d}"
        )

    fold0_dir = (
        Path(args.fold0_dir).expanduser().resolve()
        if args.fold0_dir
        else common.default_fold0_dir(args.feature_preset, args.target_col)
    )
    predictions, checkpoints, source_manifest = common.predict_checkpoint_set(
        feature_result.X,
        row_indices,
        fold0_dir,
        args.top_n,
        metadata={
            "source_model_fold": 0,
            "live_trade_date": trade_date.strftime("%Y-%m-%d"),
            "shared_across_fixed_signals": True,
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
        "model_data": str(model_data_path),
        "feature_file": str(feature_file),
        "fold0_dir": str(fold0_dir),
        "top_n": args.top_n,
        "prediction_rows": int(len(predictions)),
        "prediction_columns": [str(column) for column in predictions.columns],
        "prediction_file": str(prediction_file),
        "checkpoints_file": str(checkpoints_file),
        "source_model_manifest": source_manifest,
        "feature_report": feature_result.report,
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
