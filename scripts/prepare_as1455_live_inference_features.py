#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare one shared low-memory Chapter-17 feature matrix for live inference."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_as1455_live_strict_oos_monitor as live  # noqa: E402
from utils import as1455_ch17_common as common  # noqa: E402
from utils.as1455_live_inference_lowmem import (  # noqa: E402
    build_current_day_inference_features,
    load_live_history_context,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trade-date", required=True)
    p.add_argument("--feature-preset", default="rotation_addon_onehot")
    p.add_argument(
        "--model-data",
        default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5",
    )
    p.add_argument("--feature-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--report-file", default=None)
    p.add_argument("--hdf-chunksize", type=int, default=100_000)
    return p.parse_args()


def rss_peak_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


def main() -> None:
    args = parse_args()
    trade_date = live.parse_trade_date(args.trade_date)
    model_data = Path(args.model_data).expanduser().resolve()
    feature_file = Path(args.feature_file).expanduser().resolve()
    out_file = Path(args.out_file).expanduser().resolve()
    report_file = (
        Path(args.report_file).expanduser().resolve()
        if args.report_file
        else out_file.with_suffix(out_file.suffix + ".report.json")
    )

    context = load_live_history_context(
        model_data,
        trade_date,
        chunksize=args.hdf_chunksize,
    )
    live_base = live.load_live_base_features(
        feature_file,
        pd.Index([context.symbol_sample]),
        trade_date,
    )

    target_manifests: dict[str, dict] = {}
    required_union: list[str] = []
    for target_col in common.TARGET_SPECS:
        fold0_dir = common.default_fold0_dir(args.feature_preset, target_col)
        _, manifest = common.load_preprocess(fold0_dir)
        target_manifests[target_col] = manifest
        for column in manifest["feature_cols_final"]:
            if column not in required_union:
                required_union.append(column)

    result = build_current_day_inference_features(
        live_base,
        context,
        args.feature_preset,
        required_feature_columns=required_union,
    )
    del context, live_base
    gc.collect()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    result.X.to_pickle(out_file, protocol=4)
    report = {
        "status": "ok",
        "protocol": "as1455_live_shared_inference_features_lowmem_v1",
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "feature_preset": args.feature_preset,
        "model_data": str(model_data),
        "source_feature_file": str(feature_file),
        "prepared_feature_file": str(out_file),
        "rows": int(len(result.X)),
        "columns": int(result.X.shape[1]),
        "required_union_columns": required_union,
        "target_feature_column_counts": {
            target: len(manifest["feature_cols_final"])
            for target, manifest in target_manifests.items()
        },
        "peak_rss_mb": rss_peak_mb(),
        **result.report,
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "trade_date": report["trade_date"],
        "rows": report["rows"],
        "columns": report["columns"],
        "history_read_mode": report.get("history_read_mode"),
        "peak_rss_mb": report["peak_rss_mb"],
        "prepared_feature_file": str(out_file),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
