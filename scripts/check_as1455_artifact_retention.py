#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic check for prediction artifact retention."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_artifact_retention import compact_prediction_dir  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "00_predictions"
        root.mkdir()
        index = pd.MultiIndex.from_tuples(
            [("000001.SZ", pd.Timestamp("2026-07-10"))],
            names=["symbol", "date"],
        )
        predictions = pd.DataFrame({0: [0.1]}, index=index)
        hdf = root / "fold0_forward_preds.h5"
        csv = root / "fold0_forward_preds.csv"
        actual = root / "actual_r05_fwd.csv"
        predictions.to_hdf(hdf, "predictions", mode="w")
        predictions.to_csv(csv)
        pd.Series([float("nan")], index=index, name="r05_fwd").to_csv(actual)
        manifest = root / "fold0_forward_prediction_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "prediction_file": str(hdf),
                    "prediction_csv": str(csv),
                    "actual_file": str(actual),
                }
            ),
            encoding="utf-8",
        )

        result = compact_prediction_dir(root, apply=True)
        assert not csv.exists()
        assert hdf.exists()
        assert actual.exists(), "actual labels are not duplicated by prediction HDF"
        updated = json.loads(manifest.read_text(encoding="utf-8"))
        assert updated["prediction_csv"] is None
        assert updated["prediction_csv_retained"] is False
        assert updated["actual_file"] == str(actual)
        assert result["actual_labels_retained"] is True
    print("[PASS] AS1455 prediction artifact retention")


if __name__ == "__main__":
    main()
