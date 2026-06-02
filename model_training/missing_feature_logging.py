#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small helpers for logging feature missingness before model fit/predict."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def feature_missing_report(
    df: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    max_missing: float | None = None,
    sample_path: str | Path | None = None,
) -> pd.DataFrame:
    rows = []
    for group_name, cols in groups.items():
        for col in cols:
            if col not in df.columns:
                rows.append({
                    "sample_path": str(sample_path or ""),
                    "feature_group": group_name,
                    "feature": col,
                    "action": "dropped",
                    "reason": "missing_column",
                    "rows": int(len(df)),
                    "non_null_count": 0,
                    "missing_count": int(len(df)),
                    "missing_rate": 1.0 if len(df) else np.nan,
                    "max_missing": np.nan if max_missing is None else float(max_missing),
                })
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            missing_count = int(s.isna().sum())
            non_null_count = int(s.notna().sum())
            missing_rate = float(s.isna().mean()) if len(s) else np.nan
            if max_missing is not None and missing_rate > float(max_missing):
                action = "dropped"
                reason = "missing_rate_gt_max"
            elif non_null_count < 3:
                action = "dropped"
                reason = "too_few_non_null"
            else:
                action = "kept"
                reason = "ok"
            rows.append({
                "sample_path": str(sample_path or ""),
                "feature_group": group_name,
                "feature": col,
                "action": action,
                "reason": reason,
                "rows": int(len(df)),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "max_missing": np.nan if max_missing is None else float(max_missing),
            })
    return pd.DataFrame(rows)


def log_and_write_feature_missing_report(
    report: pd.DataFrame,
    out_dir: str | Path | None,
    filename: str = "feature_missing_report.csv",
    context: str = "",
) -> None:
    label = f" {context}" if context else ""
    if report.empty:
        print(f"[MISSING]{label} feature_missing_report=empty", flush=True)
        return
    kept = int((report["action"] == "kept").sum())
    dropped = int((report["action"] == "dropped").sum())
    high_missing = int(((report["action"] == "dropped") & (report["reason"] == "missing_rate_gt_max")).sum())
    too_few = int(((report["action"] == "dropped") & (report["reason"] == "too_few_non_null")).sum())
    print(
        f"[MISSING]{label} feature_candidates={len(report)} kept={kept} dropped={dropped} "
        f"dropped_high_missing={high_missing} dropped_too_few_non_null={too_few}",
        flush=True,
    )
    worst_kept = report[report["action"] == "kept"].sort_values("missing_rate", ascending=False).head(5)
    if not worst_kept.empty:
        cols = [f"{r.feature}:{float(r.missing_rate):.3f}" for r in worst_kept.itertuples()]
        print(f"[MISSING]{label} worst_kept=" + ",".join(cols), flush=True)
    worst_dropped = report[report["action"] == "dropped"].sort_values("missing_rate", ascending=False).head(8)
    if not worst_dropped.empty:
        cols = [f"{r.feature}:{float(r.missing_rate):.3f}:{r.reason}" for r in worst_dropped.itertuples()]
        print(f"[MISSING]{label} worst_dropped=" + ",".join(cols), flush=True)
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        report.to_csv(path / filename, index=False, encoding="utf-8-sig")


def matrix_missing_stats(frame: pd.DataFrame, cols: Sequence[str], prefix: str) -> Dict[str, float | int]:
    present = [c for c in cols if c in frame.columns]
    rows = int(len(frame))
    n_features = int(len(present))
    total_cells = rows * n_features
    missing_columns = int(len(cols) - len(present))
    if rows == 0 or not present:
        return {
            f"{prefix}_missing_columns": missing_columns,
            f"{prefix}_missing_cells": 0,
            f"{prefix}_total_feature_cells": int(total_cells),
            f"{prefix}_missing_rate": np.nan,
            f"{prefix}_features_with_missing": 0,
            f"{prefix}_max_feature_missing_rate": np.nan,
        }
    x = frame.loc[:, present].apply(pd.to_numeric, errors="coerce")
    missing_by_col = x.isna().sum()
    missing_cells = int(missing_by_col.sum())
    return {
        f"{prefix}_missing_columns": missing_columns,
        f"{prefix}_missing_cells": missing_cells,
        f"{prefix}_total_feature_cells": int(total_cells),
        f"{prefix}_missing_rate": float(missing_cells / total_cells) if total_cells else np.nan,
        f"{prefix}_features_with_missing": int((missing_by_col > 0).sum()),
        f"{prefix}_max_feature_missing_rate": float(x.isna().mean().max()) if n_features else np.nan,
    }
