#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-train a small model from A-share Chapter 12 model_data.h5.

This is a pipeline validation tool, not a production model search. It verifies
that the generated /model_data can be read, split by time, transformed, fitted,
scored, and serialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce_smoke5_industry" / "model_data.h5"
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_train_smoke"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank(method="average")
    b = pd.Series(y_pred).rank(method="average")
    value = a.corr(b)
    return float(value) if pd.notna(value) else float("nan")


def split_by_date(data: pd.DataFrame, train_frac: float, valid_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(sorted(data.index.get_level_values("date").unique()))
    if len(dates) < 10:
        raise RuntimeError(f"need at least 10 unique dates for smoke split, got {len(dates)}")
    train_end = max(1, int(len(dates) * train_frac))
    valid_end = max(train_end + 1, int(len(dates) * (train_frac + valid_frac)))
    valid_end = min(valid_end, len(dates) - 1)
    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]
    if len(valid_dates) == 0 or len(test_dates) == 0:
        raise RuntimeError("empty valid/test split; adjust split fractions")
    date_index = data.index.get_level_values("date")
    return (
        data.loc[date_index.isin(train_dates)].copy(),
        data.loc[date_index.isin(valid_dates)].copy(),
        data.loc[date_index.isin(test_dates)].copy(),
    )


def split_metrics(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "split": split,
        "rows": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "spearman": spearman(y_true, y_pred),
        "y_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-train Ridge on A-share Chapter 12 model_data.h5")
    p.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--target", default="r01_fwd", choices=["r01_fwd", "r05_fwd", "r21_fwd"])
    p.add_argument("--max-rows", type=int, default=0, help="Optional tail row cap after dropna; 0 means all rows")
    p.add_argument("--train-frac", type=float, default=0.60)
    p.add_argument("--valid-frac", type=float, default=0.20)
    p.add_argument("--alpha", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_data_path = Path(args.model_data)
    out_dir = ensure_dir(Path(args.out_dir))

    data = pd.read_hdf(model_data_path, "model_data").dropna().sort_index()
    if int(args.max_rows) > 0 and len(data) > int(args.max_rows):
        data = data.tail(int(args.max_rows)).copy()
    outcomes = data.filter(like="fwd").columns.tolist()
    if args.target not in outcomes:
        raise RuntimeError(f"target {args.target} not found in outcomes={outcomes}")
    features = [c for c in data.columns if c not in outcomes]
    if not features:
        raise RuntimeError("no feature columns after dropping outcomes")

    train, valid, test = split_by_date(data, float(args.train_frac), float(args.valid_frac))
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(args.alpha))),
        ]
    )
    x_train = train.loc[:, features].to_numpy(dtype=float)
    y_train = train[args.target].to_numpy(dtype=float)
    pipe.fit(x_train, y_train)

    rows = []
    preds = []
    for split, frame in [("train", train), ("valid", valid), ("test", test)]:
        y_true = frame[args.target].to_numpy(dtype=float)
        y_pred = pipe.predict(frame.loc[:, features].to_numpy(dtype=float))
        rows.append(split_metrics(split, y_true, y_pred))
        part = frame.reset_index()[["symbol", "date"]].copy()
        part["split"] = split
        part["target"] = y_true
        part["prediction"] = y_pred
        preds.append(part)

    metrics = {
        "model_data_path": str(model_data_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "target": str(args.target),
        "rows_after_dropna": int(len(data)),
        "symbols": int(data.index.get_level_values("symbol").nunique()),
        "date_start": data.index.get_level_values("date").min().strftime("%Y-%m-%d"),
        "date_end": data.index.get_level_values("date").max().strftime("%Y-%m-%d"),
        "features": int(len(features)),
        "outcomes": outcomes,
        "model": {"family": "sklearn.linear_model.Ridge", "alpha": float(args.alpha)},
        "splits": rows,
    }
    (out_dir / "train_smoke_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "train_smoke_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_csv(out_dir / "train_smoke_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"pipeline": pipe, "features": features, "target": args.target, "outcomes": outcomes}, out_dir / "ridge_smoke_model.joblib")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
