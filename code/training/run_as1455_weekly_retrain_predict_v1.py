#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weekly rolling retrain + prediction generator for AS1455 Ch17 NN.

This script is intentionally separate from the existing CV/OOS prediction script.
It trains fixed hyper-parameter NN models once per weekly update date using only
labels observable by that update date's close, then predicts the next holding
period until the next weekly update.

Default base model specs are the five hyper-parameter rows recorded by the
20260622_cv7 run and used by prediction columns 0..4.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from ast import literal_eval
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

EXPECTED_OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]
DEFAULT_MODEL_SPECS = [
    {"prediction_col": 0, "dense_layers": "(16, 8)", "activation": "tanh", "dropout": 0.2, "batch_size": 256, "epoch": 3},
    {"prediction_col": 1, "dense_layers": "(16, 8)", "activation": "tanh", "dropout": 0.1, "batch_size": 256, "epoch": 2},
    {"prediction_col": 2, "dense_layers": "(32, 16)", "activation": "tanh", "dropout": 0.2, "batch_size": 64, "epoch": 18},
    {"prediction_col": 3, "dense_layers": "(64, 32)", "activation": "tanh", "dropout": 0.1, "batch_size": 256, "epoch": 2},
    {"prediction_col": 4, "dense_layers": "(64, 32)", "activation": "tanh", "dropout": 0.1, "batch_size": 256, "epoch": 5},
]


@dataclass
class UpdateJob:
    update_id: int
    update_date: str
    label_train_end: str
    predict_start: str
    predict_end: str
    n_train_dates: int
    n_train_rows: int
    n_pred_dates: int
    n_pred_rows: int


def require_deps() -> None:
    missing = []
    try:
        import tables  # noqa: F401
    except Exception:
        missing.append("tables/PyTables")
    try:
        import tensorflow  # noqa: F401
    except Exception:
        missing.append("tensorflow")
    if missing:
        raise SystemExit("Missing required dependency: " + ", ".join(missing))


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def clear_model_session() -> None:
    try:
        from tensorflow.keras.backend import clear_session
        clear_session()
    except Exception:
        pass
    gc.collect()


def make_model(input_dim: int, dense_layers: tuple[int, ...], activation: str, dropout: float):
    from tensorflow.keras.layers import Activation, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    for i, layer_size in enumerate(dense_layers, 1):
        if i == 1:
            model.add(Dense(layer_size, input_dim=input_dim))
            model.add(Activation(activation))
        else:
            model.add(Dense(layer_size))
            model.add(Activation(activation))
    model.add(Dropout(float(dropout)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Adam")
    return model


def load_model_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_hdf(path, "model_data")
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}; expected ['symbol', 'date']")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}; expected {EXPECTED_OUTCOMES}")
    data = data.sort_index()
    y = data["r01_fwd"].copy()
    X = data.drop(["r01_fwd", "r05_fwd", "r21_fwd"], axis=1)
    if any("fwd" in str(c) for c in X.columns):
        raise RuntimeError("feature matrix still contains fwd columns")
    if X.shape[1] != 31:
        raise RuntimeError(f"expected 31 features, got {X.shape[1]}")
    return X, y


def parse_model_specs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        specs = DEFAULT_MODEL_SPECS
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            specs = json.loads(path.read_text(encoding="utf-8"))
        else:
            specs = pd.read_csv(path).to_dict("records")
    out = []
    for i, s in enumerate(specs):
        row = dict(s)
        row.setdefault("prediction_col", i)
        row["prediction_col"] = int(row["prediction_col"])
        row["dense_layers"] = str(row["dense_layers"])
        row["activation"] = str(row.get("activation", "tanh"))
        row["dropout"] = float(row.get("dropout", 0.0))
        row["batch_size"] = int(row.get("batch_size", 64))
        row["epoch"] = int(row.get("epoch", 0))
        # Original Ch17 script stores checkpoint/prediction after each 0-based epoch.
        # Therefore selected epoch e means the model has been fit for e+1 epochs.
        row["fit_epochs"] = int(row["epoch"]) + 1
        out.append(row)
    out = sorted(out, key=lambda x: int(x["prediction_col"]))
    return out


def previous_trading_date(dates: list[pd.Timestamp], d: pd.Timestamp) -> pd.Timestamp | None:
    prev = [x for x in dates if x < d]
    return prev[-1] if prev else None


def next_trading_date(dates: list[pd.Timestamp], d: pd.Timestamp) -> pd.Timestamp | None:
    nxt = [x for x in dates if x > d]
    return nxt[0] if nxt else None


def make_weekly_update_dates(dates: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    date_frame = pd.DataFrame({"date": dates})
    # Week ending Friday. If Friday is not a trading day, use the last available trading day of that week.
    date_frame["week"] = date_frame["date"].dt.to_period("W-FRI").astype(str)
    weekly_last = date_frame.groupby("week")["date"].max().sort_values().tolist()
    before_start = [d for d in weekly_last if d < start]
    if before_start:
        initial = before_start[-1]
    else:
        p = previous_trading_date(dates, start)
        if p is None:
            raise RuntimeError("no trading date before start-date; cannot create initial weekly model")
        initial = p
    in_range_updates = [d for d in weekly_last if start <= d < end]
    updates = [initial] + in_range_updates
    dedup = []
    seen = set()
    for d in updates:
        if d not in seen:
            dedup.append(d)
            seen.add(d)
    return dedup


def build_update_jobs(X: pd.DataFrame, start_date: str | None, end_date: str | None, max_updates: int | None = None) -> list[UpdateJob]:
    all_dates = sorted(pd.to_datetime(X.index.get_level_values("date").unique()).normalize())
    start = pd.Timestamp(start_date).normalize() if start_date else all_dates[0]
    end = pd.Timestamp(end_date).normalize() if end_date else all_dates[-1]
    pred_dates = [d for d in all_dates if start <= d <= end]
    if not pred_dates:
        raise RuntimeError(f"no prediction dates in [{start.date()}, {end.date()}]")
    updates = make_weekly_update_dates(all_dates, pred_dates[0], pred_dates[-1])
    jobs: list[UpdateJob] = []
    for i, u in enumerate(updates):
        next_u = updates[i + 1] if i + 1 < len(updates) else None
        p_start = next_trading_date(all_dates, u)
        if p_start is None:
            continue
        p_start = max(p_start, pred_dates[0])
        p_end = next_u if next_u is not None else pred_dates[-1]
        p_end = min(p_end, pred_dates[-1])
        if p_start > p_end:
            continue
        label_end = previous_trading_date(all_dates, u)
        if label_end is None:
            continue
        train_mask_dates = [d for d in all_dates if d <= label_end]
        pred_mask_dates = [d for d in all_dates if p_start <= d <= p_end]
        n_train_rows = int(X.loc[X.index.get_level_values("date").isin(train_mask_dates)].dropna().shape[0])
        n_pred_rows = int(X.loc[X.index.get_level_values("date").isin(pred_mask_dates)].dropna().shape[0])
        if n_train_rows == 0 or n_pred_rows == 0:
            continue
        jobs.append(UpdateJob(
            update_id=len(jobs),
            update_date=u.strftime("%Y-%m-%d"),
            label_train_end=label_end.strftime("%Y-%m-%d"),
            predict_start=p_start.strftime("%Y-%m-%d"),
            predict_end=p_end.strftime("%Y-%m-%d"),
            n_train_dates=len(train_mask_dates),
            n_train_rows=n_train_rows,
            n_pred_dates=len(pred_mask_dates),
            n_pred_rows=n_pred_rows,
        ))
        if max_updates is not None and len(jobs) >= max_updates:
            break
    if not jobs:
        raise RuntimeError("empty weekly update job list")
    return jobs


def select_by_date(df: pd.DataFrame | pd.Series, start: str | None = None, end: str | None = None):
    dates = pd.to_datetime(df.index.get_level_values("date")).normalize()
    mask = pd.Series(True, index=np.arange(len(df)))
    if start is not None:
        mask &= dates >= pd.Timestamp(start).normalize()
    if end is not None:
        mask &= dates <= pd.Timestamp(end).normalize()
    return df.iloc[np.asarray(mask)]


def train_predict_one_update(
    X: pd.DataFrame,
    y: pd.Series,
    job: UpdateJob,
    specs: list[dict[str, Any]],
    update_dir: Path,
    seed: int,
    force: bool,
    verbose: int,
) -> pd.DataFrame:
    pred_path = update_dir / "predictions.h5"
    if pred_path.exists() and not force:
        return pd.read_hdf(pred_path, "predictions")

    update_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_X = select_by_date(X, end=job.label_train_end)
    train_y = select_by_date(y, end=job.label_train_end)
    train = train_X.join(train_y.rename("target")).dropna()
    if train.empty:
        raise RuntimeError(f"empty train data for update {job.update_date}")
    train_y_clean = train.pop("target")
    train_X_clean = train.astype("float32")

    pred_X = select_by_date(X, start=job.predict_start, end=job.predict_end).dropna().astype("float32")
    if pred_X.empty:
        raise RuntimeError(f"empty predict data for update {job.update_date}")

    model_rows = []
    pred_cols = []
    for spec in specs:
        col = int(spec["prediction_col"])
        model_seed = seed + int(job.update_id) * 1000 + col
        set_seed(model_seed)
        m_start = time.time()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_X_clean).astype("float32")
        x_pred = scaler.transform(pred_X).astype("float32")
        dense_layers = literal_eval(str(spec["dense_layers"]))
        if isinstance(dense_layers, int):
            dense_layers = (dense_layers,)
        dense_layers = tuple(int(x) for x in dense_layers)
        model = make_model(train_X_clean.shape[1], dense_layers, str(spec["activation"]), float(spec["dropout"]))
        model.fit(
            x_train,
            train_y_clean.values.astype("float32"),
            batch_size=int(spec["batch_size"]),
            epochs=int(spec["fit_epochs"]),
            verbose=verbose,
            shuffle=True,
        )
        preds = model.predict(x_pred, verbose=0).squeeze()
        pred_cols.append(pd.Series(preds, index=pred_X.index, name=col))
        model_rows.append({
            **asdict(job),
            "prediction_col": col,
            "dense_layers": spec["dense_layers"],
            "activation": spec["activation"],
            "dropout": spec["dropout"],
            "batch_size": spec["batch_size"],
            "selected_epoch": spec["epoch"],
            "fit_epochs": spec["fit_epochs"],
            "seed": model_seed,
            "elapsed_seconds": round(time.time() - m_start, 3),
        })
        del model, scaler, x_train, x_pred, preds
        clear_model_session()

    out = pd.concat(pred_cols, axis=1).sort_index()
    out.columns = [int(c) for c in out.columns]
    out.to_hdf(pred_path, "predictions", mode="w")
    pd.DataFrame(model_rows).to_csv(update_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    (update_dir / "update_meta.json").write_text(json.dumps({
        **asdict(job),
        "elapsed_seconds": round(time.time() - started, 3),
        "prediction_rows": int(len(out)),
        "prediction_cols": [int(c) for c in out.columns],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AS1455 weekly rolling retrain + prediction generator")
    p.add_argument("--model-data", default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start-date", default="2024-07-17")
    p.add_argument("--end-date", default="2026-05-15")
    p.add_argument("--model-spec-file", default=None, help="Optional CSV/JSON with dense_layers,activation,dropout,batch_size,epoch,prediction_col")
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--max-updates", type=int, default=None, help="Smoke/debug: run only the first N weekly updates")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_deps()
    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"
    updates_dir = results_dir / "weekly_updates"
    results_dir.mkdir(parents=True, exist_ok=True)
    updates_dir.mkdir(parents=True, exist_ok=True)

    specs = parse_model_specs(Path(args.model_spec_file) if args.model_spec_file else None)
    pd.DataFrame(specs).to_csv(results_dir / "best_params.csv", index=False, encoding="utf-8-sig")

    X, y = load_model_data(Path(args.model_data))
    jobs = build_update_jobs(X, args.start_date, args.end_date, max_updates=args.max_updates)
    pd.DataFrame([asdict(j) for j in jobs]).to_csv(results_dir / "model_update_schedule.csv", index=False, encoding="utf-8-sig")

    all_preds = []
    all_train_logs = []
    for job in jobs:
        print(json.dumps({"status": "update_start", **asdict(job)}, ensure_ascii=False), flush=True)
        update_dir = updates_dir / f"update_{job.update_date.replace('-', '')}"
        preds = train_predict_one_update(X, y, job, specs, update_dir, args.seed, args.force, args.verbose)
        all_preds.append(preds)
        log_path = update_dir / "training_log.csv"
        if log_path.exists():
            all_train_logs.append(pd.read_csv(log_path))
        print(json.dumps({"status": "update_done", "update_date": job.update_date, "pred_rows": int(len(preds))}, ensure_ascii=False), flush=True)

    predictions = pd.concat(all_preds).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="last")]
    predictions.to_hdf(results_dir / "weekly_predictions.h5", "predictions", mode="w")
    if all_train_logs:
        pd.concat(all_train_logs, ignore_index=True).to_csv(results_dir / "weekly_training_log.csv", index=False, encoding="utf-8-sig")

    dates = pd.to_datetime(predictions.index.get_level_values("date")).normalize()
    summary = {
        "model_data": str(Path(args.model_data)),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_update_jobs": len(jobs),
        "prediction_file": str(results_dir / "weekly_predictions.h5"),
        "prediction_rows": int(len(predictions)),
        "prediction_columns": [int(c) for c in predictions.columns],
        "prediction_date_min": dates.min().strftime("%Y-%m-%d"),
        "prediction_date_max": dates.max().strftime("%Y-%m-%d"),
        "label_rule": "At update-date close, train rows use dates <= previous trading date (label_train_end), because r01_fwd for update_date is not yet observable.",
        "execution_rule": "Model updated after update-date close; predictions from this model are used from the next trading day through the next update date.",
    }
    (results_dir / "weekly_predictions_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
