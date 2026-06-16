#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run A-share Chapter 17 NN training on Chapter 12-style model_data.

This script intentionally reproduces the Chapter 17 training data path and
model loop. It changes only paths and the train-end date.
"""
from __future__ import annotations

import argparse
import gc
import json
from ast import literal_eval as make_tuple
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DATA = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce" / "model_data.h5"
DEFAULT_OUT_DIR = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_reproduce"

LOOKAHEAD = 1
N_SPLITS = 12
TRAIN_PERIOD_LENGTH = 21 * 12 * 4
TEST_PERIOD_LENGTH = 21 * 3
EPOCHS = 20
TRAIN_IC_DATES = 24 * 21
PARAMS = ["dense_layers", "activation", "dropout", "batch_size"]
EXPECTED_OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]

DENSE_LAYER_OPTS = [(16, 8), (32, 16), (32, 32), (64, 32)]
ACTIVATION_OPTS = ["tanh"]
DROPOUT_OPTS = [0, 0.1, 0.2]
BATCH_SIZE_OPTS = [64, 256]


@dataclass
class TrainDataSummary:
    model_data_path: str
    results_dir: str
    train_end: str
    min_date: str
    max_date: str
    n_dates: int
    n_symbols: int
    n_rows_before_dropna: int
    n_rows_after_dropna: int
    X_cv_shape: list[int]
    y_cv_shape: list[int]
    n_features: int
    outcomes: list[str]
    n_splits: int
    train_period_length: int
    test_period_length: int
    lookahead: int
    param_grid_size_before_batch_size: int
    param_grid_size: int
    epochs: int


def require_runtime_deps() -> None:
    missing = []
    try:
        import tensorflow  # noqa: F401
    except Exception:
        missing.append("tensorflow")
    try:
        import tables  # noqa: F401
    except Exception:
        missing.append("tables/PyTables")
    if missing:
        raise SystemExit("Missing required dependency: " + ", ".join(missing))


def format_time(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"


class MultipleTimeSeriesCV:
    """Same splitter as ML4T utils.py."""

    def __init__(
        self,
        n_splits: int = 3,
        train_period_length: int = 126,
        test_period_length: int = 21,
        lookahead: int | None = None,
        date_idx: str = "date",
        shuffle: bool = False,
    ) -> None:
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X: pd.DataFrame, y: pd.Series | None = None, groups: Any = None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > days[train_start]) & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start]) & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()


def get_train_valid_data(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray):
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_val, y_val = X.iloc[test_idx], y.iloc[test_idx]
    return x_train, y_train, x_val, y_val


def make_model(input_dim: int, dense_layers: Iterable[int], activation: str, dropout: float):
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
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Adam")
    return model


def clear_model_session() -> None:
    from tensorflow.keras.backend import clear_session

    clear_session()
    gc.collect()


def load_training_data(model_data_path: Path, train_end: Any):
    data = pd.read_hdf(model_data_path, "model_data")
    n_rows_before_dropna = int(len(data))
    if list(data.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected index names: {data.index.names}")
    outcomes = data.filter(like="fwd").columns.tolist()
    if outcomes != EXPECTED_OUTCOMES:
        raise RuntimeError(f"unexpected outcomes: {outcomes}")

    dates = data.index.get_level_values("date")
    if train_end is None:
        train_end = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    else:
        data = data.loc[dates <= train_end]
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    n_rows_after_dropna = int(len(data))

    y_cv = data.pop("r01_fwd")
    data.drop(["r05_fwd", "r21_fwd"], axis=1, inplace=True)
    X_cv = data
    if any("fwd" in c for c in X_cv.columns):
        raise RuntimeError("X_cv contains fwd columns")
    if y_cv.empty or y_cv.isna().any():
        raise RuntimeError("y_cv is empty or contains NA")
    return X_cv, y_cv, outcomes, train_end, n_rows_before_dropna, n_rows_after_dropna


def write_train_data_summary(
    path: Path,
    model_data_path: Path,
    results_dir: Path,
    train_end: str,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    outcomes: list[str],
    n_rows_before_dropna: int,
    n_rows_after_dropna: int,
) -> TrainDataSummary:
    dates = X_cv.index.get_level_values("date")
    summary = TrainDataSummary(
        model_data_path=str(model_data_path.resolve()),
        results_dir=str(results_dir.resolve()),
        train_end=train_end,
        min_date=pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        max_date=pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
        n_dates=int(dates.nunique()),
        n_symbols=int(X_cv.index.get_level_values("symbol").nunique()),
        n_rows_before_dropna=n_rows_before_dropna,
        n_rows_after_dropna=n_rows_after_dropna,
        X_cv_shape=[int(X_cv.shape[0]), int(X_cv.shape[1])],
        y_cv_shape=[int(y_cv.shape[0])],
        n_features=int(X_cv.shape[1]),
        outcomes=outcomes,
        n_splits=N_SPLITS,
        train_period_length=TRAIN_PERIOD_LENGTH,
        test_period_length=TEST_PERIOD_LENGTH,
        lookahead=LOOKAHEAD,
        param_grid_size_before_batch_size=len(DENSE_LAYER_OPTS) * len(ACTIVATION_OPTS) * len(DROPOUT_OPTS),
        param_grid_size=len(DENSE_LAYER_OPTS) * len(ACTIVATION_OPTS) * len(DROPOUT_OPTS) * len(BATCH_SIZE_OPTS),
        epochs=EPOCHS,
    )
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_cv() -> MultipleTimeSeriesCV:
    return MultipleTimeSeriesCV(
        n_splits=N_SPLITS,
        train_period_length=TRAIN_PERIOD_LENGTH,
        test_period_length=TEST_PERIOD_LENGTH,
        lookahead=LOOKAHEAD,
    )


def write_cv_split_report(cv: MultipleTimeSeriesCV, X_cv: pd.DataFrame, y_cv: pd.Series, report_path: Path) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_cv)):
        train_index = X_cv.iloc[train_idx].index
        test_index = X_cv.iloc[test_idx].index
        rows.append(
            {
                "fold": fold,
                "train_start": pd.Timestamp(train_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
                "train_end": pd.Timestamp(train_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
                "test_start": pd.Timestamp(test_index.get_level_values("date").min()).strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(test_index.get_level_values("date").max()).strftime("%Y-%m-%d"),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_train_symbols": int(train_index.get_level_values("symbol").nunique()),
                "n_test_symbols": int(test_index.get_level_values("symbol").nunique()),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    return report


def param_grid_frame() -> pd.DataFrame:
    rows = []
    for dense_layers, activation, dropout in product(DENSE_LAYER_OPTS, ACTIVATION_OPTS, DROPOUT_OPTS):
        for batch_size in BATCH_SIZE_OPTS:
            rows.append({"dense_layers": str(dense_layers), "activation": activation, "dropout": dropout, "batch_size": batch_size})
    return pd.DataFrame(rows)


def train_cv(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    results_dir: Path,
    logs_dir: Path,
    force_train: bool,
    smoke: bool,
    full_cv_smoke: bool = False,
) -> None:
    scores_path = results_dir / "scores.h5"
    if scores_path.exists() and not force_train:
        print("Skipping NN CV training because results/scores.h5 already exists.")
        return
    if scores_path.exists() and force_train:
        scores_path.unlink()

    cv = make_cv()
    if smoke:
        dense_layer_opts = [DENSE_LAYER_OPTS[0]]
        dropout_opts = [DROPOUT_OPTS[0]]
        batch_size_opts = [BATCH_SIZE_OPTS[0]]
        n_epochs = 2
        max_folds = N_SPLITS if full_cv_smoke else 1
    else:
        dense_layer_opts = DENSE_LAYER_OPTS
        dropout_opts = DROPOUT_OPTS
        batch_size_opts = BATCH_SIZE_OPTS
        n_epochs = EPOCHS
        max_folds = N_SPLITS

    ic = []
    scaler = StandardScaler()
    param_grid = list(product(dense_layer_opts, ACTIVATION_OPTS, dropout_opts))
    np.random.shuffle(param_grid)
    for dense_layers, activation, dropout in param_grid:
        for batch_size in batch_size_opts:
            print(dense_layers, activation, dropout, batch_size, flush=True)
            checkpoint_dir = logs_dir / str(dense_layers) / activation / str(dropout) / str(batch_size)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            start = time()
            for fold, (train_idx, test_idx) in enumerate(cv.split(X_cv)):
                if fold >= max_folds:
                    break
                x_train, y_train, x_val, y_val = get_train_valid_data(X_cv, y_cv, train_idx, test_idx)
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                preds = y_val.to_frame("actual")
                r = pd.DataFrame(index=y_val.groupby(level="date").size().index)
                model = make_model(X_cv.shape[1], dense_layers, activation, dropout)
                for epoch in range(n_epochs):
                    model.fit(
                        x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=0,
                        shuffle=True,
                        validation_data=(x_val, y_val),
                    )
                    model.save_weights((checkpoint_dir / f"ckpt_{fold}_{epoch}.weights.h5").as_posix())
                    preds[epoch] = model.predict(x_val, verbose=0).squeeze()
                    r[epoch] = preds.groupby(level="date", group_keys=False).apply(lambda x: spearmanr(x.actual, x[epoch])[0]).to_frame(epoch)
                    print(format_time(time() - start), f"{fold + 1:02d} | {epoch + 1:02d} | {r[epoch].mean():7.4f} | {r[epoch].median():7.4f}", flush=True)
                ic.append(
                    r.assign(
                        dense_layers=str(dense_layers),
                        activation=activation,
                        dropout=dropout,
                        batch_size=batch_size,
                        fold=fold,
                    )
                )
                del model, x_train, y_train, x_val, y_val, preds, r
                clear_model_session()
            pd.concat(ic).to_hdf(scores_path, "ic_by_day")


def get_best_params(results_dir: Path, n: int = 5) -> list[dict[str, Any]]:
    ic = pd.read_hdf(results_dir / "scores.h5", "ic_by_day").drop("fold", axis=1)
    dates = sorted(ic.index.unique())
    train_dates = dates[:TRAIN_IC_DATES]
    ic = ic.loc[train_dates]
    out = (
        ic.groupby(PARAMS)
        .median()
        .stack()
        .to_frame("ic")
        .reset_index()
        .rename(columns={"level_4": "epoch"})
        .nlargest(n=n, columns="ic")
    )
    selected = out.drop("ic", axis=1)
    selected.to_csv(results_dir / "best_params.csv", index=False, encoding="utf-8-sig")
    return selected.to_dict("records")


def generate_predictions_one(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    logs_dir: Path,
    dense_layers: str,
    activation: str,
    dropout: float,
    batch_size: int,
    epoch: int,
    max_folds: int | None = None,
) -> pd.Series:
    cv = make_cv()
    scaler = StandardScaler()
    predictions = []
    dropout_text = "0" if str(dropout) == "0.0" else str(dropout)
    checkpoint_dir = logs_dir / str(dense_layers) / activation / dropout_text / str(batch_size)
    if not checkpoint_dir.exists():
        checkpoint_dir = logs_dir / str(dense_layers) / activation / str(dropout) / str(batch_size)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_cv)):
        if max_folds is not None and fold >= max_folds:
            break
        x_train, y_train, x_val, y_val = get_train_valid_data(X_cv, y_cv, train_idx, test_idx)
        x_val = scaler.fit(x_train).transform(x_val)
        model = make_model(X_cv.shape[1], make_tuple(dense_layers), activation, dropout)
        checkpoint_path = checkpoint_dir / f"ckpt_{fold}_{int(epoch)}.weights.h5"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint for fold={fold}, epoch={epoch}: {checkpoint_path}")
        status = model.load_weights(checkpoint_path.as_posix())
        if hasattr(status, "expect_partial"):
            status.expect_partial()
        predictions.append(pd.Series(model.predict(x_val, verbose=0).squeeze(), index=y_val.index))
        del model, x_train, y_train, x_val, y_val
        clear_model_session()
    return pd.concat(predictions)


def generate_predictions(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    results_dir: Path,
    logs_dir: Path,
    smoke: bool = False,
    full_cv_smoke: bool = False,
) -> pd.DataFrame:
    best_params = get_best_params(results_dir, n=1 if smoke else 5)
    predictions = []
    for i, params in enumerate(best_params):
        params = dict(params)
        params["dropout"] = float(params["dropout"])
        params["batch_size"] = int(params["batch_size"])
        params["epoch"] = int(params["epoch"])
        max_folds = N_SPLITS if (smoke and full_cv_smoke) else (1 if smoke else None)
        predictions.append(generate_predictions_one(X_cv, y_cv, logs_dir, max_folds=max_folds, **params).to_frame(i))
    out = pd.concat(predictions, axis=1)
    out.columns = list(range(len(predictions)))
    out.to_hdf(results_dir / "test_preds.h5", "predictions")
    return out


def write_score_summaries(results_dir: Path) -> None:
    ic = pd.read_hdf(results_dir / "scores.h5", "ic_by_day")
    epoch_cols = [c for c in ic.columns if isinstance(c, int) or str(c).isdigit()]
    summary = (
        ic.groupby(PARAMS)[epoch_cols]
        .median()
        .stack()
        .rename("median_daily_ic")
        .reset_index()
        .rename(columns={"level_4": "epoch"})
        .sort_values("median_daily_ic", ascending=False)
    )
    summary.to_csv(results_dir / "scores_summary.csv", index=False, encoding="utf-8-sig")


def write_predictions_summary(results_dir: Path, predictions: pd.DataFrame, best_params: pd.DataFrame | None = None) -> None:
    dates = predictions.index.get_level_values("date")
    symbols = predictions.index.get_level_values("symbol")
    summary = {
        "prediction_rows": int(len(predictions)),
        "prediction_symbols": int(symbols.nunique()),
        "prediction_columns": int(predictions.shape[1]),
        "columns": [int(c) for c in predictions.columns],
        "date_min": pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        "date_max": pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
    }
    (results_dir / "predictions_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strict ML4T Chapter 17 NN training on A-share model_data")
    p.add_argument("--model-data", default=str(DEFAULT_MODEL_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--train-end", default=None, help="YYYY or YYYY-MM-DD. Default: max date in model_data")
    p.add_argument("--force-train", action="store_true", help="Retrain even when results/scores.h5 exists")
    p.add_argument("--smoke", action="store_true", help="Run 1 param combo, 1 fold, 2 epochs")
    p.add_argument("--full-cv-smoke", action="store_true", help="With --smoke, run all CV folds while keeping 1 param combo and 2 epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_cv_smoke and not args.smoke:
        raise SystemExit("--full-cv-smoke requires --smoke")
    require_runtime_deps()
    model_data_path = Path(args.model_data)
    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    X_cv, y_cv, outcomes, train_end, n_rows_before_dropna, n_rows_after_dropna = load_training_data(model_data_path, args.train_end)
    write_train_data_summary(
        results_dir / "train_data_summary.json",
        model_data_path,
        results_dir,
        train_end,
        X_cv,
        y_cv,
        outcomes,
        n_rows_before_dropna,
        n_rows_after_dropna,
    )
    cv = make_cv()
    write_cv_split_report(cv, X_cv, y_cv, results_dir / "cv_split_report.csv")
    param_grid_frame().to_csv(results_dir / "param_grid.csv", index=False, encoding="utf-8-sig")

    train_cv(X_cv, y_cv, results_dir, logs_dir, force_train=args.force_train, smoke=args.smoke, full_cv_smoke=args.full_cv_smoke)
    write_score_summaries(results_dir)
    predictions = generate_predictions(X_cv, y_cv, results_dir, logs_dir, smoke=args.smoke, full_cv_smoke=args.full_cv_smoke)
    best = pd.read_csv(results_dir / "best_params.csv")
    write_predictions_summary(results_dir, predictions, best)
    print(json.dumps({"results_dir": str(results_dir.resolve()), "predictions_shape": list(predictions.shape)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
