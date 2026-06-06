
import argparse
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


TARGET_PATTERNS = [
    "target_1d", "target_1D", "target_01d", "return_1d", "returns_1d",
    "fwd_return_1d", "forward_return_1d", "ret_fwd_1d", "y_1d", "1d"
]


def is_date_ticker_index(df: pd.DataFrame) -> bool:
    if not isinstance(df.index, pd.MultiIndex):
        return False
    names = [str(x).lower() for x in df.index.names]
    return ("date" in names) and (("ticker" in names) or ("symbol" in names) or ("asset" in names))


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        return df
    names = list(df.index.names)
    lower = [str(x).lower() for x in names]
    date_level = lower.index("date")
    ticker_level = None
    for nm in ["ticker", "symbol", "asset"]:
        if nm in lower:
            ticker_level = lower.index(nm)
            break
    if ticker_level is None:
        return df
    if date_level != 0 or ticker_level != 1:
        order = [date_level, ticker_level] + [i for i in range(len(names)) if i not in [date_level, ticker_level]]
        df = df.reorder_levels(order).sort_index()
    df.index = df.index.set_names(["date", "ticker"] + list(df.index.names[2:]))
    return df


def read_hdf_candidates(path: Path):
    out = []
    try:
        store = pd.HDFStore(path)
        keys = store.keys()
        store.close()
    except Exception:
        return out
    for key in keys:
        try:
            df = pd.read_hdf(path, key=key)
            if isinstance(df, pd.DataFrame) and len(df) > 100 and is_date_ticker_index(df):
                out.append((f"{path}::{key}", normalize_index(df)))
        except Exception:
            pass
    return out


def read_parquet_candidate(path: Path):
    try:
        df = pd.read_parquet(path)
        if isinstance(df, pd.DataFrame):
            if {"date", "ticker"}.issubset(set(map(str.lower, df.columns))):
                cols = {str(c).lower(): c for c in df.columns}
                df = df.set_index([cols["date"], cols["ticker"]]).sort_index()
            if len(df) > 100 and is_date_ticker_index(df):
                return [(str(path), normalize_index(df))]
    except Exception:
        return []
    return []


def scan_dataframes(repo: Path):
    candidates = []
    for ext in ["*.h5", "*.hdf", "*.hdf5"]:
        for p in repo.rglob(ext):
            candidates.extend(read_hdf_candidates(p))
    for ext in ["*.parquet", "*.pq"]:
        for p in repo.rglob(ext):
            candidates.extend(read_parquet_candidate(p))
    return candidates


def read_book_predictions(repo: Path, ticker=""):
    path = repo / "07_linear_models" / "data.h5"
    if not path.exists():
        return None, ""

    frames = []
    specs = [
        ("linear", "/lr/predictions"),
        ("ridge", "/ridge/predictions"),
        ("lasso", "/lasso/predictions"),
    ]
    for family, key in specs:
        try:
            df = pd.read_hdf(path, key=key)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or not {"actuals", "predicted"}.issubset(df.columns):
            continue
        df = normalize_index(df.rename(columns={"actuals": "actual"}))
        if ticker:
            df = df.loc[df.index.get_level_values("ticker").astype(str) == ticker]
        if df.empty:
            continue
        if "alpha" in df.columns:
            model = family + "_alpha_" + df["alpha"].map(lambda x: f"{float(x):g}")
        else:
            model = family
        part = pd.DataFrame({
            "model": model,
            "actual": df["actual"].to_numpy(),
            "predicted": df["predicted"].to_numpy(),
        }, index=df.index)
        frames.append(part)

    if not frames:
        return None, ""
    return pd.concat(frames).sort_index(), str(path)


def write_prediction_outputs(pred, out, source, target="", target_candidates=None, n_features=None, n_dates=None, n_splits=None, ticker=""):
    pred.to_csv(out / "predictions.csv")

    per_date = (
        pred.groupby(["model", pd.Grouper(level="date")], sort=True)
        .apply(metric_frame)
        .reset_index()
        .rename(columns={"level_1": "date"})
    )
    per_date.to_csv(out / "per_date_rmse.csv", index=False)

    per_ticker = (
        pred.groupby(["model", pd.Grouper(level="ticker")], sort=True)
        .apply(metric_frame)
        .reset_index()
        .rename(columns={"level_1": "ticker"})
    )
    per_ticker.to_csv(out / "per_ticker_rmse.csv", index=False)

    summary = []
    for model, g in pred.groupby("model"):
        m = metric_frame(g)
        daily = per_date[per_date["model"] == model]
        ticker_rows = per_ticker[per_ticker["model"] == model]
        m["model"] = model
        m["mean_daily_rmse"] = daily["rmse"].mean()
        m["mean_daily_rmse_pct"] = daily["rmse_pct"].mean()
        m["median_ticker_rmse"] = ticker_rows["rmse"].median()
        m["median_ticker_rmse_pct"] = ticker_rows["rmse_pct"].median()
        summary.append(m)
    summary = pd.DataFrame(summary).set_index("model").sort_values("mean_daily_rmse")
    summary.to_csv(out / "model_summary.csv")

    txt = []
    txt.append(f"source={source}")
    if target:
        txt.append(f"target={target}")
    if target_candidates is not None:
        txt.append(f"target_candidates={target_candidates}")
    txt.append(f"n_prediction_rows={len(pred)}")
    if n_features is not None:
        txt.append(f"n_features={n_features}")
    if n_dates is not None:
        txt.append(f"n_dates={n_dates}")
    if n_splits is not None:
        txt.append(f"n_splits={n_splits}")
    txt.append("")
    txt.append("Top models by mean daily RMSE:")
    txt.append(summary[["mean_daily_rmse_pct", "median_ticker_rmse_pct", "rmse_norm", "spearman"]].head(10).to_string())
    if ticker:
        ticker_rows = per_ticker[per_ticker["ticker"].astype(str) == ticker]
        txt.append("")
        txt.append(f"Ticker={ticker}")
        txt.append(ticker_rows.sort_values("rmse").head(20).to_string(index=False))
    (out / "run_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


def choose_dataset(candidates, target_col=""):
    scored = []
    for source, df in candidates:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in df.columns:
            tcols = [target_col]
        else:
            tcols = []
            for c in numeric_cols:
                cl = str(c).lower()
                if any(pat.lower() == cl for pat in TARGET_PATTERNS):
                    tcols.append(c)
            if not tcols:
                for c in numeric_cols:
                    cl = str(c).lower()
                    if any(pat.lower() in cl for pat in TARGET_PATTERNS):
                        tcols.append(c)
        score = len(df) + 1000 * len(tcols) + 10 * len(numeric_cols)
        scored.append((score, source, df, tcols, numeric_cols))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        raise RuntimeError("No candidate MultiIndex(date,ticker) dataset found.")
    # Prefer datasets with target candidates
    for _, source, df, tcols, numeric_cols in scored:
        if tcols:
            return source, df, tcols[0], numeric_cols, tcols
    source, df, tcols, numeric_cols = scored[0][1], scored[0][2], scored[0][3], scored[0][4]
    raise RuntimeError(
        f"Found datasets but no target column candidate. Best source={source}. "
        f"Numeric columns sample={numeric_cols[:50]}"
    )


def make_splits(dates, train_days=63, test_days=10, lookahead=1):
    dates = list(pd.Index(dates).sort_values().unique())
    splits = []
    start = 0
    while start + train_days + lookahead + test_days <= len(dates):
        train_dates = dates[start:start + train_days]
        test_start = start + train_days + lookahead
        test_dates = dates[test_start:test_start + test_days]
        splits.append((train_dates, test_dates))
        start += test_days
    return splits


def safe_spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).correlation)


def metric_frame(g):
    y = g["actual"].values
    p = g["predicted"].values
    rmse = math.sqrt(mean_squared_error(y, p))
    std = float(np.std(y, ddof=0))
    return pd.Series({
        "n": len(g),
        "rmse": rmse,
        "rmse_pct": rmse * 100.0,
        "target_std": std,
        "rmse_norm": rmse / std if std > 0 else np.nan,
        "mae": mean_absolute_error(y, p),
        "r2": r2_score(y, p) if len(g) >= 2 and std > 0 else np.nan,
        "spearman": safe_spearman(p, y),
        "actual_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "actual_std": std,
        "pred_std": float(np.std(p, ddof=0)),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-col", default="")
    ap.add_argument("--ticker", default="")
    ap.add_argument("--train-days", type=int, default=63)
    ap.add_argument("--test-days", type=int, default=10)
    ap.add_argument("--lookahead", type=int, default=1)
    args = ap.parse_args()

    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    book_pred, book_source = read_book_predictions(repo, args.ticker)
    if book_pred is not None:
        write_prediction_outputs(
            pred=book_pred,
            out=out,
            source=book_source + "::/lr,/ridge,/lasso/predictions",
            ticker=args.ticker,
        )
        return

    candidates = scan_dataframes(repo)
    if not candidates:
        msg = (
            "No usable book model dataset found.\n"
            "Run with -RunNotebooks first, or manually execute Chapter 7 data prep notebooks.\n"
            "Expected a DataFrame with MultiIndex(date,ticker) in an HDF/parquet file."
        )
        (out / "run_summary.txt").write_text(msg, encoding="utf-8")
        raise RuntimeError(msg)

    source, df, target, numeric_cols, target_candidates = choose_dataset(candidates, args.target_col)
    df = normalize_index(df).copy()
    df = df.sort_index()

    # Keep 1 target. Exclude obvious target/forward columns from features.
    target_like = []
    for c in numeric_cols:
        cl = str(c).lower()
        if c == target or ("target" in cl) or ("forward" in cl) or ("fwd" in cl):
            target_like.append(c)

    features = [c for c in numeric_cols if c not in set(target_like)]
    # Drop all-null and constant globally; per-split imputer handles remaining missing.
    tmp = df[features]
    features = [c for c in features if tmp[c].notna().sum() > 10 and tmp[c].nunique(dropna=True) > 1]
    if not features:
        raise RuntimeError("No usable numeric features after filtering.")

    work = df[features + [target]].dropna(subset=[target]).copy()
    # Ensure dates are datetime-like when possible.
    dates = work.index.get_level_values("date")
    unique_dates = pd.Index(dates).unique().sort_values()
    splits = make_splits(unique_dates, args.train_days, args.test_days, args.lookahead)
    if not splits:
        raise RuntimeError(f"No splits generated. unique_dates={len(unique_dates)}")

    alphas = sorted(set(list(np.logspace(-4, 4, 9)) + list(np.logspace(-4, 4, 9) * 5)))
    models = [("linear", LinearRegression(fit_intercept=False))]
    for alpha in alphas:
        models.append((f"ridge_alpha_{alpha:g}", Ridge(alpha=float(alpha), fit_intercept=False)))

    preds = []
    for split_id, (train_dates, test_dates) in enumerate(splits):
        train = work.loc[work.index.get_level_values("date").isin(train_dates)]
        test = work.loc[work.index.get_level_values("date").isin(test_dates)]
        if len(train) < 50 or len(test) == 0:
            continue
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        for model_name, model in models:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
            except Exception:
                continue
            part = pd.DataFrame({
                "split_id": split_id,
                "model": model_name,
                "actual": y_test.values,
                "predicted": y_pred,
            }, index=test.index)
            preds.append(part)

    if not preds:
        raise RuntimeError("No predictions generated.")

    pred = pd.concat(preds).sort_index()
    write_prediction_outputs(
        pred=pred,
        out=out,
        source=source,
        target=target,
        target_candidates=target_candidates,
        n_features=len(features),
        n_dates=len(unique_dates),
        n_splits=len(splits),
        ticker=args.ticker,
    )


if __name__ == "__main__":
    main()
