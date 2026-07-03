#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 fold-0 NN search with first-batch stable context features.

This script reuses the existing sector-rotation search/training code and adds
features that can be generated from the current model_data_as1455.h5 only:
market regime/breadth, leave-one-out sector rotation/breadth, sector liquidity,
and stock dollar-volume ratios.

Features requiring raw OHLCV/tradability fields are listed in the output report
instead of being silently fabricated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_as1455_sector_rotation_fold0_param_search as base

DEFAULT_OUT_DIR = base.PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_first_batch_features_fold0_search"
CORE_RETURN_COLS = ["r01", "r05", "r21"]
SECTOR_LOO_RETURN_COLS = ["r01", "r05", "r21", "r63"]
UNAVAILABLE_FIRST_BATCH_FEATURES = [
    "volume_ratio_20 requires raw volume, not only model_data.dollar_vol",
    "turnover_ratio_20 requires turnover/float shares",
    "intraday_ret_open_to_1455 requires open/current AS1455 price",
    "intraday_range_high_low requires intraday high/low or AS1455 snapshot fields",
    "near_limit_up/down requires preclose and current price",
    "prev_limit_up/down and limit streaks require raw OHLC/preclose limit-price history",
    "tradestatus/is_st/days_since_resume require raw BaoStock/AKShare tradability/basic-info files",
]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_divide(num: np.ndarray, den: np.ndarray, fill: float = 0.0) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full_like(num, fill, dtype=float), where=den != 0)


def add_first_batch_context_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Add stable first-batch features available from existing model_data only.

    No forward-return columns are used.
    """
    out = X.copy()
    dates = pd.Index(out.index.get_level_values("date"), name="date")
    sectors = out["sector"].astype(int)
    key = pd.MultiIndex.from_arrays([dates, sectors], names=["date", "sector"])
    base_df = out.assign(__date=dates, __sector=sectors)
    groups: dict[str, list[str]] = {
        "market_regime": [],
        "market_breadth": [],
        "sector_rotation_leave_one_out": [],
        "sector_breadth": [],
        "sector_liquidity": [],
        "stock_liquidity": [],
    }

    # Market regime and breadth.
    for c in CORE_RETURN_COLS:
        market_mean = base_df.groupby("__date", sort=False)[c].mean()
        market_std = base_df.groupby("__date", sort=False)[c].std(ddof=0).fillna(0)
        market_pos = base_df[c].gt(0).groupby(base_df["__date"], sort=False).mean()
        cols = [f"market_{c}_mean", f"market_{c}_std", f"market_{c}_positive_rate"]
        out[cols[0]] = market_mean.reindex(dates).to_numpy()
        out[cols[1]] = market_std.reindex(dates).to_numpy()
        out[cols[2]] = market_pos.reindex(dates).to_numpy()
        groups["market_regime"] += cols[:2]
        groups["market_breadth"].append(cols[2])

    for c in ["r01", "r05"]:
        dec_col = f"{c}dec"
        top = base_df[dec_col].ge(9).groupby(base_df["__date"], sort=False).mean()
        bottom = base_df[dec_col].le(0).groupby(base_df["__date"], sort=False).mean()
        cols = [f"market_{c}_top_decile_ratio", f"market_{c}_bottom_decile_ratio"]
        out[cols[0]] = top.reindex(dates).to_numpy()
        out[cols[1]] = bottom.reindex(dates).to_numpy()
        groups["market_breadth"] += cols

    market_dv_sum = base_df.groupby("__date", sort=True)["dollar_vol"].sum().sort_index()
    market_dv_prior20 = market_dv_sum.shift(1).rolling(20, min_periods=5).mean()
    market_dv_ratio20 = safe_divide(market_dv_sum.to_numpy(), market_dv_prior20.to_numpy(), fill=1.0)
    market_dv_ratio20 = pd.Series(market_dv_ratio20, index=market_dv_sum.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["market_dollar_vol_sum"] = market_dv_sum.reindex(dates).to_numpy()
    out["market_dollar_vol_ratio_20"] = market_dv_ratio20.reindex(dates).to_numpy()
    groups["market_regime"] += ["market_dollar_vol_sum", "market_dollar_vol_ratio_20"]

    # Leave-one-out sector rotation.
    sector_count = base_df.groupby(["__date", "__sector"], sort=False).size()
    market_count = base_df.groupby("__date", sort=False).size()
    sector_sum = base_df.groupby(["__date", "__sector"], sort=False)[SECTOR_LOO_RETURN_COLS].sum()
    market_sum = base_df.groupby("__date", sort=False)[SECTOR_LOO_RETURN_COLS].sum()
    sector_mean = base_df.groupby(["__date", "__sector"], sort=False)[SECTOR_LOO_RETURN_COLS].mean()
    sector_rank = sector_mean.groupby(level=0).rank(pct=True, ascending=True)
    sector_count_values = sector_count.reindex(key).to_numpy(dtype=float)
    market_count_values = market_count.reindex(dates).to_numpy(dtype=float)

    for c in SECTOR_LOO_RETURN_COLS:
        x = out[c].to_numpy(dtype=float)
        sector_sum_values = sector_sum[c].reindex(key).to_numpy(dtype=float)
        market_sum_values = market_sum[c].reindex(dates).to_numpy(dtype=float)
        sector_mean_full = sector_mean[c].reindex(key).to_numpy(dtype=float)
        market_mean_full = safe_divide(market_sum_values, market_count_values, fill=0.0)
        sector_mean_ex_self = np.where(
            sector_count_values > 1,
            safe_divide(sector_sum_values - x, sector_count_values - 1, fill=0.0),
            sector_mean_full,
        )
        market_mean_ex_self = np.where(
            market_count_values > 1,
            safe_divide(market_sum_values - x, market_count_values - 1, fill=0.0),
            market_mean_full,
        )
        cols = [f"sector_{c}_mean_ex_self", f"sector_{c}_rel_mkt_ex_self", f"sector_{c}_rank_pct"]
        out[cols[0]] = sector_mean_ex_self
        out[cols[1]] = sector_mean_ex_self - market_mean_ex_self
        out[cols[2]] = sector_rank[c].reindex(key).to_numpy()
        groups["sector_rotation_leave_one_out"] += cols

    # Sector breadth, leave-one-out where possible.
    for c in ["r01", "r05"]:
        x_pos = out[c].gt(0).to_numpy(dtype=float)
        sector_pos_sum = base_df[c].gt(0).groupby([base_df["__date"], base_df["__sector"]], sort=False).sum()
        sector_pos_values = sector_pos_sum.reindex(key).to_numpy(dtype=float)
        pos_full = safe_divide(sector_pos_values, sector_count_values, fill=0.0)
        pos_ex_self = np.where(
            sector_count_values > 1,
            safe_divide(sector_pos_values - x_pos, sector_count_values - 1, fill=0.0),
            pos_full,
        )
        col = f"sector_{c}_positive_rate_ex_self"
        out[col] = pos_ex_self
        groups["sector_breadth"].append(col)

        dec_col = f"{c}dec"
        x_top = out[dec_col].ge(9).to_numpy(dtype=float)
        x_bottom = out[dec_col].le(0).to_numpy(dtype=float)
        sector_top_sum = base_df[dec_col].ge(9).groupby([base_df["__date"], base_df["__sector"]], sort=False).sum()
        sector_bottom_sum = base_df[dec_col].le(0).groupby([base_df["__date"], base_df["__sector"]], sort=False).sum()
        top_values = sector_top_sum.reindex(key).to_numpy(dtype=float)
        bottom_values = sector_bottom_sum.reindex(key).to_numpy(dtype=float)
        top_full = safe_divide(top_values, sector_count_values, fill=0.0)
        bottom_full = safe_divide(bottom_values, sector_count_values, fill=0.0)
        top_ex_self = np.where(
            sector_count_values > 1,
            safe_divide(top_values - x_top, sector_count_values - 1, fill=0.0),
            top_full,
        )
        bottom_ex_self = np.where(
            sector_count_values > 1,
            safe_divide(bottom_values - x_bottom, sector_count_values - 1, fill=0.0),
            bottom_full,
        )
        cols = [f"sector_{c}_top_decile_ratio_ex_self", f"sector_{c}_bottom_decile_ratio_ex_self"]
        out[cols[0]] = top_ex_self
        out[cols[1]] = bottom_ex_self
        groups["sector_breadth"] += cols

    # Sector dollar-volume heat and stock dollar-volume ratios.
    sector_dv_sum = base_df.groupby(["__date", "__sector"], sort=False)["dollar_vol"].sum()
    market_dv_sum_unsorted = base_df.groupby("__date", sort=False)["dollar_vol"].sum()
    sector_dv_rank = sector_dv_sum.groupby(level=0).rank(pct=True, ascending=True)
    dv_values = sector_dv_sum.reindex(key).to_numpy(dtype=float)
    market_dv_values = market_dv_sum_unsorted.reindex(dates).to_numpy(dtype=float)
    out["sector_dollar_vol_sum"] = dv_values
    out["sector_dollar_vol_share"] = safe_divide(dv_values, market_dv_values, fill=0.0)
    out["sector_dollar_vol_rank_pct"] = sector_dv_rank.reindex(key).to_numpy()
    groups["sector_liquidity"] += ["sector_dollar_vol_sum", "sector_dollar_vol_share", "sector_dollar_vol_rank_pct"]

    sector_dv_df = sector_dv_sum.rename("sector_dollar_vol_sum").reset_index().sort_values(["__sector", "__date"])
    sector_dv_df["prior20"] = sector_dv_df.groupby("__sector")["sector_dollar_vol_sum"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    sector_dv_df["sector_dollar_vol_ratio_20"] = safe_divide(
        sector_dv_df["sector_dollar_vol_sum"].to_numpy(), sector_dv_df["prior20"].to_numpy(), fill=1.0
    )
    sector_dv_ratio20 = sector_dv_df.set_index(["__date", "__sector"])["sector_dollar_vol_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["sector_dollar_vol_ratio_20"] = sector_dv_ratio20.reindex(key).to_numpy()
    groups["sector_liquidity"].append("sector_dollar_vol_ratio_20")

    dv = out["dollar_vol"].astype(float).replace([np.inf, -np.inf], np.nan)
    for win in [5, 20]:
        prior = dv.groupby(level="symbol").transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 4)).mean())
        col = f"dollar_vol_ratio_{win}"
        out[col] = pd.Series(safe_divide(dv.to_numpy(), prior.to_numpy(), fill=1.0), index=out.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        groups["stock_liquidity"].append(col)

    new_cols = [c for cols in groups.values() for c in cols]
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    if out[new_cols].isna().any().any():
        bad = out[new_cols].isna().sum()
        raise RuntimeError(f"NA in first-batch context features: {bad[bad > 0].to_dict()}")
    return out, new_cols, groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AS1455 first-batch stable feature fold-0 NN parameter search")
    p.add_argument("--model-data", default=str(base.DEFAULT_MODEL_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--train-end", default=None)
    p.add_argument("--fold-index", type=int, default=0, help="0=newest fold, 6=oldest fold")
    p.add_argument("--sector-encoding", choices=["numeric", "onehot"], default="onehot")
    p.add_argument("--dropna-mode", choices=["strict_original", "r01_only"], default="r01_only")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--best-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--input-check-only", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir already has files; pass --force or choose another --out-dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_base, y, meta = base.load_xy(Path(args.model_data), args.train_end, args.dropna_mode)
    X_ctx, context_cols, feature_groups = add_first_batch_context_features(X_base)
    X_final, no_scale_cols, sector_onehot_cols = base.apply_sector_encoding(X_ctx, args.sector_encoding)
    grid = base.param_grid(args.smoke)
    train_idx, test_idx, fold = base.get_fold(X_final, args.fold_index)

    write_json(out_dir / "run_summary.json", {
        **meta,
        "model_data": str(Path(args.model_data).resolve()),
        "out_dir": str(out_dir.resolve()),
        "base_feature_count": int(X_base.shape[1]),
        "context_feature_count": len(context_cols),
        "final_feature_count": int(X_final.shape[1]),
        "sector_encoding": args.sector_encoding,
        "dropna_mode": args.dropna_mode,
        "fold_index": args.fold_index,
        "epochs": args.epochs,
        "param_grid_size": len(grid),
    })
    write_json(out_dir / "fold_report.json", fold)
    pd.DataFrame([fold]).to_csv(out_dir / "fold_report.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "feature_cols_base.json", list(X_base.columns))
    write_json(out_dir / "context_feature_cols.json", context_cols)
    write_json(out_dir / "feature_group_cols.json", feature_groups)
    write_json(out_dir / "unavailable_first_batch_features.json", UNAVAILABLE_FIRST_BATCH_FEATURES)
    write_json(out_dir / "rotation_feature_cols.json", context_cols)
    write_json(out_dir / "feature_cols_final.json", list(X_final.columns))
    write_json(out_dir / "sector_onehot_cols.json", sector_onehot_cols)
    pd.DataFrame(grid).to_csv(out_dir / "param_grid.csv", index=False, encoding="utf-8-sig")

    print(f"[DATA] base={X_base.shape[1]} context={len(context_cols)} final={X_final.shape[1]}")
    print(f"[FOLD] {fold}")
    if args.input_check_only:
        print(f"[OK] input reports written to {out_dir}")
        return

    base.require_deps()
    summary = base.train_search(X_final, y, train_idx, test_idx, no_scale_cols, grid, args.epochs, args.seed, out_dir)
    best = summary.head(args.best_n).copy()
    best.to_csv(out_dir / "best_params.csv", index=False, encoding="utf-8-sig")
    print("[BEST]")
    print(best[base.PARAM_COLS + ["pooled_spearman", "daily_ic_mean", "daily_ic_median", "daily_ic_positive_rate"]].to_string(index=False))
    base.retrain_best(X_final, y, train_idx, test_idx, no_scale_cols, best[base.PARAM_COLS], args.seed, out_dir)
    print(f"[OK] written to {out_dir}")


if __name__ == "__main__":
    main()
