#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 fold-0 NN search with full sector rotation plus compact add-on features.

This script keeps the complete sector-rotation feature set from
run_as1455_sector_rotation_fold0_param_search.py and only adds compact,
stable first-batch context features that can be generated from the current
model_data_as1455.h5.

Important: this is an additive experiment. It does not remove or replace the
original 31 base features or the complete sector-rotation features.
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

DEFAULT_OUT_DIR = base.PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch17_as1455_full_rotation_plus_first_batch_compact_fold0_search"
CORE_RETURN_COLS = ["r01", "r05", "r21"]
SKIPPED_REDUNDANT_ADDON_CANDIDATES = [
    "market_r01_top_decile_ratio",
    "market_r01_bottom_decile_ratio",
    "market_r05_top_decile_ratio",
    "market_r05_bottom_decile_ratio",
    "market_dollar_vol_sum",
    "sector_dollar_vol_sum as a new model feature; full rotation already keeps its own sector_dollar_vol_sum",
]
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


def add_compact_addon_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Add compact extra features without deleting existing base/rotation columns.

    No forward-return columns are used. Existing sector-rotation columns are kept
    unchanged if they are already present in X.
    """
    out = X.copy()
    dates = pd.Index(out.index.get_level_values("date"), name="date")
    sectors = out["sector"].astype(int)
    key = pd.MultiIndex.from_arrays([dates, sectors], names=["date", "sector"])
    base_df = out.assign(__date=dates, __sector=sectors)
    groups: dict[str, list[str]] = {
        "market_regime": [],
        "market_breadth": [],
        "sector_breadth": [],
        "sector_liquidity_addon": [],
        "stock_liquidity": [],
    }

    # Market regime/breadth. Do not add market top/bottom decile ratios because
    # they are near-constant by construction when rXXdec is a daily decile.
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

    market_dv_sum = base_df.groupby("__date", sort=True)["dollar_vol"].sum().sort_index()
    market_dv_prior20 = market_dv_sum.shift(1).rolling(20, min_periods=5).mean()
    market_dv_ratio20 = safe_divide(market_dv_sum.to_numpy(), market_dv_prior20.to_numpy(), fill=1.0)
    market_dv_ratio20 = pd.Series(market_dv_ratio20, index=market_dv_sum.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["market_dollar_vol_ratio_20"] = market_dv_ratio20.reindex(dates).to_numpy()
    groups["market_regime"].append("market_dollar_vol_ratio_20")

    # Sector breadth add-ons. These do not replace full sector rotation.
    sector_count = base_df.groupby(["__date", "__sector"], sort=False).size()
    sector_count_values = sector_count.reindex(key).to_numpy(dtype=float)
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

    # Sector dollar-volume ratio add-on. Full rotation already keeps sum/share/rank.
    sector_dv_sum = base_df.groupby(["__date", "__sector"], sort=False)["dollar_vol"].sum()
    sector_dv_df = sector_dv_sum.rename("sector_dollar_vol_sum_for_ratio").reset_index().sort_values(["__sector", "__date"])
    sector_dv_df["prior20"] = sector_dv_df.groupby("__sector")["sector_dollar_vol_sum_for_ratio"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    sector_dv_df["sector_dollar_vol_ratio_20"] = safe_divide(
        sector_dv_df["sector_dollar_vol_sum_for_ratio"].to_numpy(), sector_dv_df["prior20"].to_numpy(), fill=1.0
    )
    sector_dv_ratio20 = sector_dv_df.set_index(["__date", "__sector"])["sector_dollar_vol_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["sector_dollar_vol_ratio_20"] = sector_dv_ratio20.reindex(key).to_numpy()
    groups["sector_liquidity_addon"].append("sector_dollar_vol_ratio_20")

    # Stock dollar-volume ratios.
    dv = out["dollar_vol"].astype(float).replace([np.inf, -np.inf], np.nan)
    for win in [5, 20]:
        prior = dv.groupby(level="symbol").transform(lambda s: s.shift(1).rolling(win, min_periods=max(3, win // 4)).mean())
        col = f"dollar_vol_ratio_{win}"
        out[col] = pd.Series(safe_divide(dv.to_numpy(), prior.to_numpy(), fill=1.0), index=out.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        groups["stock_liquidity"].append(col)

    new_cols = [c for cols in groups.values() for c in cols]
    duplicate_new_cols = [c for c in new_cols if list(out.columns).count(c) > 1]
    if duplicate_new_cols:
        raise RuntimeError(f"duplicate add-on columns: {duplicate_new_cols}")
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    if out[new_cols].isna().any().any():
        bad = out[new_cols].isna().sum()
        raise RuntimeError(f"NA in compact add-on features: {bad[bad > 0].to_dict()}")
    return out, new_cols, groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AS1455 full sector-rotation plus compact add-on feature fold-0 NN parameter search")
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
    X_rot, rotation_cols = base.add_sector_rotation_features(X_base)
    X_ctx, addon_cols, feature_groups = add_compact_addon_features(X_rot)
    X_final, no_scale_cols, sector_onehot_cols = base.apply_sector_encoding(X_ctx, args.sector_encoding)
    grid = base.param_grid(args.smoke)
    train_idx, test_idx, fold = base.get_fold(X_final, args.fold_index)

    write_json(out_dir / "run_summary.json", {
        **meta,
        "model_data": str(Path(args.model_data).resolve()),
        "out_dir": str(out_dir.resolve()),
        "base_feature_count": int(X_base.shape[1]),
        "rotation_feature_count": len(rotation_cols),
        "addon_feature_count": len(addon_cols),
        "final_feature_count": int(X_final.shape[1]),
        "feature_preset": "full_rotation_plus_compact_addons",
        "sector_encoding": args.sector_encoding,
        "dropna_mode": args.dropna_mode,
        "fold_index": args.fold_index,
        "epochs": args.epochs,
        "param_grid_size": len(grid),
    })
    write_json(out_dir / "fold_report.json", fold)
    pd.DataFrame([fold]).to_csv(out_dir / "fold_report.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "feature_cols_base.json", list(X_base.columns))
    write_json(out_dir / "rotation_feature_cols.json", rotation_cols)
    write_json(out_dir / "addon_feature_cols.json", addon_cols)
    write_json(out_dir / "feature_group_cols.json", feature_groups)
    write_json(out_dir / "skipped_redundant_addon_candidates.json", SKIPPED_REDUNDANT_ADDON_CANDIDATES)
    write_json(out_dir / "unavailable_first_batch_features.json", UNAVAILABLE_FIRST_BATCH_FEATURES)
    write_json(out_dir / "feature_cols_final.json", list(X_final.columns))
    write_json(out_dir / "sector_onehot_cols.json", sector_onehot_cols)
    pd.DataFrame(grid).to_csv(out_dir / "param_grid.csv", index=False, encoding="utf-8-sig")

    print(f"[DATA] base={X_base.shape[1]} rotation={len(rotation_cols)} addon={len(addon_cols)} final={X_final.shape[1]}")
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
