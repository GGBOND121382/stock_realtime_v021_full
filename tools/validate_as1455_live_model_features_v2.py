#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict validator for AS1455 live model features.

Validates the usable prediction file, while still auditing the full file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from features.as1455_live_common import (  # noqa: E402
    EXPECTED_MODEL_COLUMNS,
    load_sector_reference_from_model_data,
    normalize_symbol,
    parse_trade_date,
    write_json,
    yyyymmdd_to_dash,
)

MONTH = 21


def read_table_with_fallback(base: Path) -> pd.DataFrame:
    candidates = []
    if base.suffix:
        candidates.append(base)
        candidates.append(base.with_suffix(".csv"))
    else:
        candidates.extend([base.with_suffix(".parquet"), base.with_suffix(".csv")])
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, dtype={"symbol": str}, encoding="utf-8-sig")
    raise FileNotFoundError("none found: " + ", ".join(str(p) for p in candidates))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", required=True)
    ap.add_argument("--trade-date", default="today")
    ap.add_argument("--sector-reference", default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5")
    ap.add_argument("--min-feature-rows", type=int, default=980)
    args = ap.parse_args()

    trade_date = parse_trade_date(args.trade_date)
    t_date = pd.Timestamp(yyyymmdd_to_dash(trade_date)).normalize()
    live_dir = Path(args.live_dir)
    full_feat_path = live_dir / "11_live_model_features.csv"
    usable_feat_path = live_dir / "11_live_model_features_usable.csv"
    prediction_feat_path = live_dir / "11_live_model_features_for_prediction.csv"
    panel_base = live_dir / "10_live_feature_panel_tail.parquet"
    report_path = live_dir / "13_live_feature_strict_validation_report.json"

    full_feat = pd.read_csv(full_feat_path, dtype={"symbol": str}, encoding="utf-8-sig")
    full_feat["symbol"] = full_feat["symbol"].map(normalize_symbol)
    feat_path = usable_feat_path if usable_feat_path.exists() else full_feat_path
    feat = pd.read_csv(feat_path, dtype={"symbol": str}, encoding="utf-8-sig")
    feat["symbol"] = feat["symbol"].map(normalize_symbol)
    if prediction_feat_path.exists():
        pred_feat = pd.read_csv(prediction_feat_path, dtype={"symbol": str}, encoding="utf-8-sig")
        pred_feat["symbol"] = pred_feat["symbol"].map(normalize_symbol)
        prediction_matches_usable = pred_feat.equals(feat)
    else:
        prediction_matches_usable = False
    missing_cols = [c for c in EXPECTED_MODEL_COLUMNS if c not in feat.columns]
    full_missing_cols = [c for c in EXPECTED_MODEL_COLUMNS if c not in full_feat.columns]
    nan_by_column = {c: int(feat[c].isna().sum()) for c in EXPECTED_MODEL_COLUMNS if c in feat.columns}
    nan_cols = [c for c, n in nan_by_column.items() if n > 0]
    full_nan_by_column = {c: int(full_feat[c].isna().sum()) for c in EXPECTED_MODEL_COLUMNS if c in full_feat.columns}
    full_nan_cols = [c for c, n in full_nan_by_column.items() if n > 0]

    panel = read_table_with_fallback(panel_base)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["symbol"] = panel["symbol"].map(normalize_symbol)
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")
    pidx = panel.set_index(["date", "symbol"]).sort_index()
    dollar_vol = pidx["close"].mul(pidx["volume"].div(1e3)).div(1e3)
    dollar_vol_ma = dollar_vol.unstack("symbol").rolling(window=MONTH, min_periods=1).mean()
    expected_rank = dollar_vol_ma.rank(axis=1, ascending=False).stack()
    expected_today = expected_rank.loc[t_date]
    got = feat.set_index("symbol")["dollar_vol_rank"].astype(float)
    rank_join = pd.DataFrame({"got": got, "expected": expected_today})
    rank_join["abs_diff"] = (rank_join["got"] - rank_join["expected"]).abs()
    rank_max_abs_diff = float(rank_join["abs_diff"].max()) if len(rank_join) else None
    rank_bad_rows = int((rank_join["abs_diff"] > 1e-9).sum()) if len(rank_join) else 0

    sector_ref, sector_ref_report = load_sector_reference_from_model_data(args.sector_reference)
    sector_bad_rows = None
    sector_unmapped_rows = None
    full_sector_bad_rows = None
    full_sector_unmapped_rows = None
    if sector_ref_report.get("loaded"):
        ref = sector_ref.set_index("symbol")["sector"]
        got_sector = pd.to_numeric(feat.set_index("symbol")["sector"], errors="coerce")
        sector_join = pd.DataFrame({"got": got_sector, "expected": ref}).loc[got_sector.index]
        sector_unmapped_rows = int(sector_join["expected"].isna().sum())
        sector_bad_rows = int((sector_join["got"] != sector_join["expected"]).sum())

        full_got_sector = pd.to_numeric(full_feat.set_index("symbol")["sector"], errors="coerce")
        full_sector_join = pd.DataFrame({"got": full_got_sector, "expected": ref}).loc[full_got_sector.index]
        full_sector_unmapped_rows = int(full_sector_join["expected"].isna().sum())
        full_sector_bad_rows = int((full_sector_join["got"] != full_sector_join["expected"]).sum())
    else:
        got_sector = pd.to_numeric(feat.get("sector", pd.Series(dtype=float)), errors="coerce")
        sector_unmapped_rows = int((got_sector < 0).sum())
        full_got_sector = pd.to_numeric(full_feat.get("sector", pd.Series(dtype=float)), errors="coerce")
        full_sector_unmapped_rows = int((full_got_sector < 0).sum())

    usable_rows = int(len(feat) - feat[EXPECTED_MODEL_COLUMNS].isna().any(axis=1).sum()) if not missing_cols else 0
    dropped_rows = int(len(full_feat) - len(feat))
    passed = (
        len(feat) >= args.min_feature_rows
        and usable_rows == len(feat)
        and usable_rows >= args.min_feature_rows
        and not missing_cols
        and not nan_cols
        and not full_missing_cols
        and rank_bad_rows == 0
        and sector_unmapped_rows == 0
        and full_sector_unmapped_rows == 0
        and (sector_bad_rows in {None, 0})
        and (full_sector_bad_rows in {None, 0})
        and prediction_matches_usable
    )

    report = {
        "trade_date": trade_date,
        "live_dir": str(live_dir),
        "full_feature_path": str(full_feat_path),
        "usable_feature_path": str(feat_path),
        "prediction_feature_path": str(prediction_feat_path),
        "feature_rows_full": int(len(full_feat)),
        "feature_rows_usable": int(len(feat)),
        "dropped_rows_from_full": dropped_rows,
        "prediction_file_matches_usable": bool(prediction_matches_usable),
        "expected_feature_columns": EXPECTED_MODEL_COLUMNS,
        "missing_columns": missing_cols,
        "full_missing_columns": full_missing_cols,
        "nan_by_column": nan_by_column,
        "nan_columns": nan_cols,
        "full_nan_by_column": full_nan_by_column,
        "full_nan_columns": full_nan_cols,
        "usable_rows": usable_rows,
        "dollar_vol_rank_nonnull": int(feat["dollar_vol_rank"].notna().sum()) if "dollar_vol_rank" in feat.columns else 0,
        "dollar_vol_rank_max_abs_diff_vs_recompute": rank_max_abs_diff,
        "dollar_vol_rank_bad_rows": rank_bad_rows,
        "sector_reference": sector_ref_report,
        "sector_unmapped_rows": sector_unmapped_rows,
        "sector_bad_rows_vs_training_reference": sector_bad_rows,
        "full_sector_unmapped_rows": full_sector_unmapped_rows,
        "full_sector_bad_rows_vs_training_reference": full_sector_bad_rows,
        "sector_unique_count": int(pd.to_numeric(feat.get("sector", pd.Series(dtype=float)), errors="coerce").nunique(dropna=True)),
        "full_sector_unique_count": int(pd.to_numeric(full_feat.get("sector", pd.Series(dtype=float)), errors="coerce").nunique(dropna=True)),
        "passed": bool(passed),
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit(f"strict live feature validation failed; see {report_path}")


if __name__ == "__main__":
    main()
