#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the pre-change audit for moving next-day models to 14:55 as-of features."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS = ROOT / "saved_models"
SAVED_DATA = ROOT / "saved_data"
REPORT_DIR = ROOT / "reports" / "asof1455_migration_audit"

LAGGED_DAILY_RE = re.compile(
    r"^[A-Za-z0-9]+_(?:fut_|future_basket_|us_|us_basket_)|^[A-Za-z0-9]+_stock_vs_(?:future_basket|us_basket)_ret\d+$"
)
HIGH_RISK_RE = re.compile(
    r"(_range_pct(?:_|$)|_shock20$|_z20$|_z60$|_ret3$|_amount_z20$|_volume_z20$|_amount_shock20$|_volume_shock20$|_fut_|_future_basket_|_us_|_us_basket_|_stock_vs_future_basket_ret\d+$|_stock_vs_us_basket_ret\d+$)"
)
TAIL_SENSITIVE_RE = re.compile(
    r"^(last_30m_|last_60m_|afternoon_)|range_pct|volume_shock20|amount_shock20|volume_z20|amount_z20|external_.*_range_pct|external_.*_shock20|sector_.*_shock20|board_.*_shock20"
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_symbol(value: Any, fallback: str = "") -> str:
    text = str(value or fallback).strip().upper().replace("_", ".")
    if not text:
        return ""
    if "." in text:
        a, b = text.split(".", 1)
        if a in {"SH", "SZ"}:
            return f"{b.zfill(6)}.{a}"
        return f"{a.zfill(6)}.{b}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    market = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits.zfill(6)}.{market}"


def resolve_repo_path(value: Any) -> Path | None:
    if not value:
        return None
    text = str(value)
    p = Path(text)
    if p.exists():
        return p
    marker = "stock_realtime_v021_full"
    if marker in p.parts:
        candidate = ROOT.joinpath(*p.parts[p.parts.index(marker) + 1 :])
        if candidate.exists():
            return candidate
    candidate = ROOT / text
    if candidate.exists():
        return candidate
    return None


def expected_repo_path(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    p = Path(text)
    marker = "stock_realtime_v021_full"
    if marker in p.parts:
        return str(ROOT.joinpath(*p.parts[p.parts.index(marker) + 1 :]))
    if not p.is_absolute():
        return str(ROOT / text)
    return text


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def read_features(model_dir: Path) -> list[str]:
    path = model_dir / "feature_columns.txt"
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def audit_saved_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    sample_rows = []
    for meta_path in sorted(SAVED_MODELS.rglob("metadata.json")):
        meta = load_json(meta_path)
        model_dir = meta_path.parent
        stock = normalize_symbol(meta.get("stock_code"), model_dir.parent.name)
        artifact = str(meta.get("artifact_name") or model_dir.name)
        features = read_features(model_dir)
        sample_value = meta.get("samples", "")
        sample_path = resolve_repo_path(sample_value)
        expected_sample = expected_repo_path(sample_value)
        sample_rows.append(
            {
                "stock_code": stock,
                "artifact_name": artifact,
                "model_dir": str(model_dir),
                "metadata_path": str(meta_path),
                "metadata_samples": sample_value,
                "expected_local_samples": expected_sample,
                "samples_exists": bool(sample_path),
                "resolved_samples": str(sample_path or ""),
                "feature_count": len(features),
                "label_mode": meta.get("label_mode", ""),
                "entry_policy": meta.get("entry_policy", ""),
                "feature_group": meta.get("feature_group", ""),
                "model_name": meta.get("model_name", ""),
            }
        )
        for i, col in enumerate(features, start=1):
            feature_rows.append(
                {
                    "stock_code": stock,
                    "artifact_name": artifact,
                    "feature_index": i,
                    "feature": col,
                    "is_high_risk": bool(HIGH_RISK_RE.search(col)),
                    "is_lagged_daily": bool(LAGGED_DAILY_RE.search(col)),
                    "is_tail_sensitive": bool(TAIL_SENSITIVE_RE.search(col)),
                    "model_dir": str(model_dir),
                    "expected_local_samples": expected_sample,
                    "samples_exists": bool(sample_path),
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(sample_rows)


def audit_leaderboards() -> pd.DataFrame:
    rows = []
    for path in sorted(SAVED_DATA.rglob("final_leaderboard.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append({"leaderboard": str(path), "read_error": f"{type(exc).__name__}: {exc}"})
            continue
        for _, row in df.iterrows():
            sample_value = row.get("sample_file", "")
            sample_path = resolve_repo_path(sample_value)
            rows.append(
                {
                    "leaderboard": str(path),
                    "pipeline_dir": str(path.parents[1]),
                    "stock_code": normalize_symbol(row.get("stock_code", row.get("raw_code", ""))),
                    "feature_group": row.get("feature_group", ""),
                    "model_name": row.get("model_name", ""),
                    "label_mode": row.get("label_mode", ""),
                    "entry_policy": row.get("entry_policy", ""),
                    "sample_file": sample_value,
                    "expected_local_sample_file": expected_repo_path(sample_value),
                    "sample_exists": bool(sample_path),
                    "intraday_bars": row.get("intraday_bars", ""),
                    "intraday_bars_exists": bool(resolve_repo_path(row.get("intraday_bars", ""))),
                    "rank_score": row.get("rank_score", np.nan),
                    "trades": row.get("trades", np.nan),
                    "win_rate": row.get("win_rate", np.nan),
                    "avg_return": row.get("avg_return", np.nan),
                }
            )
    return pd.DataFrame(rows)


def read_sample_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def audit_lagged_daily_samples(sample_paths: list[Path], trade_date: str | None) -> pd.DataFrame:
    rows = []
    for path in sorted(set(sample_paths)):
        try:
            header = read_sample_header(path)
            lagged_cols = [c for c in header if LAGGED_DAILY_RE.search(c)]
            if "date" not in header:
                rows.append({"sample_path": str(path), "read_status": "missing_date_column", "lagged_col_count": len(lagged_cols)})
                continue
            usecols = ["date"] + lagged_cols
            df = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
            dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            target = pd.to_datetime(trade_date).normalize() if trade_date else dates.max()
            day = df.loc[dates == target]
            if day.empty:
                rows.append(
                    {
                        "sample_path": str(path),
                        "read_status": "ok",
                        "target_date": str(target.date()) if pd.notna(target) else "",
                        "date_t_exists": False,
                        "sample_min_date": str(dates.min().date()) if pd.notna(dates.min()) else "",
                        "sample_max_date": str(dates.max().date()) if pd.notna(dates.max()) else "",
                        "lagged_col_count": len(lagged_cols),
                        "lagged_nonempty_count": 0,
                        "lagged_missing_count": len(lagged_cols),
                        "lagged_missing_features": ",".join(lagged_cols),
                    }
                )
                continue
            last = day.tail(1)
            nonempty = [
                c for c in lagged_cols
                if c in last.columns and pd.notna(pd.to_numeric(last[c], errors="coerce").iloc[0])
            ]
            missing = [c for c in lagged_cols if c not in nonempty]
            rows.append(
                {
                    "sample_path": str(path),
                    "read_status": "ok",
                    "target_date": str(target.date()) if pd.notna(target) else "",
                    "date_t_exists": True,
                    "sample_min_date": str(dates.min().date()) if pd.notna(dates.min()) else "",
                    "sample_max_date": str(dates.max().date()) if pd.notna(dates.max()) else "",
                    "lagged_col_count": len(lagged_cols),
                    "lagged_nonempty_count": len(nonempty),
                    "lagged_missing_count": len(missing),
                    "lagged_missing_features": ",".join(missing),
                }
            )
        except Exception as exc:
            rows.append({"sample_path": str(path), "read_status": f"read_error:{type(exc).__name__}", "error": str(exc)})
    return pd.DataFrame(rows)


def audit_reconstruction_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = []
    metrics = []
    wanted = {
        "model_feature_compare_summary.csv",
        "model_feature_compare_detail.csv",
        "bar_compare_summary.csv",
    }
    files = [p for p in SAVED_DATA.rglob("*.csv") if p.name in wanted and "feature_reconstruction_audit" in str(p)]
    for path in sorted(files):
        row = {"path": str(path), "name": path.name, "size_bytes": path.stat().st_size}
        try:
            df = pd.read_csv(path)
            row.update({"rows": len(df), "columns": ",".join(df.columns.astype(str))})
            numeric_cols = [c for c in df.columns if any(token in c.lower() for token in ["p50", "p90", "mean", "max", "abs", "rel", "diff", "error"])]
            for col in numeric_cols[:30]:
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.notna().any():
                    metrics.append(
                        {
                            "path": str(path),
                            "name": path.name,
                            "metric": col,
                            "non_null": int(vals.notna().sum()),
                            "median": float(vals.median()),
                            "p90": float(vals.quantile(0.90)),
                            "max": float(vals.max()),
                        }
                    )
        except Exception as exc:
            row["read_error"] = f"{type(exc).__name__}: {exc}"
        inventory.append(row)
    return pd.DataFrame(inventory), pd.DataFrame(metrics)


def write_summary(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    summary = {}
    feature_df = tables.get("saved_model_features", pd.DataFrame())
    model_samples = tables.get("saved_model_sample_availability", pd.DataFrame())
    leaderboard = tables.get("leaderboard_inventory", pd.DataFrame())
    lagged = tables.get("lagged_daily_sample_check", pd.DataFrame())
    recon = tables.get("reconstruction_audit_inventory", pd.DataFrame())
    if not feature_df.empty:
        by_model = feature_df.groupby(["stock_code", "artifact_name"], dropna=False).agg(
            feature_count=("feature", "count"),
            high_risk_count=("is_high_risk", "sum"),
            lagged_daily_count=("is_lagged_daily", "sum"),
            tail_sensitive_count=("is_tail_sensitive", "sum"),
        ).reset_index()
        by_model["high_risk_ratio"] = by_model["high_risk_count"] / by_model["feature_count"].replace(0, np.nan)
        by_model.sort_values(["high_risk_ratio", "lagged_daily_count"], ascending=[False, False]).to_csv(
            out_dir / "saved_model_high_risk_by_model.csv", index=False, encoding="utf-8-sig"
        )
        summary["saved_models"] = {
            "model_count": int(by_model.shape[0]),
            "feature_rows": int(feature_df.shape[0]),
            "models_with_lagged_daily": int((by_model["lagged_daily_count"] > 0).sum()),
            "models_with_high_risk_ratio_gt_20pct": int((by_model["high_risk_ratio"] > 0.20).sum()),
        }
    if not model_samples.empty:
        summary["saved_model_samples"] = {
            "metadata_count": int(model_samples.shape[0]),
            "samples_exists": int(model_samples["samples_exists"].sum()),
            "samples_missing": int((~model_samples["samples_exists"]).sum()),
        }
    if not leaderboard.empty:
        summary["leaderboards"] = {
            "rows": int(leaderboard.shape[0]),
            "files": int(leaderboard["leaderboard"].nunique()),
            "sample_rows_existing": int(leaderboard["sample_exists"].sum()) if "sample_exists" in leaderboard else 0,
            "sample_rows_missing": int((~leaderboard["sample_exists"]).sum()) if "sample_exists" in leaderboard else 0,
        }
    if not lagged.empty:
        summary["lagged_daily_sample_check"] = {
            "sample_files_checked": int(lagged.shape[0]),
            "date_t_exists_count": int(lagged.get("date_t_exists", pd.Series(dtype=bool)).fillna(False).sum()),
            "files_with_missing_lagged_daily": int((pd.to_numeric(lagged.get("lagged_missing_count", 0), errors="coerce").fillna(0) > 0).sum()),
        }
    summary["reconstruction_audit"] = {
        "files_found": int(recon.shape[0]) if not recon.empty else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trade-date", help="YYYY-MM-DD or YYYYMMDD. Defaults to each sample's latest date.")
    p.add_argument("--out-dir", default=str(REPORT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = args.trade_date
    if trade_date and re.fullmatch(r"\d{8}", trade_date):
        trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    out_dir = ensure_dir(Path(args.out_dir))

    feature_df, sample_df = audit_saved_models()
    leaderboard_df = audit_leaderboards()
    sample_paths = []
    if not sample_df.empty:
        sample_paths.extend(Path(p) for p in sample_df.loc[sample_df["samples_exists"], "resolved_samples"].tolist())
    if not leaderboard_df.empty and "sample_exists" in leaderboard_df:
        existing = leaderboard_df.loc[leaderboard_df["sample_exists"], "expected_local_sample_file"].tolist()
        sample_paths.extend(Path(p) for p in existing if p)
    sample_paths.extend(SAVED_DATA.rglob("training_samples*.csv"))
    lagged_df = audit_lagged_daily_samples(sample_paths, trade_date)
    recon_inventory, recon_metrics = audit_reconstruction_outputs()

    tables = {
        "saved_model_features": feature_df,
        "saved_model_sample_availability": sample_df,
        "leaderboard_inventory": leaderboard_df,
        "lagged_daily_sample_check": lagged_df,
        "reconstruction_audit_inventory": recon_inventory,
        "reconstruction_audit_metrics": recon_metrics,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_summary(out_dir, tables)
    print(f"WROTE audit reports to {out_dir}")


if __name__ == "__main__":
    main()
