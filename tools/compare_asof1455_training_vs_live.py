#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare asof1455 training-sample features with replayed live watch features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_building.build_asof1455_training_samples import add_asof_rolling_features, build_intraday_asof  # noqa: E402
from feature_building.build_fundamental_features import apply_asof1455_valuation_policy  # noqa: E402
from model_training.optimize_nextday_vwap_model import feature_groups, is_lagged_daily_external_feature_name  # noqa: E402
from pipelines.run_intraday_nextday_signals import (  # noqa: E402
    add_scoring_features,
    cache_symbol_dir,
    fill_lagged_daily_features_from_current_sample,
    normalize_symbol,
    overlay_current_day_from_cache,
    yyyymmdd_to_iso,
)


def normalize_path(value: object) -> Path | None:
    if not value:
        return None
    text = str(value)
    p = Path(text)
    if p.exists():
        return p
    marker = "stock_realtime_v021_full"
    parts = p.parts
    if marker in parts:
        cand = ROOT.joinpath(*parts[parts.index(marker) + 1 :])
        if cand.exists():
            return cand
    cand = ROOT / text
    return cand if cand.exists() else None


def load_pipeline_meta(pipeline_dir: Path) -> dict:
    meta_path = pipeline_dir / "pipeline_summary.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def find_pipeline_dirs(root: Path) -> list[Path]:
    return sorted(
        p.parent
        for p in root.glob("*_pipeline_out/pipeline_summary.json")
        if p.parent.name[:6].isdigit()
    )


def find_existing_sample(pipeline_dir: Path) -> Path | None:
    priorities = [
        "04_external/**/*.csv",
        "03_sector/training_samples*.csv",
        "02_fundamental/training_samples*.csv",
        "01_samples_asof1455/training_samples*.csv",
        "01_samples/training_samples*.csv",
    ]
    for pat in priorities:
        hits = sorted(pipeline_dir.glob(pat))
        hits = [p for p in hits if p.name.startswith("training_samples")]
        if hits:
            return hits[-1]
    return None


def find_existing_intraday(pipeline_dir: Path, stock: str) -> Path | None:
    raw = stock.split(".", 1)[0]
    priorities = [
        f"00_base/{raw}_5m.csv",
        f"00_base/raw_cache/{raw}_5m_raw.csv",
        "**/*_5m.csv",
        "**/*_5m_raw.csv",
    ]
    for pat in priorities:
        hits = sorted(pipeline_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def ensure_asof_training_scale(samples: pd.DataFrame, intraday_path: Path, cutoff_time: str, min_bars: int = 40) -> pd.DataFrame:
    out = samples.sort_values("date").reset_index(drop=True).copy()
    if "close_asof1455" not in out.columns:
        asof = build_intraday_asof(intraday_path, cutoff_time, min_bars)
        if asof.empty:
            return out
        out = out.merge(asof, on="date", how="inner").sort_values("date").reset_index(drop=True)
        out = add_asof_rolling_features(out)
        out["feature_cutoff_time"] = cutoff_time
    out = apply_asof1455_valuation_policy(out, out)
    for col in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        if col in out.columns:
            out[f"{col}_rank252"] = pd.to_numeric(out[col], errors="coerce").rolling(252, min_periods=60).rank(pct=True)
    return out


def exchange(symbol: str) -> str:
    return normalize_symbol(symbol).split(".", 1)[1]


def write_synthetic_snapshots(stock: str, intraday_path: Path, dates: list[str], cache: Path) -> pd.DataFrame:
    bars = pd.read_csv(intraday_path, parse_dates=["datetime"])
    if bars.empty:
        return pd.DataFrame()
    bars = bars.dropna(subset=["datetime"]).sort_values("datetime").copy()
    bars["date_key"] = bars["datetime"].dt.strftime("%Y%m%d")
    rows = []
    for date_key in dates:
        day = bars[bars["date_key"] == date_key].copy()
        if day.empty:
            rows.append({"stock_code": stock, "trade_date": date_key, "status": "missing_intraday_date"})
            continue
        day["cum_volume"] = pd.to_numeric(day.get("volume"), errors="coerce").fillna(0).cumsum()
        day["cum_amount"] = pd.to_numeric(day.get("amount"), errors="coerce").fillna(0).cumsum()
        prev = bars[bars["datetime"].dt.normalize() < day["datetime"].dt.normalize().iloc[0]]
        prev_close = float(pd.to_numeric(prev["close"], errors="coerce").dropna().iloc[-1]) if not prev.empty else np.nan
        out_rows = []
        for idx, r in day.iterrows():
            dt = pd.to_datetime(r["datetime"])
            close = float(pd.to_numeric(pd.Series([r.get("close")]), errors="coerce").iloc[0])
            pct = close / prev_close - 1.0 if np.isfinite(prev_close) and prev_close > 0 else np.nan
            partial = day.loc[:idx]
            out_rows.append({
                "vendor": "synthetic_from_training_5m",
                "source": "training_5m",
                "quote_source": "training_5m",
                "collected_at": dt.isoformat(),
                "datetime": dt.isoformat(),
                "symbol": stock,
                "exchange": exchange(stock),
                "trade_date": date_key,
                "trade_time": dt.strftime("%H%M%S"),
                "phase": "continuous",
                "name": stock,
                "last_price": close,
                "open": float(pd.to_numeric(day["open"], errors="coerce").dropna().iloc[0]),
                "high": float(pd.to_numeric(partial["high"], errors="coerce").max()),
                "low": float(pd.to_numeric(partial["low"], errors="coerce").min()),
                "prev_close": prev_close,
                "volume": float(r["cum_volume"]),
                "amount": float(r["cum_amount"]),
                "pct_chg": pct,
                "pct_chg_raw": pct,
                "pct_chg_norm": pct,
                "pct_chg_source": "computed_from_prev_close",
                "pct_chg_unit": "ratio",
            })
        sym_dir = cache_symbol_dir(cache, date_key, stock)
        sym_dir.mkdir(parents=True, exist_ok=True)
        path = sym_dir / "snapshot_5level.csv"
        pd.DataFrame(out_rows).to_csv(path, index=False, encoding="utf-8-sig")
        minute = day[["datetime", "open", "high", "low", "close", "volume", "amount"]].copy()
        minute.insert(0, "symbol", stock)
        minute.insert(1, "trade_date", date_key)
        minute["source"] = "local_snapshot_5m_right_endpoint"
        minute["bar_freq"] = "5min"
        minute["bar_label"] = "right"
        minute["bar_volume"] = pd.to_numeric(minute["volume"], errors="coerce")
        minute["bar_amount"] = pd.to_numeric(minute["amount"], errors="coerce")
        minute.to_csv(sym_dir / "minute_bars_5min.csv", index=False, encoding="utf-8-sig")
        rows.append({"stock_code": stock, "trade_date": date_key, "status": "ok", "snapshot_rows": len(out_rows), "path": str(path)})
    return pd.DataFrame(rows)


def feature_category(feature: str) -> str:
    text = str(feature)
    if is_lagged_daily_external_feature_name(text):
        return "lagged_daily_external"
    if text.endswith("_asof1455") or "_asof" in text:
        if any(k in text for k in ["ret", "gap", "z"]):
            return "stock_asof_derived"
        return "stock_asof_raw"
    if text.startswith(("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "profit_", "operation_", "growth_", "solvency_", "cashflow_", "dupont_", "fund_days_since_effective")):
        return "fundamental"
    return "other_allowed"


def compare_one(pipeline_dir: Path, dates: list[str], out_cache: Path, cutoff_time: str, max_missing: float, group: str):
    meta = load_pipeline_meta(pipeline_dir)
    stock = normalize_symbol((meta.get("symbol") or {}).get("stock_code") or pipeline_dir.name.split("_", 1)[0])
    sample_path = normalize_path(meta.get("final_samples")) or find_existing_sample(pipeline_dir)
    intraday_path = normalize_path(meta.get("intraday_bars")) or find_existing_intraday(pipeline_dir, stock)
    if sample_path is None or intraday_path is None:
        return [], [], [{"stock_code": stock, "pipeline_dir": str(pipeline_dir), "status": "missing_paths", "sample_path": str(meta.get("final_samples")), "intraday_path": str(meta.get("intraday_bars"))}]

    samples = pd.read_csv(sample_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    samples = ensure_asof_training_scale(samples, intraday_path, cutoff_time)
    if "close_asof1455" not in samples.columns or samples.empty:
        return [], [], [{"stock_code": stock, "pipeline_dir": str(pipeline_dir), "status": "cannot_build_asof_samples", "sample_path": str(sample_path), "intraday_path": str(intraday_path)}]
    groups = feature_groups(samples, max_missing=max_missing, feature_time_mode="asof1455")
    cols = groups.get(group, [])
    if not cols:
        return [], [], [{"stock_code": stock, "pipeline_dir": str(pipeline_dir), "status": "empty_feature_group", "sample_path": str(sample_path)}]

    date_keys = [d.replace("-", "") for d in dates]
    snapshot_report = write_synthetic_snapshots(stock, intraday_path, date_keys, out_cache)
    summaries = []
    details = []
    reports = snapshot_report.to_dict(orient="records") if not snapshot_report.empty else []
    for date_key in date_keys:
        target_ts = pd.to_datetime(yyyymmdd_to_iso(date_key)).normalize()
        train_row = samples[pd.to_datetime(samples["date"], errors="coerce").dt.normalize() == target_ts]
        hist = samples[pd.to_datetime(samples["date"], errors="coerce").dt.normalize() < target_ts].copy()
        if train_row.empty or hist.empty:
            reports.append({"stock_code": stock, "trade_date": date_key, "status": "missing_sample_date"})
            continue
        if not (cache_symbol_dir(out_cache, date_key, stock) / "snapshot_5level.csv").exists():
            continue

        replay = overlay_current_day_from_cache(hist, stock, date_key, out_cache, cutoff_time=cutoff_time)
        replay = add_scoring_features(replay, intraday_path, out_cache, stock, cutoff_time=cutoff_time, scoring_trade_date=date_key)
        replay = add_asof_rolling_features(replay)
        replay, lagged_filled, lagged_missing = fill_lagged_daily_features_from_current_sample(replay, samples, date_key, cols)
        live_row = replay[pd.to_datetime(replay["date"], errors="coerce").dt.normalize() == target_ts].tail(1)
        if live_row.empty:
            reports.append({"stock_code": stock, "trade_date": date_key, "status": "missing_live_row"})
            continue
        live = live_row.iloc[-1]
        train = train_row.iloc[-1]
        comparable = 0
        exactish = 0
        live_missing = 0
        train_missing = 0
        abs_rel_values = []
        for col in cols:
            lv = pd.to_numeric(pd.Series([live[col] if col in live_row.columns else np.nan]), errors="coerce").iloc[0]
            tv = pd.to_numeric(pd.Series([train[col] if col in train_row.columns else np.nan]), errors="coerce").iloc[0]
            lna = pd.isna(lv)
            tna = pd.isna(tv)
            live_missing += int(lna)
            train_missing += int(tna)
            diff = np.nan
            abs_rel = np.nan
            scale_ratio = np.nan
            if not lna and not tna:
                comparable += 1
                diff = float(lv) - float(tv)
                abs_rel = abs(diff) / max(abs(float(tv)), 1e-9)
                scale_ratio = float(lv) / float(tv) if abs(float(tv)) > 1e-12 else np.nan
                abs_rel_values.append(abs_rel)
                exactish += int(abs(diff) <= 1e-9 or abs_rel <= 1e-6)
            details.append({
                "trade_date": yyyymmdd_to_iso(date_key),
                "stock_code": stock,
                "pipeline_dir": str(pipeline_dir),
                "feature": col,
                "feature_category": feature_category(col),
                "live_value": lv,
                "train_value": tv,
                "diff": diff,
                "abs_rel_diff": abs_rel,
                "scale_ratio_live_over_train": scale_ratio,
                "live_missing": bool(lna),
                "train_missing": bool(tna),
            })
        summaries.append({
            "trade_date": yyyymmdd_to_iso(date_key),
            "stock_code": stock,
            "pipeline_dir": str(pipeline_dir),
            "sample_file": str(sample_path),
            "intraday_bars": str(intraday_path),
            "feature_group": group,
            "feature_count": len(cols),
            "comparable_features": comparable,
            "live_missing": live_missing,
            "train_missing": train_missing,
            "exactish_features": exactish,
            "exactish_share": exactish / comparable if comparable else np.nan,
            "median_abs_rel_diff": float(np.nanmedian(abs_rel_values)) if abs_rel_values else np.nan,
            "p90_abs_rel_diff": float(np.nanpercentile(abs_rel_values, 90)) if abs_rel_values else np.nan,
            "p99_abs_rel_diff": float(np.nanpercentile(abs_rel_values, 99)) if abs_rel_values else np.nan,
            "max_abs_rel_diff": float(np.nanmax(abs_rel_values)) if abs_rel_values else np.nan,
            "lagged_daily_filled": len(lagged_filled),
            "lagged_daily_missing": len(lagged_missing),
        })
    return summaries, details, reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", default=str(ROOT / "saved_data"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "asof1455_feature_scale_compare"))
    ap.add_argument("--dates", default="20260519,20260520,20260521", help="Comma dates, or 'auto' for recent sample dates per pipeline")
    ap.add_argument("--cutoff-time", default="14:55")
    ap.add_argument("--max-missing", type=float, default=0.35)
    ap.add_argument("--feature-group", default="all_no_ak")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "synthetic_snapshot_cache"
    dates = [d.strip() for d in args.dates.split(",") if d.strip()] if str(args.dates).lower() != "auto" else []
    summaries = []
    details = []
    reports = []
    for pipeline_dir in find_pipeline_dirs(Path(args.pipeline_root)):
        use_dates = dates
        if not use_dates:
            meta = load_pipeline_meta(pipeline_dir)
            stock = normalize_symbol((meta.get("symbol") or {}).get("stock_code") or pipeline_dir.name.split("_", 1)[0])
            sample_path = normalize_path(meta.get("final_samples")) or find_existing_sample(pipeline_dir)
            intraday_path = normalize_path(meta.get("intraday_bars")) or find_existing_intraday(pipeline_dir, stock)
            if sample_path is not None and intraday_path is not None:
                try:
                    tmp = pd.read_csv(sample_path, parse_dates=["date"])
                    asof_tmp = ensure_asof_training_scale(tmp, intraday_path, args.cutoff_time)
                    use_dates = [d.strftime("%Y%m%d") for d in pd.to_datetime(asof_tmp["date"], errors="coerce").dropna().sort_values().tail(5)]
                except Exception:
                    use_dates = []
        s, d, r = compare_one(pipeline_dir, use_dates, cache, args.cutoff_time, args.max_missing, args.feature_group)
        summaries.extend(s)
        details.extend(d)
        reports.extend(r)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(details)
    report_df = pd.DataFrame(reports)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["stock_code", "trade_date", "feature_category", "feature"]).reset_index(drop=True)
    category_df = pd.DataFrame()
    if not detail_df.empty:
        category_df = (
            detail_df.groupby(["stock_code", "feature_category"], dropna=False)
            .agg(
                rows=("feature", "count"),
                comparable=("abs_rel_diff", lambda x: int(x.notna().sum())),
                live_missing=("live_missing", "sum"),
                train_missing=("train_missing", "sum"),
                median_abs_rel_diff=("abs_rel_diff", "median"),
                p90_abs_rel_diff=("abs_rel_diff", lambda x: float(np.nanpercentile(x.dropna(), 90)) if x.notna().any() else np.nan),
                max_abs_rel_diff=("abs_rel_diff", "max"),
            )
            .reset_index()
        )

    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(out_dir / "feature_diffs.csv", index=False, encoding="utf-8-sig")
    category_df.to_csv(out_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    report_df.to_csv(out_dir / "input_report.csv", index=False, encoding="utf-8-sig")
    print(f"WROTE {out_dir / 'summary.csv'} rows={len(summary_df)}")
    print(f"WROTE {out_dir / 'feature_diffs.csv'} rows={len(detail_df)}")
    print(f"WROTE {out_dir / 'category_summary.csv'} rows={len(category_df)}")
    print(f"WROTE {out_dir / 'input_report.csv'} rows={len(report_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
