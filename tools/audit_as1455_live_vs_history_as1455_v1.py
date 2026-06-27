#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit live AS1455 inference features against history-reconstructed AS1455.

This script compares, for a target live date T:

1) the original live AS1455 raw row used at T 14:55, usually
   live_as1455/T/08_live_raw_row_as1455.csv;
2) the AS1455 row for T reconstructed after later history update from
   ch12_as1455/as1455_daily_cache;
3) the model input features produced by replacing T's live row with the
   reconstructed AS1455 row while reusing the original T prefast state.

It intentionally does NOT compare to 15:00 full-day close. The reconstructed
row is expected to be AS1455, i.e. the same intraday cutoff semantics as live.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS_DEFAULT = [
    "dollar_vol", "dollar_vol_rank", "rsi", "bb_high", "bb_low",
    "NATR", "ATR", "PPO", "MACD", "sector",
    "r01", "r05", "r10", "r21", "r42", "r63",
    "r01dec", "r05dec", "r10dec", "r21dec", "r42dec", "r63dec",
    "r01q_sector", "r05q_sector", "r10q_sector", "r21q_sector", "r42q_sector", "r63q_sector",
    "year", "month", "weekday",
]

RAW_OUT_COLUMNS = [
    "symbol",
    "raw_open_as1455",
    "raw_high_as1455",
    "raw_low_as1455",
    "raw_close_as1455",
    "raw_volume_as1455",
    "raw_amount_as1455",
    "live_preclose",
]

RAW_COMPARE_COLUMNS = [c for c in RAW_OUT_COLUMNS if c != "symbol"]

DATE_COLUMNS = ["date", "trade_date", "dt", "datetime"]
SYMBOL_COLUMNS = ["symbol", "code", "ticker", "ts_code"]

COL_CANDIDATES = {
    "raw_open_as1455": ["raw_open_as1455", "open_as1455", "open", "raw_open", "open_1455"],
    "raw_high_as1455": ["raw_high_as1455", "high_as1455", "high", "raw_high", "high_1455"],
    "raw_low_as1455": ["raw_low_as1455", "low_as1455", "low", "raw_low", "low_1455"],
    "raw_close_as1455": ["raw_close_as1455", "close_as1455", "close", "raw_close", "close_1455", "price"],
    "raw_volume_as1455": ["raw_volume_as1455", "volume_as1455", "volume", "raw_volume", "vol", "volume_1455"],
    "raw_amount_as1455": ["raw_amount_as1455", "amount_as1455", "amount", "raw_amount", "turnover", "amount_1455"],
    "live_preclose": ["live_preclose", "preclose", "raw_preclose", "prev_close", "pre_close", "last_close"],
}


def parse_yyyymmdd(s: str) -> str:
    x = str(s).strip().replace("-", "")
    if x.lower() == "today":
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", x):
        raise ValueError(f"bad date: {s}")
    pd.Timestamp(f"{x[:4]}-{x[4:6]}-{x[6:8]}")
    return x


def dash_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def parse_date_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    digits = raw.str.replace(r"\D", "", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mask8 = digits.str.fullmatch(r"\d{8}", na=False)
    if mask8.any():
        out.loc[mask8] = pd.to_datetime(digits.loc[mask8], format="%Y%m%d", errors="coerce")
    # Full datetime like YYYYMMDDHHMMSS.
    mask_long = (~mask8) & digits.str.len().ge(14) & digits.str[:8].str.fullmatch(r"\d{8}", na=False)
    if mask_long.any():
        out.loc[mask_long] = pd.to_datetime(digits.loc[mask_long].str[:8], format="%Y%m%d", errors="coerce")
    rest = out.isna()
    if rest.any():
        out.loc[rest] = pd.to_datetime(raw.loc[rest], errors="coerce").dt.normalize()
    return out.dt.normalize()


def normalize_symbol(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.upper().replace("_", ".")
    if s.startswith("SH.") or s.startswith("SZ."):
        exch, code = s.split(".", 1)
        return f"{code}.{exch}"
    if s.startswith("SH") and re.fullmatch(r"SH\d{6}", s):
        return f"{s[2:]}.SH"
    if s.startswith("SZ") and re.fullmatch(r"SZ\d{6}", s):
        return f"{s[2:]}.SZ"
    if re.fullmatch(r"\d{6}\.SH", s) or re.fullmatch(r"\d{6}\.SZ", s):
        return s
    if re.fullmatch(r"\d{6}", s):
        return f"{s}.SH" if s.startswith("6") else f"{s}.SZ"
    m = re.search(r"(\d{6})[._-]?(SH|SZ)?", s)
    if m:
        code, exch = m.group(1), m.group(2)
        if exch is None:
            exch = "SH" if code.startswith("6") else "SZ"
        return f"{code}.{exch}"
    return s


def infer_symbol_from_path(path: Path) -> str | None:
    return normalize_symbol(path.stem)


def read_table(path: Path, columns_hint: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"failed to read parquet {path}: {e}") from e
    return pd.DataFrame()


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def load_as1455_cache_rows(cache_dir: Path, target_date: str, max_files: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"missing as1455 cache dir: {cache_dir}")
    target_ts = pd.Timestamp(dash_date(target_date))
    files = sorted([p for p in cache_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".parquet", ".pq"}])
    if max_files is not None:
        files = files[:max_files]
    rows = []
    errors = []
    scanned = 0
    with_target = 0
    for p in files:
        scanned += 1
        try:
            df = read_table(p)
            if df.empty:
                continue
            date_col = first_existing_col(df, DATE_COLUMNS)
            if date_col is None:
                continue
            dates = parse_date_series(df[date_col])
            sub = df.loc[dates.eq(target_ts)].copy()
            if sub.empty:
                continue
            with_target += 1
            sym_col = first_existing_col(sub, SYMBOL_COLUMNS)
            if sym_col is not None:
                sub["symbol"] = sub[sym_col].map(normalize_symbol)
            else:
                inferred = infer_symbol_from_path(p)
                sub["symbol"] = inferred
            out = pd.DataFrame({"symbol": sub["symbol"]})
            for out_col, candidates in COL_CANDIDATES.items():
                src = first_existing_col(sub, candidates)
                if src is None:
                    out[out_col] = np.nan
                else:
                    out[out_col] = pd.to_numeric(sub[src], errors="coerce")
            rows.append(out)
        except Exception as e:
            errors.append({"path": str(p), "error": repr(e)})
            if len(errors) >= 20:
                # keep going but do not let report blow up
                pass
    if not rows:
        meta = {"cache_dir": str(cache_dir), "files_scanned": scanned, "files_with_target_date": with_target, "errors_sample": errors[:20]}
        raise RuntimeError(f"no AS1455 rows found for {target_date} under {cache_dir}; meta={meta}")
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["symbol"])
    # Keep last row if duplicate symbol/date appears.
    out = out.drop_duplicates("symbol", keep="last").sort_values("symbol").reset_index(drop=True)
    meta = {
        "cache_dir": str(cache_dir),
        "files_scanned": scanned,
        "files_with_target_date": with_target,
        "rows_loaded": int(len(out)),
        "errors_count": int(len(errors)),
        "errors_sample": errors[:20],
        "nonnull_by_column": {c: int(out[c].notna().sum()) for c in out.columns},
    }
    return out[RAW_OUT_COLUMNS], meta


def load_csv_norm(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    if "symbol" not in df.columns:
        raise RuntimeError(f"{path} missing symbol column: {list(df.columns)}")
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df.dropna(subset=["symbol"]).drop_duplicates("symbol", keep="last").sort_values("symbol").reset_index(drop=True)


def compare_tables(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], left_name: str, right_name: str, tol: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = left.merge(right, on="symbol", how="outer", suffixes=(f"_{left_name}", f"_{right_name}"), indicator=True)
    summary = []
    details = []
    for c in columns:
        lc = f"{c}_{left_name}"
        rc = f"{c}_{right_name}"
        if lc not in m.columns or rc not in m.columns:
            summary.append({
                "field": c, "n_left_nonnull": 0, "n_right_nonnull": 0, "n_both": 0,
                "n_changed": 0, "changed_rate": np.nan, "max_abs_diff": np.nan,
                "mean_abs_diff": np.nan, "median_abs_diff": np.nan, "p95_abs_diff": np.nan,
                "note": "missing_column_after_merge",
            })
            continue
        a = pd.to_numeric(m[lc], errors="coerce")
        b = pd.to_numeric(m[rc], errors="coerce")
        both = a.notna() & b.notna()
        absdiff = (a - b).abs()
        changed = both & (absdiff > tol)
        summary.append({
            "field": c,
            "n_left_nonnull": int(a.notna().sum()),
            "n_right_nonnull": int(b.notna().sum()),
            "n_both": int(both.sum()),
            "n_changed": int(changed.sum()),
            "changed_rate": float(changed.sum() / max(int(both.sum()), 1)),
            "max_abs_diff": float(absdiff[both].max()) if both.any() else np.nan,
            "mean_abs_diff": float(absdiff[both].mean()) if both.any() else np.nan,
            "median_abs_diff": float(absdiff[both].median()) if both.any() else np.nan,
            "p95_abs_diff": float(absdiff[both].quantile(0.95)) if both.any() else np.nan,
            "note": "ok",
        })
        if changed.any():
            top = m.loc[changed, ["symbol", lc, rc]].copy()
            top["field"] = c
            top["abs_diff"] = absdiff.loc[changed]
            top = top.sort_values("abs_diff", ascending=False).head(100)
            top = top[["field", "symbol", lc, rc, "abs_diff"]]
            details.append(top)
    s = pd.DataFrame(summary).sort_values(["changed_rate", "max_abs_diff"], ascending=False, na_position="last")
    d = pd.concat(details, ignore_index=True) if details else pd.DataFrame(columns=["field", "symbol", f"value_{left_name}", f"value_{right_name}", "abs_diff"])
    return s, d, m


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit 25-day live AS1455 features vs history-reconstructed AS1455 features")
    ap.add_argument("--trade-date", required=True, help="target date, e.g. 20260625")
    ap.add_argument("--live-root", default="saved_data/ashare_ml4t/live_as1455")
    ap.add_argument("--as1455-cache-dir", default="saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache")
    ap.add_argument("--audit-dir", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--min-cache-rows", type=int, default=900)
    ap.add_argument("--min-feature-rows", type=int, default=900)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--skip-finalize", action="store_true", help="only compare raw AS1455 rows, do not recompute features")
    args = ap.parse_args()

    started = time.time()
    trade_date = parse_yyyymmdd(args.trade_date)
    live_root = Path(args.live_root)
    live_dir = live_root / trade_date
    if not live_dir.exists():
        raise FileNotFoundError(f"missing live dir: {live_dir}")
    audit_dir = Path(args.audit_dir) if args.audit_dir else live_root / f"{trade_date}_audit_history_as1455"
    audit_dir.mkdir(parents=True, exist_ok=True)

    live_raw_path = live_dir / "08_live_raw_row_as1455.csv"
    live_qfq_path = live_dir / "09_live_qfq_row_as1455.csv"
    live_feat_path = live_dir / "11_live_model_features_for_prediction.csv"
    state_path = live_dir / "06_live_feature_state_fast.npz"
    events_path = live_dir / "03_adjustment_events.csv"

    for p in [live_raw_path, live_feat_path, state_path]:
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    # 1) Build history-reconstructed AS1455 raw row for the same target date.
    hist_raw, cache_meta = load_as1455_cache_rows(Path(args.as1455_cache_dir), trade_date)
    if len(hist_raw) < args.min_cache_rows:
        raise RuntimeError(f"too few history AS1455 rows: {len(hist_raw)} < {args.min_cache_rows}. cache_meta={cache_meta}")
    hist_raw_path = audit_dir / "08_live_raw_row_as1455.csv"
    hist_raw.to_csv(hist_raw_path, index=False, encoding="utf-8-sig")

    # Copy state/events so finalize script can reuse exactly the original pre-25 state and factor events.
    shutil.copy2(state_path, audit_dir / "06_live_feature_state_fast.npz")
    if events_path.exists():
        shutil.copy2(events_path, audit_dir / "03_adjustment_events.csv")

    # Feature columns: force the reconstructed finalize path to use exactly the original live prediction columns.
    live_feat = load_csv_norm(live_feat_path)
    feature_cols = [c for c in live_feat.columns if c not in {"date", "symbol"}]
    if not feature_cols:
        feature_cols = FEATURE_COLUMNS_DEFAULT
    feature_cols_path = audit_dir / "feature_columns_from_live_prediction.json"
    feature_cols_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) Compare raw 08 rows before feature recomputation.
    live_raw = load_csv_norm(live_raw_path)
    raw_cols = [c for c in RAW_COMPARE_COLUMNS if c in live_raw.columns and c in hist_raw.columns]
    raw_summary, raw_detail, raw_merged = compare_tables(live_raw, hist_raw, raw_cols, "live1455", "hist_as1455", args.tol)
    raw_summary.to_csv(audit_dir / "08_raw_as1455_diff_summary.csv", index=False, encoding="utf-8-sig")
    raw_detail.to_csv(audit_dir / "08_raw_as1455_diff_top100_each_field.csv", index=False, encoding="utf-8-sig")
    raw_merged.to_csv(audit_dir / "08_raw_as1455_diff_merged.csv", index=False, encoding="utf-8-sig")

    # 3) Recompute features through the same fast finalizer, using original prefast state + history AS1455 row.
    finalize_ran = False
    if not args.skip_finalize:
        finalize_script = Path("features/finalize_as1455_live_features_fast.py")
        if not finalize_script.exists():
            raise FileNotFoundError(f"missing finalize script: {finalize_script}. Install fastpath first.")
        run([
            args.python,
            str(finalize_script),
            "--trade-date", trade_date,
            "--live-dir", str(audit_dir),
            "--state-file", str(audit_dir / "06_live_feature_state_fast.npz"),
            "--training-feature-columns", str(feature_cols_path),
            "--min-feature-rows", str(args.min_feature_rows),
            "--max-elapsed-seconds", "9999",
            "--warn-only-time",
        ])
        finalize_ran = True

    report: dict[str, Any] = {
        "trade_date": trade_date,
        "live_dir": str(live_dir),
        "audit_dir": str(audit_dir),
        "as1455_cache_dir": str(args.as1455_cache_dir),
        "cache_meta": cache_meta,
        "live_raw_path": str(live_raw_path),
        "hist_raw_path": str(hist_raw_path),
        "live_feature_path": str(live_feat_path),
        "feature_columns_count": len(feature_cols),
        "feature_columns": feature_cols,
        "raw_compare": {
            "live_rows": int(len(live_raw)),
            "hist_rows": int(len(hist_raw)),
            "merge_counts": {str(k): int(v) for k, v in raw_merged["_merge"].value_counts().to_dict().items()},
            "summary_csv": str(audit_dir / "08_raw_as1455_diff_summary.csv"),
            "detail_csv": str(audit_dir / "08_raw_as1455_diff_top100_each_field.csv"),
        },
        "finalize_ran": bool(finalize_ran),
    }

    # 4) Compare qfq 09 and final 11 features if available.
    hist_qfq_path = audit_dir / "09_live_qfq_row_as1455.csv"
    hist_feat_path = audit_dir / "11_live_model_features_for_prediction.csv"
    if finalize_ran and hist_qfq_path.exists() and live_qfq_path.exists():
        live_qfq = load_csv_norm(live_qfq_path)
        hist_qfq = load_csv_norm(hist_qfq_path)
        qfq_cols = [c for c in live_qfq.columns if c not in {"date", "symbol"} and c in hist_qfq.columns]
        qfq_summary, qfq_detail, qfq_merged = compare_tables(live_qfq, hist_qfq, qfq_cols, "live1455", "hist_as1455", args.tol)
        qfq_summary.to_csv(audit_dir / "09_qfq_as1455_diff_summary.csv", index=False, encoding="utf-8-sig")
        qfq_detail.to_csv(audit_dir / "09_qfq_as1455_diff_top100_each_field.csv", index=False, encoding="utf-8-sig")
        qfq_merged.to_csv(audit_dir / "09_qfq_as1455_diff_merged.csv", index=False, encoding="utf-8-sig")
        report["qfq_compare"] = {
            "live_rows": int(len(live_qfq)),
            "hist_rows": int(len(hist_qfq)),
            "merge_counts": {str(k): int(v) for k, v in qfq_merged["_merge"].value_counts().to_dict().items()},
            "summary_csv": str(audit_dir / "09_qfq_as1455_diff_summary.csv"),
            "detail_csv": str(audit_dir / "09_qfq_as1455_diff_top100_each_field.csv"),
        }
    if finalize_ran and hist_feat_path.exists():
        hist_feat = load_csv_norm(hist_feat_path)
        live_feat = load_csv_norm(live_feat_path)
        feat_cols = [c for c in feature_cols if c in live_feat.columns and c in hist_feat.columns]
        feat_summary, feat_detail, feat_merged = compare_tables(live_feat, hist_feat, feat_cols, "live1455", "hist_as1455", args.tol)
        feat_summary.to_csv(audit_dir / "11_feature_diff_summary.csv", index=False, encoding="utf-8-sig")
        feat_detail.to_csv(audit_dir / "11_feature_diff_top100_each_field.csv", index=False, encoding="utf-8-sig")
        feat_merged.to_csv(audit_dir / "11_feature_diff_merged.csv", index=False, encoding="utf-8-sig")
        report["feature_compare"] = {
            "live_rows": int(len(live_feat)),
            "hist_rows": int(len(hist_feat)),
            "merge_counts": {str(k): int(v) for k, v in feat_merged["_merge"].value_counts().to_dict().items()},
            "summary_csv": str(audit_dir / "11_feature_diff_summary.csv"),
            "detail_csv": str(audit_dir / "11_feature_diff_top100_each_field.csv"),
        }

    report["elapsed_seconds"] = round(time.time() - started, 3)
    (audit_dir / "audit_live_vs_history_as1455_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("\n[OK] audit outputs written under:", audit_dir, flush=True)
    if (audit_dir / "11_feature_diff_summary.csv").exists():
        print("\n== feature diff top fields ==", flush=True)
        s = pd.read_csv(audit_dir / "11_feature_diff_summary.csv")
        print(s.head(40).to_string(index=False), flush=True)
    else:
        print("\n== raw AS1455 diff top fields ==", flush=True)
        print(raw_summary.head(40).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
