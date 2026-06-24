#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit + selective backup + clean AS1455 5m cache files back to the old schema.

This script DOES NOT back up the full cache directory.
It only backs up files that are classified as needing cleaning.

Target old schema:
  symbol, trade_date, datetime, open, high, low, close, volume, amount,
  source, bar_freq, bar_label
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


OLD_COLS = [
    "symbol",
    "trade_date",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "bar_freq",
    "bar_label",
]

RAW_BAOSTOCK_COLS = ["date", "time", "code", "adjustflag"]
RAW_BAOSTOCK_REQUIRED = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount"]

DEFAULT_CACHE_DIR = "saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache"


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [c.strip() for c in next(csv.reader(f))]


def classify_columns(cols: List[str]) -> str:
    colset = set(cols)
    has_old_all = all(c in colset for c in OLD_COLS)
    has_raw_any = any(c in colset for c in RAW_BAOSTOCK_COLS)
    has_raw_required = all(c in colset for c in RAW_BAOSTOCK_REQUIRED)

    if cols == OLD_COLS:
        return "old_schema_ok"
    if has_old_all and has_raw_any:
        return "mixed_polluted"
    if has_raw_required and not has_old_all:
        return "raw_baostock_only"
    if has_old_all and cols != OLD_COLS:
        return "old_schema_extra_cols"
    return "unknown_schema"


def symbol_from_code(code: str) -> str:
    s = str(code).strip()
    if not s or s.lower() == "nan":
        return ""
    low = s.lower()
    if low.startswith("sh."):
        return f"{s[3:9]}.SH"
    if low.startswith("sz."):
        return f"{s[3:9]}.SZ"
    if len(s) >= 9 and s[6] == ".":
        return s[:6].upper() + "." + s[7:9].upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        code6 = digits[:6]
        return f"{code6}.SH" if code6.startswith("6") else f"{code6}.SZ"
    return ""


def symbol_from_path(path: Path) -> str:
    code6 = path.name.split("_")[0]
    return f"{code6}.SH" if code6.startswith("6") else f"{code6}.SZ"


def normalize_symbol_value(value: object, fallback: str) -> str:
    sym = symbol_from_code(str(value))
    return sym if sym else fallback


def first_existing_series(df: pd.DataFrame, names: List[str], default: object = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def parse_datetime(df: pd.DataFrame) -> pd.Series:
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if "datetime" in df.columns:
        dt_old = pd.to_datetime(df["datetime"], errors="coerce")
        dt = dt.fillna(dt_old)

    if "time" in df.columns:
        time_s = df["time"].astype(str).str.replace(r"\D", "", regex=True)
        raw14 = time_s.str.slice(0, 14)
        dt_time = pd.to_datetime(raw14, format="%Y%m%d%H%M%S", errors="coerce")
        dt = dt.fillna(dt_time)

    if "date" in df.columns and "time" in df.columns:
        date_s = df["date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
        time_s = df["time"].astype(str).str.strip()
        hhmmss = time_s.str.extract(r"^(\d{1,2}):?(\d{2}):?(\d{0,2})", expand=True)
        valid = dt.isna() & date_s.str.len().eq(8) & hhmmss[0].notna()
        if valid.any():
            hh = hhmmss.loc[valid, 0].str.zfill(2)
            mm = hhmmss.loc[valid, 1].str.zfill(2)
            ss = hhmmss.loc[valid, 2].replace("", "00").fillna("00").str.zfill(2)
            dt_short = pd.to_datetime(date_s.loc[valid] + hh + mm + ss, format="%Y%m%d%H%M%S", errors="coerce")
            dt.loc[valid] = dt_short

    return dt


def source_series(df: pd.DataFrame) -> pd.Series:
    if "source" in df.columns:
        src = df["source"].astype(str)
        bad = src.isna() | src.str.strip().eq("") | src.str.lower().eq("nan")
    else:
        src = pd.Series([""] * len(df), index=df.index)
        bad = pd.Series([True] * len(df), index=df.index)

    if "adjustflag" in df.columns:
        adj = df["adjustflag"].astype(str).str.replace(r"\D", "", regex=True)
        adj = adj.where(adj.str.len() > 0, "3")
        fill = "baostock_5m_adjustflag_" + adj
    else:
        fill = pd.Series(["baostock_5m_adjustflag_3"] * len(df), index=df.index)

    src = src.where(~bad, fill)
    return src


def fill_const_or_existing(df: pd.DataFrame, col: str, value: str) -> pd.Series:
    if col in df.columns:
        s = df[col].astype(str)
        bad = s.isna() | s.str.strip().eq("") | s.str.lower().eq("nan")
        return s.where(~bad, value)
    return pd.Series([value] * len(df), index=df.index)


def clean_file_to_old_schema(path: Path) -> Tuple[int, int, str]:
    fallback_symbol = symbol_from_path(path)
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    rows_before = len(df)

    dt = parse_datetime(df)

    raw_symbol = first_existing_series(df, ["symbol", "code"], fallback_symbol)
    symbol = raw_symbol.apply(lambda x: normalize_symbol_value(x, fallback_symbol))

    out = pd.DataFrame(index=df.index)
    out["symbol"] = symbol
    out["_datetime_sort"] = dt
    out["datetime"] = dt
    out["trade_date"] = dt.dt.strftime("%Y%m%d")

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in df.columns:
            out[c] = pd.NA
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    out["source"] = source_series(df)
    out["bar_freq"] = fill_const_or_existing(df, "bar_freq", "5min")
    out["bar_label"] = fill_const_or_existing(df, "bar_label", "right")

    required = ["_datetime_sort", "trade_date", "open", "high", "low", "close", "volume", "amount"]
    out = out.dropna(subset=required).copy()
    if out.empty:
        return rows_before, 0, "empty_after_parse"

    ohlc = out[["open", "high", "low", "close"]]
    out = out[(ohlc > 0).all(axis=1)].copy()
    out = out[(out["volume"] >= 0) & (out["amount"] >= 0)].copy()
    if out.empty:
        return rows_before, 0, "empty_after_value_filter"

    out["datetime"] = out["_datetime_sort"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.sort_values(["symbol", "_datetime_sort"])
    out = out.drop_duplicates(["symbol", "datetime"], keep="last")
    out = out[OLD_COLS].copy()

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return rows_before, len(out), "ok"


def audit_cache(cache_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in sorted(cache_dir.glob("*_5m_raw.csv")):
        try:
            cols = read_header(p)
            status = classify_columns(cols)
            rows.append({
                "file": str(p),
                "name": p.name,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 3),
                "n_cols": len(cols),
                "status": status,
                "columns": ",".join(cols),
            })
        except Exception as e:
            rows.append({
                "file": str(p),
                "name": p.name,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 3) if p.exists() else -1,
                "n_cols": -1,
                "status": f"read_error:{type(e).__name__}:{e}",
                "columns": "",
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--apply", action="store_true", help="Actually back up and clean target files. Without this, audit only.")
    ap.add_argument("--backup-dir", default=None, help="Backup directory for target files only.")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of target files to clean.")
    ap.add_argument(
        "--target-status",
        action="append",
        default=None,
        help="Status to clean. Can be repeated. Default: mixed_polluted/raw_baostock_only/old_schema_extra_cols.",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"[ERROR] missing cache dir: {cache_dir}", file=sys.stderr)
        return 2

    target_statuses = set(args.target_status or ["mixed_polluted", "raw_baostock_only", "old_schema_extra_cols"])

    audit = audit_cache(cache_dir)
    report_path = cache_dir.parent / "baostock_5m_cache_schema_audit.csv"
    audit.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("[AUDIT] status counts:")
    if audit.empty:
        print("  <no files>")
    else:
        print(audit["status"].value_counts(dropna=False).to_string())

    targets = audit[audit["status"].isin(target_statuses)].copy()
    if args.max_files is not None:
        targets = targets.head(args.max_files).copy()

    target_report_path = cache_dir.parent / "baostock_5m_cache_oldschema_clean_targets.csv"
    targets.to_csv(target_report_path, index=False, encoding="utf-8-sig")

    print(f"\n[AUDIT] report: {report_path}")
    print(f"[TARGET] report: {target_report_path}")
    print(f"[TARGET] files: {len(targets)}")
    print(f"[TARGET] total size MB: {round(float(targets['size_mb'].sum()) if not targets.empty else 0.0, 3)}")
    if not targets.empty:
        print("[TARGET] sample:")
        print(targets[["name", "size_mb", "status"]].head(30).to_string(index=False))

    if not args.apply:
        print("\n[DRY-RUN] no files changed. Re-run with --apply to back up and clean target files only.")
        return 0

    if targets.empty:
        print("\n[OK] no target files to clean.")
        return 0

    backup_dir = Path(args.backup_dir) if args.backup_dir else cache_dir.parent / f"baostock_5m_cache_backup_targets_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[BACKUP] target files only -> {backup_dir}")

    clean_rows: List[Dict[str, object]] = []
    for i, row in enumerate(targets.itertuples(index=False), 1):
        p = Path(row.file)
        b = backup_dir / p.name
        try:
            shutil.copy2(p, b)
            before_rows, after_rows, clean_status = clean_file_to_old_schema(p)

            new_cols = read_header(p)
            if new_cols != OLD_COLS:
                clean_status = f"bad_header_after_clean:{new_cols}"

            clean_rows.append({
                "file": str(p),
                "backup": str(b),
                "rows_before": before_rows,
                "rows_after": after_rows,
                "status_before": row.status,
                "clean_status": clean_status,
            })
        except Exception as e:
            clean_rows.append({
                "file": str(p),
                "backup": str(b),
                "rows_before": -1,
                "rows_after": -1,
                "status_before": row.status,
                "clean_status": f"{type(e).__name__}: {e}",
            })

        if i % 20 == 0 or i == len(targets):
            print(f"[CLEAN] processed {i}/{len(targets)}")

    clean_report = pd.DataFrame(clean_rows)
    clean_report_path = cache_dir.parent / "baostock_5m_cache_oldschema_clean_report.csv"
    clean_report.to_csv(clean_report_path, index=False, encoding="utf-8-sig")

    print(f"\n[CLEAN] report: {clean_report_path}")
    print("[CLEAN] status counts:")
    print(clean_report["clean_status"].value_counts(dropna=False).to_string())

    bad = clean_report[clean_report["clean_status"] != "ok"]
    if not bad.empty:
        print("\n[ERROR] some files failed cleaning. First failures:")
        print(bad.head(20).to_string(index=False))
        print(f"[INFO] backups are in: {backup_dir}")
        return 1

    post = audit_cache(cache_dir)
    post_report_path = cache_dir.parent / "baostock_5m_cache_schema_audit_after_clean.csv"
    post.to_csv(post_report_path, index=False, encoding="utf-8-sig")
    print(f"\n[POST-AUDIT] report: {post_report_path}")
    print(post["status"].value_counts(dropna=False).to_string())

    remaining = post[post["status"].isin(target_statuses)]
    if not remaining.empty:
        print("\n[ERROR] target-status files remain after cleaning:")
        print(remaining[["name", "size_mb", "status", "columns"]].head(20).to_string(index=False))
        print(f"[INFO] backups are in: {backup_dir}")
        return 1

    print(f"\n[OK] cleaned {len(targets)} files. Backups of cleaned files only: {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
