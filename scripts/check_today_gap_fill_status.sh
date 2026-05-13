#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
DATE="${DATE:-$(date +%Y%m%d)}"
CACHE_DIR="${CACHE_DIR:-saved_data/akshare_realtime_cache}"
OUT_DIR="${OUT_DIR:-saved_data/gap_fill_diagnostics/${DATE}}"
mkdir -p "$OUT_DIR"

"$PYTHON" - <<'PY'
import json
import math
import os
from pathlib import Path
import pandas as pd
import numpy as np

DATE = os.environ.get("DATE") or pd.Timestamp.today().strftime("%Y%m%d")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "saved_data/akshare_realtime_cache"))
OUT_DIR = Path(os.environ.get("OUT_DIR", f"saved_data/gap_fill_diagnostics/{DATE}"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

pending = CACHE_DIR / "pending" / DATE
feature_cache = CACHE_DIR / "feature_cache"

need_times_5m = ["09:35", "09:40", "09:45", "09:50", "09:55", "10:00", "10:05", "10:10", "10:15", "10:20", "10:25", "10:30"]
# Features that were the main missing cause in today's scoring.
need_feature_cols = [
    "first_30m_ret", "first_60m_ret", "first_60m_volume_share",
    "morning_ret", "morning_vwap", "morning_vwap_to_close",
    "afternoon_ret", "afternoon_vwap", "afternoon_vwap_to_close",
    "last_30m_ret", "last_60m_ret", "last_30m_vwap", "last_30m_vwap_to_close",
    "last_30m_volume_share", "morning_afternoon_reversal", "first60_last30_reversal",
]

rows = []
if not pending.exists():
    raise SystemExit(f"[ERROR] pending dir not found: {pending}")

for sym_dir in sorted([p for p in pending.iterdir() if p.is_dir()]):
    sym = sym_dir.name
    row = {"stock_code": sym}
    f5 = sym_dir / "minute_bars_5min.csv"
    f1 = sym_dir / "minute_bars_1min.csv"
    daily = sym_dir / "daily_features.csv"
    row["has_5min"] = f5.exists()
    row["has_1min"] = f1.exists()
    row["has_daily"] = daily.exists()

    times = []
    first_time = last_time = None
    n_5m = 0
    missing_times = need_times_5m.copy()
    if f5.exists():
        try:
            df5 = pd.read_csv(f5)
            n_5m = len(df5)
            dt_col = "datetime" if "datetime" in df5.columns else None
            if dt_col:
                ts = pd.to_datetime(df5[dt_col], errors="coerce").dropna()
                if len(ts):
                    first_time = ts.min().strftime("%H:%M")
                    last_time = ts.max().strftime("%H:%M")
                    times = sorted(set(ts.dt.strftime("%H:%M")))
                    missing_times = [t for t in need_times_5m if t not in set(times)]
        except Exception as e:
            row["read_5min_error"] = f"{type(e).__name__}: {e}"

    row["n_5min"] = n_5m
    row["first_5min_time"] = first_time
    row["last_5min_time"] = last_time
    row["missing_early_5min_times"] = ",".join(missing_times)
    row["early_5min_ok"] = len(missing_times) == 0

    # Check intraday feature cache today row.
    fcandidates = [feature_cache / f"{sym}_intraday_reversal_features.csv", feature_cache / f"{sym.replace('.', '_')}_intraday_reversal_features.csv"]
    fc = next((p for p in fcandidates if p.exists()), None)
    row["feature_cache_file"] = str(fc) if fc else ""
    row["has_feature_cache"] = fc is not None
    row["today_feature_row"] = False
    row["missing_intraday_feature_cols"] = ""
    row["nan_intraday_feature_cols"] = ""

    if fc is not None:
        try:
            fdf = pd.read_csv(fc)
            if "date" in fdf.columns:
                dates = pd.to_datetime(fdf["date"], errors="coerce")
                mask = dates.dt.strftime("%Y%m%d") == DATE
                row["today_feature_row"] = bool(mask.any())
                if mask.any():
                    last = fdf.loc[mask].iloc[-1]
                    missing_cols = [c for c in need_feature_cols if c not in fdf.columns]
                    nan_cols = []
                    for c in need_feature_cols:
                        if c in fdf.columns:
                            v = pd.to_numeric(pd.Series([last[c]]), errors="coerce").iloc[0]
                            if not np.isfinite(v):
                                nan_cols.append(c)
                    row["missing_intraday_feature_cols"] = ",".join(missing_cols)
                    row["nan_intraday_feature_cols"] = ",".join(nan_cols)
        except Exception as e:
            row["read_feature_cache_error"] = f"{type(e).__name__}: {e}"

    # Check prev_close/pct_chg/amount from daily/snapshot.
    row["daily_has_amount"] = False
    row["daily_has_prev_close"] = False
    row["daily_has_pct_chg"] = False
    if daily.exists():
        try:
            ddf = pd.read_csv(daily)
            if not ddf.empty:
                dlast = ddf.iloc[-1]
                for col, key in [("amount","daily_has_amount"),("prev_close","daily_has_prev_close"),("pct_chg","daily_has_pct_chg")]:
                    if col in ddf.columns:
                        v = pd.to_numeric(pd.Series([dlast[col]]), errors="coerce").iloc[0]
                        row[key] = bool(np.isfinite(v))
        except Exception as e:
            row["read_daily_error"] = f"{type(e).__name__}: {e}"

    # Summarize whether this symbol is ready for today's previous missing set.
    blocking = []
    if not row["has_5min"]:
        blocking.append("missing_5min_file")
    elif not row["early_5min_ok"]:
        blocking.append("missing_early_5min_bars")
    if not row["today_feature_row"]:
        blocking.append("missing_today_feature_cache_row")
    if row.get("nan_intraday_feature_cols"):
        blocking.append("nan_intraday_features")
    if not row["daily_has_amount"]:
        blocking.append("daily_amount_missing")
    # prev_close/pct_chg may be computed from snapshot by scoring patch, so not always hard blocking.
    row["blocking_status"] = ";".join(blocking) if blocking else "ok"
    rows.append(row)

out = pd.DataFrame(rows)
out_csv = OUT_DIR / "gap_fill_status.csv"
out.to_csv(out_csv, index=False, encoding="utf-8-sig")

summary = {
    "date": DATE,
    "symbols": int(len(out)),
    "ok_symbols": int((out["blocking_status"] == "ok").sum()) if len(out) else 0,
    "problem_symbols": int((out["blocking_status"] != "ok").sum()) if len(out) else 0,
    "status_counts": out["blocking_status"].value_counts(dropna=False).to_dict() if len(out) else {},
    "output_csv": str(out_csv),
}
(OUT_DIR / "gap_fill_status_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
if len(out):
    cols = ["stock_code", "first_5min_time", "last_5min_time", "early_5min_ok", "today_feature_row", "nan_intraday_feature_cols", "daily_has_amount", "daily_has_prev_close", "daily_has_pct_chg", "blocking_status"]
    print("\n=== per-symbol status ===")
    print(out[cols].to_string(index=False))
PY

echo

echo "Wrote: $OUT_DIR/gap_fill_status.csv"
echo "Wrote: $OUT_DIR/gap_fill_status_summary.json"
