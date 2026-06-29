#!/usr/bin/env bash
set -euo pipefail

# Repair holes inside as1455_daily_cache from existing raw 5m cache.
# This script does NOT download data and does NOT modify raw_5m/raw_daily caches.
# Run from repository root:
#   DATES=2026-06-16,2026-06-17,2026-06-18,2026-06-22,2026-06-23 \
#   bash scripts/repair_as1455_missing_dates_from_raw5m_v1.sh

UNIVERSE="${UNIVERSE:-saved_data/ashare_ml4t/ch12_as1455/as1455_model_universe_from_h5.csv}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache}"
DATES="${DATES:-2026-06-16,2026-06-17,2026-06-18,2026-06-22,2026-06-23}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455/repair_missing_dates_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_SYMBOLS="${EXPECTED_SYMBOLS:-1000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
export UNIVERSE RAW_5M_CACHE_DIR AS1455_DAILY_CACHE_DIR DATES OUT_ROOT DRY_RUN EXPECTED_SYMBOLS PROGRESS_EVERY

mkdir -p "$OUT_ROOT"

python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

import pandas as pd

from features.as1455_live_common import (
    aggregate_as1455_from_5m,
    load_universe,
    merge_dedup_csv,
    normalize_symbol,
    read_5m_csv,
    symbol_code,
)

universe_path = Path(os.environ["UNIVERSE"])
raw_5m_dir = Path(os.environ["RAW_5M_CACHE_DIR"])
as1455_dir = Path(os.environ["AS1455_DAILY_CACHE_DIR"])
out_root = Path(os.environ["OUT_ROOT"])
dry_run = str(os.environ.get("DRY_RUN", "0")) == "1"
expected_symbols = int(os.environ.get("EXPECTED_SYMBOLS", "1000"))
progress_every = int(os.environ.get("PROGRESS_EVERY", "50"))

def parse_dates(s: str) -> list[str]:
    out = []
    for x in str(s).replace(";", ",").split(","):
        x = x.strip()
        if not x:
            continue
        ts = pd.to_datetime(x, errors="raise").normalize()
        out.append(ts.strftime("%Y-%m-%d"))
    return sorted(set(out))

target_dates = parse_dates(os.environ["DATES"])
if not target_dates:
    raise SystemExit("DATES is empty")

if not universe_path.exists():
    raise FileNotFoundError(universe_path)
if not raw_5m_dir.exists():
    raise FileNotFoundError(raw_5m_dir)
if not as1455_dir.exists():
    raise FileNotFoundError(as1455_dir)

universe = load_universe(universe_path, None)
universe["symbol"] = universe["symbol"].map(normalize_symbol)
symbols = sorted(universe["symbol"].dropna().unique().tolist())
if len(symbols) != expected_symbols:
    raise RuntimeError(f"unexpected universe size: {len(symbols)}; expected {expected_symbols}")

def as_path_for_symbol(sym: str) -> Path:
    return as1455_dir / f"{symbol_code(sym)}_as1455_daily.csv"

def raw5m_path_for_symbol(sym: str) -> Path:
    return raw_5m_dir / f"{symbol_code(sym)}_5m_raw.csv"

def existing_date_counts(path: Path) -> dict[str, int]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    except Exception:
        return {}
    if "date" not in header.columns:
        return {}
    df = pd.read_csv(path, usecols=["date"], encoding="utf-8-sig")
    d = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return d.value_counts(dropna=True).to_dict()

rows = []
repair_frames_by_path: dict[Path, list[pd.DataFrame]] = {}

for idx, sym in enumerate(symbols, 1):
    as_path = as_path_for_symbol(sym)
    raw_path = raw5m_path_for_symbol(sym)
    counts = existing_date_counts(as_path)
    missing_dates = [d for d in target_dates if counts.get(d, 0) == 0]
    duplicate_dates = [d for d in target_dates if counts.get(d, 0) > 1]
    cached_dates = [d for d in target_dates if counts.get(d, 0) == 1]

    status = "cached" if not missing_dates and not duplicate_dates else "needs_repair"
    error = ""
    written_rows = 0
    aggregate_rows = 0

    if missing_dates:
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            status = "raw_5m_missing"
            error = str(raw_path)
        else:
            try:
                bars = read_5m_csv(raw_path, symbol=sym)
                bars = bars[bars["date"].dt.strftime("%Y-%m-%d").isin(target_dates)].copy()
                if bars.empty:
                    status = "raw_5m_no_target_bars"
                else:
                    agg = aggregate_as1455_from_5m(
                        bars,
                        symbol=sym,
                        start_date=min(missing_dates),
                        end_date=max(missing_dates),
                    )
                    aggregate_rows = int(len(agg))
                    if not agg.empty:
                        agg_dates = pd.to_datetime(agg["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                        new = agg.loc[agg_dates.isin(missing_dates)].copy()
                    else:
                        new = agg
                    written_rows = int(len(new))
                    got_dates = set(pd.to_datetime(new["date"], errors="coerce").dt.strftime("%Y-%m-%d")) if written_rows else set()
                    still_missing = [d for d in missing_dates if d not in got_dates]
                    if still_missing:
                        status = "aggregate_missing_dates"
                        error = ",".join(still_missing)
                    elif written_rows:
                        status = "would_update" if dry_run else "updated_merge_dedup"
                        if not dry_run:
                            repair_frames_by_path.setdefault(as_path, []).append(new)
                    else:
                        status = "aggregate_empty"
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

    rows.append({
        "symbol": sym,
        "code": symbol_code(sym),
        "as1455_path": str(as_path),
        "raw_5m_path": str(raw_path),
        "cached_dates": ",".join(cached_dates),
        "missing_dates_before": ",".join(missing_dates),
        "duplicate_dates_before": ",".join(duplicate_dates),
        "status": status,
        "aggregate_rows": aggregate_rows,
        "written_rows": written_rows,
        "error": error,
    })

    if progress_every > 0 and idx % progress_every == 0:
        print(f"[INFO] scanned {idx}/{len(symbols)} symbols", flush=True)

# Write repairs after scan so failed symbols do not leave partially constructed per-symbol state.
if not dry_run:
    for p, frames in repair_frames_by_path.items():
        new_df = pd.concat(frames, ignore_index=True, sort=False)
        merged = merge_dedup_csv(p, new_df, subset=["symbol", "date"])
        merged.to_csv(p, index=False, encoding="utf-8-sig")

# Validate target date row counts after repair. Read each per-symbol AS1455 file once.
validation_map = {d: {"rows": 0, "bad_symbols": []} for d in target_dates}
for sym in symbols:
    p = as_path_for_symbol(sym)
    c = existing_date_counts(p)
    for d in target_dates:
        n = int(c.get(d, 0))
        validation_map[d]["rows"] += n
        if n != 1:
            validation_map[d]["bad_symbols"].append(f"{sym}:{n}")

validation_rows = []
for d in target_dates:
    bad_symbols = validation_map[d]["bad_symbols"]
    validation_rows.append({
        "date": d,
        "rows": int(validation_map[d]["rows"]),
        "expected_rows": len(symbols),
        "bad_symbol_count": len(bad_symbols),
        "bad_symbol_sample": ";".join(bad_symbols[:50]),
    })

by_symbol = pd.DataFrame(rows)
validation = pd.DataFrame(validation_rows)
by_symbol_path = out_root / "repair_missing_dates_by_symbol.csv"
validation_path = out_root / "repair_missing_dates_validation.csv"
report_path = out_root / "repair_missing_dates_report.json"
by_symbol.to_csv(by_symbol_path, index=False, encoding="utf-8-sig")
validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

status_counts = by_symbol["status"].value_counts(dropna=False).to_dict()
report = {
    "dates": target_dates,
    "dry_run": dry_run,
    "universe": str(universe_path),
    "raw_5m_cache_dir": str(raw_5m_dir),
    "as1455_daily_cache_dir": str(as1455_dir),
    "n_symbols": int(len(symbols)),
    "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    "symbols_with_missing_before": int((by_symbol["missing_dates_before"].astype(str) != "").sum()),
    "symbols_with_duplicates_before": int((by_symbol["duplicate_dates_before"].astype(str) != "").sum()),
    "written_rows_sum": int(by_symbol["written_rows"].sum()),
    "validation": validation.to_dict("records"),
    "by_symbol_report": str(by_symbol_path),
    "validation_report": str(validation_path),
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))

bad_after = validation[validation["bad_symbol_count"] > 0]
if len(bad_after):
    raise SystemExit(f"repair incomplete; see {validation_path}")
PY
