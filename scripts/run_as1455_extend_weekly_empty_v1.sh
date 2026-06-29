#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Rebuild AS1455 model_data_as1455.h5 from as1455_daily_cache, then run
# empty-account weekly retrain/backtest for the extension period.
#
# Default target: START_DATE=2026-05-16, END_DATE=2026-06-26.
# Run from repo root: ~/stock_realtime_v021_full

set -Eeuo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_as1455_extend_weekly_empty_v1.sh

Environment overrides:
  PYTHON=python3
  START_DATE=2026-05-16
  END_DATE=2026-06-26
  OUT_ROOT=saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_${START_DATE}_to_${END_DATE}
  MODEL_DATA=saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
  AS1455_CACHE=saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache
  UNIVERSE=saved_data/ashare_ml4t/ch12_as1455/as1455_model_universe_from_h5.csv
  WEEKLY_SCRIPT=scripts/run_as1455_top5_weekly_retrain_full_v7.sh
  SKIP_AS1455_PRECHECK=0
  FORCE=0

This script:
  1) verifies AS1455 daily cache has exactly one END_DATE row per symbol;
  2) backs up model_data_as1455.h5;
  3) rebuilds model_data_as1455.h5 from AS1455 daily cache;
  4) validates HDF schema required by weekly retrain;
  5) runs empty-account weekly retrain/backtest;
  6) prints leaderboard summary.
EOF
  exit 0
fi

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2026-05-16}"
END_DATE="${END_DATE:-2026-06-26}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
AS1455_CACHE="${AS1455_CACHE:-saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_ml4t/ch12_as1455/as1455_model_universe_from_h5.csv}"
WEEKLY_SCRIPT="${WEEKLY_SCRIPT:-scripts/run_as1455_top5_weekly_retrain_full_v7.sh}"
SKIP_AS1455_PRECHECK="${SKIP_AS1455_PRECHECK:-0}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_${START_DATE}_to_${END_DATE}}"
REPORT="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455_rebuild_report_${END_DATE}.json"
LOG_DIR="${OUT_ROOT}/logs"
RUN_LOG="${LOG_DIR}/run_as1455_extend_weekly_empty_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

log() { echo "[INFO] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || fail "missing dir: $1"; }

exec > >(tee -a "${RUN_LOG}") 2>&1

log "repo=$(pwd)"
log "START_DATE=${START_DATE}"
log "END_DATE=${END_DATE}"
log "MODEL_DATA=${MODEL_DATA}"
log "AS1455_CACHE=${AS1455_CACHE}"
log "UNIVERSE=${UNIVERSE}"
log "OUT_ROOT=${OUT_ROOT}"
log "RUN_LOG=${RUN_LOG}"

require_file "features/as1455_live_common.py"
require_file "${WEEKLY_SCRIPT}"
require_file "${UNIVERSE}"
require_dir "${AS1455_CACHE}"
require_file "${MODEL_DATA}"

log "checking Python imports needed for rebuild"
"${PYTHON}" - <<'PY'
import importlib
for name in ["pandas", "numpy", "tables"]:
    importlib.import_module(name)
from features.as1455_live_common import compute_ch12_features, load_universe, normalize_symbol
print("[OK] rebuild imports available")
PY

if [[ "${SKIP_AS1455_PRECHECK}" != "1" ]]; then
  log "prechecking AS1455 daily cache coverage for END_DATE=${END_DATE}"
  "${PYTHON}" - "${AS1455_CACHE}" "${END_DATE}" <<'PY'
import sys
import pandas as pd
from pathlib import Path

root = Path(sys.argv[1])
end_date = sys.argv[2]
rows_end = 0
max_dates = []
bad = []
files = sorted(root.glob("*_as1455_daily.csv"))
for p in files:
    code = p.name.split("_")[0]
    try:
        df = pd.read_csv(p, usecols=["date"], encoding="utf-8-sig")
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        if d.empty:
            bad.append((code, "empty_date"))
            continue
        max_dates.append(d.max().normalize())
        n = int(d.dt.strftime("%Y-%m-%d").eq(end_date).sum())
        rows_end += n
        if n != 1:
            bad.append((code, f"rows_{end_date}={n}"))
    except Exception as exc:
        bad.append((code, f"{type(exc).__name__}: {exc}"))

s = pd.Series(max_dates)
print("files:", len(max_dates))
print("global max:", s.max().date() if len(s) else None)
print(f"rows_{end_date}:", rows_end)
print("bad count:", len(bad))
if bad:
    print("bad head:", bad[:50])
if len(max_dates) != 1000 or str(s.max().date()) != end_date or rows_end != 1000 or bad:
    raise SystemExit(f"AS1455 cache is not clean to {end_date}; stop before rebuilding model_data")
print("[OK] AS1455 cache precheck passed")
PY
else
  log "SKIP_AS1455_PRECHECK=1; AS1455 cache precheck skipped"
fi

backup="${MODEL_DATA}.bak_before_extend_${END_DATE}_$(date +%Y%m%d_%H%M%S)"
log "backing up MODEL_DATA to ${backup}"
cp -av "${MODEL_DATA}" "${backup}"

log "rebuilding ${MODEL_DATA} from ${AS1455_CACHE}"
"${PYTHON}" - "${AS1455_CACHE}" "${MODEL_DATA}" "${REPORT}" "${UNIVERSE}" "${END_DATE}" <<'PY'
import json
import sys
import pandas as pd
from pathlib import Path

from features.as1455_live_common import (
    compute_ch12_features,
    load_universe,
    normalize_symbol,
)

AS1455_CACHE = Path(sys.argv[1])
OUT_H5 = Path(sys.argv[2])
REPORT = Path(sys.argv[3])
UNIVERSE_PATH = sys.argv[4]
END_DATE = sys.argv[5]

EXPECTED_COLUMNS = [
    "dollar_vol",
    "dollar_vol_rank",
    "rsi",
    "bb_high",
    "bb_low",
    "NATR",
    "ATR",
    "PPO",
    "MACD",
    "sector",
    "r01",
    "r05",
    "r10",
    "r21",
    "r42",
    "r63",
    "r01dec",
    "r05dec",
    "r10dec",
    "r21dec",
    "r42dec",
    "r63dec",
    "r01q_sector",
    "r05q_sector",
    "r10q_sector",
    "r21q_sector",
    "r42q_sector",
    "r63q_sector",
    "r01_fwd",
    "r05_fwd",
    "r21_fwd",
    "year",
    "month",
    "weekday",
]

universe = load_universe(UNIVERSE_PATH, None)
universe["symbol"] = universe["symbol"].map(normalize_symbol)
symbols = sorted(universe["symbol"].dropna().unique().tolist())
if len(symbols) != 1000:
    raise RuntimeError(f"expected 1000 universe symbols, got {len(symbols)}")

frames = []
missing = []
for i, sym in enumerate(symbols, 1):
    code = sym.split(".")[0]
    p = AS1455_CACHE / f"{code}_as1455_daily.csv"
    if not p.exists() or p.stat().st_size == 0:
        missing.append(sym)
        continue
    df = pd.read_csv(p, dtype={"symbol": str}, encoding="utf-8-sig")
    if df.empty:
        missing.append(sym)
        continue
    need = [
        "date",
        "symbol",
        "raw_open_as1455",
        "raw_high_as1455",
        "raw_low_as1455",
        "raw_close_as1455",
        "raw_volume_as1455",
    ]
    miss_cols = [c for c in need if c not in df.columns]
    if miss_cols:
        raise RuntimeError(f"{p} missing columns: {miss_cols}")
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"].eq(sym)].copy()
    if df.empty:
        missing.append(sym)
        continue
    part = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
        "symbol": df["symbol"],
        "open": pd.to_numeric(df["raw_open_as1455"], errors="coerce"),
        "high": pd.to_numeric(df["raw_high_as1455"], errors="coerce"),
        "low": pd.to_numeric(df["raw_low_as1455"], errors="coerce"),
        "close": pd.to_numeric(df["raw_close_as1455"], errors="coerce"),
        "volume": pd.to_numeric(df["raw_volume_as1455"], errors="coerce"),
    })
    part = part.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])
    part = part[(part[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frames.append(part)
    if i % 100 == 0:
        print(f"[INFO] loaded {i}/{len(symbols)} symbols", flush=True)

if missing:
    raise RuntimeError(f"missing AS1455 cache symbols: {missing[:20]} ... total={len(missing)}")

panel = pd.concat(frames, ignore_index=True, sort=False)
panel = panel.drop_duplicates(["date", "symbol"], keep="last")
panel = panel.sort_values(["date", "symbol"])

prices = panel.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]].sort_index()
features, outliers = compute_ch12_features(prices, universe, include_forward_labels=True)
features = features.reset_index()

miss = [c for c in EXPECTED_COLUMNS if c not in features.columns]
if miss:
    raise RuntimeError(f"computed features missing columns: {miss}")

model_data = features[["symbol", "date"] + EXPECTED_COLUMNS].copy()
model_data["symbol"] = model_data["symbol"].map(normalize_symbol)
model_data["date"] = pd.to_datetime(model_data["date"], errors="coerce").dt.normalize()
model_data = model_data.dropna(subset=["symbol", "date"])
model_data = model_data.drop_duplicates(["symbol", "date"], keep="last")
model_data = model_data.sort_values(["symbol", "date"])
model_data = model_data.set_index(["symbol", "date"])

if list(model_data.index.names) != ["symbol", "date"]:
    raise RuntimeError(f"bad index names: {model_data.index.names}")
outcomes = model_data.filter(like="fwd").columns.tolist()
if outcomes != ["r01_fwd", "r05_fwd", "r21_fwd"]:
    raise RuntimeError(f"bad outcomes: {outcomes}")
X = model_data.drop(["r01_fwd", "r05_fwd", "r21_fwd"], axis=1)
if X.shape[1] != 31:
    raise RuntimeError(f"expected 31 features, got {X.shape[1]}")
if any("fwd" in str(c) for c in X.columns):
    raise RuntimeError("feature matrix still contains fwd columns")

dates = pd.to_datetime(model_data.index.get_level_values("date"))
if str(dates.max().date()) != END_DATE:
    raise RuntimeError(f"model_data max date is {dates.max().date()}, expected {END_DATE}")
rows_end = int((dates == pd.Timestamp(END_DATE)).sum())
if rows_end < 900:
    raise RuntimeError(f"too few model_data rows on {END_DATE}: {rows_end}")

OUT_H5.parent.mkdir(parents=True, exist_ok=True)
model_data.to_hdf(OUT_H5, "model_data", mode="w", format="table")

report = {
    "out_h5": str(OUT_H5),
    "rows": int(len(model_data)),
    "symbols": int(model_data.index.get_level_values("symbol").nunique()),
    "date_min": str(dates.min().date()),
    "date_max": str(dates.max().date()),
    "n_dates": int(dates.nunique()),
    "rows_on_end_date": rows_end,
    "n_columns": int(model_data.shape[1]),
    "x_columns": int(X.shape[1]),
    "outcomes": outcomes,
    "outlier_symbols": outliers["symbol"].tolist() if not outliers.empty else [],
    "n_outlier_symbols": int(len(outliers)),
    "x_nan_rows": int(X.isna().any(axis=1).sum()),
    "label_nan_rows": int(model_data[["r01_fwd", "r05_fwd", "r21_fwd"]].isna().any(axis=1).sum()),
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

log "validating rebuilt HDF schema"
"${PYTHON}" - "${MODEL_DATA}" "${END_DATE}" <<'PY'
import sys
import pandas as pd

p = sys.argv[1]
end_date = sys.argv[2]
df = pd.read_hdf(p, "model_data")
dates = pd.to_datetime(df.index.get_level_values("date"))
print("index names:", list(df.index.names))
print("date range:", dates.min().date(), dates.max().date(), dates.nunique())
print("rows:", len(df))
print("symbols:", df.index.get_level_values("symbol").nunique())
print("columns:", len(df.columns), list(df.columns))
print("X columns:", df.drop(columns=["r01_fwd", "r05_fwd", "r21_fwd"]).shape[1])
print(f"rows on {end_date}:", int((dates == pd.Timestamp(end_date)).sum()))
assert list(df.index.names) == ["symbol", "date"]
assert str(dates.max().date()) == end_date
assert df.drop(columns=["r01_fwd", "r05_fwd", "r21_fwd"]).shape[1] == 31
print("[OK] HDF validation passed")
PY

log "running weekly retrain/backtest from empty account"
OUT_ROOT="${OUT_ROOT}" START_DATE="${START_DATE}" END_DATE="${END_DATE}" FORCE="${FORCE}" \
  bash "${WEEKLY_SCRIPT}"

LEADERBOARD="${OUT_ROOT}/02_summary/leaderboard_by_sharpe.csv"
require_file "${LEADERBOARD}"

log "leaderboard: ${LEADERBOARD}"
cat "${LEADERBOARD}"

log "compact summary"
"${PYTHON}" - "${LEADERBOARD}" <<'PY'
import sys
import pandas as pd
p = sys.argv[1]
df = pd.read_csv(p)
cols = [
    "run_name",
    "final_nav",
    "total_return",
    "annual_return",
    "sharpe",
    "max_drawdown",
    "monthly_win_rate",
    "trade_win_rate",
    "gross_trade_amount",
    "total_fee",
]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
PY

log "DONE"
log "OUT_ROOT=${OUT_ROOT}"
log "MODEL_DATA_REPORT=${REPORT}"
log "RUN_LOG=${RUN_LOG}"
