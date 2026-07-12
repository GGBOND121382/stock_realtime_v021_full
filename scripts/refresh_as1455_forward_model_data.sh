#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
FORWARD_MODEL_DIR="${FORWARD_MODEL_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"
SKIP_HISTORY_UPDATE="${SKIP_HISTORY_UPDATE:-0}"
REBUILD_AS1455_DAILY_CACHE="${REBUILD_AS1455_DAILY_CACHE:-0}"
QFQ5M_AUDIT_SAMPLES="${QFQ5M_AUDIT_SAMPLES:-0}"
PROFILE_MEMORY="${PROFILE_MEMORY:-1}"

RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-$SOURCE_DIR/baostock_5m_cache}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-$SOURCE_DIR/as1455_daily_cache}"
LIVE_OUT_ROOT="${LIVE_OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"

if [[ "$SKIP_HISTORY_UPDATE" != "1" ]]; then
  echo "===== 1/2 update AS1455 historical caches to latest completed trading day ====="
  env_args=(
    "PYTHON=$PYTHON_BIN"
    "TRADE_DATE=$TRADE_DATE"
    "HISTORY_END_DATE=$HISTORY_END_DATE"
    "TIMEZONE=$TIMEZONE"
    "UNIVERSE=$UNIVERSE"
    "OUT_ROOT=$LIVE_OUT_ROOT"
    "RAW_5M_CACHE_DIR=$RAW_5M_CACHE_DIR"
    "RAW_DAILY_CACHE_DIR=$RAW_DAILY_CACHE_DIR"
    "AS1455_DAILY_CACHE_DIR=$AS1455_DAILY_CACHE_DIR"
  )
  if [[ -n "$MAX_SYMBOLS" ]]; then
    env_args+=("MAX_SYMBOLS=$MAX_SYMBOLS")
  fi
  env "${env_args[@]}" bash scripts/run_as1455_live_data_feature_pipeline.sh history
fi

resolved_end=""
live_date=$(
  TZ="$TIMEZONE" "$PYTHON_BIN" - "$TRADE_DATE" <<'PY'
import sys
from datetime import datetime
s = sys.argv[1]
if s.lower() == "today":
    print(datetime.now().strftime("%Y%m%d"))
else:
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
)
history_report="$LIVE_OUT_ROOT/$live_date/00_history_update_report.json"
if [[ -s "$history_report" ]]; then
  resolved_end=$(
    "$PYTHON_BIN" - "$history_report" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
print(obj.get("history_end_date", ""))
PY
  )
fi

if [[ -z "$resolved_end" && "$HISTORY_END_DATE" != "auto" ]]; then
  resolved_end="$HISTORY_END_DATE"
fi

mkdir -p "$FORWARD_MODEL_DIR"

echo "===== 2/2 rebuild extended AS1455 model_data from refreshed caches ====="
args=(
  scripts/build_ashare_ch12_as1455_model_data.py
  --universe "$UNIVERSE"
  --out-dir "$FORWARD_MODEL_DIR"
  --bar-root "$RAW_5M_CACHE_DIR"
  --bar-glob "*_5m_raw.csv"
  --baostock-5m-cache-dir "$RAW_5M_CACHE_DIR"
  --as1455-daily-cache-dir "$AS1455_DAILY_CACHE_DIR"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --qfq5m-audit-samples "$QFQ5M_AUDIT_SAMPLES"
)
if [[ -n "$resolved_end" ]]; then
  args+=(--end-date "$resolved_end")
fi
if [[ -n "$MAX_SYMBOLS" ]]; then
  args+=(--max-symbols "$MAX_SYMBOLS" --allow-partial-coverage)
fi
if [[ "$REBUILD_AS1455_DAILY_CACHE" == "1" ]]; then
  args+=(--rebuild-as1455-daily-cache)
fi
if [[ "$PROFILE_MEMORY" == "1" ]]; then
  args+=(--profile-memory)
fi

"$PYTHON_BIN" "${args[@]}"

model_data="$FORWARD_MODEL_DIR/model_data_as1455.h5"
[[ -s "$model_data" ]] || { echo "[ERROR] model_data not generated: $model_data" >&2; exit 1; }

"$PYTHON_BIN" - "$model_data" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_hdf(path, "model_data")
dates = pd.DatetimeIndex(df.index.get_level_values("date"))
print(f"[MODEL DATA] path={path}")
print(f"[MODEL DATA] rows={len(df)} symbols={df.index.get_level_values('symbol').nunique()}")
print(f"[MODEL DATA] date_min={dates.min():%Y-%m-%d} date_max={dates.max():%Y-%m-%d}")
for col in ["r01_fwd", "r05_fwd", "r21_fwd"]:
    valid = df[col].notna()
    if valid.any():
        d = pd.DatetimeIndex(df.index.get_level_values("date")[valid])
        print(f"[MODEL DATA] {col}_valid_end={d.max():%Y-%m-%d} rows={int(valid.sum())}")
    else:
        print(f"[MODEL DATA] {col}_valid_end=<none> rows=0")
PY

echo "[DONE] refreshed forward model data: $model_data"
