#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
BAR_CACHE_DIR="${BAR_CACHE_DIR:-$OUT_DIR/baostock_5m_cache}"
DAILY_CACHE_DIR="${DAILY_CACHE_DIR:-$OUT_DIR/as1455_daily_cache}"
QFQ_CACHE_DIR="${QFQ_CACHE_DIR:-saved_data/ashare_ml4t/ch12_reproduce/baostock_qfq_daily_cache}"

if [[ ! -d "$BAR_CACHE_DIR" ]]; then
  echo "Missing completed 5min cache directory: $BAR_CACHE_DIR" >&2
  exit 1
fi

shopt -s nullglob
bar_files=("$BAR_CACHE_DIR"/*_5m_raw.csv)
shopt -u nullglob
if (( ${#bar_files[@]} == 0 )); then
  echo "No *_5m_raw.csv files found under: $BAR_CACHE_DIR" >&2
  exit 1
fi

rebuild_args=()
if [[ "${REBUILD_DAILY_CACHE:-0}" == "1" ]]; then
  rebuild_args+=(--rebuild-as1455-daily-cache)
else
  shopt -s nullglob
  daily_files=("$DAILY_CACHE_DIR"/*_as1455_daily.csv)
  shopt -u nullglob
  if (( ${#daily_files[@]} > 0 )); then
    for daily_file in "${daily_files[@]}"; do
      header="$(head -n 1 "$daily_file")"
      if [[ ",$header," != *",raw_daily_close,"* ]]; then
        echo "Existing as1455 daily cache uses the old schema; rebuilding it once."
        rebuild_args+=(--rebuild-as1455-daily-cache)
        break
      fi
    done
  fi
fi

echo "Using completed 5min cache: $BAR_CACHE_DIR (${#bar_files[@]} files)"
echo "Using qfq daily cache: $QFQ_CACHE_DIR"
echo "The builder is forced to --no-fetch-missing-baostock; 5min data will not be downloaded."

exec "$PYTHON" scripts/build_ashare_ch12_as1455_model_data.py \
  --out-dir "$OUT_DIR" \
  --bar-root "$BAR_CACHE_DIR" \
  --bar-glob "*_5m_raw.csv" \
  --baostock-5m-cache-dir "$BAR_CACHE_DIR" \
  --as1455-daily-cache-dir "$DAILY_CACHE_DIR" \
  --qfq-daily-cache-dir "$QFQ_CACHE_DIR" \
  --no-fetch-missing-baostock \
  --baostock-fetch-retries "${BAOSTOCK_FETCH_RETRIES:-2}" \
  --baostock-fetch-sleep "${BAOSTOCK_FETCH_SLEEP:-1}" \
  --baostock-query-timeout "${BAOSTOCK_QUERY_TIMEOUT:-60}" \
  --profile-memory \
  "${rebuild_args[@]}" \
  "$@"
