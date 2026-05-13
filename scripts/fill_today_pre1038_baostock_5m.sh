#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE="${DATE:-$(date +%Y%m%d)}"
CACHE_DIR="${CACHE_DIR:-saved_data/akshare_realtime_cache}"
SYMBOLS_FILE="${SYMBOLS_FILE:-}"
BEFORE_TIME="${BEFORE_TIME:-10:38}"

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import baostock
PY
then
  echo "[ERROR] baostock is not installed in this Python env." >&2
  echo "Install it first: $PYTHON -m pip install baostock" >&2
  exit 2
fi

cmd=("$PYTHON" tools/fill_intraday_gap_from_baostock.py --date "$DATE" --cache-dir "$CACHE_DIR" --before-time "$BEFORE_TIME")
if [[ -n "$SYMBOLS_FILE" ]]; then
  cmd+=(--symbols-file "$SYMBOLS_FILE")
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"

echo
summary="$CACHE_DIR/pending/$DATE/baostock_gap_fill_summary.json"
if [[ -f "$summary" ]]; then
  echo "[SUMMARY] $summary"
  cat "$summary"
fi
