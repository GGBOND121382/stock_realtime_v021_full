#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE="${DATE:-20260513}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55}"
CACHE_DIR="${CACHE_DIR:-saved_data/akshare_realtime_cache}"
SYMBOLS="${SYMBOLS:-}"
SYMBOLS_FILE="${SYMBOLS_FILE:-}"

echo "============================================================"
echo "[RUN] Rebuild today's local 5m OHLCV from snapshots"
echo "DATE        = ${DATE}"
echo "CUTOFF_TIME = ${CUTOFF_TIME}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "SYMBOLS     = ${SYMBOLS:-<auto>}"
echo "SYMBOLS_FILE= ${SYMBOLS_FILE:-<none>}"
echo "============================================================"

CMD=(
  "${PYTHON}" tools/fix_5m_ohlcv_from_snapshots.py
  --date "${DATE}"
  --cache-dir "${CACHE_DIR}"
  --cutoff-time "${CUTOFF_TIME}"
)

if [[ -n "${SYMBOLS}" ]]; then
  CMD+=(--symbols "${SYMBOLS}")
fi
if [[ -n "${SYMBOLS_FILE}" ]]; then
  CMD+=(--symbols-file "${SYMBOLS_FILE}")
fi

echo "[RUN]"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Report:"
echo "  ${CACHE_DIR}/pending/${DATE}/_5m_ohlc_snapshot_fix_report/fix_summary.csv"
echo
echo "Then compare again:"
echo "  DATE=${DATE} CUTOFF_TIME=${CUTOFF_TIME} bash scripts/compare_today_collected_vs_baostock.sh"
