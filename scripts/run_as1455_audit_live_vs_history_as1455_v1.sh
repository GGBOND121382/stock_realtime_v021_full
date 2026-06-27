#!/usr/bin/env bash
# Audit live AS1455 14:55 features against history-reconstructed AS1455 for the same date.
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-20260625}"
LIVE_ROOT="${LIVE_ROOT:-saved_data/ashare_ml4t/live_as1455}"
AS1455_CACHE_DIR="${AS1455_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache}"
AUDIT_DIR="${AUDIT_DIR:-}"
MIN_CACHE_ROWS="${MIN_CACHE_ROWS:-900}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-900}"
SKIP_FINALIZE="${SKIP_FINALIZE:-0}"

args=(
  tools/audit_as1455_live_vs_history_as1455_v1.py
  --trade-date "${TRADE_DATE}"
  --live-root "${LIVE_ROOT}"
  --as1455-cache-dir "${AS1455_CACHE_DIR}"
  --min-cache-rows "${MIN_CACHE_ROWS}"
  --min-feature-rows "${MIN_FEATURE_ROWS}"
)
if [[ -n "${AUDIT_DIR}" ]]; then
  args+=(--audit-dir "${AUDIT_DIR}")
fi
if [[ "${SKIP_FINALIZE}" == "1" ]]; then
  args+=(--skip-finalize)
fi

echo "[CONFIG]"
echo "  TRADE_DATE=${TRADE_DATE}"
echo "  LIVE_ROOT=${LIVE_ROOT}"
echo "  AS1455_CACHE_DIR=${AS1455_CACHE_DIR}"
echo "  AUDIT_DIR=${AUDIT_DIR:-${LIVE_ROOT}/${TRADE_DATE//-/}_audit_history_as1455}"
echo "  MIN_CACHE_ROWS=${MIN_CACHE_ROWS}"
echo "  MIN_FEATURE_ROWS=${MIN_FEATURE_ROWS}"
echo "  SKIP_FINALIZE=${SKIP_FINALIZE}"

"${PYTHON}" "${args[@]}"
