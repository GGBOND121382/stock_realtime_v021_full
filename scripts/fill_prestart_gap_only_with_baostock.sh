#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE="${DATE:-20260513}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55}"
CACHE_DIR="${CACHE_DIR:-saved_data/akshare_realtime_cache}"
SYMBOLS="${SYMBOLS:-}"
WRITE_IN_PLACE="${WRITE_IN_PLACE:-1}"

echo "============================================================"
echo "[FIX] fill ONLY pre-start 5m gap with BaoStock"
echo "DATE           = ${DATE}"
echo "CUTOFF_TIME    = ${CUTOFF_TIME}"
echo "CACHE_DIR      = ${CACHE_DIR}"
echo "SYMBOLS        = ${SYMBOLS:-<auto>}"
echo "WRITE_IN_PLACE = ${WRITE_IN_PLACE}"
echo "============================================================"
echo "This script preserves local collected bars from the first local timestamp onward."
echo "It does NOT overwrite post-start local bars with BaoStock."
echo

"${PYTHON}" - <<'PY'
import importlib.util
if importlib.util.find_spec("baostock") is None:
    raise SystemExit("[ERROR] baostock not installed. Run: python3 -m pip install baostock")
print("[OK] baostock installed")
PY

CMD=(
  "${PYTHON}" tools/fill_prestart_gap_only_with_baostock.py
  --date "${DATE}"
  --cache-dir "${CACHE_DIR}"
  --cutoff-time "${CUTOFF_TIME}"
)

if [[ -n "${SYMBOLS}" ]]; then
  CMD+=(--symbols "${SYMBOLS}")
fi

if [[ "${WRITE_IN_PLACE}" == "1" ]]; then
  CMD+=(--write-in-place)
fi

echo "[RUN]"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "============================================================"
echo "[DONE]"
echo "Report:"
echo "  ${CACHE_DIR}/pending/${DATE}/_baostock_prestart_gap_report/prestart_gap_only_summary.csv"
echo
echo "Important columns:"
echo "  gap_baostock_rows_added: BaoStock bars added before first local timestamp"
echo "  post_start_*_diff: diagnostics for local collection after start"
echo
echo "Next:"
echo "  Re-run comparison. You SHOULD still see post-start volume/amount differences if collection code has cumulative-bar bug."
echo "============================================================"
