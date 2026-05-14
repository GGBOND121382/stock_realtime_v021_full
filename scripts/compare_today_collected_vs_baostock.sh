#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE="${DATE:-$(date +%Y%m%d)}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55}"
CACHE_DIR="${CACHE_DIR:-saved_data/akshare_realtime_cache}"
OUT_DIR="${OUT_DIR:-saved_data/baostock_compare/${DATE}}"
SYMBOLS="${SYMBOLS:-}"

echo "============================================================"
echo "[COMPARE] local collected data vs BaoStock 5m"
echo "DATE        = ${DATE}"
echo "CUTOFF_TIME = ${CUTOFF_TIME}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "OUT_DIR     = ${OUT_DIR}"
echo "SYMBOLS     = ${SYMBOLS:-<auto>}"
echo "============================================================"

"${PYTHON}" - <<'PY'
import importlib.util
if importlib.util.find_spec("baostock") is None:
    raise SystemExit("[ERROR] baostock not installed. Run: python3 -m pip install baostock")
print("[OK] baostock installed")
PY

CMD=(
  "${PYTHON}" tools/compare_collected_vs_baostock.py
  --date "${DATE}"
  --cache-dir "${CACHE_DIR}"
  --out-dir "${OUT_DIR}"
  --cutoff-time "${CUTOFF_TIME}"
)

if [[ -n "${SYMBOLS}" ]]; then
  CMD+=(--symbols "${SYMBOLS}")
fi

echo "[RUN]"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "============================================================"
echo "[DONE]"
echo "Summary:"
echo "  ${OUT_DIR}/comparison_summary.csv"
echo "  ${OUT_DIR}/comparison_summary.json"
echo
echo "Quick view:"
echo "  python3 - <<'PY'"
echo "  import pandas as pd"
echo "  p='${OUT_DIR}/comparison_summary.csv'"
echo "  df=pd.read_csv(p)"
echo "  cols=[c for c in ['symbol','severity','collected_bars','baostock_bars','aligned_bars','only_in_baostock','close_max_abs_diff_bps','daily_vs_baostock_amount_rel_diff','daily_vs_baostock_vwap_diff_bps'] if c in df.columns]"
echo "  print(df[cols].to_string(index=False))"
echo "  PY"
echo "============================================================"
