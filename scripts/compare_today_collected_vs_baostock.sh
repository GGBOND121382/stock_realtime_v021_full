#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE="${DATE:-$(date +%Y%m%d)}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55}"
CACHE_DIR="${CACHE_DIR:-saved_data/akshare_realtime_cache}"
OUT_DIR="${OUT_DIR:-saved_data/baostock_compare/${DATE}}"
SYMBOLS="${SYMBOLS:-}"
RUN_PREMARKET_UPDATE="${RUN_PREMARKET_UPDATE:-1}"
SKIP_BAOSTOCK_QUERY="${SKIP_BAOSTOCK_QUERY:-0}"
MODELS_DIR="${MODELS_DIR:-saved_models}"
SAVED_DATA_DIR="${SAVED_DATA_DIR:-saved_data}"
WATCHLIST="${WATCHLIST:-selected_watchlist.txt}"
CONTEXT_DIR="${CONTEXT_DIR:-saved_data/realtime_context}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
SIGNAL_DIR="${SIGNAL_DIR:-saved_data/intraday_nextday_signals}"
PORTFOLIO_DIR="${PORTFOLIO_DIR:-portfolio_reports}"
BENCHMARK_SYMBOLS="${BENCHMARK_SYMBOLS:-000300.SH,000001.SH,399001.SZ,399006.SZ}"

echo "============================================================"
echo "[COMPARE] local collected data vs BaoStock 5m"
echo "DATE        = ${DATE}"
echo "CUTOFF_TIME = ${CUTOFF_TIME}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "OUT_DIR     = ${OUT_DIR}"
echo "SYMBOLS     = ${SYMBOLS:-<auto>}"
echo "RUN_PREMARKET_UPDATE = ${RUN_PREMARKET_UPDATE}"
echo "SKIP_BAOSTOCK_QUERY  = ${SKIP_BAOSTOCK_QUERY}"
echo "============================================================"

if [[ "${SKIP_BAOSTOCK_QUERY}" != "1" ]]; then
"${PYTHON}" - <<'PY'
import importlib.util
if importlib.util.find_spec("baostock") is None:
    raise SystemExit("[ERROR] baostock not installed. Run: python3 -m pip install baostock")
print("[OK] baostock installed")
PY
fi

CMD=(
  "${PYTHON}" tools/compare_collected_vs_baostock.py
  --date "${DATE}"
  --cache-dir "${CACHE_DIR}"
  --out-dir "${OUT_DIR}"
  --cutoff-time "${CUTOFF_TIME}"
  --python "${PYTHON}"
  --models-dir "${MODELS_DIR}"
  --saved-data-dir "${SAVED_DATA_DIR}"
  --watchlist "${WATCHLIST}"
  --model-policy all
  --context-dir "${CONTEXT_DIR}"
  --context-config "${CONTEXT_CONFIG}"
  --signal-dir "${SIGNAL_DIR}"
  --portfolio-dir "${PORTFOLIO_DIR}"
  --benchmark-symbols "${BENCHMARK_SYMBOLS}"
)

if [[ -n "${SYMBOLS}" ]]; then
  CMD+=(--symbols "${SYMBOLS}")
fi
if [[ "${RUN_PREMARKET_UPDATE}" == "1" ]]; then
  CMD+=(--run-premarket-update)
fi
if [[ "${SKIP_BAOSTOCK_QUERY}" == "1" ]]; then
  CMD+=(--skip-baostock-query)
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
echo "  ${OUT_DIR}/pipeline_file_inventory.csv"
echo "  ${OUT_DIR}/pipeline_vs_collected_daily.csv"
echo "  ${OUT_DIR}/pipeline_vs_collected_5m.csv"
echo "  ${OUT_DIR}/prediction_feature_diff.csv"
echo "  ${OUT_DIR}/prediction_signal_diff.csv"
echo "  ${OUT_DIR}/portfolio_signal_diff.csv"
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
