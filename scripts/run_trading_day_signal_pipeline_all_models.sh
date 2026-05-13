#!/usr/bin/env bash
set -uo pipefail

# Run trading-day signal pipeline in multi-model mode.
# This is the Scheme-B entrypoint: all saved_models artifacts are scored,
# then portfolio_confirm_from_buy_signals.py can select final stocks/orders.

PYTHON="${PYTHON:-python3}"
DATE_ARG="${DATE_ARG:-}"

DATE_FLAGS=()
if [[ -n "$DATE_ARG" ]]; then
  DATE_FLAGS=(--date "$DATE_ARG")
fi

"$PYTHON" pipelines/run_trading_day_signal_pipeline.py \
  --watchlist "${WATCHLIST:-selected_watchlist.txt}" \
  --models-dir "${MODELS_DIR:-saved_models}" \
  --model-policy all \
  --context-config "${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}" \
  --cutoff-time "${CUTOFF_TIME:-14:55}" \
  --stock-collect-until "${STOCK_COLLECT_UNTIL:-14:52}" \
  --context-collect-until "${CONTEXT_COLLECT_UNTIL:-14:52}" \
  --build-time "${BUILD_TIME:-14:52}" \
  --score-time "${SCORE_TIME:-14:54}" \
  --spot-source-priority "${SPOT_SOURCE_PRIORITY:-sina,ths,em,xq}" \
  --required-fields "${REQUIRED_FIELDS:-close,open,high,low,volume,amount}" \
  --xq-max-symbols-per-round "${XQ_MAX_SYMBOLS_PER_ROUND:-10}" \
  --xq-per-symbol-timeout-seconds "${XQ_PER_SYMBOL_TIMEOUT_SECONDS:-2}" \
  --stock-collect-wait-timeout-seconds "${STOCK_COLLECT_WAIT_TIMEOUT_SECONDS:-45}" \
  --context-collect-wait-timeout-seconds "${CONTEXT_COLLECT_WAIT_TIMEOUT_SECONDS:-45}" \
  --max-missing-features "${MAX_MISSING_FEATURES:-5}" \
  --min-amount-yuan "${MIN_AMOUNT_YUAN:-50000000}" \
  "${DATE_FLAGS[@]}"
