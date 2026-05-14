#!/usr/bin/env bash
set -euo pipefail

# Run all-model trading-day signal pipeline, then portfolio confirmation.

PYTHON="${PYTHON:-python3}"
DATE_DASH="${DATE_DASH:-$(date +%F)}"
DATE_COMPACT="${DATE_COMPACT:-$(date +%Y%m%d)}"

WATCHLIST="${WATCHLIST:-selected_watchlist.txt}"
ACCOUNT="${ACCOUNT:-account.json}"
HISTORY="${HISTORY:-history_close.csv}"

"$PYTHON" pipelines/run_trading_day_signal_pipeline.py \
  --watchlist "$WATCHLIST" \
  --models-dir saved_models \
  --model-policy all \
  --context-config configs/realtime_context_sources.toml \
  --cutoff-time "${CUTOFF_TIME:-14:55}" \
  --stock-collect-until "${STOCK_COLLECT_UNTIL:-14:52}" \
  --context-collect-until "${CONTEXT_COLLECT_UNTIL:-14:52}" \
  --build-time "${BUILD_TIME:-14:52}" \
  --score-time "${SCORE_TIME:-14:54}" \
  --spot-source-priority "${SPOT_SOURCE_PRIORITY:-sina,ths,em,xq}" \
  --required-fields "${REQUIRED_FIELDS:-close,open,high,low,volume,amount}" \
  --xq-max-symbols-per-round "${XQ_MAX_SYMBOLS_PER_ROUND:-10}" \
  --xq-per-symbol-timeout-seconds "${XQ_TIMEOUT:-2}" \
  --stock-collect-wait-timeout-seconds "${STOCK_WAIT_TIMEOUT:-45}" \
  --context-collect-wait-timeout-seconds "${CONTEXT_WAIT_TIMEOUT:-45}" \
  --max-missing-features "${MAX_MISSING_FEATURES:-5}" \
  --min-amount-yuan "${MIN_AMOUNT_YUAN:-50000000}"

DATE_DASH="$DATE_DASH" DATE_COMPACT="$DATE_COMPACT" ACCOUNT="$ACCOUNT" HISTORY="$HISTORY" \
  bash scripts/run_portfolio_confirm_from_signals.sh
