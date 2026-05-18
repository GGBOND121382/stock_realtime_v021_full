#!/usr/bin/env bash
set -euo pipefail

# Run all-model trading-day signal pipeline, then portfolio confirmation.

PYTHON="${PYTHON:-python3}"
DATE_DASH="${DATE_DASH:-}"
DATE_COMPACT="${DATE_COMPACT:-}"
if [[ -z "$DATE_DASH" && -z "$DATE_COMPACT" ]]; then
  DATE_DASH="$(date +%F)"
  DATE_COMPACT="$(date +%Y%m%d)"
elif [[ -z "$DATE_DASH" ]]; then
  DATE_DASH="${DATE_COMPACT:0:4}-${DATE_COMPACT:4:2}-${DATE_COMPACT:6:2}"
elif [[ -z "$DATE_COMPACT" ]]; then
  DATE_COMPACT="${DATE_DASH//-/}"
fi

WATCHLIST="${WATCHLIST:-selected_watchlist.txt}"
ACCOUNT="${ACCOUNT:-account.json}"
HISTORY="${HISTORY:-}"
SAVED_MODELS="${SAVED_MODELS:-saved_models}"
SAVED_DATA_DIR="${SAVED_DATA_DIR:-saved_data}"
CONFIG="${CONFIG:-configs/portfolio_confirm_config.json}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
OUT_DIR="${OUT_DIR:-portfolio_reports}"

"$PYTHON" pipelines/run_trading_day_signal_pipeline.py \
  --date "$DATE_COMPACT" \
  --watchlist "$WATCHLIST" \
  --models-dir "$SAVED_MODELS" \
  --model-policy all \
  --saved-data-dir "$SAVED_DATA_DIR" \
  --context-config "$CONTEXT_CONFIG" \
  --cutoff-time "${CUTOFF_TIME:-14:55}" \
  --stock-collect-until "${STOCK_COLLECT_UNTIL:-14:52}" \
  --context-collect-until "${CONTEXT_COLLECT_UNTIL:-14:52}" \
  --build-time "${BUILD_TIME:-14:52}" \
  --score-time "${SCORE_TIME:-14:54}" \
  --spot-source-priority "${SPOT_SOURCE_PRIORITY:-sina_batch,ths_etf,xq}" \
  --required-fields "${REQUIRED_FIELDS:-close,open,high,low,volume,amount}" \
  --xq-max-symbols-per-round "${XQ_MAX_SYMBOLS_PER_ROUND:-10}" \
  --xq-per-symbol-timeout-seconds "${XQ_TIMEOUT:-2}" \
  --stock-collect-wait-timeout-seconds "${STOCK_WAIT_TIMEOUT:-45}" \
  --context-collect-wait-timeout-seconds "${CONTEXT_WAIT_TIMEOUT:-45}" \
  --max-missing-features "${MAX_MISSING_FEATURES:-5}" \
  --min-amount-yuan "${MIN_AMOUNT_YUAN:-50000000}"

DATE_DASH="$DATE_DASH" DATE_COMPACT="$DATE_COMPACT" ACCOUNT="$ACCOUNT" HISTORY="$HISTORY" \
  SAVED_MODELS="$SAVED_MODELS" SAVED_DATA_DIR="$SAVED_DATA_DIR" CONFIG="$CONFIG" \
  CONTEXT_CONFIG="$CONTEXT_CONFIG" OUT_DIR="$OUT_DIR" \
  bash scripts/run_portfolio_confirm_from_signals.sh
