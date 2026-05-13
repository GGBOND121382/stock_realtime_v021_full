#!/usr/bin/env bash
set -euo pipefail
DATE="${DATE:-$(date +%Y%m%d)}"
PYTHON="${PYTHON:-python3}"
WATCHLIST="${WATCHLIST:-selected_watchlist.txt}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
MODEL_POLICY="${MODEL_POLICY:-all}"
MAX_MISSING_FEATURES="${MAX_MISSING_FEATURES:-5}"
MIN_AMOUNT_YUAN="${MIN_AMOUNT_YUAN:-50000000}"
OUT_DIR="${OUT_DIR:-saved_data/intraday_nextday_signals/${DATE}_replay_fixed}"

echo "[REPLAY] DATE=$DATE OUT_DIR=$OUT_DIR"
mkdir -p "$OUT_DIR"

# Re-score using existing pending cache and context outputs.  This assumes your
# main pipeline has already collected/built today's cache.  We run score-now mode
# through the same public pipeline entry when available.
$PYTHON pipelines/run_intraday_nextday_signals.py \
  --watchlist "$WATCHLIST" \
  --context-config "$CONTEXT_CONFIG" \
  --signal-out-dir saved_data/intraday_nextday_signals \
  --model-policy "$MODEL_POLICY" \
  --trade-date "$DATE" \
  --cutoff-time 14:55 \
  --score-time 00:00 \
  --max-missing-features "$MAX_MISSING_FEATURES" \
  --min-amount-yuan "$MIN_AMOUNT_YUAN" \
  --score-now-only || {
    echo "[WARN] score-now-only mode is not supported by your current script."
    echo "Run the normal trading-day pipeline after applying the patch, or use your existing replay wrapper."
    exit 1
  }
