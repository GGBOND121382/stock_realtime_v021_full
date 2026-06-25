#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PREDICTIONS="${PREDICTIONS:-saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_models_$(date +%Y%m%d)}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
OFFSET_MODE="${OFFSET_MODE:-zero}"

python3 code/backtest/run_as1455_close_auction_grid_v1.py \
  --force \
  --offset-mode "$OFFSET_MODE" \
  --out-root "$OUT_ROOT" \
  --predictions "$PREDICTIONS" \
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
  --profile close_auction_skip_limit \
  --capacity-mode "$CAPACITY_MODE" \
  --run-output-mode "$OUTPUT_MODE"

echo
echo "Output mode: $OUTPUT_MODE"
echo "Grid output: $OUT_ROOT"
ls -lh "$OUT_ROOT/02_summary" || true
