#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_ml4t/ch12_as1455/as1455_model_universe_from_h5.csv}"
BUILD_EXECUTION_SIDECAR="${BUILD_EXECUTION_SIDECAR:-1}"

if [[ "$TRADE_DATE" == "today" ]]; then
  LIVE_DATE="$(date +%Y%m%d)"
else
  LIVE_DATE="${TRADE_DATE//-/}"
fi
LIVE_DIR="${LIVE_DIR:-$OUT_ROOT/$LIVE_DATE}"

if [[ "$BUILD_EXECUTION_SIDECAR" == "1" ]]; then
  echo "[INFO] building live execution sidecar"
  "$PYTHON" tools/build_as1455_live_execution_sidecar_v1.py \
    --trade-date "$TRADE_DATE" \
    --out-root "$OUT_ROOT" \
    --live-dir "$LIVE_DIR" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --universe "$UNIVERSE"
  export PRICE_FILE="${PRICE_FILE:-$LIVE_DIR/08_live_execution_sidecar.csv}"
  echo "[INFO] PRICE_FILE=$PRICE_FILE"
fi

bash scripts/run_as1455_live_checkpoint_signal_v1.sh
