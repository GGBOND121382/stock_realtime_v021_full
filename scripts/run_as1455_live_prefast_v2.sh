#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
LIVE_KEEP_HISTORY_TAIL="${LIVE_KEEP_HISTORY_TAIL:-0}"

bash scripts/run_as1455_live_prefast_v1.sh

if [[ "$LIVE_KEEP_HISTORY_TAIL" == "0" ]]; then
  "$PYTHON" tools/cleanup_as1455_live_intermediates_v2.py \
    --trade-date "$TRADE_DATE" \
    --out-root "$OUT_ROOT"
else
  echo "[INFO] LIVE_KEEP_HISTORY_TAIL=$LIVE_KEEP_HISTORY_TAIL; keep 04/05/10"
fi
