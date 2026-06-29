#!/usr/bin/env bash
set -euo pipefail

# Repair AS1455 daily cache for one date from already-downloaded raw 5m cache.
# This does not download data and does not write raw 5m/raw daily caches.

TRADE_DATE="${TRADE_DATE:-20260629}"
HISTORY_END_DATE="${HISTORY_END_DATE:-2026-06-26}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" pipelines/as1455_update_history_to_prevday_fast_v4.py \
  --trade-date "${TRADE_DATE}" \
  --history-end-date "${HISTORY_END_DATE}" \
  --history-start-date 2020-01-01 \
  --universe saved_data/ashare_ml4t/ch12_as1455/as1455_model_universe_from_h5.csv \
  --raw-5m-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --as1455-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache \
  --out-root saved_data/ashare_ml4t/live_as1455 \
  --skip-raw-5m \
  --skip-raw-daily \
  --sleep-seconds 0
