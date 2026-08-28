#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
PRE_SCHEDULE="${PRE_SCHEDULE:-35 9 * * 1-5}"
POST_SCHEDULE="${POST_SCHEDULE:-50 14 * * 1-5}"
REFRESH_SCHEDULE="${REFRESH_SCHEDULE:-20 20 * * 1-5}"
MODEL_ROLLOVER_SCHEDULE="${MODEL_ROLLOVER_SCHEDULE:-30 21 * * 1-5}"
INSTALL_MODEL_ROLLOVER="${INSTALL_MODEL_ROLLOVER:-0}"

[[ "$EUID" -eq 0 ]] || { echo "[ERROR] cron installation requires root" >&2; exit 1; }
[[ "$INSTALL_MODEL_ROLLOVER" == "0" || "$INSTALL_MODEL_ROLLOVER" == "1" ]] || {
  echo "[ERROR] INSTALL_MODEL_ROLLOVER must be 0 or 1" >&2
  exit 2
}

env \
  TIMEZONE="$TIMEZONE" \
  MATRIX_ROOT="$MATRIX_ROOT" \
  OUT_ROOT="$OUT_ROOT" \
  PRE_SCHEDULE="$PRE_SCHEDULE" \
  POST_SCHEDULE="$POST_SCHEDULE" \
  bash scripts/install_as1455_live_nine_strategy_cron.sh

env \
  TIMEZONE="$TIMEZONE" \
  MATRIX_ROOT="$MATRIX_ROOT" \
  LIVE_ROOT="$OUT_ROOT" \
  CRON_SCHEDULE="$REFRESH_SCHEDULE" \
  bash scripts/install_as1455_dashboard_daily_refresh_cron.sh

if [[ "$INSTALL_MODEL_ROLLOVER" == "1" ]]; then
  env \
    TIMEZONE="$TIMEZONE" \
    CRON_SCHEDULE="$MODEL_ROLLOVER_SCHEDULE" \
    bash scripts/install_as1455_model_rollover_cron.sh
fi

echo "[PASS] AS1455 strategy dashboard automation installed"
echo "[PASS] 09:35 prepare; 14:50 collect -> 14:55 plan; 20:20 incremental account refresh"
if [[ "$INSTALL_MODEL_ROLLOVER" == "1" ]]; then
  echo "[PASS] rolling model checker/retrain enabled at $MODEL_ROLLOVER_SCHEDULE"
else
  echo "[INFO] rolling model cron not installed (set INSTALL_MODEL_ROLLOVER=1 after server smoke validation)"
fi
echo "[PASS] timezone=$TIMEZONE"
