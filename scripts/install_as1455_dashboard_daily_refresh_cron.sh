#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PROJECT_ROOT="$PWD"
CRON_SCHEDULE="${CRON_SCHEDULE:-30 18 * * 1-5}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
TAG="# AS1455_DASHBOARD_DAILY_REFRESH"
COMMAND="cd $PROJECT_ROOT && MATRIX_ROOT=$MATRIX_ROOT SKIP_DATA_REFRESH=0 TIMEZONE=$TIMEZONE bash scripts/run_as1455_dashboard_refresh.sh"
LINE="$CRON_SCHEDULE $COMMAND $TAG"

current="$(crontab -l 2>/dev/null || true)"
filtered="$(printf '%s\n' "$current" | grep -vF "$TAG" || true)"
{
  printf 'CRON_TZ=%s\n' "$TIMEZONE"
  printf '%s\n' "$filtered" | grep -v '^CRON_TZ=' || true
  printf '%s\n' "$LINE"
} | sed '/^[[:space:]]*$/d' | crontab -

echo "[PASS] installed weekday AS1455 dashboard refresh"
echo "[PASS] schedule=$CRON_SCHEDULE timezone=$TIMEZONE"
echo "[PASS] command=$COMMAND"
