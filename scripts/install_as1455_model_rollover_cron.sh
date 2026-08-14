#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

[[ "$EUID" -eq 0 ]] || { echo "[ERROR] cron installation requires root" >&2; exit 1; }

TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
CRON_SCHEDULE="${CRON_SCHEDULE:-30 21 * * 1-5}"
RUN_USER="${RUN_USER:-root}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CRON_FILE="${CRON_FILE:-/etc/cron.d/as1455-model-rollover}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/as1455_model_rollover_cron.log}"
mkdir -p "$(dirname "$LOG_FILE")"

cat > "$CRON_FILE" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CRON_TZ=$TIMEZONE
$CRON_SCHEDULE $RUN_USER cd $PROJECT_ROOT && bash scripts/run_as1455_model_rollover_job.sh >> $LOG_FILE 2>&1
EOF
chmod 0644 "$CRON_FILE"

echo "[PASS] installed $CRON_FILE"
echo "[PASS] schedule=$CRON_SCHEDULE timezone=$TIMEZONE"
echo "[PASS] the job exits quickly unless the current production period has reached 63 successful live trading days"
