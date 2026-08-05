#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PROJECT_ROOT="$PWD"
CRON_SCHEDULE="${CRON_SCHEDULE:-30 18 * * 1-5}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
CRON_FILE="${CRON_FILE:-/etc/cron.d/as1455-dashboard-refresh}"

[[ "$EUID" -eq 0 ]] || { echo "[ERROR] writing $CRON_FILE requires root" >&2; exit 1; }
cat > "$CRON_FILE" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CRON_TZ=$TIMEZONE
$CRON_SCHEDULE root cd $PROJECT_ROOT && MATRIX_ROOT=$MATRIX_ROOT SKIP_DATA_REFRESH=0 TIMEZONE=$TIMEZONE bash scripts/run_as1455_dashboard_refresh.sh
EOF
chmod 0644 "$CRON_FILE"

echo "[PASS] installed weekday AS1455 dashboard refresh"
echo "[PASS] file=$CRON_FILE schedule=$CRON_SCHEDULE timezone=$TIMEZONE"
