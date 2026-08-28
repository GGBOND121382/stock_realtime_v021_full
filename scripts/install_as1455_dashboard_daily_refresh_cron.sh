#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PROJECT_ROOT="$PWD"
CRON_SCHEDULE="${CRON_SCHEDULE:-20 20 * * 1-5}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
LIVE_ROOT="${LIVE_ROOT:-saved_data/ashare_ml4t/live_as1455}"
CRON_FILE="${CRON_FILE:-/etc/cron.d/as1455-dashboard-refresh}"

[[ "$EUID" -eq 0 ]] || { echo "[ERROR] writing $CRON_FILE requires root" >&2; exit 1; }
cat > "$CRON_FILE" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CRON_TZ=$TIMEZONE
$CRON_SCHEDULE root cd $PROJECT_ROOT && MATRIX_ROOT=$MATRIX_ROOT LIVE_ROOT=$LIVE_ROOT SKIP_DATA_REFRESH=0 TRACKING_MODE=incremental TIMEZONE=$TIMEZONE bash scripts/run_as1455_dashboard_refresh.sh
EOF
chmod 0644 "$CRON_FILE"

echo "[PASS] installed weekday AS1455 incremental tracking refresh"
echo "[PASS] file=$CRON_FILE schedule=$CRON_SCHEDULE timezone=$TIMEZONE"
echo "[PASS] default is 20:20 Asia/Shanghai on weekdays (after the expected 20:00 BaoStock update)"
echo "[PASS] if BaoStock is late, the tracker remains at the latest completed market date and advances on the next run"
