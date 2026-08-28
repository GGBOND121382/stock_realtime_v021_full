#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PROJECT_ROOT="$PWD"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
PRE_SCHEDULE="${PRE_SCHEDULE:-35 9 * * 1-5}"
POST_SCHEDULE="${POST_SCHEDULE:-50 14 * * 1-5}"
DEFERRED_SCHEDULE="${DEFERRED_SCHEDULE:-20 15 * * 1-5}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
CRON_FILE="${CRON_FILE:-/etc/cron.d/as1455-nine-strategy-live}"
PRODUCTION_EXPERIMENT="${PRODUCTION_EXPERIMENT:-r21_best_reb21_fold0_4_forward}"

[[ "$EUID" -eq 0 ]] || { echo "[ERROR] writing $CRON_FILE requires root" >&2; exit 1; }
cat > "$CRON_FILE" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CRON_TZ=$TIMEZONE
$PRE_SCHEDULE root cd $PROJECT_ROOT && MATRIX_ROOT=$MATRIX_ROOT OUT_ROOT=$OUT_ROOT TIMEZONE=$TIMEZONE PRODUCTION_EXPERIMENT=$PRODUCTION_EXPERIMENT bash scripts/run_as1455_live_nine_strategy_job.sh pre
$POST_SCHEDULE root cd $PROJECT_ROOT && MATRIX_ROOT=$MATRIX_ROOT OUT_ROOT=$OUT_ROOT TIMEZONE=$TIMEZONE PRODUCTION_EXPERIMENT=$PRODUCTION_EXPERIMENT bash scripts/run_as1455_live_nine_strategy_job.sh post
$DEFERRED_SCHEDULE root cd $PROJECT_ROOT && OUT_ROOT=$OUT_ROOT TIMEZONE=$TIMEZONE bash scripts/run_as1455_deferred_nine_strategy_predictions.sh
EOF
chmod 0644 "$CRON_FILE"

echo "[PASS] installed AS1455 production + research-monitor jobs"
echo "[PASS] file=$CRON_FILE timezone=$TIMEZONE"
echo "[PASS] production_experiment=$PRODUCTION_EXPERIMENT"
echo "[PASS] pre=$PRE_SCHEDULE (shared preparation + nine tracking accounts)"
echo "[PASS] post=$POST_SCHEDULE (14:55 critical path: production strategy only)"
echo "[PASS] deferred=$DEFERRED_SCHEDULE (after-close Top-5 predictions for all nine research-monitor strategies)"
