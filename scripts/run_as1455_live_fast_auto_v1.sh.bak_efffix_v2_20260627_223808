#!/usr/bin/env bash
# AS1455 live one-key fast monitor pipeline.
# Production entry:
#   TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_v1.sh
#
# This script keeps slow feature rebuild out of the 14:55 path:
#   before collect window: pre -> prefast -> wait -> postfast
#   inside/after collect window: requires prefast state, then postfast
set -Eeuo pipefail

MODE="${1:-auto}"
PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
PREPARE_TIME="${PREPARE_TIME:-09:35:00}"
COLLECT_START_TIME="${COLLECT_START_TIME:-14:50:00}"
UNTIL="${UNTIL:-14:55:05}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
LOG_ROOT="${LOG_ROOT:-logs}"
mkdir -p "$LOG_ROOT"

live_date() {
  "$PYTHON" - <<PY
from datetime import datetime
s = "${TRADE_DATE}"
if s.lower() == "today":
    print(datetime.now().strftime("%Y%m%d"))
else:
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
}

LIVE_DATE="$(live_date)"
LIVE_DIR="${OUT_ROOT}/${LIVE_DATE}"
STATE_PATH="${LIVE_DIR}/06_live_feature_state_fast.npz"

info() { echo "[INFO] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

seconds_now() {
  TZ="${TIMEZONE}" "$PYTHON" - <<'PY'
from datetime import datetime
n = datetime.now()
print(n.hour * 3600 + n.minute * 60 + n.second)
PY
}

time_to_seconds() {
  "$PYTHON" - "$1" <<'PY'
import sys
p = sys.argv[1].split(':')
if len(p) == 2:
    h, m = map(int, p); s = 0
elif len(p) == 3:
    h, m, s = map(int, p)
else:
    raise SystemExit(f"bad time: {sys.argv[1]}")
print(h * 3600 + m * 60 + s)
PY
}

wait_until_time() {
  local target="$1" target_sec now_sec sleep_sec
  target_sec="$(time_to_seconds "$target")"
  now_sec="$(seconds_now)"
  if (( now_sec >= target_sec )); then
    info "current time already >= ${target}; no wait"
    return 0
  fi
  sleep_sec=$(( target_sec - now_sec ))
  info "waiting ${sleep_sec}s until ${target} (${TIMEZONE})"
  sleep "$sleep_sec"
}

check_scripts() {
  [[ -f scripts/run_as1455_live_data_feature_pipeline.sh ]] || fail "missing scripts/run_as1455_live_data_feature_pipeline.sh"
  [[ -f scripts/run_as1455_live_prefast_v2.sh ]] || fail "missing scripts/run_as1455_live_prefast_v2.sh; install as1455_live_fastpath_onekey first"
  [[ -f scripts/run_as1455_live_postfast_v1.sh ]] || fail "missing scripts/run_as1455_live_postfast_v1.sh; install as1455_live_fastpath_onekey first"
}

run_pre_fast() {
  info "running slow pre stage: history + prepare"
  TRADE_DATE="${TRADE_DATE}" OUT_ROOT="${OUT_ROOT}" bash scripts/run_as1455_live_data_feature_pipeline.sh pre
  info "building fast feature state before collect window"
  TRADE_DATE="${TRADE_DATE}" OUT_ROOT="${OUT_ROOT}" bash scripts/run_as1455_live_prefast_v2.sh
  [[ -f "$STATE_PATH" ]] || fail "fast feature state not generated: $STATE_PATH"
}

run_post_fast() {
  [[ -f "$STATE_PATH" ]] || fail "missing fast feature state: $STATE_PATH. Run prefast before 14:50."
  info "running fast post stage: collect + finalize prediction features"
  TRADE_DATE="${TRADE_DATE}" OUT_ROOT="${OUT_ROOT}" UNTIL="${UNTIL}" bash scripts/run_as1455_live_postfast_v1.sh
}

print_config() {
  cat <<EOF
[CONFIG]
  MODE=${MODE}
  TRADE_DATE=${TRADE_DATE}
  LIVE_DATE=${LIVE_DATE}
  LIVE_DIR=${LIVE_DIR}
  TIMEZONE=${TIMEZONE}
  PREPARE_TIME=${PREPARE_TIME}
  COLLECT_START_TIME=${COLLECT_START_TIME}
  UNTIL=${UNTIL}
  STATE_PATH=${STATE_PATH}
EOF
}

main() {
  check_scripts
  print_config
  case "$MODE" in
    pre|prefast)
      run_pre_fast
      ;;
    post|postfast)
      run_post_fast
      ;;
    auto|all)
      local now_sec collect_sec prepare_sec
      now_sec="$(seconds_now)"
      prepare_sec="$(time_to_seconds "$PREPARE_TIME")"
      collect_sec="$(time_to_seconds "$COLLECT_START_TIME")"

      if (( now_sec < collect_sec )); then
        if (( now_sec < prepare_sec )); then
          info "before prepare time; running history now is allowed, then wait for prepare time via pipeline pre"
        fi
        run_pre_fast
        wait_until_time "$COLLECT_START_TIME"
        run_post_fast
      else
        info "already in/after collect window; using existing fast state and running postfast only"
        run_post_fast
      fi
      ;;
    status)
      ls -lh "$LIVE_DIR"/06_live_feature_state_fast.npz 2>/dev/null || true
      ls -lh "$LIVE_DIR"/11_live_model_features_for_prediction.csv 2>/dev/null || true
      [[ -f "$LIVE_DIR/12_feature_build_report.json" ]] && cat "$LIVE_DIR/12_feature_build_report.json" || true
      [[ -f "$LIVE_DIR/13_live_feature_strict_validation_report.json" ]] && cat "$LIVE_DIR/13_live_feature_strict_validation_report.json" || true
      ;;
    *)
      cat <<EOF
Usage:
  TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_v1.sh [auto|pre|post|status]

Default mode is auto.
  auto : pre + prefast before 14:50, wait, then postfast
  pre  : history + prepare + prefast only
  post : collect + fast finalize only; requires 06_live_feature_state_fast.npz
EOF
      exit 2
      ;;
  esac
}

main "$@"
