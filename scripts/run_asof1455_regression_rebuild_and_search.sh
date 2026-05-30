#!/usr/bin/env bash
set -uo pipefail

# One-click orchestration for the asof1455 pooled regression workflow.
#
# Default behavior:
#   1) inspect saved_data cleanup plan, but do not move files
#   2) rebuild the 43-symbol canonical asof1455 universe
#   3) run pooled regression model search
#   4) fail the wrapper if either summary contains failed/timeout rows
#
# Common overrides:
#   SMOKE=1                         run first 2 symbols and first 2 model jobs
#   DRY_RUN=1                       print child commands without rebuilding/training
#   CLEANUP_MODE=skip|plan|execute  default: plan
#   RUN_BUILD=0                     skip sample rebuild
#   RUN_MODEL=0                     skip model search
#   PYTHON=.venv/bin/python         python executable
#   JOB_TIMEOUT=8h                  per-symbol/per-model timeout

USER_JOB_TIMEOUT="${JOB_TIMEOUT:-}"

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_MODEL="${RUN_MODEL:-1}"
CLEANUP_MODE="${CLEANUP_MODE:-plan}"
CHECK_SUMMARIES="${CHECK_SUMMARIES:-1}"
OUT_ROOT="${OUT_ROOT:-saved_data/ml4t_asof1455_lgbm_pipeline_out}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-${OUT_ROOT}/logs/rebuild_and_search_$(date +%Y%m%d_%H%M%S)}"

if [[ "$SMOKE" == "1" ]]; then
  MAX_SYMBOLS="${MAX_SYMBOLS:-2}"
  MAX_RUNS="${MAX_RUNS:-2}"
  if [[ -z "$USER_JOB_TIMEOUT" ]]; then
    JOB_TIMEOUT="1m"
  fi
else
  MAX_SYMBOLS="${MAX_SYMBOLS:-0}"
  MAX_RUNS="${MAX_RUNS:-0}"
fi

mkdir -p "$MASTER_LOG_DIR"
MASTER_SUMMARY="${MASTER_LOG_DIR}/workflow_summary.txt"

log() {
  local msg="$1"
  echo "[$(date '+%F %T')] $msg" | tee -a "$MASTER_SUMMARY"
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  log "CMD $(quote_cmd "$@")"
  "$@" 2>&1 | tee -a "${MASTER_LOG_DIR}/workflow.log"
  return "${PIPESTATUS[0]}"
}

csv_bad_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 1
    return 0
  fi
  awk -F',' 'NR > 1 && $0 != "" && $0 !~ /,ok,/ { n += 1 } END { print n + 0 }' "$path"
}

csv_status_report() {
  local label="$1"
  local path="$2"
  local status_col="$3"
  if [[ ! -f "$path" ]]; then
    log "${label} summary missing: ${path}"
    return 1
  fi
  log "${label} summary: ${path}"
  awk -F',' -v status_col="$status_col" '
    NR == 1 { next }
    $0 == "" { next }
    { status[$status_col] += 1 }
    END {
      for (s in status) {
        printf "  %s=%d\n", s, status[s]
      }
    }
  ' "$path" | tee -a "$MASTER_SUMMARY"
}

cleanup_saved_data() {
  case "$CLEANUP_MODE" in
    skip)
      log "cleanup skipped"
      ;;
    plan)
      log "cleanup plan only"
      run_cmd "$PYTHON" tools/cleanup_saved_data_layout.py --saved-data saved_data
      ;;
    execute)
      log "cleanup execute"
      run_cmd "$PYTHON" tools/cleanup_saved_data_layout.py --saved-data saved_data --execute
      ;;
    *)
      log "invalid CLEANUP_MODE=${CLEANUP_MODE}; expected skip, plan, or execute"
      return 2
      ;;
  esac
}

build_universe() {
  local build_log_dir="${MASTER_LOG_DIR}/build_universe"
  log "build universe start: START_DATE=${START_DATE} END_DATE=${END_DATE} MAX_SYMBOLS=${MAX_SYMBOLS} DRY_RUN=${DRY_RUN}"
  LOG_DIR="$build_log_dir" \
  PYTHON="$PYTHON" \
  START_DATE="$START_DATE" \
  END_DATE="$END_DATE" \
  JOB_TIMEOUT="$JOB_TIMEOUT" \
  DRY_RUN="$DRY_RUN" \
  MAX_SYMBOLS="$MAX_SYMBOLS" \
    bash scripts/build_asof1455_regression_universe.sh
  local rc=$?
  local summary="${build_log_dir}/build_summary.csv"
  csv_status_report "build" "$summary" 5
  if [[ "$CHECK_SUMMARIES" == "1" && "$(csv_bad_count "$summary")" != "0" ]]; then
    log "build summary contains failed/timeout rows"
    return 1
  fi
  return "$rc"
}

run_model_pool() {
  local model_log_dir="${MASTER_LOG_DIR}/regression_pool"
  log "model pool start: FEATURE_GROUPS=${FEATURE_GROUPS:-default} MODEL_FAMILIES=${MODEL_FAMILIES:-default} MAX_RUNS=${MAX_RUNS} DRY_RUN=${DRY_RUN}"
  LOG_DIR="$model_log_dir" \
  OUT_ROOT="$OUT_ROOT" \
  PYTHON="$PYTHON" \
  JOB_TIMEOUT="$JOB_TIMEOUT" \
  DRY_RUN="$DRY_RUN" \
  MAX_RUNS="$MAX_RUNS" \
  TRAIN_DAYS="${TRAIN_DAYS:-756}" \
  VALID_DAYS="${VALID_DAYS:-0}" \
  TEST_DAYS="${TEST_DAYS:-21}" \
  EMBARGO_DAYS="${EMBARGO_DAYS:-1}" \
  STEP_DAYS="${STEP_DAYS:-21}" \
  MAX_POSITIONS="${MAX_POSITIONS:-3}" \
  MIN_DAILY_CANDIDATES="${MIN_DAILY_CANDIDATES:-10}" \
  MIN_PRED_RETURN_BPS="${MIN_PRED_RETURN_BPS:-0.0}" \
  MAX_MISSING="${MAX_MISSING:-0.70}" \
  FEATURE_GROUPS="${FEATURE_GROUPS:-ml4t_intraday,ml4t_fundamental,ml4t_sector,ml4t_external}" \
  MODEL_FAMILIES="${MODEL_FAMILIES:-ridge,elasticnet,extratrees,lgbm_l1,lgbm_huber,lgbm_quantile,catboost_rmse,catboost_huber,catboost_quantile,randomforest,lgbm_l2}" \
    bash scripts/run_asof1455_regression_model_pool.sh
  local rc=$?
  local summary="${model_log_dir}/regression_pool_summary.csv"
  csv_status_report "model" "$summary" 3
  if [[ "$CHECK_SUMMARIES" == "1" && "$(csv_bad_count "$summary")" != "0" ]]; then
    log "model summary contains failed/timeout rows"
    return 1
  fi
  return "$rc"
}

main() {
  log "workflow start"
  log "MASTER_LOG_DIR=${MASTER_LOG_DIR}"
  log "PYTHON=${PYTHON}"
  log "CLEANUP_MODE=${CLEANUP_MODE} RUN_BUILD=${RUN_BUILD} RUN_MODEL=${RUN_MODEL} SMOKE=${SMOKE}"

  cleanup_saved_data || return $?

  if [[ "$RUN_BUILD" == "1" ]]; then
    build_universe || return $?
  else
    log "build skipped"
  fi

  if [[ "$RUN_MODEL" == "1" ]]; then
    run_model_pool || return $?
  else
    log "model pool skipped"
  fi

  log "workflow done"
  log "summary: ${MASTER_SUMMARY}"
}

main "$@"
