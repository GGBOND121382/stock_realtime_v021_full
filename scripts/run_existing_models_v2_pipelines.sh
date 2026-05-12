#!/usr/bin/env bash
set -uo pipefail

# Re-run pipelines for the retained existing/planned models into isolated v2 output roots.
# Output roots: saved_data/<code>_pipeline_out_${RUN_TAG}, default saved_data/<code>_pipeline_out_v2_models
# This does not overwrite existing saved_data/<code>_pipeline_out.
#
# Run from project root:
#   chmod +x scripts/run_existing_models_v2_pipelines.sh
#   PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_existing_models_v2_pipelines.sh

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
RUN_TAG="${RUN_TAG:-v2_models}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/existing_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --run-tag "$RUN_TAG"
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --feature-pipeline fundamental,sector
  --search-targets hit50,hit80,close_profit
  --entry-policies vwap_low,all_days
  --groups reversal_fundamental_regime,reversal_fundamental_regime_sector,reversal_fundamental_regime_sector_external,all_no_ak
  --models xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_600_lr002_mcw3,xgb_d4_500_lr002_mcw5,lgbm_leaves7_400,lgbm_leaves15_700,extra_trees_600_d3,random_forest_600_d4
  --quantiles 0.5,0.6,0.7,0.8
  --train-rows 756
  --valid-rows 126
  --test-rows 63
  --min-valid-trades 8
  --min-train-entries 80
  --external-lag-days 1
  --resume
  --excel
)

SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "symbol,sector,external,run_tag,status,returncode,start_time,end_time,log_file" > "$SUMMARY_FILE"

run_one() {
  local symbol="$1"
  local sector="$2"
  local external="$3"
  local safe_symbol="${symbol//./_}"
  local ext_label="${external:-none}"
  local log_file="$LOG_DIR/${safe_symbol}_${ext_label}.log"
  local start_time
  local end_time
  local rc
  local external_args=()

  if [[ -n "$external" ]]; then
    external_args+=(--external "$external")
  fi

  start_time="$(date '+%F %T')"
  echo "============================================================" | tee -a "$log_file"
  echo "[START] ${start_time} symbol=${symbol}, sector=${sector}, external=${ext_label}, run_tag=${RUN_TAG}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
    --symbol "$symbol" \
    --sector-symbol "$sector" \
    "${external_args[@]}" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_file"
  rc=${PIPESTATUS[0]}

  end_time="$(date '+%F %T')"
  if [[ "$rc" -eq 0 ]]; then
    echo "[DONE] ${end_time} symbol=${symbol}, status=ok" | tee -a "$log_file"
    echo "${symbol},${sector},${ext_label},${RUN_TAG},ok,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} symbol=${symbol}, timeout=${JOB_TIMEOUT}" | tee -a "$log_file"
    echo "${symbol},${sector},${ext_label},${RUN_TAG},timeout,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  else
    echo "[FAIL] ${end_time} symbol=${symbol}, returncode=${rc}" | tee -a "$log_file"
    echo "${symbol},${sector},${ext_label},${RUN_TAG},failed,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  fi
}

FAILED=0

# Retained existing/planned model set, excluding 中国巨石 and 万华化学.
run_one 002270.SZ 电网设备     ""                 || FAILED=$((FAILED + 1))
run_one 002311.SZ 农产品加工   "feed,hog"         || FAILED=$((FAILED + 1))
run_one 002714.SZ 养殖业       "hog,muyuan_hk"    || FAILED=$((FAILED + 1))
run_one 600276.SH 化学制药     ""                 || FAILED=$((FAILED + 1))
run_one 600312.SH 电网设备     ""                 || FAILED=$((FAILED + 1))
run_one 601899.SH 贵金属       "zijin_external"   || FAILED=$((FAILED + 1))

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[FAILED SYMBOLS] ${FAILED}"
echo "[SUMMARY] ${SUMMARY_FILE}"
echo "[PIPELINE OUTPUT ROOTS] saved_data/<code>_pipeline_out_${RUN_TAG}"
echo "============================================================"

exit 0
