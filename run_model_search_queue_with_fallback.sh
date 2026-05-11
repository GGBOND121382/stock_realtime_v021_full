#!/usr/bin/env bash
set -uo pipefail

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --start-date 2018-01-01
  --end-date "$END_DATE"
  --feature-pipeline fundamental,sector
  --search-targets hit50,hit80,close_profit
  --entry-policies vwap_low,all_days
  --groups reversal_fundamental_regime,reversal_fundamental_regime_sector,all_no_ak
  --models xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_600_lr002_mcw3,xgb_d4_500_lr002_mcw5,lgbm_leaves7_400,lgbm_leaves15_700,extra_trees_600_d3,random_forest_600_d4
  --quantiles 0.5,0.6,0.7,0.8
  --train-rows 756
  --valid-rows 126
  --test-rows 63
  --min-valid-trades 8
  --min-train-entries 80
  --resume
  --excel
)

SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "symbol,sector,mode,external,status,returncode,start_time,end_time,log_file" > "$SUMMARY_FILE"

run_pipeline() {
  local symbol="$1"
  local sector="$2"
  local mode="$3"
  local external="${4:-}"

  local safe_symbol="${symbol//./_}"
  local log_file="$LOG_DIR/${safe_symbol}_${mode}.log"
  local start_time
  local end_time
  local rc

  start_time="$(date '+%F %T')"

  echo "============================================================" | tee -a "$log_file"
  echo "[START] ${start_time} symbol=${symbol}, sector=${sector}, mode=${mode}, external=${external:-none}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  if [[ -n "$external" ]]; then
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
      --symbol "$symbol" \
      --sector-symbol "$sector" \
      --external "$external" \
      "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  else
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
      --symbol "$symbol" \
      --sector-symbol "$sector" \
      "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"

  if [[ "$rc" -eq 0 ]]; then
    echo "[DONE] ${end_time} symbol=${symbol}, mode=${mode}, status=ok" | tee -a "$log_file"
    echo "${symbol},${sector},${mode},${external:-},ok,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} symbol=${symbol}, mode=${mode}, timeout=${JOB_TIMEOUT}" | tee -a "$log_file"
    echo "${symbol},${sector},${mode},${external:-},timeout,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  else
    echo "[FAIL] ${end_time} symbol=${symbol}, mode=${mode}, returncode=${rc}" | tee -a "$log_file"
    echo "${symbol},${sector},${mode},${external:-},failed,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  fi
}

run_one() {
  local symbol="$1"
  local sector="$2"
  local external="${3:-}"

  echo
  echo "############################################################"
  echo "# QUEUE symbol=${symbol}, sector=${sector}, external=${external:-none}"
  echo "############################################################"

  if [[ -n "$external" ]]; then
    if run_pipeline "$symbol" "$sector" "external" "$external"; then
      echo "[QUEUE] ${symbol}: external succeeded, skip baseline fallback."
      return 0
    else
      echo "[QUEUE] ${symbol}: external failed/timeout, fallback to baseline without external."
      if run_pipeline "$symbol" "$sector" "baseline" ""; then
        echo "[QUEUE] ${symbol}: baseline fallback succeeded."
        return 0
      else
        echo "[QUEUE] ${symbol}: baseline fallback also failed. Continue next symbol."
        return 1
      fi
    fi
  else
    if run_pipeline "$symbol" "$sector" "baseline" ""; then
      echo "[QUEUE] ${symbol}: baseline succeeded."
      return 0
    else
      echo "[QUEUE] ${symbol}: baseline failed. Continue next symbol."
      return 1
    fi
  fi
}

FAILED=0

run_one 002270.SZ 电网设备 || FAILED=$((FAILED + 1))
run_one 600276.SH 化学制药 || FAILED=$((FAILED + 1))
run_one 002311.SZ 饲料 "feed,hog" || FAILED=$((FAILED + 1))
run_one 002714.SZ 养殖业 "hog,muyuan_hk" || FAILED=$((FAILED + 1))
run_one 600176.SH 玻璃玻纤 || FAILED=$((FAILED + 1))
run_one 600309.SH 化学制品 || FAILED=$((FAILED + 1))

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[FAILED SYMBOLS] ${FAILED}"
echo "[SUMMARY] ${SUMMARY_FILE}"
echo "============================================================"

exit 0
