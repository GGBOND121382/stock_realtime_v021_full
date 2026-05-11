#!/usr/bin/env bash
# run_muyuan_hog_only_search.sh
#
# 牧原股份 002714.SZ 重跑脚本：
#   主方案：external=hog
#   fallback：无 external baseline
#
# 目的：
#   避免使用不稳定的港股 proxy 特征；
#   生成更适合 14:55 实盘的 hog-only / sector / fundamental 模型。
#
# 注意：
#   本脚本默认不使用 --resume，避免复用之前 04_external/muyuan_hk 或 hog_hk_proxy 的旧输出。
#
# 用法：
#   PYTHON=python3 ./run_muyuan_hog_only_search.sh
#   JOB_TIMEOUT=8h PYTHON=python3 ./run_muyuan_hog_only_search.sh

set -uo pipefail

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/muyuan_hog_only_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --symbol 002714.SZ
  --sector-symbol 养殖业
  --start-date 2018-01-01
  --end-date "$END_DATE"
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
  --excel
)

SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "symbol,mode,external,status,returncode,start_time,end_time,log_file" > "$SUMMARY_FILE"

run_cmd() {
  local mode="$1"
  local external="${2:-}"
  local log_file="$LOG_DIR/002714_${mode}.log"
  local start_time
  local end_time
  local rc

  start_time="$(date '+%F %T')"

  echo "============================================================" | tee -a "$log_file"
  echo "[START] ${start_time} 002714.SZ mode=${mode}, external=${external:-none}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  if [[ -n "$external" ]]; then
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --feature-pipeline fundamental,sector \
      --external "$external" \
      2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  else
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --feature-pipeline fundamental,sector \
      2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"

  if [[ "$rc" -eq 0 ]]; then
    echo "[DONE] ${end_time} mode=${mode}, status=ok" | tee -a "$log_file"
    echo "002714.SZ,${mode},${external:-},ok,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} mode=${mode}, timeout=${JOB_TIMEOUT}" | tee -a "$log_file"
    echo "002714.SZ,${mode},${external:-},timeout,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  else
    echo "[FAIL] ${end_time} mode=${mode}, returncode=${rc}" | tee -a "$log_file"
    echo "002714.SZ,${mode},${external:-},failed,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  fi
}

if run_cmd "hog_only" "hog"; then
  echo "[QUEUE] hog-only succeeded."
  echo "[SUMMARY] ${SUMMARY_FILE}"
  exit 0
fi

echo "[QUEUE] hog-only failed/timeout, fallback to baseline without external."

if run_cmd "baseline" ""; then
  echo "[QUEUE] baseline succeeded."
  echo "[SUMMARY] ${SUMMARY_FILE}"
  exit 0
fi

echo "[QUEUE] all modes failed."
echo "[SUMMARY] ${SUMMARY_FILE}"
exit 0
