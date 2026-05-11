#!/usr/bin/env bash
# Retry Muyuan model search with fixed HK data source.
# Order:
#   1) hog,muyuan_hk external
#   2) hog-only external fallback
#   3) baseline without external fallback
# No --resume here, to avoid reusing stale failed external outputs.

set -uo pipefail

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/muyuan_hk_retry_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --symbol 002714.SZ
  --sector-symbol 养殖业
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
  --excel
)

run_mode() {
  local mode="$1"
  local external="${2:-}"
  local log_file="$LOG_DIR/002714_${mode}.log"
  echo "============================================================" | tee -a "$log_file"
  echo "[START] $(date '+%F %T') mode=${mode}, external=${external:-none}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  if [[ -n "$external" ]]; then
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --external "$external" 2>&1 | tee -a "$log_file"
  else
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
      "${COMMON_ARGS[@]}" 2>&1 | tee -a "$log_file"
  fi
  local rc=${PIPESTATUS[0]}
  echo "[END] $(date '+%F %T') mode=${mode}, rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

if run_mode "hog_muyuan_hk" "hog,muyuan_hk"; then
  echo "[OK] hog,muyuan_hk succeeded. logs=$LOG_DIR"
  exit 0
fi

echo "[WARN] hog,muyuan_hk failed; fallback to hog-only."
if run_mode "hog_only" "hog"; then
  echo "[OK] hog-only fallback succeeded. logs=$LOG_DIR"
  exit 0
fi

echo "[WARN] hog-only failed; fallback to no external baseline."
if run_mode "baseline" ""; then
  echo "[OK] baseline fallback succeeded. logs=$LOG_DIR"
  exit 0
fi

echo "[FAIL] all Muyuan modes failed. logs=$LOG_DIR"
exit 1
