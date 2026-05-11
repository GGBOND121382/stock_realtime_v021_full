#!/usr/bin/env bash
# run_muyuan_proxy_search_fixed.sh
set -uo pipefail
PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/muyuan_proxy_fixed_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"
SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "symbol,mode,external,status,returncode,start_time,end_time,log_file" > "$SUMMARY_FILE"
start_time="$(date '+%F %T')"
echo "[START] $start_time 002714.SZ external=hog,muyuan_hk" | tee -a "$LOG_FILE"
timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
  --symbol 002714.SZ \
  --sector-symbol 养殖业 \
  --start-date 2018-01-01 \
  --end-date "$END_DATE" \
  --feature-pipeline fundamental,sector \
  --external hog,muyuan_hk \
  --search-targets hit50,hit80,close_profit \
  --entry-policies vwap_low,all_days \
  --groups reversal_fundamental_regime,reversal_fundamental_regime_sector,all_no_ak \
  --models xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_600_lr002_mcw3,xgb_d4_500_lr002_mcw5,lgbm_leaves7_400,lgbm_leaves15_700,extra_trees_600_d3,random_forest_600_d4 \
  --quantiles 0.5,0.6,0.7,0.8 \
  --train-rows 756 \
  --valid-rows 126 \
  --test-rows 63 \
  --min-valid-trades 8 \
  --min-train-entries 80 \
  --excel 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
end_time="$(date '+%F %T')"
if [[ "$rc" -eq 0 ]]; then status="ok"; elif [[ "$rc" -eq 124 ]]; then status="timeout"; else status="failed"; fi
echo "002714.SZ,proxy_fixed,hog,muyuan_hk,${status},${rc},${start_time},${end_time},${LOG_FILE}" >> "$SUMMARY_FILE"
echo "[DONE] status=$status rc=$rc summary=$SUMMARY_FILE"
exit 0
