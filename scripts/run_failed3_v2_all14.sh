#!/usr/bin/env bash
set -uo pipefail

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-2026-05-12}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
RUN_TAG="${RUN_TAG:-v2_all14}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/failed3_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "symbol,sector,external,run_tag,status,returncode,start_time,end_time,elapsed_seconds,log_file" > "$SUMMARY_FILE"

COMMON_ARGS=(
  --run-tag "$RUN_TAG"
  --start-date 2018-01-01
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
  --stock-external-domestic-lag-days 0
  --stock-external-future-lag-days 1
  --stock-external-us-lag-days 1
  --resume
  --excel
)

run_one() {
  local symbol="$1"
  local sector="$2"
  local external="$3"
  local safe_symbol="${symbol//./_}"
  local log_file="$LOG_DIR/${safe_symbol}_${external}.log"
  local start_ts end_ts start_epoch end_epoch elapsed rc
  start_ts="$(date '+%F %T')"
  start_epoch="$(date +%s)"
  local cmd=(timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py
    --symbol "$symbol"
    --sector-symbol "$sector"
    --external "$external"
    "${COMMON_ARGS[@]}")

  {
    echo "============================================================"
    echo "[START] $start_ts symbol=$symbol sector=$sector external=$external run_tag=$RUN_TAG"
    printf '[CMD]'
    printf ' %q' "${cmd[@]}"
    echo
    echo "============================================================"
  } | tee "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    rc=0
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_ts="$(date '+%F %T')"
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))
  if [[ "$rc" -eq 0 ]]; then
    echo "[DONE] $end_ts symbol=$symbol elapsed=${elapsed}s" | tee -a "$log_file"
    echo "$symbol,$sector,$external,$RUN_TAG,ok,$rc,$start_ts,$end_ts,$elapsed,$log_file" >> "$SUMMARY_FILE"
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] $end_ts symbol=$symbol elapsed=${elapsed}s" | tee -a "$log_file"
    echo "$symbol,$sector,$external,$RUN_TAG,timeout,$rc,$start_ts,$end_ts,$elapsed,$log_file" >> "$SUMMARY_FILE"
  else
    echo "[FAIL] $end_ts symbol=$symbol returncode=$rc elapsed=${elapsed}s" | tee -a "$log_file"
    echo "$symbol,$sector,$external,$RUN_TAG,failed,$rc,$start_ts,$end_ts,$elapsed,$log_file" >> "$SUMMARY_FILE"
  fi
  return 0
}

run_one 002518.SZ 其他电源设备 storage_power
run_one 600522.SH 通信设备 optical_cable_grid
run_one 600487.SH 通信设备 optical_cable_grid

echo "[SUMMARY] $SUMMARY_FILE"
exit 0
