#!/usr/bin/env bash
set -uo pipefail

# Serially run the 6 retained existing-model v2 pipelines + 8 new external v2 pipelines.
# Designed for the stock_external_nextday_v2_patch.zip patch.
#
# Run from project root:
#   chmod +x scripts/run_all_14_v2_pipelines.sh
#   PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_all_14_v2_pipelines.sh
#
# Optional controls:
#   RUN_TAG=v2_all14              # log/summary tag; output roots stay saved_data/<code>_pipeline_out
#   ENABLE_YF_FOR_AI=1            # enable yfinance for 601138.SH ai_compute; US features are forced T-1 aligned
#   DRY_RUN=1                     # print commands without executing
#   LOG_DIR=...                   # override log directory
#
# Notes:
#   - Failures/timeouts for one stock do not stop the remaining queue.
#   - Existing model external builders keep legacy --external-lag-days=1.
#   - New stock external builders use source-specific lags:
#       A-share/ETF/THS board lag = 0
#       domestic futures lag      = 1
#       U.S. yfinance lag         = 1, forced inside builder to avoid leakage.

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
RUN_TAG="${RUN_TAG:-v2_all14}"
ENABLE_YF_FOR_AI="${ENABLE_YF_FOR_AI:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/all14_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

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
  --stock-external-domestic-lag-days 0
  --stock-external-future-lag-days 1
  --stock-external-us-lag-days 1
  --resume
  --excel
)

SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "phase,symbol,sector,external,run_tag,enable_us_yf,status,returncode,start_time,end_time,elapsed_seconds,log_file" > "$SUMMARY_FILE"

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local phase="$1"
  local symbol="$2"
  local sector="$3"
  local external="$4"
  local enable_us_yf="${5:-0}"

  local safe_symbol="${symbol//./_}"
  local ext_label="${external:-none}"
  local log_file="$LOG_DIR/${phase}_${safe_symbol}_${ext_label}.log"
  local start_time end_time start_epoch end_epoch elapsed rc
  local cmd=()
  local external_args=()
  local extra_args=()

  if [[ -n "$external" ]]; then
    external_args+=(--external "$external")
  fi
  if [[ "$enable_us_yf" == "1" ]]; then
    extra_args+=(--enable-us-yf)
  fi

  cmd=(
    "$PYTHON" pipelines/run_nextday_pipeline.py
    --symbol "$symbol"
    --sector-symbol "$sector"
    "${external_args[@]}"
    "${COMMON_ARGS[@]}"
    "${extra_args[@]}"
  )

  start_time="$(date '+%F %T')"
  start_epoch="$(date +%s)"

  {
    echo "============================================================"
    echo "[START] ${start_time} phase=${phase}, symbol=${symbol}, sector=${sector}, external=${ext_label}, run_tag=${RUN_TAG}, enable_us_yf=${enable_us_yf}"
    echo "[CMD] $(quote_cmd "${cmd[@]}")"
    echo "============================================================"
  } | tee -a "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    rc=0
    echo "[DRY-RUN] skipped execution" | tee -a "$log_file"
  else
    timeout --foreground "$JOB_TIMEOUT" "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))

  if [[ "$rc" -eq 0 ]]; then
    echo "[DONE] ${end_time} phase=${phase}, symbol=${symbol}, status=ok, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${phase},${symbol},${sector},${ext_label},${RUN_TAG},${enable_us_yf},ok,${rc},${start_time},${end_time},${elapsed},${log_file}" >> "$SUMMARY_FILE"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} phase=${phase}, symbol=${symbol}, timeout=${JOB_TIMEOUT}, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${phase},${symbol},${sector},${ext_label},${RUN_TAG},${enable_us_yf},timeout,${rc},${start_time},${end_time},${elapsed},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  else
    echo "[FAIL] ${end_time} phase=${phase}, symbol=${symbol}, returncode=${rc}, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${phase},${symbol},${sector},${ext_label},${RUN_TAG},${enable_us_yf},failed,${rc},${start_time},${end_time},${elapsed},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  fi
}

FAILED=0

# -----------------------------------------------------------------------------
# Phase 1: retained existing/planned models v2, excluding 中国巨石 and 万华化学.
# -----------------------------------------------------------------------------
run_one existing 002270.SZ 电网设备     ""               0 || FAILED=$((FAILED + 1))
run_one existing 002311.SZ 农产品加工   "feed,hog"       0 || FAILED=$((FAILED + 1))
run_one existing 002714.SZ 养殖业       "hog,muyuan_hk"  0 || FAILED=$((FAILED + 1))
run_one existing 600276.SH 化学制药     ""               0 || FAILED=$((FAILED + 1))
run_one existing 600312.SH 电网设备     ""               0 || FAILED=$((FAILED + 1))
run_one existing 601899.SH 贵金属       "zijin_external" 0 || FAILED=$((FAILED + 1))

# -----------------------------------------------------------------------------
# Phase 2: new 8 stocks with realistic external profiles.
# -----------------------------------------------------------------------------
run_one next8 601138.SH 消费电子     "ai_compute"             "$ENABLE_YF_FOR_AI" || FAILED=$((FAILED + 1))
run_one next8 002080.SZ 建筑材料     "material_wind_battery"  0 || FAILED=$((FAILED + 1))
run_one next8 601985.SH 电力         "power_utility_rate"      0 || FAILED=$((FAILED + 1))
run_one next8 600096.SH 农化制品     "fertilizer"              0 || FAILED=$((FAILED + 1))
run_one next8 002518.SZ 其他电源设备 "storage_power"           0 || FAILED=$((FAILED + 1))
run_one next8 603308.SH 通用设备     "aero_nuclear_equipment"  0 || FAILED=$((FAILED + 1))
run_one next8 600522.SH 通信设备     "optical_cable_grid"      0 || FAILED=$((FAILED + 1))
run_one next8 600487.SH 通信设备     "optical_cable_grid"      0 || FAILED=$((FAILED + 1))

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[FAILED SYMBOLS] ${FAILED}"
echo "[SUMMARY] ${SUMMARY_FILE}"
echo "[PIPELINE OUTPUT ROOTS] saved_data/<code>_pipeline_out"
echo "============================================================"

# Keep exit 0 so a long queue can finish and you can inspect queue_summary.csv.
exit 0
