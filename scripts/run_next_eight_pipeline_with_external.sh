#!/usr/bin/env bash
set -uo pipefail

# Batch-run the next 8 stock pipelines with realistic non-EM external profiles.
# Output is isolated from old runs by --run-tag (default: v2_external).
# Optional yfinance is used only when explicitly requested per stock; by default it is enabled for 工业富联.
#
# Run from project root:
#   chmod +x scripts/run_next_eight_pipeline_with_external.sh
#   PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_next_eight_pipeline_with_external.sh
#
# Install optional U.S. data dependency when enabling yfinance:
#   python3 -m pip install -U yfinance

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
RUN_TAG="${RUN_TAG:-v2_external}"
ENABLE_YF_FOR_AI="${ENABLE_YF_FOR_AI:-1}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/next_eight_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

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
  # Old external builders keep legacy lag; new stock builder uses source-specific lags below.
  --external-lag-days 1
  --stock-external-domestic-lag-days 0
  --stock-external-future-lag-days 1
  --stock-external-us-lag-days 1
  --resume
  --excel
)

SUMMARY_FILE="$LOG_DIR/queue_summary.csv"
echo "symbol,sector,external,run_tag,enable_us_yf,status,returncode,start_time,end_time,log_file" > "$SUMMARY_FILE"

run_one() {
  local symbol="$1"
  local sector="$2"
  local external="$3"
  local enable_us_yf="${4:-0}"
  local safe_symbol="${symbol//./_}"
  local log_file="$LOG_DIR/${safe_symbol}_${external}.log"
  local start_time
  local end_time
  local rc
  local extra_args=()

  if [[ "$enable_us_yf" == "1" ]]; then
    extra_args+=(--enable-us-yf)
  fi

  start_time="$(date '+%F %T')"

  echo "============================================================" | tee -a "$log_file"
  echo "[START] ${start_time} symbol=${symbol}, sector=${sector}, external=${external}, run_tag=${RUN_TAG}, enable_us_yf=${enable_us_yf}" | tee -a "$log_file"
  echo "============================================================" | tee -a "$log_file"

  timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py \
    --symbol "$symbol" \
    --sector-symbol "$sector" \
    --external "$external" \
    "${COMMON_ARGS[@]}" \
    "${extra_args[@]}" 2>&1 | tee -a "$log_file"
  rc=${PIPESTATUS[0]}

  end_time="$(date '+%F %T')"
  if [[ "$rc" -eq 0 ]]; then
    echo "[DONE] ${end_time} symbol=${symbol}, status=ok" | tee -a "$log_file"
    echo "${symbol},${sector},${external},${RUN_TAG},${enable_us_yf},ok,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} symbol=${symbol}, timeout=${JOB_TIMEOUT}" | tee -a "$log_file"
    echo "${symbol},${sector},${external},${RUN_TAG},${enable_us_yf},timeout,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  else
    echo "[FAIL] ${end_time} symbol=${symbol}, returncode=${rc}" | tee -a "$log_file"
    echo "${symbol},${sector},${external},${RUN_TAG},${enable_us_yf},failed,${rc},${start_time},${end_time},${log_file}" >> "$SUMMARY_FILE"
    return "$rc"
  fi
}

FAILED=0

# External profile mapping:
# 601138.SH 工业富联 -> AI compute/server/cloud-capex proxy; yfinance optional and T-1 aligned.
# 002080.SZ 中材科技 -> fiberglass/material + wind + battery proxy
# 601985.SH 中国核电 -> utility/power/defensive style proxy
# 600096.SH 云天化   -> fertilizer/agri-chemical commodities and peer proxy
# 002518.SZ 科士达   -> storage/photovoltaic inverter/data-center power proxy
# 603308.SH 应流股份 -> aero-engine/military/nuclear-equipment/high-alloy proxy
# 600522.SH 中天科技 -> optical communication + cable/grid/offshore-wind proxy
# 600487.SH 亨通光电 -> same optical/cable/grid proxy

run_one 601138.SH 消费电子     ai_compute             "$ENABLE_YF_FOR_AI" || FAILED=$((FAILED + 1))
run_one 002080.SZ 建筑材料     material_wind_battery  0 || FAILED=$((FAILED + 1))
run_one 601985.SH 电力         power_utility_rate      0 || FAILED=$((FAILED + 1))
run_one 600096.SH 农化制品     fertilizer              0 || FAILED=$((FAILED + 1))
run_one 002518.SZ 其他电源设备 storage_power           0 || FAILED=$((FAILED + 1))
run_one 603308.SH 通用设备     aero_nuclear_equipment  0 || FAILED=$((FAILED + 1))
run_one 600522.SH 通信设备     optical_cable_grid      0 || FAILED=$((FAILED + 1))
run_one 600487.SH 通信设备     optical_cable_grid      0 || FAILED=$((FAILED + 1))

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[FAILED SYMBOLS] ${FAILED}"
echo "[SUMMARY] ${SUMMARY_FILE}"
echo "[PIPELINE OUTPUT ROOTS] saved_data/<code>_pipeline_out"
echo "============================================================"

exit 0
