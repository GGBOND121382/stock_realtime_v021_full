#!/usr/bin/env bash
set -uo pipefail

# Build canonical asof1455 data for the full historical regression universe.
#
# Output policy:
#   saved_data/<code>_pipeline_out
#
# No run_tag is used here.  This is intentional: the regression pipeline reads
# only canonical per-stock pipeline directories.

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
ENABLE_YF_FOR_AI="${ENABLE_YF_FOR_AI:-1}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
FORCE_REFRESH="${FORCE_REFRESH:-0}"
LOG_DIR="${LOG_DIR:-saved_data/ml4t_asof1455_lgbm_pipeline_out/logs/build_universe_$(date +%Y%m%d_%H%M%S)}"
MAX_SYMBOLS="${MAX_SYMBOLS:-0}"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --feature-time-mode asof1455
  --feature-cutoff-time 14:55
  --feature-pipeline fundamental,sector
  --skip-akshare-fund-flow
  --external-lag-days 1
  --stock-external-domestic-lag-days 0
  --stock-external-future-lag-days 1
  --stock-external-us-lag-days 1
  --continue-on-error
)

if [[ "$RESUME" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

if [[ "$FORCE_REFRESH" == "1" ]]; then
  COMMON_ARGS+=(--force-refresh)
fi

SUMMARY_FILE="$LOG_DIR/build_summary.csv"
echo "symbol,sector,external,enable_us_yf,status,returncode,start_time,end_time,elapsed_seconds,log_file" > "$SUMMARY_FILE"
RUN_COUNT=0

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

external_stages() {
  local external="$1"
  if [[ -z "$external" ]]; then
    printf ''
    return 0
  fi
  local out=""
  IFS=',' read -ra parts <<< "$external"
  for part in "${parts[@]}"; do
    case "$part" in
      storage_power) out+=",external_storage_power" ;;
      power_utility_rate) out+=",external_power_utility_rate" ;;
      ai_compute) out+=",external_ai_compute" ;;
      feed) out+=",external_feed" ;;
      hog) out+=",external_hog" ;;
      fertilizer) out+=",external_fertilizer" ;;
      zijin_external) out+=",external_zijin_external" ;;
      optical_cable_grid) out+=",external_optical_cable_grid" ;;
      aero_nuclear_equipment) out+=",external_aero_nuclear_equipment" ;;
      material_wind_battery) out+=",external_material_wind_battery" ;;
      muyuan_hk) out+=",external_muyuan_hk" ;;
      "") ;;
      *) echo "[WARN] unsupported external profile in stage mapper: $part" >&2 ;;
    esac
  done
  printf '%s' "$out"
}

run_one() {
  local symbol="$1"
  local sector="$2"
  local external="${3:-}"
  local enable_us_yf="${4:-0}"

  local raw="${symbol%%.*}"
  local out_root="saved_data/${raw}_pipeline_out"
  local safe_symbol="${symbol//./_}"
  local ext_label="${external:-none}"
  local ext_summary="${ext_label//,/+}"
  local log_file="$LOG_DIR/${safe_symbol}_${ext_label}.log"
  local start_time end_time start_epoch end_epoch elapsed rc stages
  local cmd=()
  local external_args=()
  local extra_args=()

  if [[ "$MAX_SYMBOLS" != "0" && "$RUN_COUNT" -ge "$MAX_SYMBOLS" ]]; then
    return 0
  fi
  RUN_COUNT=$((RUN_COUNT + 1))

  stages="update_data,samples,asof_samples,fundamental,sector$(external_stages "$external")"

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
    --out-root "$out_root"
    "${external_args[@]}"
    "${COMMON_ARGS[@]}"
    --only-stages "$stages"
    "${extra_args[@]}"
  )

  start_time="$(date '+%F %T')"
  start_epoch="$(date +%s)"

  {
    echo "============================================================"
    echo "[START] ${start_time} symbol=${symbol}, sector=${sector}, external=${ext_label}, out_root=${out_root}"
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
    echo "[DONE] ${end_time} symbol=${symbol}, status=ok, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${symbol},${sector},${ext_summary},${enable_us_yf},ok,${rc},${start_time},${end_time},${elapsed},${log_file}" >> "$SUMMARY_FILE"
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} symbol=${symbol}, timeout=${JOB_TIMEOUT}, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${symbol},${sector},${ext_summary},${enable_us_yf},timeout,${rc},${start_time},${end_time},${elapsed},${log_file}" >> "$SUMMARY_FILE"
  else
    echo "[FAIL] ${end_time} symbol=${symbol}, returncode=${rc}, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${symbol},${sector},${ext_summary},${enable_us_yf},failed,${rc},${start_time},${end_time},${elapsed},${log_file}" >> "$SUMMARY_FILE"
  fi
  return 0
}

# Full historical pipeline/search universe, de-duplicated.
run_one 000657.SZ 小金属 zijin_external 0
run_one 000786.SZ 建筑材料 material_wind_battery 0
run_one 002028.SZ 电网设备 storage_power 0
run_one 002080.SZ 建筑材料 material_wind_battery 0
run_one 002128.SZ 煤炭开采加工 power_utility_rate 0
run_one 002261.SZ 软件开发 ai_compute "$ENABLE_YF_FOR_AI"
run_one 002270.SZ 电网设备 "" 0
run_one 002297.SZ 军工装备 aero_nuclear_equipment 0
run_one 002311.SZ 农产品加工 feed,hog 0
run_one 002364.SZ 其他电源设备 storage_power 0
run_one 002460.SZ 能源金属 zijin_external 0
run_one 002518.SZ 其他电源设备 storage_power 0
run_one 002601.SZ 化学原料 "" 0
run_one 002714.SZ 养殖业 hog,muyuan_hk 0
run_one 002895.SZ 农化制品 fertilizer 0
run_one 003816.SZ 电力 power_utility_rate 0
run_one 600016.SH 银行 "" 0
run_one 600030.SH 证券 "" 0
run_one 600096.SH 农化制品 fertilizer 0
run_one 600176.SH 建筑材料 "" 0
run_one 600276.SH 化学制药 "" 0
run_one 600309.SH 化学制品 "" 0
run_one 600312.SH 电网设备 "" 0
run_one 600361.SH 工业金属 zijin_external 0
run_one 600438.SH 光伏设备 material_wind_battery 0
run_one 600487.SH 通信设备 optical_cable_grid 0
run_one 600522.SH 通信设备 optical_cable_grid 0
run_one 600584.SH 半导体 ai_compute "$ENABLE_YF_FOR_AI"
run_one 600885.SH 电网设备 storage_power 0
run_one 600919.SH 银行 "" 0
run_one 601100.SH 工程机械 aero_nuclear_equipment 0
run_one 601138.SH 消费电子 ai_compute "$ENABLE_YF_FOR_AI"
run_one 601186.SH 建筑装饰 "" 0
run_one 601336.SH 保险 "" 0
run_one 601390.SH 建筑装饰 "" 0
run_one 601818.SH 银行 "" 0
run_one 601899.SH 贵金属 zijin_external 0
run_one 601985.SH 电力 power_utility_rate 0
run_one 601991.SH 电力 power_utility_rate 0
run_one 603259.SH 医疗服务 "" 0
run_one 603308.SH 通用设备 aero_nuclear_equipment 0
run_one 603986.SH 半导体 ai_compute "$ENABLE_YF_FOR_AI"
run_one 605499.SH 饮料制造 "" 0

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[SUMMARY] ${SUMMARY_FILE}"
echo "[CANONICAL OUTPUT ROOTS] saved_data/<code>_pipeline_out"
echo "============================================================"

exit 0
