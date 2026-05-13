#!/usr/bin/env bash
set -uo pipefail

# train_overwrite_official_models.sh
#
# Purpose:
#   Train the selected official models and overwrite the old artifacts in:
#     saved_models/<stock_code>/<artifact_name>/
#
# Design:
#   - NO run-tag by default.
#   - Pipeline output uses canonical folders:
#       saved_data/<code>_pipeline_out/
#   - Model output uses canonical folder:
#       saved_models/
#   - Existing artifact directories are removed before saving.
#   - Muyuan 002714 is kept as a low-weight/observation model.
#
# Default official set:
#   600312.SH  平高电气      main
#   601899.SH  紫金矿业      main
#   002311.SZ  海大集团      main
#   603308.SH  应流股份      main, needs v2 external patch
#   600096.SH  云天化        main
#   601985.SH  中国核电      steady
#   002714.SZ  牧原股份      observation, kept by request
#
# Optional:
#   INCLUDE_HENGRUI=1 adds 600276.SH hit80 auxiliary model.
#
# Requirements:
#   1) Run from project root.
#   2) Apply these patches first:
#        stock_external_nextday_v2_patch.zip
#        stock_external_v2_board_prefix_fix.zip
#   3) Python command is python3 by default.
#
# Example:
#   chmod +x scripts/train_overwrite_official_models.sh
#   PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/train_overwrite_official_models.sh
#
# Useful env options:
#   DRY_RUN=1                  Print commands only.
#   SKIP_PIPELINE=1            Only save from existing pipeline outputs.
#   SKIP_SAVE=1                Only run pipeline/search.
#   CLEAN_PIPELINE=0           Do not remove saved_data/<code>_pipeline_out before running.
#   ONLY=600312.SH,601899.SH   Only process selected symbols.
#   INCLUDE_HENGRUI=1          Also overwrite/save HengRui hit80 auxiliary model.
#   JOB_TIMEOUT=8h             Timeout per stock pipeline.
#
# Notes:
#   - Default CLEAN_PIPELINE=1 is intentional: it avoids mixing old outputs with the new v2 external policy.
#   - This script overwrites artifacts, not Git files.

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
MODEL_OUT_DIR="${MODEL_OUT_DIR:-saved_models}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/official_overwrite_$(date +%Y%m%d_%H%M%S)}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_SAVE="${SKIP_SAVE:-0}"
CLEAN_PIPELINE="${CLEAN_PIPELINE:-1}"
INCLUDE_HENGRUI="${INCLUDE_HENGRUI:-0}"
ONLY="${ONLY:-}"

mkdir -p "$LOG_DIR" "$MODEL_OUT_DIR"

PIPELINE_SUMMARY="$LOG_DIR/pipeline_summary.csv"
SAVE_SUMMARY="$LOG_DIR/save_summary.csv"
echo "symbol,sector,external,status,returncode,start_time,end_time,log_file" > "$PIPELINE_SUMMARY"
echo "symbol,artifact,status,returncode,samples,intraday_bars,artifact_dir,start_time,end_time,log_file" > "$SAVE_SUMMARY"

COMMON_PIPELINE_ARGS=(
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

SAVE_COMMON_ARGS=(
  --entry-vwap-premium-bps 50
  --round-trip-cost-bps 1.7
  --valid-rows 252
  --min-train-entries 80
  --min-valid-trades 8
  --quantiles 0.5,0.6,0.7,0.8
)

contains_only() {
  local symbol="$1"
  if [[ -z "$ONLY" ]]; then
    return 0
  fi
  local raw="${symbol%%.*}"
  IFS=',' read -ra arr <<< "$ONLY"
  for x in "${arr[@]}"; do
    x="$(echo "$x" | tr '[:lower:]' '[:upper:]' | xargs)"
    if [[ "$x" == "${symbol^^}" || "$x" == "$raw" ]]; then
      return 0
    fi
  done
  return 1
}

print_cmd() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
}

run_pipeline() {
  local symbol="$1"
  local sector="$2"
  local external="$3"

  if ! contains_only "$symbol"; then
    echo "[SKIP] pipeline $symbol not selected by ONLY=$ONLY"
    return 0
  fi

  if [[ "$SKIP_PIPELINE" == "1" ]]; then
    echo "[SKIP] pipeline $symbol because SKIP_PIPELINE=1"
    return 0
  fi

  local raw="${symbol%%.*}"
  local safe_symbol="${symbol//./_}"
  local ext_label="${external:-none}"
  local out_root="saved_data/${raw}_pipeline_out"
  local log_file="$LOG_DIR/pipeline_${safe_symbol}_${ext_label}.log"
  local start_time end_time rc
  local external_args=()

  if [[ -n "$external" ]]; then
    external_args+=(--external "$external")
  fi

  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[PIPELINE START] $start_time symbol=$symbol sector=$sector external=$ext_label"
  echo "[OUT_ROOT] $out_root"
  echo "============================================================"

  if [[ "$CLEAN_PIPELINE" == "1" ]]; then
    echo "[CLEAN] removing pipeline output: $out_root"
    if [[ "$DRY_RUN" != "1" ]]; then
      rm -rf "$out_root"
    fi
  fi

  local cmd=(
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py
    --symbol "$symbol"
    --sector-symbol "$sector"
    --out-root "$out_root"
    "${external_args[@]}"
    "${COMMON_PIPELINE_ARGS[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}" | tee -a "$log_file"
    rc=0
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"
  if [[ "$rc" -eq 0 ]]; then
    echo "[PIPELINE DONE] $symbol"
    echo "${symbol},${sector},${ext_label},ok,${rc},${start_time},${end_time},${log_file}" >> "$PIPELINE_SUMMARY"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[PIPELINE TIMEOUT] $symbol timeout=$JOB_TIMEOUT"
    echo "${symbol},${sector},${ext_label},timeout,${rc},${start_time},${end_time},${log_file}" >> "$PIPELINE_SUMMARY"
    return "$rc"
  else
    echo "[PIPELINE FAIL] $symbol returncode=$rc"
    echo "${symbol},${sector},${ext_label},failed,${rc},${start_time},${end_time},${log_file}" >> "$PIPELINE_SUMMARY"
    return "$rc"
  fi
}

save_model() {
  local symbol="$1"
  local artifact="$2"
  local samples="$3"
  local intraday="$4"
  local feature_group="$5"
  local model_name="$6"
  local label_mode="$7"
  local entry_policy="$8"
  local target_hit_bps="${9:-50}"

  if ! contains_only "$symbol"; then
    echo "[SKIP] save $symbol not selected by ONLY=$ONLY"
    return 0
  fi

  if [[ "$SKIP_SAVE" == "1" ]]; then
    echo "[SKIP] save $symbol because SKIP_SAVE=1"
    return 0
  fi

  local artifact_dir="$MODEL_OUT_DIR/$symbol/$artifact"
  local safe_symbol="${symbol//./_}"
  local log_file="$LOG_DIR/save_${safe_symbol}_${artifact}.log"
  local start_time end_time rc

  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[SAVE START] $start_time $symbol -> $artifact"
  echo "[ARTIFACT_DIR] $artifact_dir"
  echo "============================================================"

  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -f "$samples" ]]; then
      echo "[SAVE ERROR] missing samples: $samples"
      end_time="$(date '+%F %T')"
      echo "${symbol},${artifact},missing_samples,2,${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
      return 2
    fi
    if [[ ! -f "$intraday" ]]; then
      echo "[SAVE ERROR] missing intraday bars: $intraday"
      end_time="$(date '+%F %T')"
      echo "${symbol},${artifact},missing_intraday,2,${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
      return 2
    fi

    if [[ -d "$artifact_dir" ]]; then
      echo "[OVERWRITE] removing old artifact: $artifact_dir"
      rm -rf "$artifact_dir"
    fi
  else
    print_cmd rm -rf "$artifact_dir"
  fi

  local cmd=(
    "$PYTHON" model_saving/save_nextday_model.py
    --stock-code "$symbol"
    --artifact-name "$artifact"
    --samples "$samples"
    --intraday-bars "$intraday"
    --out-dir "$MODEL_OUT_DIR"
    --feature-group "$feature_group"
    --model-name "$model_name"
    --label-mode "$label_mode"
    --entry-policy "$entry_policy"
    --target-hit-bps "$target_hit_bps"
    "${SAVE_COMMON_ARGS[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}" | tee -a "$log_file"
    rc=0
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"
  if [[ "$rc" -eq 0 ]]; then
    echo "[SAVE DONE] $symbol -> $artifact"
    echo "${symbol},${artifact},ok,${rc},${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
    return 0
  else
    echo "[SAVE FAIL] $symbol -> $artifact returncode=$rc"
    echo "${symbol},${artifact},failed,${rc},${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
    return "$rc"
  fi
}

PIPE_FAILED=0
SAVE_FAILED=0

# -----------------------------
# 1) Pipeline/search stage
# -----------------------------

run_pipeline 600312.SH 电网设备     ""                       || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 601899.SH 贵金属       "zijin_external"         || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 002311.SZ 农产品加工   "feed,hog"               || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 603308.SH 通用设备     "aero_nuclear_equipment" || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 600096.SH 农化制品     "fertilizer"             || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 601985.SH 电力         ""                       || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 002714.SZ 养殖业       "hog,muyuan_hk"          || PIPE_FAILED=$((PIPE_FAILED + 1))

if [[ "$INCLUDE_HENGRUI" == "1" ]]; then
  run_pipeline 600276.SH 化学制药   ""                       || PIPE_FAILED=$((PIPE_FAILED + 1))
fi

# -----------------------------
# 2) Save selected official artifacts
# -----------------------------

# 平高电气：覆盖旧主模型
save_model \
  600312.SH \
  nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_v1 \
  "saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/600312_pipeline_out/00_base/600312_5m.csv" \
  reversal_fundamental_regime \
  xgb_d3_400_lr003_mcw3 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 紫金矿业：覆盖旧 vwap_low ExtraTrees 主模型
save_model \
  601899.SH \
  nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_zijin_v1 \
  "saved_data/601899_pipeline_out/04_external/zijin_external/training_samples_with_zijin_external.csv" \
  "saved_data/601899_pipeline_out/00_base/601899_5m.csv" \
  reversal_fundamental_regime_sector \
  extra_trees_600_d3 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 海大集团：覆盖旧 hog 主模型
save_model \
  002311.SZ \
  nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_sector_hog_v1 \
  "saved_data/002311_pipeline_out/04_external/hog/training_samples_with_hog_industry.csv" \
  "saved_data/002311_pipeline_out/00_base/002311_5m.csv" \
  reversal_fundamental_regime_sector \
  xgb_d3_600_lr002_mcw3 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 应流股份：新增正式模型
save_model \
  603308.SH \
  nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_sector_external_aero_nuclear_v1 \
  "saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment/training_samples_with_aero_nuclear_equipment_external.csv" \
  "saved_data/603308_pipeline_out/00_base/603308_5m.csv" \
  reversal_fundamental_regime_sector_external \
  xgb_d3_400_lr003_mcw3 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 云天化：新增正式模型；pipeline 生成 fertilizer 供后续复核，但保存当前更稳的 sector 版本
save_model \
  600096.SH \
  nextday_vwap_low_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
  "saved_data/600096_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/600096_pipeline_out/00_base/600096_5m.csv" \
  reversal_fundamental_regime_sector \
  xgb_d4_500_lr002_mcw5 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 中国核电：新增正式模型
save_model \
  601985.SH \
  nextday_all_days_close_profit_extra_trees_reversal_fundamental_regime_v1 \
  "saved_data/601985_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/601985_pipeline_out/00_base/601985_5m.csv" \
  reversal_fundamental_regime \
  extra_trees_600_d3 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 牧原股份：保留一个观察模型，覆盖旧 artifact
save_model \
  002714.SZ \
  nextday_vwap_low_close_profit_random_forest_reversal_fundamental_regime_sector_muyuan_hk_v1 \
  "saved_data/002714_pipeline_out/04_external/muyuan_hk/training_samples_with_muyuan_hk_external.csv" \
  "saved_data/002714_pipeline_out/00_base/002714_5m.csv" \
  reversal_fundamental_regime_sector \
  random_forest_600_d4 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# 恒瑞医药：可选 hit80 辅助模型
if [[ "$INCLUDE_HENGRUI" == "1" ]]; then
  save_model \
    600276.SH \
    nextday_all_days_hit80_extra_trees_reversal_fundamental_regime_sector_v1 \
    "saved_data/600276_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/600276_pipeline_out/00_base/600276_5m.csv" \
    reversal_fundamental_regime_sector \
    extra_trees_600_d3 \
    hit \
    all_days \
    80 || SAVE_FAILED=$((SAVE_FAILED + 1))
fi

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[MODEL_OUT_DIR] $MODEL_OUT_DIR"
echo "[PIPELINE_FAILED] $PIPE_FAILED"
echo "[SAVE_FAILED] $SAVE_FAILED"
echo "[PIPELINE_SUMMARY] $PIPELINE_SUMMARY"
echo "[SAVE_SUMMARY] $SAVE_SUMMARY"
echo "============================================================"

if [[ "$PIPE_FAILED" -gt 0 || "$SAVE_FAILED" -gt 0 ]]; then
  exit 1
fi
exit 0
