#!/usr/bin/env bash
set -uo pipefail

# canonicalize_rebuild_and_save_models_only_stages.sh
#
# FINAL SIMPLE FLOW:
#
#   1) Copy/merge:
#        legacy suffixed pipeline dirs
#      into canonical:
#        saved_data/<code>_pipeline_out/
#
#   2) Rebuild/complete data in canonical folders using:
#        run_nextday_pipeline.py --only-stages update_data,samples,fundamental,sector[,external_<external_name>]
#
#      This does NOT run model search.
#
#   3) Train/save already-selected fixed artifacts into:
#        saved_models/<stock_code>/<artifact_name>/
#
# No selected_good.
# No production run-tag folders.
# No fallback search.
# No deletion of v2_all14 by default.

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
MODEL_OUT_DIR="${MODEL_OUT_DIR:-saved_models}"

SOURCE_SUFFIX="${SOURCE_SUFFIX:-_v2_all14}"
DELETE_SOURCE="${DELETE_SOURCE:-0}"
SYNC_DELETE="${SYNC_DELETE:-1}"
SKIP_MIGRATE="${SKIP_MIGRATE:-0}"
SKIP_REBUILD="${SKIP_REBUILD:-0}"
SKIP_SAVE="${SKIP_SAVE:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
CLEAN_MODEL_STOCK_DIR="${CLEAN_MODEL_STOCK_DIR:-1}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"

TRAIN_THREADS="${TRAIN_THREADS:-1}"
export OMP_NUM_THREADS="$TRAIN_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_THREADS"
export MKL_NUM_THREADS="$TRAIN_THREADS"
export NUMEXPR_NUM_THREADS="$TRAIN_THREADS"

LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/canonicalize_only_stages_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" "$MODEL_OUT_DIR"

MIGRATE_SUMMARY="$LOG_DIR/migrate_summary.csv"
REBUILD_SUMMARY="$LOG_DIR/rebuild_summary.csv"
SAVE_SUMMARY="$LOG_DIR/save_summary.csv"
echo "symbol,source,destination,status,start_time,end_time" > "$MIGRATE_SUMMARY"
echo "symbol,sector,external,only_stages,status,returncode,start_time,end_time,log_file" > "$REBUILD_SUMMARY"
echo "symbol,artifact,status,returncode,samples,intraday_bars,artifact_dir,start_time,end_time,log_file" > "$SAVE_SUMMARY"

SYMBOLS=(
  600312.SH
  601899.SH
  603308.SH
  600096.SH
  002311.SZ
  601985.SH
  600276.SH
  002714.SZ
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

run_cmd_or_dry() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "$@"
    return 0
  fi
  "$@"
}

pipeline_root_for_symbol() {
  local symbol="$1"
  local raw="${symbol%%.*}"
  echo "saved_data/${raw}_pipeline_out"
}

source_root_for_symbol() {
  local symbol="$1"
  local raw="${symbol%%.*}"
  echo "saved_data/${raw}_pipeline_out${SOURCE_SUFFIX}"
}

path_samples_sector() {
  local symbol="$1"
  echo "$(pipeline_root_for_symbol "$symbol")/03_sector/training_samples_with_sector.csv"
}

path_samples_external() {
  local symbol="$1"
  local external="$2"
  local filename="$3"
  echo "$(pipeline_root_for_symbol "$symbol")/04_external/${external}/${filename}"
}

path_intraday() {
  local symbol="$1"
  local raw="${symbol%%.*}"
  echo "$(pipeline_root_for_symbol "$symbol")/00_base/${raw}_5m.csv"
}

require_only_stages() {
  local help_text
  help_text="$("$PYTHON" pipelines/run_nextday_pipeline.py --help 2>/dev/null || true)"
  if ! echo "$help_text" | grep -q -- "--only-stages"; then
    echo "[ERROR] pipelines/run_nextday_pipeline.py does not expose --only-stages." >&2
    echo "[ERROR] This script refuses to rebuild data by running search." >&2
    echo "[ERROR] Add --only-stages support first, or use an older manual pipeline command intentionally." >&2
    return 2
  fi
  return 0
}

migrate_one() {
  local symbol="$1"
  if ! contains_only "$symbol"; then
    echo "[SKIP] migrate $symbol not selected by ONLY=$ONLY"
    return 0
  fi
  if [[ "$SKIP_MIGRATE" == "1" ]]; then
    echo "[SKIP] migrate $symbol because SKIP_MIGRATE=1"
    return 0
  fi

  local src dst start_time end_time
  src="$(source_root_for_symbol "$symbol")"
  dst="$(pipeline_root_for_symbol "$symbol")"
  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[MIGRATE] $symbol"
  echo "[FROM] $src"
  echo "[TO]   $dst"
  echo "============================================================"

  if [[ ! -d "$src" ]]; then
    echo "[MIGRATE SKIP] source not found: $src"
    end_time="$(date '+%F %T')"
    echo "${symbol},${src},${dst},source_missing,${start_time},${end_time}" >> "$MIGRATE_SUMMARY"
    return 0
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$dst"
  else
    print_cmd mkdir -p "$dst"
  fi

  if command -v rsync >/dev/null 2>&1; then
    local rsync_args=(-a)
    if [[ "$SYNC_DELETE" == "1" ]]; then
      rsync_args+=(--delete)
    fi
    run_cmd_or_dry rsync "${rsync_args[@]}" "$src"/ "$dst"/
  else
    if [[ "$SYNC_DELETE" == "1" ]]; then
      run_cmd_or_dry rm -rf "$dst"
      run_cmd_or_dry mkdir -p "$dst"
    fi
    run_cmd_or_dry cp -a "$src"/. "$dst"/
  fi

  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "[MIGRATE FAIL] $symbol rc=$rc"
    end_time="$(date '+%F %T')"
    echo "${symbol},${src},${dst},failed:${rc},${start_time},${end_time}" >> "$MIGRATE_SUMMARY"
    return "$rc"
  fi

  if [[ "$DELETE_SOURCE" == "1" ]]; then
    echo "[DELETE SOURCE] $src"
    run_cmd_or_dry rm -rf "$src"
  fi

  end_time="$(date '+%F %T')"
  echo "${symbol},${src},${dst},ok,${start_time},${end_time}" >> "$MIGRATE_SUMMARY"
  return 0
}

rebuild_data() {
  local symbol="$1"
  local sector="$2"
  local external="$3"
  local only_stages="$4"

  if ! contains_only "$symbol"; then
    echo "[SKIP] rebuild $symbol not selected by ONLY=$ONLY"
    return 0
  fi
  if [[ "$SKIP_REBUILD" == "1" ]]; then
    echo "[SKIP] rebuild $symbol because SKIP_REBUILD=1"
    return 0
  fi

  local safe_symbol="${symbol//./_}"
  local ext_label="${external:-none}"
  local out_root
  out_root="$(pipeline_root_for_symbol "$symbol")"

  local log_file="$LOG_DIR/rebuild_${safe_symbol}_${ext_label}.log"
  local start_time end_time rc
  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[REBUILD DATA ONLY] $symbol sector=$sector external=$ext_label"
  echo "[ONLY_STAGES] $only_stages"
  echo "[CANONICAL OUT_ROOT] $out_root"
  echo "============================================================"

  if [[ "$FORCE_REBUILD" == "1" ]]; then
    echo "[CLEAN CANONICAL BEFORE REBUILD] $out_root"
    run_cmd_or_dry rm -rf "$out_root"
  fi

  local external_args=()
  if [[ -n "$external" ]]; then
    external_args+=(--external "$external")
  fi

  local cmd=(
    timeout --foreground "$JOB_TIMEOUT"
    "$PYTHON" pipelines/run_nextday_pipeline.py
    --symbol "$symbol"
    --sector-symbol "$sector"
    "${external_args[@]}"
    --feature-pipeline fundamental,sector
    --only-stages "$only_stages"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --external-lag-days 1
    --stock-external-domestic-lag-days 0
    --stock-external-future-lag-days 1
    --stock-external-us-lag-days 1
    --resume
    --excel
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
    echo "[REBUILD DONE] $symbol"
    echo "${symbol},${sector},${ext_label},${only_stages},ok,${rc},${start_time},${end_time},${log_file}" >> "$REBUILD_SUMMARY"
    return 0
  else
    echo "[REBUILD FAIL] $symbol rc=$rc"
    echo "${symbol},${sector},${ext_label},${only_stages},failed,${rc},${start_time},${end_time},${log_file}" >> "$REBUILD_SUMMARY"
    return "$rc"
  fi
}

clean_model_stock_dir_once() {
  local symbol="$1"
  if ! contains_only "$symbol"; then
    return 0
  fi
  if [[ "$SKIP_SAVE" == "1" ]]; then
    return 0
  fi
  if [[ "$CLEAN_MODEL_STOCK_DIR" != "1" ]]; then
    return 0
  fi

  local stock_dir="$MODEL_OUT_DIR/$symbol"
  echo
  echo "============================================================"
  echo "[CLEAN MODEL STOCK DIR] $stock_dir"
  echo "============================================================"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd rm -rf "$stock_dir"
    print_cmd mkdir -p "$stock_dir"
  else
    rm -rf "$stock_dir"
    mkdir -p "$stock_dir"
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
  local safe_artifact
  safe_artifact="$(echo "$artifact" | tr '/:' '__')"
  local log_file="$LOG_DIR/save_${safe_symbol}_${safe_artifact}.log"
  local start_time end_time rc
  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[SAVE MODEL] $symbol -> $artifact"
  echo "[SAMPLES] $samples"
  echo "[INTRADAY] $intraday"
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
      echo "[OVERWRITE] $artifact_dir"
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
    echo "[SAVE FAIL] $symbol -> $artifact rc=$rc"
    echo "${symbol},${artifact},failed,${rc},${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
    return "$rc"
  fi
}

MIGRATE_FAILED=0
REBUILD_FAILED=0
SAVE_FAILED=0

echo
echo "============================================================"
echo "[FINAL CANONICAL DATA-ONLY FLOW]"
echo "migrate from:  saved_data/<code>_pipeline_out${SOURCE_SUFFIX}/"
echo "canonical:     saved_data/<code>_pipeline_out/"
echo "models:        $MODEL_OUT_DIR/"
echo "v2 source kept: DELETE_SOURCE=$DELETE_SOURCE"
echo "============================================================"

require_only_stages || exit 2

# Step 1: copy/merge v2_all14 into canonical. Source is kept by default.
for s in "${SYMBOLS[@]}"; do
  migrate_one "$s" || MIGRATE_FAILED=$((MIGRATE_FAILED + 1))
done

# Step 2: rebuild/complete required data only.
rebuild_data 600312.SH 电网设备 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 601899.SH 贵金属 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 603308.SH 通用设备 "aero_nuclear_equipment" "update_data,samples,fundamental,sector,external_aero_nuclear_equipment" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 600096.SH 农化制品 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 002311.SZ 农产品加工 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 601985.SH 电力 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 600276.SH 化学制药 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

rebuild_data 002714.SZ 养殖业 "" "update_data,samples,fundamental,sector" \
  || REBUILD_FAILED=$((REBUILD_FAILED + 1))

# Step 3: save selected fixed models.
for s in "${SYMBOLS[@]}"; do
  clean_model_stock_dir_once "$s"
done

# 600312 平高
save_model 600312.SH nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_v1 \
  "$(path_samples_sector 600312.SH)" "$(path_intraday 600312.SH)" \
  reversal_fundamental_regime xgb_d3_600_lr002_mcw3 close_profit vwap_low 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 600312.SH nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 600312.SH)" "$(path_intraday 600312.SH)" \
  reversal_fundamental_regime_sector xgb_d3_400_lr003_mcw3 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 600312.SH nextday_all_days_close_profit_xgb_d3_600_reversal_fundamental_regime_v1 \
  "$(path_samples_sector 600312.SH)" "$(path_intraday 600312.SH)" \
  reversal_fundamental_regime xgb_d3_600_lr002_mcw3 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 601899 紫金
save_model 601899.SH nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_zijin_v1 \
  "$(path_samples_sector 601899.SH)" "$(path_intraday 601899.SH)" \
  reversal_fundamental_regime_sector extra_trees_600_d3 close_profit vwap_low 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 601899.SH nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 601899.SH)" "$(path_intraday 601899.SH)" \
  reversal_fundamental_regime_sector xgb_d3_400_lr003_mcw3 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 603308 应流
save_model 603308.SH nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_external_aero_nuclear_v1 \
  "$(path_samples_external 603308.SH aero_nuclear_equipment training_samples_with_aero_nuclear_equipment_external.csv)" "$(path_intraday 603308.SH)" \
  reversal_fundamental_regime_sector_external xgb_d3_400_lr003_mcw3 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 603308.SH nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_sector_external_aero_nuclear_v1 \
  "$(path_samples_external 603308.SH aero_nuclear_equipment training_samples_with_aero_nuclear_equipment_external.csv)" "$(path_intraday 603308.SH)" \
  reversal_fundamental_regime_sector_external xgb_d3_600_lr002_mcw3 close_profit vwap_low 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 603308.SH nextday_all_days_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 603308.SH)" "$(path_intraday 603308.SH)" \
  reversal_fundamental_regime_sector xgb_d4_500_lr002_mcw5 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 600096 云天化
save_model 600096.SH nextday_vwap_low_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 600096.SH)" "$(path_intraday 600096.SH)" \
  reversal_fundamental_regime_sector xgb_d4_500_lr002_mcw5 close_profit vwap_low 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 600096.SH nextday_all_days_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 600096.SH)" "$(path_intraday 600096.SH)" \
  reversal_fundamental_regime_sector xgb_d4_500_lr002_mcw5 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 600096.SH nextday_all_days_hit80_xgb_d3_400_reversal_fundamental_regime_v1 \
  "$(path_samples_sector 600096.SH)" "$(path_intraday 600096.SH)" \
  reversal_fundamental_regime xgb_d3_400_lr003_mcw3 hit all_days 80 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 002311 海大
save_model 002311.SZ nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 002311.SZ)" "$(path_intraday 002311.SZ)" \
  reversal_fundamental_regime_sector xgb_d3_600_lr002_mcw3 close_profit vwap_low 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 002311.SZ nextday_all_days_close_profit_random_forest_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 002311.SZ)" "$(path_intraday 002311.SZ)" \
  reversal_fundamental_regime_sector random_forest_600_d4 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 002311.SZ nextday_all_days_hit80_lgbm_leaves15_reversal_fundamental_regime_v1 \
  "$(path_samples_sector 002311.SZ)" "$(path_intraday 002311.SZ)" \
  reversal_fundamental_regime lgbm_leaves15_700 hit all_days 80 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))
save_model 002311.SZ nextday_vwap_low_hit80_lgbm_leaves15_reversal_fundamental_regime_v1 \
  "$(path_samples_sector 002311.SZ)" "$(path_intraday 002311.SZ)" \
  reversal_fundamental_regime lgbm_leaves15_700 hit vwap_low 80 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 601985 中国核电
save_model 601985.SH nextday_all_days_close_profit_extra_trees_reversal_fundamental_regime_v1 \
  "$(path_samples_sector 601985.SH)" "$(path_intraday 601985.SH)" \
  reversal_fundamental_regime extra_trees_600_d3 close_profit all_days 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 600276 恒瑞
save_model 600276.SH nextday_all_days_hit80_extra_trees_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 600276.SH)" "$(path_intraday 600276.SH)" \
  reversal_fundamental_regime_sector extra_trees_600_d3 hit all_days 80 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

# 002714 牧原
save_model 002714.SZ nextday_vwap_low_close_profit_random_forest_reversal_fundamental_regime_sector_v1 \
  "$(path_samples_sector 002714.SZ)" "$(path_intraday 002714.SZ)" \
  reversal_fundamental_regime_sector random_forest_600_d4 close_profit vwap_low 50 \
  || SAVE_FAILED=$((SAVE_FAILED + 1))

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[MIGRATE_FAILED] $MIGRATE_FAILED"
echo "[REBUILD_FAILED] $REBUILD_FAILED"
echo "[SAVE_FAILED] $SAVE_FAILED"
echo "[MIGRATE_SUMMARY] $MIGRATE_SUMMARY"
echo "[REBUILD_SUMMARY] $REBUILD_SUMMARY"
echo "[SAVE_SUMMARY] $SAVE_SUMMARY"
echo "============================================================"

if [[ "$MIGRATE_FAILED" -gt 0 || "$REBUILD_FAILED" -gt 0 || "$SAVE_FAILED" -gt 0 ]]; then
  exit 1
fi
exit 0
