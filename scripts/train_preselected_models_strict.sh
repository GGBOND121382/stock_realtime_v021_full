#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/train_preselected_models_strict.sh
#
# Periodic retraining script for selected models only.
#
# It trains from existing/latest sample files under saved_data.
# It does NOT run pipeline.
# It does NOT scan leaderboard.
# It does NOT delete data.
#
# Defaults:
#   - train/save only 603308.SH selected models
#
# Optional:
#   INCLUDE_VETTED_NEW=1       add 600522.SH and 600487.SH vetted models
#   INCLUDE_EXISTING_STRONG=1  add 600312.SH and 601899.SH strong existing models
#   INCLUDE_600487_HIT=1       add 600487.SH hit80 auxiliary model
#   INCLUDE_RISKY_002518=1     add 002518.SZ high-drawdown model; off by default
#
# Usage:
#   DRY_RUN=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/train_preselected_models_strict.sh
#   PYTHON=python3 END_DATE=2026-05-15 bash scripts/train_preselected_models_strict.sh

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
MODELS_DIR="${MODELS_DIR:-saved_models}"
LOG_ROOT="${LOG_ROOT:-saved_data/model_update_logs/preselected_strict_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"
ONLY="${ONLY:-}"

INCLUDE_VETTED_NEW="${INCLUDE_VETTED_NEW:-0}"
INCLUDE_EXISTING_STRONG="${INCLUDE_EXISTING_STRONG:-0}"
INCLUDE_600487_HIT="${INCLUDE_600487_HIT:-0}"
INCLUDE_RISKY_002518="${INCLUDE_RISKY_002518:-0}"

DATE_TAG="${END_DATE//-/}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-strict_${DATE_TAG}}"

mkdir -p "$LOG_ROOT"
REPORT="$LOG_ROOT/preselected_strict_train_report.csv"
echo "stock_code,artifact,status,samples,intraday_bars,log_path,backup_dir" > "$REPORT"

contains_only() {
  local stock="$1"
  if [[ -z "$ONLY" ]]; then
    return 0
  fi
  local raw="${stock%%.*}"
  IFS=',' read -ra arr <<< "$ONLY"
  for x in "${arr[@]}"; do
    x="$(echo "$x" | xargs | tr '[:lower:]' '[:upper:]')"
    if [[ "$x" == "$stock" || "$x" == "$raw" ]]; then
      return 0
    fi
  done
  return 1
}

save_model() {
  local stock="$1"
  local artifact="$2"
  local samples="$3"
  local intraday="$4"
  local feature_group="$5"
  local model_name="$6"
  local label_mode="$7"
  local entry_policy="$8"
  local target_hit_bps="${9:-50}"

  if ! contains_only "$stock"; then
    echo "[SKIP ONLY] $stock $artifact"
    return 0
  fi

  local artifact_dir="$MODELS_DIR/$stock/$artifact"
  local backup_dir=""
  local safe_artifact="${artifact//[^A-Za-z0-9_]/_}"
  local log_path="$LOG_ROOT/save_${stock//./_}_${safe_artifact}.log"

  echo "============================================================"
  echo "[MODEL] $stock -> $artifact"
  echo "samples=$samples"
  echo "intraday=$intraday"
  echo "feature_group=$feature_group"
  echo "model_name=$model_name"
  echo "label_mode=$label_mode"
  echo "entry_policy=$entry_policy"
  echo "============================================================"

  if [[ ! -f "$samples" ]]; then
    echo "[MISSING_SAMPLES] $samples"
    echo "$stock,$artifact,missing_samples,$samples,$intraday,$log_path," >> "$REPORT"
    return 0
  fi

  if [[ ! -f "$intraday" ]]; then
    echo "[MISSING_INTRADAY] $intraday"
    echo "$stock,$artifact,missing_intraday,$samples,$intraday,$log_path," >> "$REPORT"
    return 0
  fi

  if [[ -d "$artifact_dir" ]]; then
    if [[ "$OVERWRITE_EXISTING" != "1" ]]; then
      echo "[SKIP_EXISTING] $artifact_dir"
      echo "$stock,$artifact,skipped_existing,$samples,$intraday,$log_path," >> "$REPORT"
      return 0
    fi
    backup_dir="$LOG_ROOT/existing_model_backups/$stock/$artifact"
    mkdir -p "$(dirname "$backup_dir")"
    echo "[BACKUP_EXISTING] $artifact_dir -> $backup_dir"
    if [[ "$DRY_RUN" != "1" ]]; then
      mv "$artifact_dir" "$backup_dir"
    fi
  fi

  cmd=(
    "$PYTHON" model_saving/save_nextday_model.py
    --stock-code "$stock"
    --artifact-name "$artifact"
    --samples "$samples"
    --intraday-bars "$intraday"
    --out-dir "$MODELS_DIR"
    --feature-group "$feature_group"
    --model-name "$model_name"
    --label-mode "$label_mode"
    --entry-policy "$entry_policy"
    --target-hit-bps "$target_hit_bps"
    --entry-vwap-premium-bps 50
    --round-trip-cost-bps 1.7
    --valid-rows 252
    --min-train-entries 80
    --min-valid-trades 8
    --quantiles 0.5,0.6,0.7,0.8
  )

  printf '[RUN]'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$stock,$artifact,dry_run,$samples,$intraday,$log_path,$backup_dir" >> "$REPORT"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "$log_path"
  rc=${PIPESTATUS[0]}
  if [[ "$rc" == "0" ]]; then
    echo "$stock,$artifact,ok,$samples,$intraday,$log_path,$backup_dir" >> "$REPORT"
  else
    echo "$stock,$artifact,failed_rc_${rc},$samples,$intraday,$log_path,$backup_dir" >> "$REPORT"
  fi
}

echo "============================================================"
echo "[STRICT PRESELECTED MODEL TRAINER]"
echo "Default: only 603308.SH."
echo "No automatic picking. No pipeline run. No data removal."
echo "END_DATE=$END_DATE"
echo "ARTIFACT_SUFFIX=$ARTIFACT_SUFFIX"
echo "LOG_ROOT=$LOG_ROOT"
echo "ONLY=$ONLY"
echo "INCLUDE_VETTED_NEW=$INCLUDE_VETTED_NEW"
echo "INCLUDE_EXISTING_STRONG=$INCLUDE_EXISTING_STRONG"
echo "INCLUDE_600487_HIT=$INCLUDE_600487_HIT"
echo "INCLUDE_RISKY_002518=$INCLUDE_RISKY_002518"
echo "============================================================"

# Default: 603308 only.
save_model \
  603308.SH \
  "nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_external_aero_nuclear_${ARTIFACT_SUFFIX}" \
  "saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment/training_samples_with_aero_nuclear_equipment_external.csv" \
  "saved_data/603308_pipeline_out/00_base/603308_5m.csv" \
  reversal_fundamental_regime_sector_external \
  xgb_d3_400_lr003_mcw3 \
  close_profit \
  all_days \
  50

save_model \
  603308.SH \
  "nextday_all_days_close_profit_xgb_d4_reversal_fundamental_regime_sector_${ARTIFACT_SUFFIX}" \
  "saved_data/603308_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/603308_pipeline_out/00_base/603308_5m.csv" \
  reversal_fundamental_regime_sector \
  xgb_d4_500_lr002_mcw5 \
  close_profit \
  all_days \
  50

if [[ "$INCLUDE_VETTED_NEW" == "1" ]]; then
  save_model \
    600522.SH \
    "nextday_all_days_close_profit_random_forest_600_d4_reversal_fundamental_regime_sector_external_optical_cable_grid_${ARTIFACT_SUFFIX}" \
    "saved_data/600522_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
    "saved_data/600522_pipeline_out/00_base/600522_5m.csv" \
    reversal_fundamental_regime_sector_external \
    random_forest_600_d4 \
    close_profit \
    all_days \
    50

  save_model \
    600487.SH \
    "nextday_vwap_low_close_profit_extra_trees_600_d3_reversal_fundamental_regime_sector_external_optical_cable_grid_${ARTIFACT_SUFFIX}" \
    "saved_data/600487_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
    "saved_data/600487_pipeline_out/00_base/600487_5m.csv" \
    reversal_fundamental_regime_sector_external \
    extra_trees_600_d3 \
    close_profit \
    vwap_low \
    50
fi

if [[ "$INCLUDE_EXISTING_STRONG" == "1" ]]; then
  save_model \
    600312.SH \
    "nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_${ARTIFACT_SUFFIX}" \
    "saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/600312_pipeline_out/00_base/600312_5m.csv" \
    reversal_fundamental_regime \
    xgb_d3_600_lr002_mcw3 \
    close_profit \
    vwap_low \
    50

  save_model \
    600312.SH \
    "nextday_all_days_close_profit_xgb_d3_600_reversal_fundamental_regime_${ARTIFACT_SUFFIX}" \
    "saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/600312_pipeline_out/00_base/600312_5m.csv" \
    reversal_fundamental_regime \
    xgb_d3_600_lr002_mcw3 \
    close_profit \
    all_days \
    50

  save_model \
    601899.SH \
    "nextday_vwap_low_close_profit_extra_trees_600_d3_reversal_fundamental_regime_sector_zijin_${ARTIFACT_SUFFIX}" \
    "saved_data/601899_pipeline_out/04_external/zijin_external/training_samples_with_zijin_external.csv" \
    "saved_data/601899_pipeline_out/00_base/601899_5m.csv" \
    reversal_fundamental_regime_sector \
    extra_trees_600_d3 \
    close_profit \
    vwap_low \
    50
fi

if [[ "$INCLUDE_600487_HIT" == "1" ]]; then
  save_model \
    600487.SH \
    "nextday_all_days_hit80_xgb_d2_200_reversal_fundamental_regime_optical_cable_grid_${ARTIFACT_SUFFIX}" \
    "saved_data/600487_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
    "saved_data/600487_pipeline_out/00_base/600487_5m.csv" \
    reversal_fundamental_regime \
    xgb_d2_200_lr003_mcw5 \
    hit \
    all_days \
    80
fi

if [[ "$INCLUDE_RISKY_002518" == "1" ]]; then
  save_model \
    002518.SZ \
    "nextday_all_days_close_profit_xgb_d2_200_reversal_fundamental_regime_sector_storage_power_${ARTIFACT_SUFFIX}" \
    "saved_data/002518_pipeline_out/04_external/storage_power/training_samples_with_storage_power_external.csv" \
    "saved_data/002518_pipeline_out/00_base/002518_5m.csv" \
    reversal_fundamental_regime_sector \
    xgb_d2_200_lr003_mcw5 \
    close_profit \
    all_days \
    50
fi

echo "============================================================"
echo "[DONE]"
echo "[REPORT] $REPORT"
echo "============================================================"
