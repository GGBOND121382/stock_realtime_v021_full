#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/train_selected_models_safe.sh
#
# Optional selected/new model trainer.
# Existing model-library refresh should use:
#   bash scripts/model_library_maintenance_safe.sh retrain-existing
#
# This script does not run pipeline and does not delete data.
#
# IMPORTANT for 603308:
#   external_full is a search/summary experiment name.
#   The actual training samples are produced by the external stage under:
#     saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment/
#
# Default 603308 sample:
#   saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment/training_samples_with_aero_nuclear_equipment_external.csv
#
# Optional override:
#   EXTERNAL_FULL_SAMPLES=/path/to/training_samples.csv

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
MODELS_DIR="${MODELS_DIR:-saved_models}"
LOG_ROOT="${LOG_ROOT:-saved_data/model_update_logs/train_selected_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
APPLY="${APPLY:-0}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"
INCLUDE_VETTED_NEW="${INCLUDE_VETTED_NEW:-0}"
INCLUDE_EXTERNAL_FULL="${INCLUDE_EXTERNAL_FULL:-0}"
ONLY="${ONLY:-}"
DATE_TAG="${END_DATE//-/}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-selected_${DATE_TAG}}"
PIPELINE_603308_ROOT="${PIPELINE_603308_ROOT:-saved_data/603308_pipeline_out}"

mkdir -p "$LOG_ROOT"
REPORT="$LOG_ROOT/train_selected_report.csv"
echo "stock_code,artifact,status,samples,intraday_bars,log_path,backup_dir" > "$REPORT"

contains_only() {
  local stock="$1"
  if [[ -z "$ONLY" ]]; then return 0; fi
  local raw="${stock%%.*}"
  IFS=',' read -ra arr <<< "$ONLY"
  for x in "${arr[@]}"; do
    x="$(echo "$x" | xargs | tr '[:lower:]' '[:upper:]')"
    if [[ "$x" == "$stock" || "$x" == "$raw" ]]; then return 0; fi
  done
  return 1
}

get_603308_external_samples() {
  if [[ -n "${EXTERNAL_FULL_SAMPLES:-}" ]]; then
    echo "$EXTERNAL_FULL_SAMPLES"
    return 0
  fi

  echo "$PIPELINE_603308_ROOT/04_external/aero_nuclear_equipment/training_samples_with_aero_nuclear_equipment_external.csv"
}

save_model() {
  local stock="$1"; local artifact="$2"; local samples="$3"; local intraday="$4"
  local feature_group="$5"; local model_name="$6"; local label_mode="$7"; local entry_policy="$8"; local target_hit_bps="${9:-50}"

  if ! contains_only "$stock"; then
    echo "[SKIP ONLY] $stock $artifact"
    return 0
  fi

  local artifact_dir="$MODELS_DIR/$stock/$artifact"
  local backup_dir=""
  local safe_artifact="${artifact//[^A-Za-z0-9_]/_}"
  local log_path="$LOG_ROOT/save_${stock//./_}_${safe_artifact}.log"

  echo "============================================================"
  echo "[MODEL] $stock $artifact"
  echo "samples=$samples"
  echo "intraday=$intraday"
  echo "feature_group=$feature_group"
  echo "model_name=$model_name"
  echo "label_mode=$label_mode"
  echo "entry_policy=$entry_policy"
  echo "============================================================"

  if [[ -z "$samples" || ! -f "$samples" ]]; then
    echo "[MISSING_SAMPLES] $samples"
    echo "$stock,$artifact,missing_samples,$samples,$intraday,$log_path," >> "$REPORT"
    return 0
  fi
  if [[ -z "$intraday" || ! -f "$intraday" ]]; then
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
    if [[ "$DRY_RUN" != "1" && "$APPLY" == "1" ]]; then
      mv "$artifact_dir" "$backup_dir"
    fi
  fi

  cmd=(
    "$PYTHON" model_saving/save_nextday_model.py
    --stock-code "$stock" --artifact-name "$artifact"
    --samples "$samples" --intraday-bars "$intraday"
    --out-dir "$MODELS_DIR"
    --feature-group "$feature_group" --model-name "$model_name"
    --label-mode "$label_mode" --entry-policy "$entry_policy"
    --target-hit-bps "$target_hit_bps"
    --entry-vwap-premium-bps 50 --round-trip-cost-bps 1.7
    --valid-rows 252 --min-train-entries 80 --min-valid-trades 8
    --quantiles 0.5,0.6,0.7,0.8
  )

  printf '[RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'

  if [[ "$DRY_RUN" == "1" || "$APPLY" != "1" ]]; then
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
echo "[SELECTED TRAIN]"
echo "Default: no extra model unless INCLUDE_* flag is set"
echo "APPLY=$APPLY"
echo "PIPELINE_603308_ROOT=$PIPELINE_603308_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "============================================================"

if [[ "$INCLUDE_VETTED_NEW" == "1" ]]; then
  save_model 600522.SH "nextday_all_days_close_profit_random_forest_600_d4_reversal_fundamental_regime_sector_external_optical_cable_grid_${ARTIFACT_SUFFIX}" \
    "saved_data/600522_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
    "saved_data/600522_pipeline_out/00_base/600522_5m.csv" \
    reversal_fundamental_regime_sector_external random_forest_600_d4 close_profit all_days 50

  save_model 600487.SH "nextday_vwap_low_close_profit_extra_trees_600_d3_reversal_fundamental_regime_sector_external_optical_cable_grid_${ARTIFACT_SUFFIX}" \
    "saved_data/600487_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
    "saved_data/600487_pipeline_out/00_base/600487_5m.csv" \
    reversal_fundamental_regime_sector_external extra_trees_600_d3 close_profit vwap_low 50
fi

if [[ "$INCLUDE_EXTERNAL_FULL" == "1" ]]; then
  samples="$(get_603308_external_samples)"
  echo "[603308_EXTERNAL_SAMPLES] $samples"

  save_model 603308.SH "nextday_all_days_close_profit_xgb_d3_600_reversal_fundamental_regime_external_ane_live_board_v2_external_full_${ARTIFACT_SUFFIX}" \
    "$samples" \
    "$PIPELINE_603308_ROOT/00_base/603308_5m.csv" \
    reversal_fundamental_regime_external xgb_d3_600_lr002_mcw3 close_profit all_days 50

  save_model 603308.SH "nextday_all_days_close_profit_extra_trees_600_external_ane_live_board_v2_external_full_${ARTIFACT_SUFFIX}" \
    "$samples" \
    "$PIPELINE_603308_ROOT/00_base/603308_5m.csv" \
    external extra_trees_600_d3 close_profit all_days 50
fi

echo "[REPORT] $REPORT"
