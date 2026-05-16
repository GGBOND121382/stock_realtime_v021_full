#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/rebuild_603308_pipeline_safe.sh
#
# Rebuild ONLY the data needed to train/save 603308.SH models.
#
# IMPORTANT:
#   This script DOES NOT run model search.
#   It rebuilds only:
#     - 00_base / raw bars
#     - samples
#     - fundamental
#     - sector
#     - external_aero_nuclear_equipment
#   Then it directly trains/saves selected 603308 models via save_nextday_model.py.
#
# It refuses any OUT_ROOT other than saved_data/603308_pipeline_out.
# It does not remove pipeline folders.

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-2026-05-15}"
OUT_ROOT="${OUT_ROOT:-saved_data/603308_pipeline_out}"
MODELS_DIR="${MODELS_DIR:-saved_models}"
LOG_ROOT="${LOG_ROOT:-saved_data/model_update_logs/rebuild_603308_no_search_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DATA_REBUILD="${SKIP_DATA_REBUILD:-0}"
SAVE_SECTOR_MODEL="${SAVE_SECTOR_MODEL:-1}"
SAVE_EXTERNAL_MODEL="${SAVE_EXTERNAL_MODEL:-1}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-603308_rebuilt_data_${END_DATE//-/}}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

STOCK="603308.SH"
SECTOR="通用设备"
EXTERNAL="aero_nuclear_equipment"

EXTERNAL_SAMPLES="$OUT_ROOT/04_external/$EXTERNAL/training_samples_with_${EXTERNAL}_external.csv"
SECTOR_SAMPLES="$OUT_ROOT/03_sector/training_samples_with_sector.csv"
INTRADAY_BARS="$OUT_ROOT/00_base/603308_5m.csv"

EXT_ARTIFACT="nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_external_aero_nuclear_${ARTIFACT_SUFFIX}"
SECTOR_ARTIFACT="nextday_all_days_close_profit_xgb_d4_reversal_fundamental_regime_sector_${ARTIFACT_SUFFIX}"

die() {
  echo "[FATAL] $*" >&2
  exit 1
}

run_cmd() {
  printf '[RUN]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

[[ "$OUT_ROOT" == "saved_data/603308_pipeline_out" ]] || die "OUT_ROOT must be saved_data/603308_pipeline_out, got: $OUT_ROOT"

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "[603308 DATA REBUILD + MODEL TRAIN]"
echo "No model search will be run."
echo "STOCK=$STOCK"
echo "OUT_ROOT=$OUT_ROOT"
echo "MODELS_DIR=$MODELS_DIR"
echo "END_DATE=$END_DATE"
echo "LOG_ROOT=$LOG_ROOT"
echo "SAVE_EXTERNAL_MODEL=$SAVE_EXTERNAL_MODEL"
echo "SAVE_SECTOR_MODEL=$SAVE_SECTOR_MODEL"
echo "ARTIFACT_SUFFIX=$ARTIFACT_SUFFIX"
echo "============================================================"

if [[ "$SKIP_DATA_REBUILD" != "1" ]]; then
  echo "[STEP] Rebuild only training data stages for 603308"
  cmd=(
    "$PYTHON" pipelines/run_nextday_pipeline.py
    --symbol "$STOCK"
    --sector-symbol "$SECTOR"
    --out-root "$OUT_ROOT"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --cache-mode incremental
    --feature-cache-mode incremental
    --feature-pipeline fundamental,sector
    --external "$EXTERNAL"
    --external-lag-days 1
    --stock-external-domestic-lag-days 0
    --stock-external-future-lag-days 1
    --stock-external-us-lag-days 1
    --only-stages update_data,samples,fundamental,sector,external_aero_nuclear_equipment
    --resume
  )

  printf '[RUN]'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/rebuild_603308_training_data.log"
    rc=${PIPESTATUS[0]}
    [[ "$rc" == "0" ]] || die "training data rebuild failed; log=$LOG_ROOT/rebuild_603308_training_data.log"
  fi
else
  echo "[SKIP] data rebuild because SKIP_DATA_REBUILD=1"
fi

echo "[STEP] Check required training files"
missing=0
for f in "$INTRADAY_BARS" "$EXTERNAL_SAMPLES"; do
  if [[ -f "$f" ]]; then
    echo "[OK] $f"
  else
    echo "[MISSING] $f"
    missing=1
  fi
done
if [[ "$SAVE_SECTOR_MODEL" == "1" ]]; then
  if [[ -f "$SECTOR_SAMPLES" ]]; then
    echo "[OK] $SECTOR_SAMPLES"
  else
    echo "[MISSING] $SECTOR_SAMPLES"
    missing=1
  fi
fi
[[ "$missing" == "0" || "$DRY_RUN" == "1" ]] || die "required 603308 training files missing"

backup_existing_artifact() {
  local artifact="$1"
  local artifact_dir="$MODELS_DIR/$STOCK/$artifact"
  if [[ ! -d "$artifact_dir" ]]; then
    return 0
  fi
  if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    die "artifact already exists: $artifact_dir. Use a new ARTIFACT_SUFFIX or set ALLOW_OVERWRITE=1."
  fi
  local backup_dir="$LOG_ROOT/existing_model_backups/$artifact"
  mkdir -p "$(dirname "$backup_dir")"
  echo "[BACKUP] $artifact_dir -> $backup_dir"
  if [[ "$DRY_RUN" != "1" ]]; then
    mv "$artifact_dir" "$backup_dir"
  fi
}

save_model() {
  local artifact="$1"
  local samples="$2"
  local feature_group="$3"
  local model_name="$4"
  local label_mode="$5"
  local entry_policy="$6"
  local target_hit_bps="${7:-50}"

  [[ -f "$samples" || "$DRY_RUN" == "1" ]] || die "samples missing: $samples"
  [[ -f "$INTRADAY_BARS" || "$DRY_RUN" == "1" ]] || die "intraday bars missing: $INTRADAY_BARS"

  backup_existing_artifact "$artifact"

  echo "============================================================"
  echo "[SAVE MODEL] $artifact"
  echo "samples=$samples"
  echo "feature_group=$feature_group"
  echo "model_name=$model_name"
  echo "label_mode=$label_mode"
  echo "entry_policy=$entry_policy"
  echo "============================================================"

  cmd=(
    "$PYTHON" model_saving/save_nextday_model.py
    --stock-code "$STOCK"
    --artifact-name "$artifact"
    --samples "$samples"
    --intraday-bars "$INTRADAY_BARS"
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

  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "$LOG_ROOT/save_${artifact}.log"
    rc=${PIPESTATUS[0]}
    [[ "$rc" == "0" ]] || die "save model failed: $artifact"
  fi
}

if [[ "$SAVE_EXTERNAL_MODEL" == "1" ]]; then
  save_model \
    "$EXT_ARTIFACT" \
    "$EXTERNAL_SAMPLES" \
    reversal_fundamental_regime_sector_external \
    xgb_d3_400_lr003_mcw3 \
    close_profit \
    all_days \
    50
fi

if [[ "$SAVE_SECTOR_MODEL" == "1" ]]; then
  save_model \
    "$SECTOR_ARTIFACT" \
    "$SECTOR_SAMPLES" \
    reversal_fundamental_regime_sector \
    xgb_d4_500_lr002_mcw5 \
    close_profit \
    all_days \
    50
fi

echo "[STEP] Write metadata summary"
if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" - "$MODELS_DIR" "$STOCK" "$LOG_ROOT" <<'PY'
import json, sys
from pathlib import Path
models_dir, stock, log_root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
rows = []
for meta_path in sorted((models_dir / stock).glob("*/metadata.json")):
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows.append({"artifact": meta_path.parent.name, "error": f"{type(exc).__name__}: {exc}"})
        continue
    m = meta.get("validation_tail_trade_metrics", {}) or {}
    rows.append({
        "artifact": meta_path.parent.name,
        "created_at": meta.get("artifact_created_at", ""),
        "date_max": meta.get("date_max", ""),
        "feature_group": meta.get("feature_group", ""),
        "model_name": meta.get("model_name", ""),
        "label_mode": meta.get("label_mode", ""),
        "entry_policy": meta.get("entry_policy", ""),
        "trades": m.get("trades", ""),
        "win_rate": m.get("win_rate", ""),
        "avg_return": m.get("avg_return", ""),
        "median_return": m.get("median_return", ""),
        "max_drawdown": m.get("max_drawdown", ""),
        "profit_factor": m.get("profit_factor", ""),
        "samples": meta.get("samples", ""),
    })
out = log_root / "603308_model_metadata_summary.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[WROTE] {out}")
PY
fi

echo "============================================================"
echo "[DONE] 603308 training data rebuilt and models saved. No search was run."
echo "[LOG_ROOT] $LOG_ROOT"
echo "============================================================"
