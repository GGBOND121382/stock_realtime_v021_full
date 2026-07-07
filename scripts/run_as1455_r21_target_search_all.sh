#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6}"
TARGET_COL="${TARGET_COL:-r21_fwd}"
EPOCHS="${EPOCHS:-20}"
BEST_N="${BEST_N:-5}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-saved_data/ashare_ml4t/ch17_as1455_target_search/logs}"
FORCE="${FORCE:-0}"
SMOKE="${SMOKE:-0}"
INPUT_CHECK_ONLY="${INPUT_CHECK_ONLY:-0}"
RETRAIN_BEST="${RETRAIN_BEST:-0}"

mkdir -p "$LOG_DIR"

extra_args=()
if [[ "$FORCE" == "1" ]]; then
  extra_args+=(--force)
fi
if [[ "$SMOKE" == "1" ]]; then
  extra_args+=(--smoke)
fi
if [[ "$INPUT_CHECK_ONLY" == "1" ]]; then
  extra_args+=(--input-check-only)
fi
if [[ "$RETRAIN_BEST" == "1" ]]; then
  extra_args+=(--retrain-best)
fi

for preset in $FEATURE_PRESETS; do
  for fold in $FOLDS; do
    log="$LOG_DIR/${preset}_${TARGET_COL}_fold${fold}_search.log"
    echo "===== preset=${preset} target=${TARGET_COL} fold=${fold} ====="
    "$PYTHON_BIN" scripts/run_as1455_target_fold_param_search.py \
      --feature-preset "$preset" \
      --target-col "$TARGET_COL" \
      --model-data "$MODEL_DATA" \
      --fold-index "$fold" \
      --sector-encoding onehot \
      --dropna-mode target_only \
      --epochs "$EPOCHS" \
      --best-n "$BEST_N" \
      --seed "$SEED" \
      "${extra_args[@]}" 2>&1 | tee "$log"
  done
done

echo "[DONE] r21 target searches finished. Logs: $LOG_DIR"
