#!/usr/bin/env bash
set -uo pipefail

# Per-symbol, single-target regression search.
# Target: target_next_close_bps = next_day_close / close_asof1455 - cost.

PYTHON="${PYTHON:-python3}"
JOB_TIMEOUT="${JOB_TIMEOUT:-24h}"
OUT_DIR="${OUT_DIR:-saved_data/single_target_asof1455_model_search_out/search_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_GLOBS="${SAMPLE_GLOBS:-saved_data/**/*_pipeline_out/04_external/*/training_samples_with_*external*.csv;saved_data/**/*_pipeline_out/03_sector/training_samples_with_sector.csv;saved_data/**/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv;saved_data/**/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv}"
SAMPLE_GLOB="${SAMPLE_GLOB:-}"
SYMBOLS="${SYMBOLS:-}"
MAX_SYMBOLS="${MAX_SYMBOLS:-0}"
MODELS="${MODELS:-constant,ewma,ols,ridge,lasso,elasticnet,tree,randomforest,lgbm_l2,lgbm_l1,lgbm_huber,catboost_rmse,catboost_mae,catboost_huber}"
FEATURE_SETS="${FEATURE_SETS:-all}"
TRAIN_WINDOWS="${TRAIN_WINDOWS:-756,504}"
TEST_DAYS="${TEST_DAYS:-21}"
EMBARGO_DAYS="${EMBARGO_DAYS:-1}"
ROUND_TRIP_COST_BPS="${ROUND_TRIP_COST_BPS:-1.7}"
MAX_MISSING="${MAX_MISSING:-0.70}"
MIN_ROWS="${MIN_ROWS:-600}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-252}"
N_JOBS="${N_JOBS:-1}"

mkdir -p "$OUT_DIR"

cmd=(
  "$PYTHON" scripts/search_single_target_asof1455_models.py
  --out-dir "$OUT_DIR"
  --models "$MODELS"
  --feature-sets "$FEATURE_SETS"
  --train-windows "$TRAIN_WINDOWS"
  --test-days "$TEST_DAYS"
  --embargo-days "$EMBARGO_DAYS"
  --round-trip-cost-bps "$ROUND_TRIP_COST_BPS"
  --max-missing "$MAX_MISSING"
  --min-rows "$MIN_ROWS"
  --min-train-rows "$MIN_TRAIN_ROWS"
  --n-jobs "$N_JOBS"
)

if [[ -n "$SAMPLE_GLOB" ]]; then
  cmd+=(--sample-glob "$SAMPLE_GLOB")
else
  IFS=';' read -r -a sample_glob_array <<< "$SAMPLE_GLOBS"
  for pat in "${sample_glob_array[@]}"; do
    if [[ -n "$pat" ]]; then
      cmd+=(--sample-glob "$pat")
    fi
  done
fi

if [[ -n "$SYMBOLS" ]]; then
  cmd+=(--symbols "$SYMBOLS")
fi
if [[ "$MAX_SYMBOLS" != "0" ]]; then
  cmd+=(--max-symbols "$MAX_SYMBOLS")
fi

{
  echo "[START] $(date '+%F %T')"
  printf '[CMD]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
} | tee "$OUT_DIR/run.log"

timeout --foreground "$JOB_TIMEOUT" "${cmd[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"
rc=${PIPESTATUS[0]}

echo "[END] $(date '+%F %T') returncode=${rc}" | tee -a "$OUT_DIR/run.log"
exit "$rc"
