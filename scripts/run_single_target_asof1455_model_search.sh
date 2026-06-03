#!/usr/bin/env bash
set -uo pipefail

# Per-symbol, single-target Ridge-only regression search.
# Target: target_next_close_bps = 10000 * (next_day_close / close_asof1455 - 1) - cost_bps.

PYTHON="${PYTHON:-python3}"
JOB_TIMEOUT="${JOB_TIMEOUT:-24h}"
OUT_DIR="${OUT_DIR:-saved_data/single_target_asof1455_ridge_search_out/search_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_GLOBS="${SAMPLE_GLOBS:-saved_data/*_pipeline_out/04_external/*/training_samples_with_*external*.csv;saved_data/*_pipeline_out/03_sector/training_samples_with_sector.csv;saved_data/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv;saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv}"
SAMPLE_GLOB="${SAMPLE_GLOB:-}"
SYMBOLS="${SYMBOLS:-}"
MAX_SYMBOLS="${MAX_SYMBOLS:-0}"
MODEL_TYPES="${MODEL_TYPES:-all}"
TRAIN_WINDOWS="${TRAIN_WINDOWS:-756,504}"
VALID_DAYS="${VALID_DAYS:-63}"
TEST_DAYS="${TEST_DAYS:-21}"
EMBARGO_DAYS="${EMBARGO_DAYS:-1}"
COST_BPS="${COST_BPS:-1.7}"
ALPHA_GRID="${ALPHA_GRID:-logspace:-6:8:29}"
KERNEL_ALPHA_GRID="${KERNEL_ALPHA_GRID:-logspace:-4:10:29}"
N_COMPONENTS_GRID="${N_COMPONENTS_GRID:-64,128,256,512,1024}"
SVD_COMPONENTS_GRID="${SVD_COMPONENTS_GRID:-16,32,64,128,256}"
MAX_POLY_FEATURES="${MAX_POLY_FEATURES:-20000}"
MAX_EXACT_KERNEL_TRAIN_ROWS="${MAX_EXACT_KERNEL_TRAIN_ROWS:-1200}"
MIN_ROWS="${MIN_ROWS:-300}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-120}"
N_JOBS="${N_JOBS:-1}"

mkdir -p "$OUT_DIR"

cmd=(
  "$PYTHON" scripts/search_single_target_asof1455_ridge_models.py
  --out-dir "$OUT_DIR"
  --model-types "$MODEL_TYPES"
  --train-window "$TRAIN_WINDOWS"
  --valid-window "$VALID_DAYS"
  --test-window "$TEST_DAYS"
  --embargo "$EMBARGO_DAYS"
  --cost-bps "$COST_BPS"
  --alpha-grid "$ALPHA_GRID"
  --kernel-alpha-grid "$KERNEL_ALPHA_GRID"
  --n-components-grid "$N_COMPONENTS_GRID"
  --svd-components-grid "$SVD_COMPONENTS_GRID"
  --max-poly-features "$MAX_POLY_FEATURES"
  --max-exact-kernel-train-rows "$MAX_EXACT_KERNEL_TRAIN_ROWS"
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
