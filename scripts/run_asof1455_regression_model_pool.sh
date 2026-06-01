#!/usr/bin/env bash
set -uo pipefail

# Run pooled asof1455 regression backtests over the canonical universe.
#
# Assumes data has been built by:
#   scripts/build_asof1455_regression_universe.sh

PYTHON="${PYTHON:-python3}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-saved_data/ml4t_asof1455_lgbm_pipeline_out/logs/regression_pool_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-saved_data/ml4t_asof1455_lgbm_pipeline_out}"

TRAIN_DAYS="${TRAIN_DAYS:-756}"
VALID_DAYS="${VALID_DAYS:-0}"
TEST_DAYS="${TEST_DAYS:-21}"
EMBARGO_DAYS="${EMBARGO_DAYS:-1}"
STEP_DAYS="${STEP_DAYS:-21}"
MAX_POSITIONS="${MAX_POSITIONS:-3}"
MIN_DAILY_CANDIDATES="${MIN_DAILY_CANDIDATES:-10}"
MIN_PRED_RETURN_BPS="${MIN_PRED_RETURN_BPS:-0.0}"
MAX_MISSING="${MAX_MISSING:-0.70}"
MAX_RUNS="${MAX_RUNS:-0}"
N_JOBS="${N_JOBS:-1}"
RIDGE_SOLVER="${RIDGE_SOLVER:-lsqr}"
LAZY_WINDOW_LOAD="${LAZY_WINDOW_LOAD:-0}"
LAZY_WINDOW_LOOKBACK_DAYS="${LAZY_WINDOW_LOOKBACK_DAYS:-180}"
LAZY_CHUNK_ROWS="${LAZY_CHUNK_ROWS:-100000}"
LAZY_USECOLS="${LAZY_USECOLS:-1}"

FEATURE_GROUPS="${FEATURE_GROUPS:-ml4t_intraday,ml4t_fundamental,ml4t_sector,ml4t_external}"
MODEL_FAMILIES="${MODEL_FAMILIES:-ridge,elasticnet,extratrees,lgbm_l1,lgbm_huber,lgbm_quantile,catboost_rmse,catboost_huber,catboost_quantile,randomforest,lgbm_l2}"

mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/regression_pool_summary.csv"
echo "feature_group,model_family,status,returncode,start_time,end_time,elapsed_seconds,log_file,out_dir" > "$SUMMARY_FILE"
RUN_COUNT=0

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

sample_glob_for_group() {
  local group="$1"
  case "$group" in
    ml4t|core|ml4t_core|ml4t_intraday|core_intraday)
      echo "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv"
      ;;
    fundamental|ml4t_fundamental|core_fundamental)
      echo "saved_data/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv|saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv"
      ;;
    sector|ml4t_sector|core_sector)
      echo "saved_data/*_pipeline_out/03_sector/training_samples_with_sector.csv|saved_data/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv|saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv"
      ;;
    external|ml4t_external|core_external|core_sector_external_lagged|all)
      echo "saved_data/*_pipeline_out/04_external/*/training_samples*.csv|saved_data/*_pipeline_out/03_sector/training_samples_with_sector.csv|saved_data/*_pipeline_out/02_fundamental/training_samples_with_fundamentals.csv|saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv"
      ;;
    *)
      echo "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv"
      ;;
  esac
}

objective_alpha_for_model() {
  local model="$1"
  case "$model" in
    lgbm_quantile|catboost_quantile) echo "0.50" ;;
    lgbm_huber) echo "0.90" ;;
    *) echo "0.90" ;;
  esac
}

run_one() {
  local feature_group="$1"
  local model_family="$2"
  local safe_group="${feature_group//[^A-Za-z0-9_]/_}"
  local safe_model="${model_family//[^A-Za-z0-9_]/_}"
  local run_name="reg_${safe_group}_${safe_model}_train${TRAIN_DAYS}_test${TEST_DAYS}_pos${MAX_POSITIONS}"
  local out_dir="${OUT_ROOT}/99_summary/${run_name}"
  local log_file="$LOG_DIR/${run_name}.log"
  local start_time end_time start_epoch end_epoch elapsed rc sample_globs alpha

  if [[ "$MAX_RUNS" != "0" && "$RUN_COUNT" -ge "$MAX_RUNS" ]]; then
    return 0
  fi
  RUN_COUNT=$((RUN_COUNT + 1))

  sample_globs="$(sample_glob_for_group "$feature_group")"
  alpha="$(objective_alpha_for_model "$model_family")"

  local cmd=(
    "$PYTHON" scripts/backtest_ml4t_asof1455_lgbm.py
    --bars-glob "saved_data/*_pipeline_out/00_base/*_5m.csv"
    --out-root "$OUT_ROOT"
    --run-name "$run_name"
    --feature-group "$feature_group"
    --model-family "$model_family"
    --entry-price-col close_asof1455
    --exit-price-col next_day_close
    --entry-policy all_days
    --round-trip-cost-bps 1.7
    --train-days "$TRAIN_DAYS"
    --valid-days "$VALID_DAYS"
    --test-days "$TEST_DAYS"
    --embargo-days "$EMBARGO_DAYS"
    --step-days "$STEP_DAYS"
    --selection-rule strict_top_decile_positive
    --min-pred-return-bps "$MIN_PRED_RETURN_BPS"
    --min-daily-candidates "$MIN_DAILY_CANDIDATES"
    --max-positions "$MAX_POSITIONS"
    --max-missing "$MAX_MISSING"
    --winsorize-target
    --objective-alpha "$alpha"
    --n-jobs "$N_JOBS"
    --ridge-solver "$RIDGE_SOLVER"
  )
  if [[ "$LAZY_WINDOW_LOAD" == "1" ]]; then
    cmd+=(
      --lazy-window-load
      --lazy-window-lookback-days "$LAZY_WINDOW_LOOKBACK_DAYS"
      --lazy-chunk-rows "$LAZY_CHUNK_ROWS"
    )
    if [[ "$LAZY_USECOLS" == "0" ]]; then
      cmd+=(--no-lazy-usecols)
    fi
  fi
  IFS='|' read -ra SAMPLE_PATTERNS <<< "$sample_globs"
  for pat in "${SAMPLE_PATTERNS[@]}"; do
    cmd+=(--sample-glob "$pat")
  done

  start_time="$(date '+%F %T')"
  start_epoch="$(date +%s)"

  {
    echo "============================================================"
    echo "[START] ${start_time} feature_group=${feature_group}, model_family=${model_family}, sample_globs=${sample_globs}"
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
    echo "[DONE] ${end_time} feature_group=${feature_group}, model_family=${model_family}, status=ok, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${feature_group},${model_family},ok,${rc},${start_time},${end_time},${elapsed},${log_file},${out_dir}" >> "$SUMMARY_FILE"
  elif [[ "$rc" -eq 124 ]]; then
    echo "[TIMEOUT] ${end_time} feature_group=${feature_group}, model_family=${model_family}, timeout=${JOB_TIMEOUT}, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${feature_group},${model_family},timeout,${rc},${start_time},${end_time},${elapsed},${log_file},${out_dir}" >> "$SUMMARY_FILE"
  else
    echo "[FAIL] ${end_time} feature_group=${feature_group}, model_family=${model_family}, returncode=${rc}, elapsed=${elapsed}s" | tee -a "$log_file"
    echo "${feature_group},${model_family},failed,${rc},${start_time},${end_time},${elapsed},${log_file},${out_dir}" >> "$SUMMARY_FILE"
  fi
  return 0
}

IFS=',' read -ra FEATURE_GROUP_LIST <<< "$FEATURE_GROUPS"
IFS=',' read -ra MODEL_FAMILY_LIST <<< "$MODEL_FAMILIES"

for group in "${FEATURE_GROUP_LIST[@]}"; do
  group="$(echo "$group" | xargs)"
  [[ -z "$group" ]] && continue
  for model in "${MODEL_FAMILY_LIST[@]}"; do
    model="$(echo "$model" | xargs)"
    [[ -z "$model" ]] && continue
    run_one "$group" "$model"
  done
done

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[SUMMARY] ${SUMMARY_FILE}"
echo "============================================================"

exit 0
