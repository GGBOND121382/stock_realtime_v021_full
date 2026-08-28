#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-post}"
[[ "$MODE" == "check" || "$MODE" == "post" || "$MODE" == "status" ]] || {
  echo "Usage: bash scripts/run_as1455_live_r21_best_pipeline.sh [check|post|status]" >&2
  exit 2
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

TRADE_DATE="${TRADE_DATE:-today}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
MODEL_REGISTRY_ROOT="${MODEL_REGISTRY_ROOT:-saved_data/ashare_ml4t/ch17_as1455_model_registry}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
UNTIL="${UNTIL:-14:55:05}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55:00}"
MIN_VALID_RATE="${MIN_VALID_RATE:-0.98}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-980}"
MAX_FINALIZE_SECONDS="${MAX_FINALIZE_SECONDS:-40}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
ALLOW_INDICATOR_FALLBACK="${ALLOW_INDICATOR_FALLBACK:-0}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-0.05}"
PRODUCTION_EXPERIMENT="${PRODUCTION_EXPERIMENT:-r21_best_reb21_fold0_4_forward}"
PRODUCTION_TARGET="r21"
PRODUCTION_TARGET_COL="r21_fwd"
PRODUCTION_TOP_N=1

live_date() {
  "$PYTHON_BIN" - <<PY
from datetime import datetime
from zoneinfo import ZoneInfo
s = "${TRADE_DATE}"
if s.lower() == "today":
    print(datetime.now(ZoneInfo("${TIMEZONE}")).strftime("%Y%m%d"))
else:
    s=s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
}

LIVE_DATE="$(live_date)"
LIVE_DIR="$OUT_ROOT/$LIVE_DATE"
NINE_ROOT="$LIVE_DIR/nine_strategy"
PRED_ROOT="$NINE_ROOT/shared_predictions"
FEATURE_FILE="$LIVE_DIR/11_live_model_features_for_prediction.csv"
SIDECAR_FILE="$LIVE_DIR/08_live_execution_sidecar.csv"
CALENDAR_FILE="$LIVE_DIR/05_execution_calendar.csv"
PREPARED_FEATURE_FILE="$LIVE_DIR/12_live_ch17_inference_features.pkl"
PREPARED_FEATURE_REPORT="$LIVE_DIR/12_live_ch17_inference_features_report.json"
ACTIVE_MODEL_SNAPSHOT="$LIVE_DIR/13_active_model_snapshot.json"

fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

check_files() {
  local files=(
    scripts/invalidate_as1455_execution_ready.py
    scripts/prepare_as1455_live_inference_features.py
    scripts/run_as1455_live_target_predictions.py
    scripts/run_as1455_live_production_strategy_planner_entry.py
    scripts/snapshot_as1455_active_models.py
    data_collection/collect_as1455_live_quotes_as1455.py
    features/finalize_as1455_live_features_fast.py
    tools/build_as1455_live_execution_sidecar_v1.py
    utils/as1455_live_inference_lowmem.py
    utils/as1455_model_registry.py
    utils/as1455_model_roll.py
  )
  local f
  for f in "${files[@]}"; do [[ -f "$f" ]] || fail "missing $f"; done
  "$PYTHON_BIN" -m py_compile \
    scripts/invalidate_as1455_execution_ready.py \
    scripts/prepare_as1455_live_inference_features.py \
    scripts/run_as1455_live_target_predictions.py \
    scripts/run_as1455_live_production_strategy_planner_entry.py \
    scripts/snapshot_as1455_active_models.py \
    data_collection/collect_as1455_live_quotes_as1455.py \
    features/finalize_as1455_live_features_fast.py \
    tools/build_as1455_live_execution_sidecar_v1.py \
    utils/as1455_live_inference_lowmem.py \
    utils/as1455_model_registry.py \
    utils/as1455_model_roll.py
  [[ -f "$MATRIX_ROOT/expected_experiments.txt" ]] || fail "missing matrix results: $MATRIX_ROOT"
  [[ -f "$MATRIX_ROOT/strict_forward_latest_states_manifest.json" ]] || fail \
    "missing latest strategy account states; run the tracking refresh first"
  echo "[PASS] r21-best production live pipeline static check"
}

run_post() {
  check_files

  # READY lifecycle starts here, before collection/inference.  A rerun can never
  # leave the previous same-day execution batch visible while new work is running.
  info "invalidating any existing same-day READY batch before recompute"
  "$PYTHON_BIN" scripts/invalidate_as1455_execution_ready.py --out-root "$NINE_ROOT"

  [[ -f "$LIVE_DIR/06_live_feature_state_fast.npz" ]] || fail \
    "missing prefast state; run the 09:35 pre job before the 14:50 collection window"
  mkdir -p "$PRED_ROOT/$PRODUCTION_TARGET"

  if [[ "$SKIP_COLLECT" != "1" ]]; then
    info "collecting 14:50-14:55 quotes and freezing latest snapshot <= $CUTOFF_TIME"
    "$PYTHON_BIN" data_collection/collect_as1455_live_quotes_as1455.py collect-loop \
      --trade-date "$LIVE_DATE" \
      --universe "$UNIVERSE" \
      --out-root "$OUT_ROOT" \
      --until "$UNTIL" \
      --cutoff-time "$CUTOFF_TIME" \
      --min-valid-rate "$MIN_VALID_RATE"
  else
    info "SKIP_COLLECT=1; reusing existing live snapshot"
  fi

  info "finalizing current-day model rows once"
  finalize_args=(
    --trade-date "$LIVE_DATE"
    --out-root "$OUT_ROOT"
    --min-feature-rows "$MIN_FEATURE_ROWS"
    --max-elapsed-seconds "$MAX_FINALIZE_SECONDS"
  )
  [[ "$ALLOW_INDICATOR_FALLBACK" == "1" ]] && finalize_args+=(--allow-indicator-fallback)
  "$PYTHON_BIN" features/finalize_as1455_live_features_fast.py "${finalize_args[@]}"

  info "building shared live execution sidecar once"
  "$PYTHON_BIN" tools/build_as1455_live_execution_sidecar_v1.py \
    --trade-date "$LIVE_DATE" \
    --out-root "$OUT_ROOT" \
    --live-dir "$LIVE_DIR" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --universe "$UNIVERSE"

  [[ -f "$FEATURE_FILE" ]] || fail "missing finalized live feature file: $FEATURE_FILE"
  [[ -f "$SIDECAR_FILE" ]] || fail "missing live execution sidecar: $SIDECAR_FILE"
  [[ -f "$CALENDAR_FILE" ]] || fail "missing execution calendar: $CALENDAR_FILE"

  info "freezing production model-generation snapshot"
  "$PYTHON_BIN" scripts/snapshot_as1455_active_models.py \
    --trade-date "$LIVE_DATE" \
    --feature-preset "$FEATURE_PRESET" \
    --registry-root "$MODEL_REGISTRY_ROOT" \
    --out-file "$ACTIVE_MODEL_SNAPSHOT"
  [[ -f "$ACTIVE_MODEL_SNAPSHOT" ]] || fail "missing active model snapshot: $ACTIVE_MODEL_SNAPSHOT"

  info "preparing one low-memory inference matrix for live + deferred research inference"
  "$PYTHON_BIN" scripts/prepare_as1455_live_inference_features.py \
    --trade-date "$LIVE_DATE" \
    --feature-preset "$FEATURE_PRESET" \
    --model-data "$MODEL_DATA" \
    --feature-file "$FEATURE_FILE" \
    --model-snapshot "$ACTIVE_MODEL_SNAPSHOT" \
    --out-file "$PREPARED_FEATURE_FILE" \
    --report-file "$PREPARED_FEATURE_REPORT"
  [[ -f "$PREPARED_FEATURE_FILE" ]] || fail "missing prepared inference matrix: $PREPARED_FEATURE_FILE"

  info "latency-critical inference: target=$PRODUCTION_TARGET_COL Top-$PRODUCTION_TOP_N only"
  "$PYTHON_BIN" scripts/run_as1455_live_target_predictions.py \
    --trade-date "$LIVE_DATE" \
    --target-col "$PRODUCTION_TARGET_COL" \
    --feature-preset "$FEATURE_PRESET" \
    --model-data "$MODEL_DATA" \
    --feature-file "$FEATURE_FILE" \
    --prepared-feature-file "$PREPARED_FEATURE_FILE" \
    --prepared-feature-report "$PREPARED_FEATURE_REPORT" \
    --model-snapshot "$ACTIVE_MODEL_SNAPSHOT" \
    --out-dir "$PRED_ROOT/$PRODUCTION_TARGET" \
    --top-n "$PRODUCTION_TOP_N"

  info "planning production strategy and committing READY last: $PRODUCTION_EXPERIMENT"
  "$PYTHON_BIN" scripts/run_as1455_live_production_strategy_planner_entry.py \
    --production-experiment "$PRODUCTION_EXPERIMENT" \
    --model-registry-root "$MODEL_REGISTRY_ROOT" \
    --trade-date "$LIVE_DATE" \
    --matrix-root "$MATRIX_ROOT" \
    --prediction-root "$PRED_ROOT" \
    --execution-sidecar "$SIDECAR_FILE" \
    --execution-calendar "$CALENDAR_FILE" \
    --out-root "$NINE_ROOT" \
    --feature-preset "$FEATURE_PRESET" \
    --capacity-mode "$CAPACITY_MODE" \
    --participation-rate "$PARTICIPATION_RATE"

  # No required/fallible state mutation belongs after execution_batch.json READY.
  echo "[PASS] production 14:55 planning complete"
  echo "[PASS] production_experiment=$PRODUCTION_EXPERIMENT"
  echo "[PASS] critical_path_models=1"
  echo "[PASS] READY is the final production commit"
}

status() {
  echo "[STATUS] live_date=$LIVE_DATE"
  echo "[STATUS] production_experiment=$PRODUCTION_EXPERIMENT"
  ls -lh \
    "$FEATURE_FILE" \
    "$PREPARED_FEATURE_FILE" \
    "$ACTIVE_MODEL_SNAPSHOT" \
    "$PRED_ROOT/$PRODUCTION_TARGET/top5_live_predictions.csv" \
    "$NINE_ROOT/live_nine_strategy_summary.csv" \
    "$NINE_ROOT/live_nine_strategy_manifest.json" 2>/dev/null || true
  [[ -f "$NINE_ROOT/live_nine_strategy_summary.csv" ]] && cat "$NINE_ROOT/live_nine_strategy_summary.csv"
}

case "$MODE" in
  check) check_files ;;
  post) run_post ;;
  status) status ;;
esac
