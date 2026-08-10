#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-auto}"
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

fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

check_files() {
  local files=(
    scripts/run_as1455_live_strict_oos_pipeline.sh
    scripts/run_as1455_live_target_predictions.py
    scripts/run_as1455_live_nine_strategy_planner.py
    scripts/run_as1455_live_nine_strategy_planner_entry.py
    data_collection/collect_as1455_live_quotes_as1455.py
    features/finalize_as1455_live_features_fast.py
    tools/build_as1455_live_execution_sidecar_v1.py
  )
  local f
  for f in "${files[@]}"; do [[ -f "$f" ]] || fail "missing $f"; done
  "$PYTHON_BIN" -m py_compile \
    scripts/run_as1455_live_target_predictions.py \
    scripts/run_as1455_live_nine_strategy_planner.py \
    scripts/run_as1455_live_nine_strategy_planner_entry.py \
    data_collection/collect_as1455_live_quotes_as1455.py \
    features/finalize_as1455_live_features_fast.py \
    tools/build_as1455_live_execution_sidecar_v1.py
  [[ -f "$MATRIX_ROOT/expected_experiments.txt" ]] || fail "missing matrix results: $MATRIX_ROOT"
  [[ -f "$MATRIX_ROOT/strict_forward_latest_states_manifest.json" ]] || fail \
    "missing latest strategy account states; run the 20:00 backtest refresh first"
  echo "[PASS] nine-strategy live pipeline static check"
}

run_pre() {
  check_files
  info "running shared pre-14:55 preparation through the canonical live pipeline"
  env \
    PYTHON="$PYTHON_BIN" \
    TRADE_DATE="$LIVE_DATE" \
    TIMEZONE="$TIMEZONE" \
    OUT_ROOT="$OUT_ROOT" \
    UNIVERSE="$UNIVERSE" \
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
    MODEL_DATA="$MODEL_DATA" \
    bash scripts/run_as1455_live_strict_oos_pipeline.sh pre
}

run_post() {
  check_files
  [[ -f "$LIVE_DIR/06_live_feature_state_fast.npz" ]] || fail \
    "missing prefast state; run pre before the 14:50 collection window"
  mkdir -p "$PRED_ROOT"

  if [[ "$SKIP_COLLECT" != "1" ]]; then
    info "collecting 14:50-14:55 quotes and freezing the latest snapshot <= $CUTOFF_TIME"
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

  info "building the shared live execution sidecar once"
  "$PYTHON_BIN" tools/build_as1455_live_execution_sidecar_v1.py \
    --trade-date "$LIVE_DATE" \
    --out-root "$OUT_ROOT" \
    --live-dir "$LIVE_DIR" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --universe "$UNIVERSE"

  [[ -f "$FEATURE_FILE" ]] || fail "missing finalized live feature file: $FEATURE_FILE"
  [[ -f "$SIDECAR_FILE" ]] || fail "missing live execution sidecar: $SIDECAR_FILE"
  [[ -f "$CALENDAR_FILE" ]] || fail "missing execution calendar: $CALENDAR_FILE"

  for target in r01 r05 r21; do
    info "shared fold0 inference target=${target}_fwd"
    "$PYTHON_BIN" scripts/run_as1455_live_target_predictions.py \
      --trade-date "$LIVE_DATE" \
      --target-col "${target}_fwd" \
      --feature-preset "$FEATURE_PRESET" \
      --model-data "$MODEL_DATA" \
      --feature-file "$FEATURE_FILE" \
      --out-dir "$PRED_ROOT/$target" \
      --top-n 5
  done

  info "planning all nine fixed-signal strategies with T-1 simulated account states"
  "$PYTHON_BIN" scripts/run_as1455_live_nine_strategy_planner_entry.py \
    --trade-date "$LIVE_DATE" \
    --matrix-root "$MATRIX_ROOT" \
    --prediction-root "$PRED_ROOT" \
    --execution-sidecar "$SIDECAR_FILE" \
    --execution-calendar "$CALENDAR_FILE" \
    --out-root "$NINE_ROOT" \
    --feature-preset "$FEATURE_PRESET" \
    --capacity-mode "$CAPACITY_MODE" \
    --participation-rate "$PARTICIPATION_RATE"

  echo "[PASS] nine-strategy 14:55 planning complete"
  echo "[PASS] summary=$NINE_ROOT/live_nine_strategy_summary.csv"
  echo "[PASS] rebalance_only=$NINE_ROOT/live_rebalance_strategies.csv"
}

status() {
  echo "[STATUS] live_date=$LIVE_DATE"
  echo "[STATUS] live_dir=$LIVE_DIR"
  ls -lh \
    "$LIVE_DIR/05_prepare_report.json" \
    "$LIVE_DIR/08_collection_report.json" \
    "$FEATURE_FILE" \
    "$NINE_ROOT/live_nine_strategy_summary.csv" \
    "$NINE_ROOT/live_nine_strategy_manifest.json" 2>/dev/null || true
  [[ -f "$NINE_ROOT/live_nine_strategy_summary.csv" ]] && cat "$NINE_ROOT/live_nine_strategy_summary.csv"
}

case "$MODE" in
  check) check_files ;;
  pre) run_pre ;;
  post) run_post ;;
  auto|all)
    run_pre
    now="$($PYTHON_BIN - <<PY
from datetime import datetime
from zoneinfo import ZoneInfo
n=datetime.now(ZoneInfo("$TIMEZONE")); print(n.hour*3600+n.minute*60+n.second)
PY
)"
    target=$((14*3600+50*60))
    if (( now < target )); then
      info "waiting $((target-now)) seconds until 14:50 (${TIMEZONE})"
      sleep "$((target-now))"
    fi
    run_post
    ;;
  status) status ;;
  *)
    echo "Usage: bash scripts/run_as1455_live_nine_strategy_pipeline.sh [check|pre|post|auto|status]" >&2
    exit 2
    ;;
esac
