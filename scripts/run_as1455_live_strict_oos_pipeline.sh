#!/usr/bin/env bash
# AS1455 clean live monitor: T-1 refresh -> 09:35 prepare -> 14:55 collect
# -> clean strict-OOS inference -> canonical v7 single-day plan.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-auto}"
PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
TARGET_COL="${TARGET_COL:-r05_fwd}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
POSITIONS_FILE="${POSITIONS_FILE:-}"
CASH="${CASH:-}"
CASH_FILE="${CASH_FILE:-}"
PREPARE_TIME="${PREPARE_TIME:-09:35:00}"
COLLECT_START_TIME="${COLLECT_START_TIME:-14:50:00}"
UNTIL="${UNTIL:-14:55:05}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55:00}"
MIN_VALID_RATE="${MIN_VALID_RATE:-0.98}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-980}"
MAX_FINALIZE_SECONDS="${MAX_FINALIZE_SECONDS:-40}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
ALLOW_MISSING_BUY_DATE="${ALLOW_MISSING_BUY_DATE:-0}"
ALLOW_INDICATOR_FALLBACK="${ALLOW_INDICATOR_FALLBACK:-0}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-0.05}"
SELECTION_BACKTEST_ROOT="${SELECTION_BACKTEST_ROOT:-}"
FOLD0_DIR="${FOLD0_DIR:-}"

live_date() {
  "$PYTHON" - <<PY
from datetime import datetime
from zoneinfo import ZoneInfo
s = "${TRADE_DATE}"
if s.lower() == "today": print(datetime.now(ZoneInfo("${TIMEZONE}")).strftime("%Y%m%d"))
else:
    s=s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
}
LIVE_DATE="$(live_date)"
LIVE_DIR="${OUT_ROOT}/${LIVE_DATE}"

fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
seconds_now() {
  TZ="$TIMEZONE" "$PYTHON" - <<'PY'
from datetime import datetime
n=datetime.now(); print(n.hour*3600+n.minute*60+n.second)
PY
}
time_to_seconds() {
  "$PYTHON" - "$1" <<'PY'
import sys
p=list(map(int,sys.argv[1].split(':')))
if len(p)==2: p.append(0)
if len(p)!=3: raise SystemExit('bad clock')
print(p[0]*3600+p[1]*60+p[2])
PY
}
wait_until() {
  local target="$1" now target_s
  now="$(seconds_now)"; target_s="$(time_to_seconds "$target")"
  if (( now < target_s )); then
    info "waiting $((target_s-now)) seconds until ${target} (${TIMEZONE})"
    sleep "$((target_s-now))"
  fi
}

max_symbol_args=()
[[ -n "$MAX_SYMBOLS" ]] && max_symbol_args=(--max-symbols "$MAX_SYMBOLS")

check_files() {
  local files=(
    scripts/run_as1455_live_data_feature_pipeline.sh
    pipelines/as1455_live_prepare.py
    data_collection/collect_as1455_live_quotes_as1455.py
    features/as1455_live_common.py
    features/build_as1455_live_feature_state_fast.py
    features/finalize_as1455_live_features_fast.py
    tools/build_as1455_live_execution_sidecar_v1.py
    utils/as1455_live_inference.py
    code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
    scripts/run_as1455_live_strict_oos_monitor.py
  )
  local f
  for f in "${files[@]}"; do [[ -f "$f" ]] || fail "missing $f"; done
  "$PYTHON" -m py_compile "${files[@]:1}"
  [[ -f "$UNIVERSE" ]] || fail "missing universe: $UNIVERSE"
  echo "[PASS] live strict-OOS pipeline static check"
}

run_pre() {
  info "refreshing historical caches to T-1 with the clean history updater"
  TRADE_DATE="$LIVE_DATE" UNIVERSE="$UNIVERSE" \
    bash scripts/run_as1455_live_data_feature_pipeline.sh history
  info "preparing current-day preclose, adjustment state, and qfq history tail"
  "$PYTHON" pipelines/as1455_live_prepare.py \
    --trade-date "$LIVE_DATE" --history-end-date auto \
    --universe "$UNIVERSE" "${max_symbol_args[@]}" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --as1455-daily-cache-dir "$AS1455_DAILY_CACHE_DIR" \
    --out-root "$OUT_ROOT"
  info "building compact pre-14:55 feature state"
  "$PYTHON" features/build_as1455_live_feature_state_fast.py \
    --trade-date "$LIVE_DATE" --out-root "$OUT_ROOT" \
    --sector-reference "$MODEL_DATA"
  [[ -f "$LIVE_DIR/06_live_feature_state_fast.npz" ]] || fail "prefast state missing"
}

require_account_state() {
  [[ -n "$POSITIONS_FILE" ]] || fail "set POSITIONS_FILE to broker/current holdings CSV"
  [[ -f "$POSITIONS_FILE" ]] || fail "positions file not found: $POSITIONS_FILE"
  [[ -n "$CASH" || -n "$CASH_FILE" ]] || fail "set CASH or CASH_FILE to current available cash"
  [[ -z "$CASH_FILE" || -f "$CASH_FILE" ]] || fail "cash file not found: $CASH_FILE"
}

run_plan() {
  require_account_state
  local args=(
    --trade-date "$LIVE_DATE" --live-dir "$LIVE_DIR"
    --model-data "$MODEL_DATA" --target-col "$TARGET_COL"
    --feature-preset "$FEATURE_PRESET" --positions-file "$POSITIONS_FILE"
    --capacity-mode "$CAPACITY_MODE" --participation-rate "$PARTICIPATION_RATE"
  )
  [[ -n "$CASH" ]] && args+=(--cash "$CASH")
  [[ -n "$CASH_FILE" ]] && args+=(--cash-file "$CASH_FILE")
  [[ -n "$SELECTION_BACKTEST_ROOT" ]] && args+=(--selection-backtest-root "$SELECTION_BACKTEST_ROOT")
  [[ -n "$FOLD0_DIR" ]] && args+=(--fold0-dir "$FOLD0_DIR")
  [[ "$ALLOW_MISSING_BUY_DATE" == "1" ]] && args+=(--allow-missing-buy-date)
  "$PYTHON" scripts/run_as1455_live_strict_oos_monitor.py "${args[@]}"
}

run_post() {
  [[ -f "$LIVE_DIR/06_live_feature_state_fast.npz" ]] || fail "missing prefast state; run pre before 14:50"
  if [[ "$SKIP_COLLECT" != "1" ]]; then
    info "collecting current quotes and freezing the latest valid snapshot <= ${CUTOFF_TIME}"
    "$PYTHON" data_collection/collect_as1455_live_quotes_as1455.py collect-loop \
      --trade-date "$LIVE_DATE" --universe "$UNIVERSE" "${max_symbol_args[@]}" \
      --out-root "$OUT_ROOT" --until "$UNTIL" --cutoff-time "$CUTOFF_TIME" \
      --min-valid-rate "$MIN_VALID_RATE"
  fi
  info "finalizing only the current-day model rows"
  finalize_args=(
    --trade-date "$LIVE_DATE" --out-root "$OUT_ROOT"
    --min-feature-rows "$MIN_FEATURE_ROWS"
    --max-elapsed-seconds "$MAX_FINALIZE_SECONDS"
  )
  [[ "$ALLOW_INDICATOR_FALLBACK" == "1" ]] && finalize_args+=(--allow-indicator-fallback)
  "$PYTHON" features/finalize_as1455_live_features_fast.py "${finalize_args[@]}"
  info "building the live execution sidecar"
  "$PYTHON" tools/build_as1455_live_execution_sidecar_v1.py \
    --trade-date "$LIVE_DATE" --out-root "$OUT_ROOT" --live-dir "$LIVE_DIR" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" --universe "$UNIVERSE"
  run_plan
}

status() {
  echo "[STATUS] LIVE_DIR=$LIVE_DIR"
  ls -lh "$LIVE_DIR"/05_prepare_report.json \
         "$LIVE_DIR"/06_live_feature_state_fast.npz \
         "$LIVE_DIR"/08_collection_report.json \
         "$LIVE_DIR"/11_live_model_features_for_prediction.csv \
         "$LIVE_DIR"/17_live_strict_oos_manifest.json 2>/dev/null || true
  [[ -f "$LIVE_DIR/17_live_strict_oos_manifest.json" ]] && cat "$LIVE_DIR/17_live_strict_oos_manifest.json"
}

case "$MODE" in
  check) check_files ;;
  pre) check_files; run_pre ;;
  post) check_files; run_post ;;
  plan) check_files; run_plan ;;
  auto|all)
    check_files
    now="$(seconds_now)"; collect="$(time_to_seconds "$COLLECT_START_TIME")"
    if (( now < collect )); then
      wait_until "$PREPARE_TIME"
      run_pre
      wait_until "$COLLECT_START_TIME"
      run_post
    else
      info "already at/after collect window; reusing existing prefast state"
      run_post
    fi
    ;;
  status) status ;;
  *)
    cat <<EOF
Usage: bash scripts/run_as1455_live_strict_oos_pipeline.sh [check|pre|post|plan|auto|status]

Required for post/plan/auto:
  POSITIONS_FILE=/path/current_positions.csv
  CASH=12345.67  # or CASH_FILE=/path/current_cash.txt

Strategy defaults (override explicitly when needed):
  TARGET_COL=$TARGET_COL
  FEATURE_PRESET=$FEATURE_PRESET
EOF
    exit 2
    ;;
esac
