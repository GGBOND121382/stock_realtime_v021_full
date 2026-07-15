#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-help}"
PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
HISTORY_START_DATE="${HISTORY_START_DATE:-2020-01-01}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
DRY_RUN="${DRY_RUN:-0}"
NO_BAOSTOCK_CALENDAR="${NO_BAOSTOCK_CALENDAR:-0}"
SKIP_AS1455_AGGREGATE="${SKIP_AS1455_AGGREGATE:-0}"
LOG_ROOT="${LOG_ROOT:-logs}"
SCRIPT_COMMON="features/as1455_live_common.py"
SCRIPT_BASE="pipelines/as1455_update_history_to_prevday.py"
SCRIPT_HISTORY="pipelines/as1455_update_history_to_prevday_fast_v4.py"

mkdir -p "$LOG_ROOT"

fail() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }

live_date() {
  "$PYTHON" - "$TRADE_DATE" <<'PY'
import sys
from datetime import datetime
s = sys.argv[1]
if s.lower() == "today":
    print(datetime.now().strftime("%Y%m%d"))
else:
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
}

check_files() {
  require_file "$SCRIPT_COMMON"
  require_file "$SCRIPT_BASE"
  require_file "$SCRIPT_HISTORY"
  require_file "$UNIVERSE"
  "$PYTHON" -m py_compile "$SCRIPT_COMMON" "$SCRIPT_BASE" "$SCRIPT_HISTORY"
}

print_context() {
  cat <<EOF
[CONFIG]
  MODE=$MODE
  PYTHON=$PYTHON
  TRADE_DATE=$TRADE_DATE
  HISTORY_END_DATE=$HISTORY_END_DATE
  HISTORY_START_DATE=$HISTORY_START_DATE
  UNIVERSE=$UNIVERSE
  RAW_5M_CACHE_DIR=$RAW_5M_CACHE_DIR
  RAW_DAILY_CACHE_DIR=$RAW_DAILY_CACHE_DIR
  AS1455_DAILY_CACHE_DIR=$AS1455_DAILY_CACHE_DIR
  MAX_SYMBOLS=${MAX_SYMBOLS:-<none>}
EOF
}

run_history() {
  check_files
  local args=(
    "$SCRIPT_HISTORY"
    --trade-date "$TRADE_DATE"
    --history-end-date "$HISTORY_END_DATE"
    --history-start-date "$HISTORY_START_DATE"
    --universe "$UNIVERSE"
    --raw-5m-cache-dir "$RAW_5M_CACHE_DIR"
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
    --as1455-daily-cache-dir "$AS1455_DAILY_CACHE_DIR"
    --out-root "$OUT_ROOT"
    --sleep-seconds "$SLEEP_SECONDS"
  )
  [[ -n "$MAX_SYMBOLS" ]] && args+=(--max-symbols "$MAX_SYMBOLS")
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  [[ "$NO_BAOSTOCK_CALENDAR" == "1" ]] && args+=(--no-baostock-calendar)
  [[ "$SKIP_AS1455_AGGREGATE" == "1" ]] && args+=(--skip-as1455-aggregate)

  local stamp log report
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_ROOT/as1455_history_${stamp}.log"
  print_context
  "$PYTHON" "${args[@]}" 2>&1 | tee "$log"

  if [[ "$DRY_RUN" != "1" ]]; then
    report="$OUT_ROOT/$(live_date)/00_history_update_report.json"
    require_file "$report"
    "$PYTHON" - "$report" <<'PY'
import json, sys
path = sys.argv[1]
obj = json.load(open(path, encoding="utf-8"))
errors = int(obj.get("errors", -1))
if errors != 0:
    raise SystemExit(f"history update has errors={errors}; see {path}")
print(f"[PASS] history caches updated through {obj.get('history_end_date')} for {obj.get('n_symbols')} symbols")
PY
  fi
}

status() {
  local report="$OUT_ROOT/$(live_date)/00_history_update_report.json"
  print_context
  if [[ -s "$report" ]]; then
    cat "$report"
  else
    echo "[MISSING] $report"
  fi
}

case "$MODE" in
  history) run_history ;;
  check) check_files; echo "[PASS] AS1455 history pipeline check passed" ;;
  status) status ;;
  help|-h|--help)
    cat <<'EOF'
Usage:
  bash scripts/run_as1455_live_data_feature_pipeline.sh history
  bash scripts/run_as1455_live_data_feature_pipeline.sh check
  bash scripts/run_as1455_live_data_feature_pipeline.sh status
EOF
    ;;
  *) fail "unknown mode: $MODE" ;;
esac
