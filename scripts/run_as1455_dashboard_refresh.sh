#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
LIVE_ROOT="${LIVE_ROOT:-saved_data/ashare_ml4t/live_as1455}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-$SOURCE_DIR/baostock_5m_cache}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-$SOURCE_DIR/as1455_daily_cache}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
SKIP_DATA_REFRESH="${SKIP_DATA_REFRESH:-0}"
TRACKING_MODE="${TRACKING_MODE:-incremental}"
TRACKING_START_DATE="${TRACKING_START_DATE:-}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
FULL_RESEARCH_REFRESH="${FULL_RESEARCH_REFRESH:-0}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
TRADE_DATE="${TRADE_DATE:-today}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
HISTORY_WORKERS="${HISTORY_WORKERS:-3}"
SYMBOL_RETRIES="${SYMBOL_RETRIES:-2}"
HEAVY_LOCK_FILE="${AS1455_HEAVY_LOCK_FILE:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
HEAVY_LOCK_WAIT_SECONDS="${AS1455_HEAVY_LOCK_WAIT_SECONDS:-0}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
STATE_DIR="$MATRIX_ROOT/.dashboard"
STATUS_FILE="$STATE_DIR/refresh_status.json"
LOCK_FILE="$STATE_DIR/refresh.lock"
mkdir -p "$STATE_DIR" "$(dirname "$HEAVY_LOCK_FILE")"

[[ "$TRACKING_MODE" == "incremental" || "$TRACKING_MODE" == "rebuild" ]] || {
  echo "[ERROR] TRACKING_MODE must be incremental or rebuild" >&2
  exit 2
}
[[ "$HISTORY_WORKERS" =~ ^[1-8]$ ]] || { echo "[ERROR] HISTORY_WORKERS must be 1..8" >&2; exit 2; }
[[ "$SYMBOL_RETRIES" =~ ^[0-9]+$ ]] || { echo "[ERROR] SYMBOL_RETRIES must be non-negative" >&2; exit 2; }
[[ "$HEAVY_LOCK_WAIT_SECONDS" =~ ^[0-9]+$ ]] || { echo "[ERROR] AS1455_HEAVY_LOCK_WAIT_SECONDS must be non-negative" >&2; exit 2; }

write_status() {
  local status="$1"
  local exit_code="${2:-}"
  local finished_at="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATUS="$status" EXIT_CODE="$exit_code" \
    FINISHED_AT="$finished_at" LOG_FILE="$LOG_FILE" STARTED_AT="$STARTED_AT" \
    SKIP_DATA_REFRESH="$SKIP_DATA_REFRESH" PID_VALUE="$$" \
    TRACKING_MODE="$TRACKING_MODE" FULL_RESEARCH_REFRESH="$FULL_RESEARCH_REFRESH" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
payload = {
    "status": os.environ["STATUS"],
    "pid": int(os.environ["PID_VALUE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ.get("FINISHED_AT") or None,
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ.get("EXIT_CODE") else None,
    "skip_data_refresh": os.environ["SKIP_DATA_REFRESH"] == "1",
    "tracking_mode": os.environ["TRACKING_MODE"],
    "materialize_live_plans": os.environ["TRACKING_MODE"] == "rebuild",
    "full_research_refresh": os.environ["FULL_RESEARCH_REFRESH"] == "1",
    "log_file": os.environ["LOG_FILE"],
    "daily_refresh_semantics": (
        "nightly raw-daily close update only, then append new tracking-account dates; "
        "changing/rebuilding tracking start additionally materializes all existing 14:55 plans once; "
        "historical Fold/Grid and canonical old forward results are not recomputed"
    ),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
LOG_FILE="$STATE_DIR/refresh_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S).log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s\n' "[BLOCKED] another dashboard refresh holds $LOCK_FILE" > "$LOG_FILE"
  write_status blocked 75 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 75
fi

exec 8>"$HEAVY_LOCK_FILE"
if [[ "$HEAVY_LOCK_WAIT_SECONDS" -gt 0 ]]; then
  flock -w "$HEAVY_LOCK_WAIT_SECONDS" 8 || {
    printf '%s\n' "[BLOCKED] live/heavy AS1455 compute holds $HEAVY_LOCK_FILE" > "$LOG_FILE"
    write_status blocked 76 "$(TZ="$TIMEZONE" date -Iseconds)"
    exit 76
  }
else
  flock -n 8 || {
    printf '%s\n' "[BLOCKED] live/heavy AS1455 compute holds $HEAVY_LOCK_FILE" > "$LOG_FILE"
    write_status blocked 76 "$(TZ="$TIMEZONE" date -Iseconds)"
    exit 76
  }
fi

write_status running
exec > >(tee -a "$LOG_FILE") 2>&1

completed=0
on_exit() {
  local rc=$?
  if [[ "$completed" != "1" && "$rc" -ne 0 ]]; then
    write_status failed "$rc" "$(TZ="$TIMEZONE" date -Iseconds)"
  fi
}
trap on_exit EXIT

echo "[START] AS1455 daily refresh started_at=$STARTED_AT"
echo "[MODE] skip_data_refresh=$SKIP_DATA_REFRESH tracking_mode=$TRACKING_MODE full_research_refresh=$FULL_RESEARCH_REFRESH"
echo "[RESOURCE] acquired shared heavy-compute lock: $HEAVY_LOCK_FILE"

echo "===== 0/3 validate frozen nine-strategy matrix ====="
[[ -f "$MATRIX_ROOT/expected_experiments.txt" ]] || {
  echo "[ERROR] missing $MATRIX_ROOT/expected_experiments.txt" >&2
  exit 1
}
count="$(grep -cve '^[[:space:]]*$' "$MATRIX_ROOT/expected_experiments.txt")"
[[ "$count" -eq 9 ]] || { echo "[ERROR] expected 9 experiments, got $count" >&2; exit 1; }
"$PYTHON_BIN" -m py_compile \
  scripts/update_as1455_tracking_accounts.py \
  scripts/resolve_as1455_nightly_history_end.py \
  scripts/materialize_as1455_start_date_plans.py \
  dashboard/as1455_plan_compute.py \
  dashboard/as1455_plan_preview.py \
  utils/as1455_materialized_plan.py \
  utils/as1455_tracking.py \
  pipelines/as1455_history_parallel_dispatch.py

if [[ "$SKIP_DATA_REFRESH" != "1" ]]; then
  echo "===== 1/3 update completed BaoStock raw-daily closes only ====="
  resolved_history_end="$HISTORY_END_DATE"
  if [[ "${HISTORY_END_DATE,,}" == "auto" ]]; then
    resolved_history_end="$(
      "$PYTHON_BIN" scripts/resolve_as1455_nightly_history_end.py \
        --trade-date "$TRADE_DATE" \
        --timezone "$TIMEZONE" \
        --universe "$UNIVERSE"
    )"
  fi
  [[ "$resolved_history_end" == "auto" || "$resolved_history_end" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ || "$resolved_history_end" =~ ^[0-9]{8}$ ]] || {
    echo "[ERROR] invalid resolved_history_end=$resolved_history_end" >&2
    exit 1
  }
  echo "[BAOSTOCK] requested_history_end=$HISTORY_END_DATE resolved_history_end=$resolved_history_end"
  PROJECT_PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  PYTHONPATH="$PROJECT_PYTHONPATH" "$PYTHON_BIN" pipelines/as1455_history_parallel_dispatch.py \
    --trade-date "$TRADE_DATE" \
    --history-end-date "$resolved_history_end" \
    --universe "$UNIVERSE" \
    --raw-5m-cache-dir "$RAW_5M_CACHE_DIR" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --as1455-daily-cache-dir "$AS1455_DAILY_CACHE_DIR" \
    --out-root "$LIVE_ROOT" \
    --workers "$HISTORY_WORKERS" \
    --symbol-retries "$SYMBOL_RETRIES" \
    --skip-raw-5m \
    --skip-as1455-aggregate

  trade_token="$($PYTHON_BIN - "$TRADE_DATE" "$TIMEZONE" <<'PY'
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
value, timezone = sys.argv[1:]
if value.lower() == "today":
    print(datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d"))
else:
    print(value.replace("-", ""))
PY
)"
  history_report="$LIVE_ROOT/$trade_token/00_history_update_report.json"
  "$PYTHON_BIN" - "$history_report" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"nightly raw-daily report missing: {path}")
obj = json.loads(path.read_text(encoding="utf-8"))
errors = int(obj.get("errors", 0))
if errors:
    raise SystemExit(
        f"nightly raw-daily update incomplete: errors={errors} "
        f"unresolved={obj.get('unresolved_symbols', [])}; see {path}"
    )
if not obj.get("skip_raw_5m"):
    raise SystemExit("nightly refresh unexpectedly downloaded 5m data")
print(
    f"[PASS] nightly raw-daily update history_end={obj.get('history_end_date')} "
    f"symbols={obj.get('n_symbols')} new_rows={obj.get('raw_daily_new_rows_sum')}"
)
PY
else
  echo "[SKIP] BaoStock raw-daily refresh disabled"
fi

if [[ "$FULL_RESEARCH_REFRESH" == "1" ]]; then
  echo "===== optional full research refresh ====="
  env \
    MATRIX_ROOT="$MATRIX_ROOT" \
    SKIP_DATA_REFRESH=1 \
    REQUIRE_HISTORICAL_REUSE=1 \
    FORCE_HISTORICAL_GRID=0 \
    FORCE_HISTORICAL_PREDICTIONS=0 \
    bash scripts/run_ch17_as1455_full_rebuild.sh refresh-all-fixed-signals
fi

echo "===== 2/3 advance tracking accounts ====="
tracking_args=(
  scripts/update_as1455_tracking_accounts.py
  --matrix-root "$MATRIX_ROOT"
  --live-root "$LIVE_ROOT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --mode "$TRACKING_MODE"
)
[[ -n "$TRACKING_START_DATE" ]] && tracking_args+=(--tracking-start-date "$TRACKING_START_DATE")
"$PYTHON_BIN" "${tracking_args[@]}"

if [[ "$TRACKING_MODE" == "rebuild" ]]; then
  echo "===== 3/3 materialize all saved 14:55 plans for the new start date ====="
  materialize_args=(
    scripts/materialize_as1455_start_date_plans.py
    --matrix-root "$MATRIX_ROOT"
    --live-root "$LIVE_ROOT"
    --feature-preset "$FEATURE_PRESET"
  )
  [[ -n "$TRACKING_START_DATE" ]] && materialize_args+=(--tracking-start-date "$TRACKING_START_DATE")
  "$PYTHON_BIN" "${materialize_args[@]}"
else
  echo "===== 3/3 keep existing start-date plan cache ====="
  echo "[SKIP] incremental refresh does not rebuild historical 14:55 plan cache"
fi

finished_at="$(TZ="$TIMEZONE" date -Iseconds)"
completed=1
write_status success 0 "$finished_at"
echo "[PASS] AS1455 daily refresh finished_at=$finished_at"
echo "[PASS] historical Fold/Grid recomputed: no"
echo "[PASS] canonical old forward window recomputed: no"
echo "[PASS] nightly 5m/model inference recomputed: no"
echo "[PASS] tracking account mode=$TRACKING_MODE"
[[ "$TRACKING_MODE" == "rebuild" ]] && echo "[PASS] saved 14:55 plans materialized for current tracking start: yes"
