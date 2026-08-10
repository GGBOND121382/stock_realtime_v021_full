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
FULL_RESEARCH_REFRESH="${FULL_RESEARCH_REFRESH:-0}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
TRADE_DATE="${TRADE_DATE:-today}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
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
mkdir -p "$STATE_DIR"

[[ "$TRACKING_MODE" == "incremental" || "$TRACKING_MODE" == "rebuild" ]] || {
  echo "[ERROR] TRACKING_MODE must be incremental or rebuild" >&2
  exit 2
}

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
    "full_research_refresh": os.environ["FULL_RESEARCH_REFRESH"] == "1",
    "log_file": os.environ["LOG_FILE"],
    "daily_refresh_semantics": (
        "update completed BaoStock caches, then advance only new tracking-account dates; "
        "historical Fold/Grid and canonical old forward results are not recomputed"
    ),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
  LOG_FILE="$STATE_DIR/refresh_blocked_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S).log"
  printf '%s\n' "[BLOCKED] another dashboard refresh holds $LOCK_FILE" > "$LOG_FILE"
  write_status blocked 75 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 75
fi

STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
LOG_FILE="$STATE_DIR/refresh_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S).log"
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

echo "===== 0/2 validate frozen nine-strategy matrix ====="
[[ -f "$MATRIX_ROOT/expected_experiments.txt" ]] || {
  echo "[ERROR] missing $MATRIX_ROOT/expected_experiments.txt" >&2
  exit 1
}
count="$(grep -cve '^[[:space:]]*$' "$MATRIX_ROOT/expected_experiments.txt")"
[[ "$count" -eq 9 ]] || { echo "[ERROR] expected 9 experiments, got $count" >&2; exit 1; }
"$PYTHON_BIN" -m py_compile \
  scripts/update_as1455_tracking_accounts.py \
  scripts/resolve_as1455_nightly_history_end.py \
  utils/as1455_tracking.py

if [[ "$SKIP_DATA_REFRESH" != "1" ]]; then
  echo "===== 1/2 update completed BaoStock caches only ====="
  resolved_history_end="$HISTORY_END_DATE"
  if [[ "${HISTORY_END_DATE,,}" == "auto" ]]; then
    resolved_history_end="$(
      "$PYTHON_BIN" scripts/resolve_as1455_nightly_history_end.py \
        --trade-date "$TRADE_DATE" \
        --timezone "$TIMEZONE" \
        --universe "$UNIVERSE"
    )"
  fi
  echo "[BAOSTOCK] requested_history_end=$HISTORY_END_DATE resolved_history_end=$resolved_history_end"
  # The 14:55 predictions are already persisted during the trading day.  The
  # night job only needs completed execution/close data; rebuilding multi-year
  # model_data and rerunning TensorFlow inference is intentionally skipped.
  env \
    PYTHON="$PYTHON_BIN" \
    TRADE_DATE="$TRADE_DATE" \
    HISTORY_END_DATE="$resolved_history_end" \
    TIMEZONE="$TIMEZONE" \
    UNIVERSE="$UNIVERSE" \
    OUT_ROOT="$LIVE_ROOT" \
    RAW_5M_CACHE_DIR="$RAW_5M_CACHE_DIR" \
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
    AS1455_DAILY_CACHE_DIR="$AS1455_DAILY_CACHE_DIR" \
    bash scripts/run_as1455_live_data_feature_pipeline.sh history
else
  echo "[SKIP] BaoStock cache refresh disabled"
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

echo "===== 2/2 advance tracking accounts ====="
tracking_args=(
  scripts/update_as1455_tracking_accounts.py
  --matrix-root "$MATRIX_ROOT"
  --live-root "$LIVE_ROOT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --mode "$TRACKING_MODE"
)
[[ -n "$TRACKING_START_DATE" ]] && tracking_args+=(--tracking-start-date "$TRACKING_START_DATE")
"$PYTHON_BIN" "${tracking_args[@]}"

finished_at="$(TZ="$TIMEZONE" date -Iseconds)"
completed=1
write_status success 0 "$finished_at"
echo "[PASS] AS1455 daily refresh finished_at=$finished_at"
echo "[PASS] historical Fold/Grid recomputed: no"
echo "[PASS] canonical old forward window recomputed: no"
echo "[PASS] tracking account mode=$TRACKING_MODE"
