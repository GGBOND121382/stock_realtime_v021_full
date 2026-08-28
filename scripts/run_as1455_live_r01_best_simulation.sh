#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

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
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
MODEL_REGISTRY_ROOT="${MODEL_REGISTRY_ROOT:-saved_data/ashare_ml4t/ch17_as1455_model_registry}"
SIMULATION_EXPERIMENT="${R01_SIMULATION_EXPERIMENT:-r01_best_reb1_fold0_5_forward}"
SIMULATION_TARGET="r01"
SIMULATION_TARGET_COL="r01_fwd"
SIMULATION_TOP_N=1
CAPACITY_MODE="${CAPACITY_MODE:-none}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-0.05}"
HEAVY_LOCK_FILE="${AS1455_HEAVY_LOCK_FILE:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
HEAVY_LOCK_WAIT_SECONDS="${AS1455_HEAVY_LOCK_WAIT_SECONDS:-900}"

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
STAGING_ROOT="$NINE_ROOT/_simulation_staging/$SIMULATION_EXPERIMENT"
READY_FILE="$NINE_ROOT/strategies/$SIMULATION_EXPERIMENT/execution_batch.json"
STATE_DIR="$OUT_ROOT/.dashboard"
STATUS_FILE="$STATE_DIR/r01_best_simulation_status.json"
LOCK_FILE="$STATE_DIR/r01_best_simulation.lock"
mkdir -p "$STATE_DIR" "$(dirname "$HEAVY_LOCK_FILE")"

STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
LOG_FILE="$STATE_DIR/r01_best_simulation_${LIVE_DATE}_$(date +%H%M%S).log"

write_status() {
  local state="$1"
  local rc="${2:-}"
  local finished="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATE="$state" EXIT_CODE="$rc" FINISHED_AT="$finished" \
  STARTED_AT="$STARTED_AT" LOG_FILE="$LOG_FILE" TRADE_DATE_VALUE="$LIVE_DATE" \
  EXPERIMENT_VALUE="$SIMULATION_EXPERIMENT" READY_FILE_VALUE="$READY_FILE" \
  PID_VALUE="$$" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path=Path(os.environ["STATUS_FILE"])
payload={
    "status": os.environ["STATE"],
    "trade_date": os.environ["TRADE_DATE_VALUE"],
    "experiment": os.environ["EXPERIMENT_VALUE"],
    "pid": int(os.environ["PID_VALUE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ.get("FINISHED_AT") or None,
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ.get("EXIT_CODE") else None,
    "log_file": os.environ["LOG_FILE"],
    "ready_file": os.environ["READY_FILE_VALUE"],
    "production_r21_mutated": False,
}
tmp=path.with_suffix(path.suffix+".tmp")
tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
tmp.replace(path)
PY
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[BLOCKED] another r01 simulation job holds $LOCK_FILE" >&2
  write_status blocked 75 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 75
fi

# Standalone/manual runs must serialize with other heavy AS1455 work.  When the
# production post job invokes us, it already owns this lock and explicitly says so.
if [[ "${AS1455_PARENT_HEAVY_LOCK_HELD:-0}" != "1" ]]; then
  exec 8>"$HEAVY_LOCK_FILE"
  if ! flock -w "$HEAVY_LOCK_WAIT_SECONDS" 8; then
    echo "[BLOCKED] heavy AS1455 compute is busy: $HEAVY_LOCK_FILE" >&2
    write_status blocked 76 "$(TZ="$TIMEZONE" date -Iseconds)"
    exit 76
  fi
fi

write_status running
exec > >(tee -a "$LOG_FILE") 2>&1
completed=0
on_exit() {
  local rc=$?
  if [[ "$completed" != "1" && "$rc" -ne 0 ]]; then
    write_status failed "$rc" "$(TZ="$TIMEZONE" date -Iseconds)" || true
  fi
}
trap on_exit EXIT

required=(
  "$FEATURE_FILE"
  "$SIDECAR_FILE"
  "$CALENDAR_FILE"
  "$PREPARED_FEATURE_FILE"
  "$PREPARED_FEATURE_REPORT"
  "$ACTIVE_MODEL_SNAPSHOT"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "[ERROR] missing r01 simulation prerequisite: $path" >&2; exit 1; }
done

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_live_target_predictions.py \
  scripts/run_as1455_live_simulation_strategy_planner_entry.py

# Fail closed for explicit committed-only mobile fetches while this run is building.
rm -f "$READY_FILE"
rm -rf "$STAGING_ROOT"
mkdir -p "$PRED_ROOT/$SIMULATION_TARGET" "$STAGING_ROOT"

echo "[START] r01-best simulation trade_date=$LIVE_DATE"
echo "[SIMULATION] reusing finalized 14:55 features and prepared inference matrix"
echo "[SIMULATION] inference target=$SIMULATION_TARGET_COL Top-$SIMULATION_TOP_N"
"$PYTHON_BIN" scripts/run_as1455_live_target_predictions.py \
  --trade-date "$LIVE_DATE" \
  --target-col "$SIMULATION_TARGET_COL" \
  --feature-preset "$FEATURE_PRESET" \
  --model-data "$MODEL_DATA" \
  --feature-file "$FEATURE_FILE" \
  --prepared-feature-file "$PREPARED_FEATURE_FILE" \
  --prepared-feature-report "$PREPARED_FEATURE_REPORT" \
  --model-snapshot "$ACTIVE_MODEL_SNAPSHOT" \
  --out-dir "$PRED_ROOT/$SIMULATION_TARGET" \
  --top-n "$SIMULATION_TOP_N"

echo "[SIMULATION] planning isolated strategy=$SIMULATION_EXPERIMENT"
"$PYTHON_BIN" scripts/run_as1455_live_simulation_strategy_planner_entry.py \
  --simulation-experiment "$SIMULATION_EXPERIMENT" \
  --publish-root "$NINE_ROOT" \
  --trade-date "$LIVE_DATE" \
  --matrix-root "$MATRIX_ROOT" \
  --prediction-root "$PRED_ROOT" \
  --execution-sidecar "$SIDECAR_FILE" \
  --execution-calendar "$CALENDAR_FILE" \
  --out-root "$STAGING_ROOT" \
  --feature-preset "$FEATURE_PRESET" \
  --capacity-mode "$CAPACITY_MODE" \
  --participation-rate "$PARTICIPATION_RATE"

[[ -f "$READY_FILE" ]] || { echo "[ERROR] r01 simulation READY batch was not published" >&2; exit 1; }

finished_at="$(TZ="$TIMEZONE" date -Iseconds)"
completed=1
write_status success 0 "$finished_at"
echo "[PASS] r01-best simulation READY: $READY_FILE"
echo "[PASS] r21 production READY/root manifests were not rewritten"
