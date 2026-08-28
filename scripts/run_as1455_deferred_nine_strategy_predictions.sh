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
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
HEAVY_LOCK_FILE="${AS1455_HEAVY_LOCK_FILE:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
HEAVY_LOCK_WAIT_SECONDS="${AS1455_HEAVY_LOCK_WAIT_SECONDS:-900}"
[[ "$HEAVY_LOCK_WAIT_SECONDS" =~ ^[0-9]+$ ]] || {
  echo "AS1455_HEAVY_LOCK_WAIT_SECONDS must be a non-negative integer" >&2
  exit 2
}

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
PREPARED_FEATURE_FILE="$LIVE_DIR/12_live_ch17_inference_features.pkl"
PREPARED_FEATURE_REPORT="$LIVE_DIR/12_live_ch17_inference_features_report.json"
ACTIVE_MODEL_SNAPSHOT="$LIVE_DIR/13_active_model_snapshot.json"
STATE_DIR="$OUT_ROOT/.dashboard"
STATUS_FILE="$STATE_DIR/deferred_nine_strategy_predictions_status.json"
LOCK_FILE="$STATE_DIR/deferred_nine_strategy_predictions.lock"
mkdir -p "$STATE_DIR" "$(dirname "$HEAVY_LOCK_FILE")"

STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
LOG_FILE="$STATE_DIR/deferred_nine_strategy_predictions_${LIVE_DATE}_$(date +%H%M%S).log"

write_status() {
  local state="$1"
  local rc="${2:-}"
  local finished="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATE="$state" EXIT_CODE="$rc" FINISHED_AT="$finished" \
  STARTED_AT="$STARTED_AT" LOG_FILE="$LOG_FILE" TRADE_DATE_VALUE="$LIVE_DATE" \
  PID_VALUE="$$" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path=Path(os.environ["STATUS_FILE"])
payload={
    "status": os.environ["STATE"],
    "trade_date": os.environ["TRADE_DATE_VALUE"],
    "pid": int(os.environ["PID_VALUE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ.get("FINISHED_AT") or None,
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ.get("EXIT_CODE") else None,
    "log_file": os.environ["LOG_FILE"],
    "critical_1455_path": False,
    "purpose": "restore full Top-5 predictions for nine-strategy research tracking after the live decision window",
}
tmp=path.with_suffix(path.suffix+".tmp")
tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
tmp.replace(path)
PY
}

# If there was no successful 14:55 feature preparation today (weekend/holiday or
# failed live job), do not fabricate research predictions.
required=(
  "$FEATURE_FILE"
  "$PREPARED_FEATURE_FILE"
  "$PREPARED_FEATURE_REPORT"
  "$ACTIVE_MODEL_SNAPSHOT"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[SKIP] deferred predictions: missing live artifact $path"
    exit 0
  fi
done

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[BLOCKED] another deferred prediction job holds $LOCK_FILE" >&2
  exit 75
fi
exec 8>"$HEAVY_LOCK_FILE"
if ! flock -w "$HEAVY_LOCK_WAIT_SECONDS" 8; then
  echo "[BLOCKED] heavy AS1455 compute is busy: $HEAVY_LOCK_FILE" >&2
  write_status blocked 76 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 76
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

"$PYTHON_BIN" -m py_compile scripts/run_as1455_live_target_predictions.py

echo "[START] deferred full Top-5 inference trade_date=$LIVE_DATE"
for target in r01 r05 r21; do
  echo "[DEFERRED] target=${target}_fwd Top-5"
  mkdir -p "$PRED_ROOT/$target"
  "$PYTHON_BIN" scripts/run_as1455_live_target_predictions.py \
    --trade-date "$LIVE_DATE" \
    --target-col "${target}_fwd" \
    --feature-preset "$FEATURE_PRESET" \
    --model-data "$MODEL_DATA" \
    --feature-file "$FEATURE_FILE" \
    --prepared-feature-file "$PREPARED_FEATURE_FILE" \
    --prepared-feature-report "$PREPARED_FEATURE_REPORT" \
    --model-snapshot "$ACTIVE_MODEL_SNAPSHOT" \
    --out-dir "$PRED_ROOT/$target" \
    --top-n 5
done

finished_at="$(TZ="$TIMEZONE" date -Iseconds)"
completed=1
write_status success 0 "$finished_at"
echo "[PASS] deferred full Top-5 predictions complete"
echo "[PASS] 14:55 production plan was not recomputed"
echo "[PASS] nine-strategy tracking can consume these predictions at the next refresh"
