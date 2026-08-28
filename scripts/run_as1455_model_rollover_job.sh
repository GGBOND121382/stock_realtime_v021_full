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
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
MODEL_REGISTRY_ROOT="${MODEL_REGISTRY_ROOT:-saved_data/ashare_ml4t/ch17_as1455_model_registry}"
FORWARD_MODEL_DIR="${FORWARD_MODEL_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"
MODEL_DATA="${MODEL_DATA:-$FORWARD_MODEL_DIR/model_data_as1455.h5}"
HEAVY_LOCK_FILE="${AS1455_HEAVY_LOCK_FILE:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
HEAVY_LOCK_WAIT_SECONDS="${AS1455_MODEL_ROLL_LOCK_WAIT_SECONDS:-0}"
STATUS_DIR="$MODEL_REGISTRY_ROOT/.dashboard"
STATUS_FILE="$STATUS_DIR/rollover_status.json"
mkdir -p "$STATUS_DIR" "$(dirname "$HEAVY_LOCK_FILE")"
STAMP="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
LOG_FILE="$STATUS_DIR/rollover_${STAMP}.log"
STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"

write_status() {
  local state="$1"
  local rc="${2:-}"
  local detail_file="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATE="$state" EXIT_CODE="$rc" DETAIL_FILE="$detail_file" \
  STARTED_AT="$STARTED_AT" LOG_FILE="$LOG_FILE" PID_VALUE="$$" \
  "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["STATUS_FILE"])
payload = {
    "status": os.environ["STATE"],
    "pid": int(os.environ["PID_VALUE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": None if os.environ["STATE"] == "running" else __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ.get("EXIT_CODE") else None,
    "log_file": os.environ["LOG_FILE"],
    "detail_file": os.environ.get("DETAIL_FILE") or None,
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

exec 8>"$HEAVY_LOCK_FILE"
if ! flock -w "$HEAVY_LOCK_WAIT_SECONDS" 8; then
  echo "[SKIP] AS1455 heavy compute is busy: $HEAVY_LOCK_FILE" > "$LOG_FILE"
  write_status blocked 76
  exit 76
fi

write_status running
exec > >(tee -a "$LOG_FILE") 2>&1
completed=0
on_exit() {
  local rc=$?
  if [[ "$completed" != "1" && "$rc" -ne 0 ]]; then
    write_status failed "$rc"
  fi
}
trap on_exit EXIT

echo "[START] rolling model check at $STARTED_AT"
CHECK_JSON="$STATUS_DIR/rollover_check_${STAMP}.json"
"$PYTHON_BIN" scripts/check_as1455_model_rollover.py \
  --registry-root "$MODEL_REGISTRY_ROOT" \
  --feature-preset "$FEATURE_PRESET" > "$CHECK_JSON"
cat "$CHECK_JSON"

due="$($PYTHON_BIN - "$CHECK_JSON" <<'PY'
import json, sys
obj=json.load(open(sys.argv[1], encoding='utf-8'))
print('1' if obj.get('due') else '0')
PY
)"
if [[ "$due" != "1" ]]; then
  completed=1
  write_status waiting 0 "$CHECK_JSON"
  echo "[PASS] rollover not due"
  exit 0
fi

period_end="$($PYTHON_BIN - "$CHECK_JSON" <<'PY'
import json, sys
obj=json.load(open(sys.argv[1], encoding='utf-8'))
value=obj.get('rollover_boundary')
if not value:
    raise SystemExit('missing rollover_boundary')
print(value)
PY
)"

echo "[DUE] period boundary=$period_end; refreshing extended model_data"
env \
  PYTHON_BIN="$PYTHON_BIN" \
  FORWARD_MODEL_DIR="$FORWARD_MODEL_DIR" \
  bash scripts/refresh_as1455_forward_model_data.sh

[[ -s "$MODEL_DATA" ]] || {
  echo "[ERROR] rolling model_data missing after refresh: $MODEL_DATA" >&2
  exit 1
}

echo "[TRAIN] fixed Top-5 recipes; no Fold/Grid search; three targets serially"
"$PYTHON_BIN" scripts/train_as1455_rolling_generation.py \
  --model-data "$MODEL_DATA" \
  --registry-root "$MODEL_REGISTRY_ROOT" \
  --feature-preset "$FEATURE_PRESET" \
  --period-end "$period_end" \
  --activate \
  --force

POST_JSON="$STATUS_DIR/rollover_after_${STAMP}.json"
"$PYTHON_BIN" scripts/check_as1455_model_rollover.py \
  --registry-root "$MODEL_REGISTRY_ROOT" \
  --feature-preset "$FEATURE_PRESET" > "$POST_JSON"
cat "$POST_JSON"
completed=1
write_status success 0 "$POST_JSON"
echo "[PASS] rolling generation activated; next live day will freeze the new generation"
