#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-post}"
[[ "$STAGE" == "pre" || "$STAGE" == "post" ]] || {
  echo "Usage: bash scripts/run_as1455_live_nine_strategy_job.sh [pre|post]" >&2
  exit 2
}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
STATE_DIR="$OUT_ROOT/.dashboard"
STATUS_FILE="$STATE_DIR/nine_strategy_${STAGE}_status.json"
LOCK_FILE="$STATE_DIR/nine_strategy.lock"
mkdir -p "$STATE_DIR"

trade_date="$($PYTHON_BIN - <<PY
from datetime import datetime
from zoneinfo import ZoneInfo
print(datetime.now(ZoneInfo("$TIMEZONE")).strftime("%Y%m%d"))
PY
)"
STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
LOG_FILE="$STATE_DIR/nine_strategy_${STAGE}_${trade_date}_$(date +%H%M%S).log"

write_status() {
  local state="$1"
  local rc="${2:-}"
  local finished="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATE="$state" EXIT_CODE="$rc" FINISHED_AT="$finished" \
  STARTED_AT="$STARTED_AT" LOG_FILE="$LOG_FILE" STAGE="$STAGE" TRADE_DATE_VALUE="$trade_date" \
  PID_VALUE="$$" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path=Path(os.environ["STATUS_FILE"])
payload={
    "status": os.environ["STATE"],
    "stage": os.environ["STAGE"],
    "trade_date": os.environ["TRADE_DATE_VALUE"],
    "pid": int(os.environ["PID_VALUE"]),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ.get("FINISHED_AT") or None,
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ.get("EXIT_CODE") else None,
    "log_file": os.environ["LOG_FILE"],
}
tmp=path.with_suffix(path.suffix+".tmp")
tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
tmp.replace(path)
PY
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s\n' "[BLOCKED] another nine-strategy live job holds $LOCK_FILE" > "$LOG_FILE"
  write_status blocked 75 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 75
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

echo "[START] stage=$STAGE trade_date=$trade_date started_at=$STARTED_AT"
set +e
env \
  PYTHON_BIN="$PYTHON_BIN" \
  TRADE_DATE="$trade_date" \
  TIMEZONE="$TIMEZONE" \
  OUT_ROOT="$OUT_ROOT" \
  MATRIX_ROOT="$MATRIX_ROOT" \
  bash scripts/run_as1455_live_nine_strategy_pipeline.sh "$STAGE"
rc=$?
set -e
finished_at="$(TZ="$TIMEZONE" date -Iseconds)"
if [[ "$rc" -eq 0 ]]; then
  completed=1
  write_status success 0 "$finished_at"
  echo "[PASS] stage=$STAGE finished_at=$finished_at"
else
  write_status failed "$rc" "$finished_at"
  echo "[FAILED] stage=$STAGE exit_code=$rc finished_at=$finished_at" >&2
fi
exit "$rc"
