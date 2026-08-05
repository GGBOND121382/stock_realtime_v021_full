#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
SKIP_DATA_REFRESH="${SKIP_DATA_REFRESH:-0}"
STATE_DIR="$MATRIX_ROOT/.dashboard"
STATUS_FILE="$STATE_DIR/refresh_status.json"
LOCK_FILE="$STATE_DIR/refresh.lock"
mkdir -p "$STATE_DIR"

write_status() {
  local status="$1"
  local exit_code="${2:-}"
  local finished_at="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATUS="$status" EXIT_CODE="$exit_code" \
    FINISHED_AT="$finished_at" LOG_FILE="$LOG_FILE" STARTED_AT="$STARTED_AT" \
    SKIP_DATA_REFRESH="$SKIP_DATA_REFRESH" PID_VALUE="$$" python3 - <<'PY'
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
    "log_file": os.environ["LOG_FILE"],
    "command": "bash scripts/run_ch17_as1455_full_rebuild.sh refresh-all-fixed-signals",
    "require_historical_reuse": True,
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  STARTED_AT="$(date -Iseconds)"
  LOG_FILE="$STATE_DIR/refresh_blocked_$(date +%Y%m%d_%H%M%S).log"
  printf '%s\n' "[BLOCKED] another dashboard refresh holds $LOCK_FILE" > "$LOG_FILE"
  write_status blocked 75 "$(date -Iseconds)"
  exit 75
fi

STARTED_AT="$(date -Iseconds)"
LOG_FILE="$STATE_DIR/refresh_$(date +%Y%m%d_%H%M%S).log"
write_status running
exec > >(tee -a "$LOG_FILE") 2>&1

completed=0
on_exit() {
  local rc=$?
  if [[ "$completed" != "1" && "$rc" -ne 0 ]]; then
    write_status failed "$rc" "$(date -Iseconds)"
  fi
}
trap on_exit EXIT

echo "[START] dashboard refresh started_at=$STARTED_AT"
echo "[MODE] skip_data_refresh=$SKIP_DATA_REFRESH"
echo "[MODE] require_historical_reuse=1 force_historical_grid=0"

# Fail closed before any market-data refresh. The dashboard must never turn a
# routine daily refresh into an unexpected 30/150/630-row historical Grid.
PLAN_JSON="$STATE_DIR/dashboard_fold_availability_plan.json"
eval "$(
  python3 scripts/resolve_as1455_fixed_signal_matrix_folds.py \
    --top-n 5 \
    --output-json "$PLAN_JSON" \
    --format shell
)"

require_history() {
  local target_col="$1"
  local rebalance_every="$2"
  local target_folds="$3"
  local signal_kind="$4"
  local found
  found="$(
    python3 scripts/find_as1455_compatible_historical_result.py \
      --target-col "$target_col" \
      --signal-kind "$signal_kind" \
      --rebalance-every "$rebalance_every" \
      --target-folds "$target_folds" \
      --format path
  )"
  if [[ -z "$found" || ! -d "$found" ]]; then
    echo "[BLOCKED] no validated historical Grid for target=$target_col signal=$signal_kind folds=$target_folds" >&2
    return 1
  fi
  echo "[HISTORY REUSE] target=$target_col signal=$signal_kind root=$found"
}

for signal in all5 first3 best; do
  require_history r01_fwd 1 "$TARGET_FOLDS_R01" "$signal"
  require_history r05_fwd 5 "$TARGET_FOLDS_R05" "$signal"
  require_history r21_fwd 21 "$TARGET_FOLDS_R21" "$signal"
done

set +e
env \
  MATRIX_ROOT="$MATRIX_ROOT" \
  SKIP_DATA_REFRESH="$SKIP_DATA_REFRESH" \
  REQUIRE_HISTORICAL_REUSE=1 \
  FORCE_HISTORICAL_GRID=0 \
  FORCE_HISTORICAL_PREDICTIONS=0 \
  bash scripts/run_ch17_as1455_full_rebuild.sh refresh-all-fixed-signals
rc=$?
set -e

finished_at="$(date -Iseconds)"
if [[ "$rc" -eq 0 ]]; then
  completed=1
  write_status success 0 "$finished_at"
  echo "[PASS] dashboard refresh finished_at=$finished_at"
else
  write_status failed "$rc" "$finished_at"
  echo "[FAILED] dashboard refresh exit_code=$rc finished_at=$finished_at" >&2
fi
exit "$rc"
