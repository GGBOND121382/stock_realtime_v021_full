#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-post}"
[[ "$STAGE" == "pre" || "$STAGE" == "post" ]] || {
  echo "Usage: bash scripts/run_as1455_live_production_job.sh [pre|post]" >&2
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
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-0.05}"
PRODUCTION_EXPERIMENT="${PRODUCTION_EXPERIMENT:-r21_best_reb21_fold0_4_forward}"
R01_SIMULATION_ENABLED="${R01_SIMULATION_ENABLED:-1}"
HEAVY_LOCK_FILE="${AS1455_HEAVY_LOCK_FILE:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
HEAVY_LOCK_WAIT_SECONDS="${AS1455_HEAVY_LOCK_WAIT_SECONDS:-900}"
[[ "$HEAVY_LOCK_WAIT_SECONDS" =~ ^[0-9]+$ ]] || {
  echo "AS1455_HEAVY_LOCK_WAIT_SECONDS must be a non-negative integer" >&2
  exit 2
}
[[ "$R01_SIMULATION_ENABLED" == "0" || "$R01_SIMULATION_ENABLED" == "1" ]] || {
  echo "R01_SIMULATION_ENABLED must be 0 or 1" >&2
  exit 2
}

STATE_DIR="$OUT_ROOT/.dashboard"
STATUS_FILE="$STATE_DIR/nine_strategy_${STAGE}_status.json"
LOCK_FILE="$STATE_DIR/nine_strategy.lock"
mkdir -p "$STATE_DIR" "$(dirname "$HEAVY_LOCK_FILE")"

trade_date="$($PYTHON_BIN - <<PY
from datetime import datetime
from zoneinfo import ZoneInfo
print(datetime.now(ZoneInfo("$TIMEZONE")).strftime("%Y%m%d"))
PY
)"
LIVE_DIR="$OUT_ROOT/$trade_date"
NINE_ROOT="$LIVE_DIR/nine_strategy"
CALENDAR_FILE="$LIVE_DIR/05_execution_calendar.csv"
STARTED_AT="$(TZ="$TIMEZONE" date -Iseconds)"
LOG_FILE="$STATE_DIR/nine_strategy_${STAGE}_${trade_date}_$(date +%H%M%S).log"

write_status() {
  local state="$1"
  local rc="${2:-}"
  local finished="${3:-}"
  STATUS_FILE="$STATUS_FILE" STATE="$state" EXIT_CODE="$rc" FINISHED_AT="$finished" \
  STARTED_AT="$STARTED_AT" LOG_FILE="$LOG_FILE" STAGE="$STAGE" TRADE_DATE_VALUE="$trade_date" \
  PRODUCTION_EXPERIMENT_VALUE="$PRODUCTION_EXPERIMENT" PID_VALUE="$$" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path=Path(os.environ["STATUS_FILE"])
payload={
    "status": os.environ["STATE"],
    "stage": os.environ["STAGE"],
    "trade_date": os.environ["TRADE_DATE_VALUE"],
    "production_experiment": os.environ["PRODUCTION_EXPERIMENT_VALUE"],
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

sync_tracking_accounts() {
  echo "[TRACKING] advancing all nine research-monitor accounts to latest completed market date"
  "$PYTHON_BIN" scripts/update_as1455_tracking_accounts.py \
    --matrix-root "$MATRIX_ROOT" \
    --live-root "$OUT_ROOT" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --feature-preset "$FEATURE_PRESET" \
    --mode incremental \
    --capacity-mode none \
    --participation-rate "$PARTICIPATION_RATE"
}

tracking_ready_for_post() {
  [[ -f "$CALENDAR_FILE" ]] || {
    echo "[TRACKING] execution calendar missing: $CALENDAR_FILE" >&2
    return 1
  }
  "$PYTHON_BIN" - "$MATRIX_ROOT" "$CALENDAR_FILE" "$trade_date" <<'PY'
import json
import math
import sys
from pathlib import Path
import pandas as pd

matrix_root = Path(sys.argv[1])
calendar_file = Path(sys.argv[2])
trade_date = pd.to_datetime(sys.argv[3], format="%Y%m%d").normalize()
config_file = matrix_root / ".dashboard" / "user_config.json"
manifest_file = matrix_root / "tracking_matrix_manifest.json"

try:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    start = pd.Timestamp(config["tracking_start_date"]).normalize()
    initial_cash = float(config["tracking_initial_cash"])
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError(f"invalid tracking_initial_cash={initial_cash}")
except Exception as exc:
    print(
        f"[TRACKING] production account config invalid; legacy fallback forbidden: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

if start > trade_date:
    print(
        f"[TRACKING] tracking start is after trade date: start={start:%Y-%m-%d} trade={trade_date:%Y-%m-%d}",
        file=sys.stderr,
    )
    raise SystemExit(1)
if start == trade_date:
    print(f"[TRACKING] first tracking day; no prior account state required: start={start:%Y-%m-%d}")
    raise SystemExit(0)

calendar = pd.read_csv(calendar_file, encoding="utf-8-sig")
if "date" not in calendar.columns:
    print(f"[TRACKING] calendar lacks date column: {calendar_file}", file=sys.stderr)
    raise SystemExit(1)
dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"], errors="coerce").dropna()).normalize().unique().sort_values()
prior = dates[(dates >= start) & (dates < trade_date)]
if len(prior) == 0:
    print(f"[TRACKING] no prior market day required from start={start:%Y-%m-%d}")
    raise SystemExit(0)
expected = pd.Timestamp(prior[-1]).normalize()

try:
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"[TRACKING] tracking manifest unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)
asofs = {pd.Timestamp(value).normalize() for value in manifest.get("asof_dates", []) if value}
ready = (
    manifest.get("status") == "ok"
    and int(manifest.get("completed_experiment_count", 0) or 0) == 9
    and asofs == {expected}
)
if ready:
    print(f"[TRACKING] all nine accounts ready through T-1={expected:%Y-%m-%d}")
    raise SystemExit(0)
print(
    f"[TRACKING] stale before post: expected={expected:%Y-%m-%d} "
    f"status={manifest.get('status')} completed={manifest.get('completed_experiment_count')} "
    f"asof_dates={sorted(str(x.date()) for x in asofs)}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s\n' "[BLOCKED] another AS1455 live job holds $LOCK_FILE" > "$LOG_FILE"
  write_status blocked 75 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 75
fi

exec 8>"$HEAVY_LOCK_FILE"
if ! flock -w "$HEAVY_LOCK_WAIT_SECONDS" 8; then
  printf '%s\n' "[BLOCKED] heavy AS1455 compute is busy: $HEAVY_LOCK_FILE wait=${HEAVY_LOCK_WAIT_SECONDS}s" > "$LOG_FILE"
  write_status blocked 76 "$(TZ="$TIMEZONE" date -Iseconds)"
  exit 76
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

echo "[START] stage=$STAGE trade_date=$trade_date production=$PRODUCTION_EXPERIMENT"
echo "[RESOURCE] acquired shared heavy-compute lock: $HEAVY_LOCK_FILE"
rc=0
set +e
if [[ "$STAGE" == "pre" ]]; then
  env \
    PYTHON_BIN="$PYTHON_BIN" TRADE_DATE="$trade_date" TIMEZONE="$TIMEZONE" \
    OUT_ROOT="$OUT_ROOT" MATRIX_ROOT="$MATRIX_ROOT" RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
    FEATURE_PRESET="$FEATURE_PRESET" \
    bash scripts/run_as1455_live_nine_strategy_pipeline.sh pre
  rc=$?
  if [[ "$rc" -eq 0 ]]; then
    sync_tracking_accounts
    rc=$?
  fi
else
  # READY must disappear before any new post-run readiness catch-up or inference.
  "$PYTHON_BIN" scripts/invalidate_as1455_execution_ready.py --out-root "$NINE_ROOT"
  rc=$?
  if [[ "$rc" -eq 0 ]] && ! tracking_ready_for_post; then
    echo "[TRACKING] stale state detected at 14:50; one all-nine catch-up before collection"
    sync_tracking_accounts
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      tracking_ready_for_post
      rc=$?
    fi
  fi
  if [[ "$rc" -eq 0 ]]; then
    env \
      PYTHON_BIN="$PYTHON_BIN" TRADE_DATE="$trade_date" TIMEZONE="$TIMEZONE" \
      OUT_ROOT="$OUT_ROOT" MATRIX_ROOT="$MATRIX_ROOT" RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
      FEATURE_PRESET="$FEATURE_PRESET" PRODUCTION_EXPERIMENT="$PRODUCTION_EXPERIMENT" \
      bash scripts/run_as1455_live_r21_best_pipeline.sh post
    rc=$?
  fi
  if [[ "$rc" -eq 0 && "$R01_SIMULATION_ENABLED" == "1" ]]; then
    echo "[SIMULATION] r21 production READY; starting isolated r01-best simulation"
    env \
      PYTHON_BIN="$PYTHON_BIN" TRADE_DATE="$trade_date" TIMEZONE="$TIMEZONE" \
      OUT_ROOT="$OUT_ROOT" MATRIX_ROOT="$MATRIX_ROOT" FEATURE_PRESET="$FEATURE_PRESET" \
      PARTICIPATION_RATE="$PARTICIPATION_RATE" AS1455_PARENT_HEAVY_LOCK_HELD=1 \
      bash scripts/run_as1455_live_r01_best_simulation.sh
    sim_rc=$?
    if [[ "$sim_rc" -ne 0 ]]; then
      echo "[WARN] r01-best simulation failed exit_code=$sim_rc; r21 production READY remains valid" >&2
    else
      echo "[PASS] r01-best simulation READY after r21 production"
    fi
  fi
fi
set -e

finished_at="$(TZ="$TIMEZONE" date -Iseconds)"
if [[ "$rc" -eq 0 ]]; then
  # execution_batch.json, when present, was already the final required commit.
  # Status bookkeeping is therefore best-effort and must never retroactively
  # turn a published READY order set into a FAILED production run.
  completed=1
  write_status success 0 "$finished_at" || \
    echo "[WARN] READY may be published but success status file could not be updated" >&2
  echo "[PASS] stage=$STAGE finished_at=$finished_at" || true
else
  write_status failed "$rc" "$finished_at" || true
  echo "[FAILED] stage=$STAGE exit_code=$rc finished_at=$finished_at" >&2 || true
fi
exit "$rc"
