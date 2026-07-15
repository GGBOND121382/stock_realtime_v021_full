#!/usr/bin/env bash
set -Eeuo pipefail

REAL_PYTHON="${AS1455_REAL_PYTHON:?AS1455_REAL_PYTHON is required}"
MIN_AVAILABLE_MEMORY_MB="${MIN_AVAILABLE_MEMORY_MB:-2048}"
MEMORY_WAIT_ATTEMPTS="${MEMORY_WAIT_ATTEMPTS:-30}"
MEMORY_WAIT_SECONDS="${MEMORY_WAIT_SECONDS:-10}"
TRAIN_COOLDOWN_SECONDS="${TRAIN_COOLDOWN_SECONDS:-20}"
BACKTEST_COOLDOWN_SECONDS="${BACKTEST_COOLDOWN_SECONDS:-20}"
DATA_COOLDOWN_SECONDS="${DATA_COOLDOWN_SECONDS:-30}"

command -v "$REAL_PYTHON" >/dev/null 2>&1 || {
  printf '[ERROR] real Python is unavailable: %s\n' "$REAL_PYTHON" >&2
  exit 127
}

mem_available_mb() {
  awk '/^MemAvailable:/ {printf "%d\n", $2 / 1024}' /proc/meminfo
}

memory_snapshot() {
  local label="$1" available swap_free
  available="$(mem_available_mb)"
  swap_free="$(awk '/^SwapFree:/ {printf "%d", $2 / 1024}' /proc/meminfo)"
  printf '[MEMORY] %s MemAvailable=%sMB SwapFree=%sMB\n' "$label" "$available" "${swap_free:-0}"
  free -h || true
}

wait_for_memory() {
  local label="$1" attempt available
  for attempt in $(seq 1 "$MEMORY_WAIT_ATTEMPTS"); do
    available="$(mem_available_mb)"
    if (( available >= MIN_AVAILABLE_MEMORY_MB )); then
      printf '[MEMORY] %s ready: MemAvailable=%sMB threshold=%sMB\n' "$label" "$available" "$MIN_AVAILABLE_MEMORY_MB"
      return 0
    fi
    printf '[MEMORY] %s waiting: MemAvailable=%sMB threshold=%sMB attempt=%s/%s\n' \
      "$label" "$available" "$MIN_AVAILABLE_MEMORY_MB" "$attempt" "$MEMORY_WAIT_ATTEMPTS"
    sleep "$MEMORY_WAIT_SECONDS"
  done
  printf '[ERROR] memory did not recover for %s: MemAvailable=%sMB threshold=%sMB\n' \
    "$label" "$(mem_available_mb)" "$MIN_AVAILABLE_MEMORY_MB" >&2
  return 1
}

joined=" $* "
heavy_kind=""
cooldown_seconds=0
case "$joined" in
  *" scripts/run_as1455_target_fold_param_search.py "*)
    heavy_kind="training"; cooldown_seconds="$TRAIN_COOLDOWN_SECONDS" ;;
  *" scripts/run_as1455_target_one_lag_backtest.py "*|*" scripts/run_as1455_fold0_forward_backtest.py "*|*" scripts/materialize_as1455_best_run.py "*)
    heavy_kind="backtest"; cooldown_seconds="$BACKTEST_COOLDOWN_SECONDS" ;;
  *" pipelines/as1455_update_history_to_prevday_fast_v4.py "*|*" pipelines/as1455_history_parallel_dispatch.py "*|*" scripts/build_ashare_ch12_as1455_model_data.py "*)
    heavy_kind="data"; cooldown_seconds="$DATA_COOLDOWN_SECONDS" ;;
esac

if [[ -n "$heavy_kind" ]]; then
  wait_for_memory "$heavy_kind before"
  memory_snapshot "$heavy_kind before"
fi

set +e
"$REAL_PYTHON" "$@"
rc=$?
set -e

if [[ -n "$heavy_kind" ]]; then
  memory_snapshot "$heavy_kind process-exited rc=$rc"
  if (( cooldown_seconds > 0 )); then
    printf '[MEMORY] %s cooldown=%ss\n' "$heavy_kind" "$cooldown_seconds"
    sleep "$cooldown_seconds"
  fi
  wait_for_memory "$heavy_kind after"
  memory_snapshot "$heavy_kind cooldown-complete"
fi

exit "$rc"
