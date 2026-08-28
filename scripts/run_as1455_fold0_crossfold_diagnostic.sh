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

"$PYTHON_BIN" -m py_compile scripts/run_as1455_fold0_crossfold_diagnostic.py

# Diagnostic only: never compete with live inference/dashboard rebuild/rolling
# retraining for memory.  If production heavy compute is active, fail fast.
HEAVY_LOCK="${AS1455_HEAVY_LOCK:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
mkdir -p "$(dirname "$HEAVY_LOCK")"
exec 9>"$HEAVY_LOCK"
if ! flock -n 9; then
  echo "[BLOCKED] AS1455 heavy compute is already running; diagnostic not started" >&2
  exit 76
fi

if [[ "${1:-}" == "--all" ]]; then
  shift
  for target in r01_fwd r05_fwd r21_fwd; do
    echo "===== diagnostic target=$target signals=best,first3,all5 ====="
    "$PYTHON_BIN" scripts/run_as1455_fold0_crossfold_diagnostic.py \
      --target "$target" \
      --signals best,first3,all5 \
      "$@"
  done
  echo "[PASS] all 9 target/signal diagnostics finished"
  exit 0
fi

exec "$PYTHON_BIN" scripts/run_as1455_fold0_crossfold_diagnostic.py "$@"
