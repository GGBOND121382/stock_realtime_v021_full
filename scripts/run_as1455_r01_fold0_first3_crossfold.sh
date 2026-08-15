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

# Diagnostic-only experiment. It does not train models, run Grid search,
# mutate tracking/live/model-registry state, or install any automation.
"$PYTHON_BIN" -m py_compile scripts/run_as1455_r01_fold0_first3_crossfold.py

# Reuse the production heavy-compute lock so this optional TensorFlow diagnostic
# never competes with live inference, dashboard rebuilds, or rolling retraining.
HEAVY_LOCK="${AS1455_HEAVY_LOCK:-saved_data/ashare_ml4t/.as1455_heavy_compute.lock}"
mkdir -p "$(dirname "$HEAVY_LOCK")"
exec 9>"$HEAVY_LOCK"
if ! flock -n 9; then
  echo "[BLOCKED] AS1455 heavy compute is already running; diagnostic not started" >&2
  exit 76
fi

# The production historical fixed-signal Grid does not pass --universe, so keep
# that exact execution-data semantics by default. A caller may still override
# it explicitly by supplying --universe <file>.
universe_args=(--universe "")
for arg in "$@"; do
  if [[ "$arg" == "--universe" || "$arg" == --universe=* ]]; then
    universe_args=()
    break
  fi
done

exec "$PYTHON_BIN" scripts/run_as1455_r01_fold0_first3_crossfold.py \
  "${universe_args[@]}" "$@"
