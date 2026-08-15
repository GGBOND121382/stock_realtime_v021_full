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

# Diagnostic-only experiment.  It does not train models, run Grid search,
# mutate tracking/live/model-registry state, or install any automation.
"$PYTHON_BIN" -m py_compile scripts/run_as1455_r01_fold0_first3_crossfold.py

# The production historical fixed-signal Grid does not pass --universe, so keep
# that exact execution-data semantics by default.  A caller may still override
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
