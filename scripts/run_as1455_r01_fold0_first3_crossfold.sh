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
exec "$PYTHON_BIN" scripts/run_as1455_r01_fold0_first3_crossfold.py "$@"
