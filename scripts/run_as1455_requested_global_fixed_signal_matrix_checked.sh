#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="${BASE_PYTHON:-python3}"
  fi
fi

"$PYTHON_BIN" -m py_compile scripts/check_as1455_requested_fixed_signal_matrix.py
"$PYTHON_BIN" scripts/check_as1455_requested_fixed_signal_matrix.py
exec bash scripts/run_as1455_requested_global_fixed_signal_matrix.sh
