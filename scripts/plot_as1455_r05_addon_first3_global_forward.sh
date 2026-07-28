#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="${BASE_PYTHON:-python3}"
  fi
fi

OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fold_selection/r05_addon_first3_ensemble_fold0_5_forward_v1}"
"$PYTHON_BIN" -m py_compile scripts/finalize_as1455_global_fold_forward_results.py
args=(
  "$PYTHON_BIN"
  scripts/finalize_as1455_global_fold_forward_results.py
  --out-root "$OUT_ROOT"
)
[[ -n "${PREDICTION_SOURCE_ROOT:-}" ]] && args+=(--prediction-source-root "$PREDICTION_SOURCE_ROOT")
"${args[@]}"
