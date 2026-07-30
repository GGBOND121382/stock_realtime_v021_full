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

MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
MIGRATION_MODE="${MIGRATION_MODE:-symlink}"
TOP_N="${TOP_N:-5}"

"$PYTHON_BIN" -m py_compile \
  scripts/find_as1455_compatible_historical_result.py \
  scripts/resolve_as1455_fixed_signal_matrix_folds.py \
  scripts/migrate_as1455_r05_historical_results_to_unified.py

args=(
  --matrix-root "$MATRIX_ROOT"
  --top-n "$TOP_N"
  --mode "$MIGRATION_MODE"
)
[[ -n "${TARGET_FOLDS:-}" ]] && args+=(--target-folds "$TARGET_FOLDS")
[[ "${REPLACE:-0}" == "1" ]] && args+=(--replace)
[[ "${ALLOW_MISSING:-0}" == "1" ]] && args+=(--allow-missing)

"$PYTHON_BIN" scripts/migrate_as1455_r05_historical_results_to_unified.py "${args[@]}" "$@"
