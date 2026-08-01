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
KEEP_FORWARD_FULL_AUDIT="${KEEP_FORWARD_FULL_AUDIT:-0}"

"$PYTHON_BIN" -m py_compile \
  scripts/export_as1455_global_forward_rebalance_positions.py

# The historical grids keep their existing compact/summary retention.  Only the
# single frozen strict-forward run is temporarily written in full mode so its
# post-rebalance position snapshots can be extracted exactly.
export FORWARD_OUTPUT_MODE=full
bash scripts/run_as1455_refresh_all_fixed_signal_backtests.sh

export_args=(
  "$PYTHON_BIN"
  scripts/export_as1455_global_forward_rebalance_positions.py
  --matrix-root "$MATRIX_ROOT"
)
if [[ "$KEEP_FORWARD_FULL_AUDIT" != "1" ]]; then
  export_args+=(--prune-full-audit)
fi
"${export_args[@]}"

echo "[PASS] refreshed nine fixed-signal backtests with complete forward rebalance holdings"
echo "[PASS] output=$MATRIX_ROOT"
