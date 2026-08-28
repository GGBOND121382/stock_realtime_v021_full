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
  scripts/export_as1455_global_forward_rebalance_positions.py \
  scripts/export_as1455_global_forward_latest_states.py

# The historical grids keep their existing compact/summary retention. Only the
# single frozen strict-forward run is temporarily written in full mode so both
# rebalance-day snapshots and the latest T-1 account state can be extracted.
export FORWARD_OUTPUT_MODE=full
bash scripts/run_as1455_refresh_all_fixed_signal_backtests.sh

# First export the rebalance snapshots without pruning, then capture the latest
# daily positions/cash.  The latter are consumed by the next trade day's 14:55
# nine-strategy monitor.
"$PYTHON_BIN" scripts/export_as1455_global_forward_rebalance_positions.py \
  --matrix-root "$MATRIX_ROOT"
"$PYTHON_BIN" scripts/export_as1455_global_forward_latest_states.py \
  --matrix-root "$MATRIX_ROOT"

if [[ "$KEEP_FORWARD_FULL_AUDIT" != "1" ]]; then
  # A second pass only prunes large all-date audit CSVs.  Stable rebalance and
  # latest-state files written above remain at the experiment roots.
  "$PYTHON_BIN" scripts/export_as1455_global_forward_rebalance_positions.py \
    --matrix-root "$MATRIX_ROOT" \
    --prune-full-audit
fi

echo "[PASS] refreshed nine fixed-signal backtests with complete forward rebalance holdings"
echo "[PASS] retained latest strict-forward account state for next-day live monitoring"
echo "[PASS] output=$MATRIX_ROOT"
