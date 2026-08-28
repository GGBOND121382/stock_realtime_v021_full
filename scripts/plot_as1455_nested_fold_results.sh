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

OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1}"
PLOTS_DIR="${PLOTS_DIR:-$OUT_ROOT/plots}"

[[ -d "$OUT_ROOT" ]] || { echo "[ERROR] nested result root not found: $OUT_ROOT" >&2; exit 1; }
[[ -f "$OUT_ROOT/nested_fold_target_results.csv" ]] || { echo "[ERROR] incomplete nested result root: $OUT_ROOT" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile scripts/plot_as1455_nested_fold_results.py

args=(
  "$PYTHON_BIN"
  scripts/plot_as1455_nested_fold_results.py
  --out-root "$OUT_ROOT"
  --plots-dir "$PLOTS_DIR"
  --overwrite
)
[[ "${SKIP_PER_SEGMENT_PLOTS:-0}" == "1" ]] && args+=(--skip-per-segment)
[[ "${SKIP_CONTINUOUS_PLOTS:-0}" == "1" ]] && args+=(--skip-continuous)

"${args[@]}"
echo "[PASS] nested fold plots generated"
echo "[PASS] plots=$PLOTS_DIR"
