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

OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r21_fixed_signal_nested_v1}"

"$PYTHON_BIN" -m py_compile \
  scripts/plot_as1455_nested_fold_results.py \
  scripts/plot_as1455_nested_fold_results_dynamic.py

[[ -s "$OUT_ROOT/r21_fixed_signal_nested_matrix_manifest.json" ]] || {
  echo "[ERROR] missing completed matrix manifest: $OUT_ROOT/r21_fixed_signal_nested_matrix_manifest.json" >&2
  exit 1
}

for signal_kind in all5 first3 best; do
  signal_root="$OUT_ROOT/$signal_kind"
  [[ -s "$signal_root/nested_fold_target_results.csv" ]] || {
    echo "[ERROR] missing completed nested result table: $signal_root/nested_fold_target_results.csv" >&2
    exit 1
  }

  args=(
    "$PYTHON_BIN"
    scripts/plot_as1455_nested_fold_results_dynamic.py
    --out-root "$signal_root"
    --plots-dir "$signal_root/plots"
    --overwrite
  )
  [[ "${SKIP_PER_SEGMENT_PLOTS:-0}" == "1" ]] && args+=(--skip-per-segment)
  [[ "${SKIP_CONTINUOUS_PLOTS:-0}" == "1" ]] && args+=(--skip-continuous)

  echo "[PLOT] signal=$signal_kind root=$signal_root"
  "${args[@]}"
done

echo "[PASS] r21 fixed-signal nested plots regenerated"
echo "[PASS] output=$OUT_ROOT"
