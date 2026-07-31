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

SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
HISTORICAL_MODEL_DATA="${HISTORICAL_MODEL_DATA:-$SOURCE_DIR/model_data_as1455.h5}"
FORWARD_MODEL_DATA="${FORWARD_MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r21_fixed_signal_nested_v1}"
SOURCE_FOLDS="${SOURCE_FOLDS:-auto}"
INITIAL_CASH="${INITIAL_CASH:-200000}"
VALIDATION_OUTPUT_MODE="${VALIDATION_OUTPUT_MODE:-summary}"
TARGET_OUTPUT_MODE="${TARGET_OUTPUT_MODE:-compact}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

[[ -s "$HISTORICAL_MODEL_DATA" ]] || { echo "[ERROR] missing historical model_data: $HISTORICAL_MODEL_DATA" >&2; exit 1; }
[[ -s "$FORWARD_MODEL_DATA" ]] || { echo "[ERROR] missing forward model_data: $FORWARD_MODEL_DATA" >&2; exit 1; }
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_nested_fold_protocol.py \
  scripts/run_as1455_r21_fixed_signal_nested_matrix.py \
  scripts/run_as1455_r21_fixed_signal_nested_matrix_entry.py \
  scripts/plot_as1455_nested_fold_results_dynamic.py \
  scripts/run_as1455_close_auction_grid_fixed_all5_ensemble.py \
  scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py \
  scripts/run_as1455_close_auction_grid_fixed_best_model.py

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_ROOT" \
  --min-free-gb "$MIN_FREE_GB" \
  --label r21-fixed-signal-nested-matrix

printf '%s\n' \
  "[MODE] r21 per-source-fold validation grid -> next fold / latest forward" \
  "[MODE] fixed_signals=all5,first3,best" \
  "[MODE] source_folds=$SOURCE_FOLDS (auto uses fold6 only when valid)" \
  "[MODE] validation_grid_per_source_per_signal=5x6x21=630" \
  "[MODE] predictions_shared_across_signals=true" \
  "[MODE] training=false data_refresh=false" \
  "[MODE] historical_model_data=$HISTORICAL_MODEL_DATA" \
  "[MODE] forward_model_data=$FORWARD_MODEL_DATA" \
  "[MODE] out_root=$OUT_ROOT"

args=(
  "$PYTHON_BIN"
  scripts/run_as1455_r21_fixed_signal_nested_matrix_entry.py
  --historical-model-data "$HISTORICAL_MODEL_DATA"
  --forward-model-data "$FORWARD_MODEL_DATA"
  --feature-preset rotation_addon_onehot
  --source-folds "$SOURCE_FOLDS"
  --out-root "$OUT_ROOT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --capacity-mode none
  --initial-cash "$INITIAL_CASH"
  --validation-output-mode "$VALIDATION_OUTPUT_MODE"
  --target-output-mode "$TARGET_OUTPUT_MODE"
)

[[ -n "${START_DATE:-}" ]] && args+=(--start-date "$START_DATE")
[[ -n "${END_DATE:-}" ]] && args+=(--end-date "$END_DATE")
[[ "${FORCE:-0}" == "1" ]] && args+=(--force)
[[ "${REUSE_FORWARD_PREDICTIONS:-0}" == "1" ]] && args+=(--reuse-forward-predictions)
[[ "${REUSE_FORWARD_RESULTS:-0}" == "1" ]] && args+=(--reuse-forward-results)
[[ "${SKIP_PARITY_CHECK:-0}" == "1" ]] && args+=(--skip-parity-check)
[[ "${SKIP_CONTINUOUS:-0}" == "1" ]] && args+=(--skip-continuous)
[[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)

"${args[@]}"

if [[ "${SKIP_PLOTS:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  for signal_kind in all5 first3 best; do
    plot_args=(
      "$PYTHON_BIN"
      scripts/plot_as1455_nested_fold_results_dynamic.py
      --out-root "$OUT_ROOT/$signal_kind"
      --plots-dir "$OUT_ROOT/$signal_kind/plots"
      --overwrite
    )
    [[ "${SKIP_PER_SEGMENT_PLOTS:-0}" == "1" ]] && plot_args+=(--skip-per-segment)
    if [[ "${SKIP_CONTINUOUS:-0}" == "1" || "${SKIP_CONTINUOUS_PLOTS:-0}" == "1" ]]; then
      plot_args+=(--skip-continuous)
    fi
    "${plot_args[@]}"
  done
fi

echo "[PASS] r21 fixed-signal nested matrix finished"
echo "[PASS] output=$OUT_ROOT"
