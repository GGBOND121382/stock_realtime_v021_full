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

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
MODEL_DATA="${MODEL_DATA:-$SOURCE_DIR/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-$SOURCE_DIR/baostock_5m_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/rotation_addon_onehot_r05_fwd_first3_ensemble_$RUN_STAMP}"
INITIAL_CASH="${INITIAL_CASH:-200000}"
VALIDATION_OUTPUT_MODE="${VALIDATION_OUTPUT_MODE:-summary}"
TARGET_OUTPUT_MODE="${TARGET_OUTPUT_MODE:-compact}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
GRID_SCRIPT="$PWD/scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py"

[[ -f "$MODEL_DATA" ]] || { echo "[ERROR] missing model_data: $MODEL_DATA" >&2; exit 1; }
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_nested_fold_protocol.py \
  scripts/check_as1455_nested_fold_protocol.py \
  scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py \
  scripts/plot_as1455_nested_fold_results.py
"$PYTHON_BIN" scripts/check_as1455_nested_fold_protocol.py

"$PYTHON_BIN" - <<'PY'
from scripts.run_as1455_close_auction_grid_fixed_first3_ensemble import (
    FIXED_SIGNAL_SPEC,
    replace_signal_specs,
)
args = replace_signal_specs([
    "--foo", "bar",
    "--signal-spec", "model_0:0:single",
    "--signal-spec=ensemble_all5_mean:0,1,2,3,4:mean",
])
assert args == ["--foo", "bar", "--signal-spec", FIXED_SIGNAL_SPEC], args
print("[PASS] fixed first3 ensemble adapter")
PY

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_ROOT" \
  --min-free-gb "$MIN_FREE_GB" \
  --label nested-r05-addon-first3-ensemble

printf '%s\n' \
  "[MODE] nested per-source-fold trading-grid selection" \
  "[MODE] fixed_signal=ensemble_first3_mean" \
  "[MODE] checkpoint_rule=top3_by_source_fold_checkpoint_ranking_then_equal_mean" \
  "[MODE] validation_grid_per_fold=5_max_positions_x_6_sell_ranks_x_5_offsets=150" \
  "[MODE] source_fold_validation_grids=7" \
  "[MODE] target_fold_grids=0" \
  "[MODE] frozen_target_backtests=7" \
  "[MODE] source_fold6->target_fold5 ... source_fold1->target_fold0" \
  "[MODE] source_fold0->strict_oos_forward" \
  "[MODE] capacity_mode=$CAPACITY_MODE" \
  "[MODE] plots=$([[ ${SKIP_PLOTS:-0} == 1 ]] && echo false || echo true)" \
  "[MODE] training=false data_refresh=false model_data_rebuild=false" \
  "[MODE] out_root=$OUT_ROOT"

args=(
  "$PYTHON_BIN"
  scripts/run_as1455_nested_fold_protocol.py
  --model-data "$MODEL_DATA"
  --feature-preset rotation_addon_onehot
  --target-col r05_fwd
  --out-root "$OUT_ROOT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --grid-script "$GRID_SCRIPT"
  --capacity-mode "$CAPACITY_MODE"
  --initial-cash "$INITIAL_CASH"
  --validation-output-mode "$VALIDATION_OUTPUT_MODE"
  --target-output-mode "$TARGET_OUTPUT_MODE"
)

if [[ "$CAPACITY_MODE" != "none" ]]; then
  [[ -d "$RAW_5M_CACHE_DIR" ]] || { echo "[ERROR] capacity mode requires raw 5m cache: $RAW_5M_CACHE_DIR" >&2; exit 1; }
  args+=(--raw-5m-cache-dir "$RAW_5M_CACHE_DIR")
fi
[[ -n "${LAST5_PANEL:-}" ]] && args+=(--last5-panel "$LAST5_PANEL")
[[ -n "${UNIVERSE:-}" ]] && args+=(--universe "$UNIVERSE")
[[ -n "${ST_SYMBOLS:-}" ]] && args+=(--st-symbols "$ST_SYMBOLS")
[[ -n "${ST_STATUS:-}" ]] && args+=(--st-status "$ST_STATUS")
[[ -n "${CORPORATE_ACTIONS:-}" ]] && args+=(--corporate-actions "$CORPORATE_ACTIONS")
[[ -n "${START_DATE:-}" ]] && args+=(--start-date "$START_DATE")
[[ -n "${END_DATE:-}" ]] && args+=(--end-date "$END_DATE")
[[ "${FORCE:-0}" == "1" ]] && args+=(--force)
[[ "${SKIP_PARITY_CHECK:-0}" == "1" ]] && args+=(--skip-parity-check)
[[ "${SKIP_CONTINUOUS:-0}" == "1" ]] && args+=(--skip-continuous)

"${args[@]}"
if [[ "${SKIP_PLOTS:-0}" != "1" ]]; then
  plot_args=(
    "$PYTHON_BIN"
    scripts/plot_as1455_nested_fold_results.py
    --out-root "$OUT_ROOT"
    --plots-dir "${PLOTS_DIR:-$OUT_ROOT/plots}"
    --overwrite
  )
  [[ "${SKIP_PER_SEGMENT_PLOTS:-0}" == "1" ]] && plot_args+=(--skip-per-segment)
  if [[ "${SKIP_CONTINUOUS:-0}" == "1" || "${SKIP_CONTINUOUS_PLOTS:-0}" == "1" ]]; then
    plot_args+=(--skip-continuous)
  fi
  "${plot_args[@]}"
fi

echo "[PASS] nested r05 addon first3 ensemble experiment finished"
echo "[PASS] output=$OUT_ROOT"
