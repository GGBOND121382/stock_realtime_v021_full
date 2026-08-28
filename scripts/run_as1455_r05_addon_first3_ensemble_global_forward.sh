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
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fold_selection/r05_addon_first3_ensemble_fold0_5_forward_$RUN_STAMP}"
HISTORICAL_ROOT="$OUT_ROOT/historical_fold0_to_fold5_selection"
FORWARD_ROOT="$OUT_ROOT/strict_oos_forward"
FIXED_GRID_SCRIPT="$PWD/scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py"
DEFAULT_PREDICTION_SOURCE="saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1"
if [[ -z "${PREDICTION_SOURCE_ROOT+x}" ]]; then
  [[ -d "$DEFAULT_PREDICTION_SOURCE" ]] && PREDICTION_SOURCE_ROOT="$DEFAULT_PREDICTION_SOURCE" || PREDICTION_SOURCE_ROOT=""
fi
CAPACITY_MODE="${CAPACITY_MODE:-none}"
HISTORICAL_OUTPUT_MODE="${HISTORICAL_OUTPUT_MODE:-compact}"
FORWARD_OUTPUT_MODE="${FORWARD_OUTPUT_MODE:-compact}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"

[[ -f "$MODEL_DATA" ]] || { echo "[ERROR] missing model_data: $MODEL_DATA" >&2; exit 1; }
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2; exit 1; }
[[ "$CAPACITY_MODE" == "none" ]] || { echo "[ERROR] this wrapper currently requires CAPACITY_MODE=none" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile \
  scripts/reuse_as1455_nested_predictions_for_global_grid.py \
  scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py \
  scripts/run_as1455_target_one_lag_backtest.py \
  scripts/run_as1455_fold0_forward_backtest.py \
  scripts/plot_as1455_backtest_return_curves.py \
  scripts/finalize_as1455_global_fold_forward_results.py \
  scripts/add_as1455_rebalance_markers_to_global_plots.py

"$PYTHON_BIN" - <<'PYTEST'
from scripts.run_as1455_close_auction_grid_fixed_first3_ensemble import (
    FIXED_SIGNAL_SPEC,
    replace_signal_specs,
)
args = replace_signal_specs([
    "--signal-spec", "model_0:0:single",
    "--signal-spec=ensemble_all5_mean:0,1,2,3,4:mean",
])
assert args == ["--signal-spec", FIXED_SIGNAL_SPEC], args
print("[PASS] fixed first3 signal adapter")
PYTEST

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_ROOT" --min-free-gb "$MIN_FREE_GB" \
  --label first3-global-fold0-5-forward

printf '%s\n' \
  "[MODE] signal=ensemble_first3_mean only" \
  "[MODE] historical development set=target_fold5..target_fold0" \
  "[MODE] one global grid=5 max_positions x 6 sell_ranks x 5 offsets=150" \
  "[MODE] strict forward backtest=1 frozen configuration" \
  "[MODE] prediction_source_root=${PREDICTION_SOURCE_ROOT:-<regenerate>}" \
  "[MODE] out_root=$OUT_ROOT"

common_historical_args=(
  --feature-preset rotation_addon_onehot
  --target-col r05_fwd
  --target-folds 0,1,2,3,4,5
  --rebalance-every 5
  --offset-mode full
  --top-n 5
  --out-root "$HISTORICAL_ROOT"
  --grid-script "$FIXED_GRID_SCRIPT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --capacity-mode "$CAPACITY_MODE"
  --output-mode "$HISTORICAL_OUTPUT_MODE"
  --max-positions-list 5,10,15,20,25
  --sell-rank-list 75,100,150,200,250,300
  --python-bin "$PYTHON_BIN"
)
[[ "${FORCE:-0}" == "1" ]] && common_historical_args+=(--force-grid)

if [[ -n "$PREDICTION_SOURCE_ROOT" ]]; then
  reuse_args=(
    "$PYTHON_BIN"
    scripts/reuse_as1455_nested_predictions_for_global_grid.py
    --nested-root "$PREDICTION_SOURCE_ROOT"
    --out-root "$HISTORICAL_ROOT"
  )
  [[ "${FORCE:-0}" == "1" ]] && reuse_args+=(--force)
  "${reuse_args[@]}"
  "$PYTHON_BIN" scripts/run_as1455_target_one_lag_backtest.py \
    "${common_historical_args[@]}" \
    --skip-predictions \
    --prediction-file "$HISTORICAL_ROOT/00_predictions/test_preds.h5"
else
  "$PYTHON_BIN" scripts/run_as1455_target_one_lag_backtest.py \
    "${common_historical_args[@]}" \
    --model-data "$MODEL_DATA"
fi

forward_args=(
  "$PYTHON_BIN"
  scripts/run_as1455_fold0_forward_backtest.py
  --feature-preset rotation_addon_onehot
  --target-col r05_fwd
  --rebalance-every 5
  --model-selection-mode strict_oos
  --selection-backtest-root "$HISTORICAL_ROOT"
  --selection-rank-metric sharpe
  --out-root "$FORWARD_ROOT"
  --grid-script "$FIXED_GRID_SCRIPT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --capacity-mode "$CAPACITY_MODE"
  --output-mode "$FORWARD_OUTPUT_MODE"
  --python-bin "$PYTHON_BIN"
  --model-data "$MODEL_DATA"
)
[[ "${FORCE:-0}" == "1" ]] && forward_args+=(--force-grid)
[[ -n "${START_DATE:-}" ]] && forward_args+=(--start-date "$START_DATE")
[[ -n "${END_DATE:-}" ]] && forward_args+=(--end-date "$END_DATE")

if [[ -n "$PREDICTION_SOURCE_ROOT" ]]; then
  FORWARD_PREDICTION="$PREDICTION_SOURCE_ROOT/source_fold0/forward/00_predictions/forward_preds.h5"
  if [[ -f "$FORWARD_PREDICTION" ]]; then
    forward_args+=(--skip-predictions --prediction-file "$FORWARD_PREDICTION")
  else
    echo "[WARN] reusable forward prediction missing; fold0 inference will run"
  fi
fi
"${forward_args[@]}"

"$PYTHON_BIN" scripts/plot_as1455_backtest_return_curves.py \
  --backtest-root "$FORWARD_ROOT" \
  --label "Strict forward: first3 ensemble selected on folds0-5" \
  --rank-metric sharpe \
  --out-dir "$OUT_ROOT/plots" \
  --title-prefix "AS1455 fixed first3 ensemble strict forward" \
  --show-selected

finalize_args=(
  "$PYTHON_BIN"
  scripts/finalize_as1455_global_fold_forward_results.py
  --out-root "$OUT_ROOT"
)
[[ -n "$PREDICTION_SOURCE_ROOT" ]] && finalize_args+=(--prediction-source-root "$PREDICTION_SOURCE_ROOT")
"${finalize_args[@]}"
"$PYTHON_BIN" scripts/add_as1455_rebalance_markers_to_global_plots.py \
  --out-root "$OUT_ROOT"

echo "[PASS] fixed first3 ensemble global fold0..5 -> strict forward finished"
echo "[PASS] output=$OUT_ROOT"
