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

OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fold_selection/r05_addon_best_model_fold0_5_forward_v1}"
HISTORICAL_ROOT="$OUT_ROOT/historical_fold0_to_fold5_selection"
FORWARD_ROOT="$OUT_ROOT/strict_oos_forward"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
FORWARD_MODEL_DIR="${FORWARD_MODEL_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"
FORWARD_MODEL_DATA="${FORWARD_MODEL_DATA:-$FORWARD_MODEL_DIR/model_data_as1455.h5}"
FIXED_GRID_SCRIPT="$PWD/scripts/run_as1455_close_auction_grid_fixed_best_model.py"
SKIP_DATA_REFRESH="${SKIP_DATA_REFRESH:-0}"
KEEP_FORWARD_BACKUP="${KEEP_FORWARD_BACKUP:-0}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
FORWARD_OUTPUT_MODE="${FORWARD_OUTPUT_MODE:-compact}"
REFRESH_STAMP="${REFRESH_STAMP:-$(date +%Y%m%d_%H%M%S)}"

[[ -d "$HISTORICAL_ROOT" ]] || { echo "[ERROR] missing historical selection root: $HISTORICAL_ROOT" >&2; exit 1; }
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2; exit 1; }
[[ "$CAPACITY_MODE" == "none" ]] || { echo "[ERROR] refresh wrapper currently requires CAPACITY_MODE=none" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_fold0_forward_backtest.py \
  scripts/run_as1455_close_auction_grid_fixed_best_model.py \
  scripts/plot_as1455_backtest_return_curves.py \
  scripts/finalize_as1455_best_model_global_fold_forward_results.py \
  scripts/add_as1455_best_model_rebalance_markers_to_global_plots.py

if [[ "$SKIP_DATA_REFRESH" != "1" ]]; then
  echo "===== 1/3 refresh historical cache and forward model_data ====="
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    SOURCE_DIR="$SOURCE_DIR" \
    FORWARD_MODEL_DIR="$FORWARD_MODEL_DIR" \
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
    TRADE_DATE="${TRADE_DATE:-today}" \
    HISTORY_END_DATE="${HISTORY_END_DATE:-auto}" \
    TIMEZONE="${TIMEZONE:-Asia/Shanghai}" \
    SKIP_HISTORY_UPDATE="${SKIP_HISTORY_UPDATE:-0}" \
    REBUILD_AS1455_DAILY_CACHE="${REBUILD_AS1455_DAILY_CACHE:-0}" \
    MIN_FREE_GB="${MIN_FREE_GB:-5}" \
    bash scripts/refresh_as1455_forward_model_data.sh
else
  echo "[SKIP] data refresh disabled; reuse $FORWARD_MODEL_DATA"
fi
[[ -s "$FORWARD_MODEL_DATA" ]] || { echo "[ERROR] missing refreshed model_data: $FORWARD_MODEL_DATA" >&2; exit 1; }

backup_root=""
if [[ -d "$FORWARD_ROOT" ]]; then
  backup_root="$OUT_ROOT/strict_oos_forward_backup_$REFRESH_STAMP"
  mv "$FORWARD_ROOT" "$backup_root"
  echo "[BACKUP] previous forward result -> $backup_root"
fi
restore_on_error() {
  status=$?
  if [[ $status -ne 0 ]]; then
    rm -rf "$FORWARD_ROOT"
    if [[ -n "$backup_root" && -d "$backup_root" ]]; then
      mv "$backup_root" "$FORWARD_ROOT"
      echo "[RESTORE] previous forward result restored after failure" >&2
    fi
  fi
  exit $status
}
trap restore_on_error ERR

mkdir -p "$FORWARD_ROOT"
echo "===== 2/3 rebuild fold0 forward predictions and frozen backtest ====="
forward_args=(
  "$PYTHON_BIN" scripts/run_as1455_fold0_forward_backtest.py
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
  --model-data "$FORWARD_MODEL_DATA"
  --force-grid
)
[[ -n "${START_DATE:-}" ]] && forward_args+=(--start-date "$START_DATE")
[[ -n "${END_DATE:-}" ]] && forward_args+=(--end-date "$END_DATE")
"${forward_args[@]}"

echo "===== 3/3 rebuild fold-return, forward, and rebalance-marker plots ====="
"$PYTHON_BIN" scripts/plot_as1455_backtest_return_curves.py \
  --backtest-root "$FORWARD_ROOT" \
  --label "Strict forward: best model selected on folds0-5" \
  --rank-metric sharpe \
  --out-dir "$OUT_ROOT/plots" \
  --title-prefix "AS1455 fixed best-model strict forward" \
  --show-selected
"$PYTHON_BIN" scripts/finalize_as1455_best_model_global_fold_forward_results.py \
  --out-root "$OUT_ROOT" \
  --prediction-source-root "refreshed_fold0_inference:$FORWARD_MODEL_DATA"
"$PYTHON_BIN" scripts/add_as1455_best_model_rebalance_markers_to_global_plots.py \
  --out-root "$OUT_ROOT"

"$PYTHON_BIN" - "$FORWARD_MODEL_DATA" "$OUT_ROOT/strict_forward_result.csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd
model_data, result_file = map(Path, sys.argv[1:])
df = pd.read_hdf(model_data, "model_data")
model_dates = pd.DatetimeIndex(df.index.get_level_values("date"))
result = pd.read_csv(result_file)
print(f"[LATEST DATA] model_data_end={model_dates.max():%Y-%m-%d}")
print(f"[LATEST FORWARD] result_end={result.iloc[0]['forward_end']}")
PY

trap - ERR
if [[ -n "$backup_root" && "$KEEP_FORWARD_BACKUP" != "1" ]]; then
  rm -rf "$backup_root"
  echo "[CLEAN] removed previous forward backup"
fi

echo "[PASS] refreshed best-model strict forward and plots"
echo "[PASS] output=$OUT_ROOT"
