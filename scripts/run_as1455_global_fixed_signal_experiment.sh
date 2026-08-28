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

FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
TARGET_COL="${TARGET_COL:?TARGET_COL is required}"
SIGNAL_KIND="${SIGNAL_KIND:?SIGNAL_KIND is required: all5, first3, or best}"
TOP_N="${TOP_N:-5}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
HISTORICAL_MODEL_DATA="${HISTORICAL_MODEL_DATA:-$SOURCE_DIR/model_data_as1455.h5}"
FORWARD_MODEL_DATA="${FORWARD_MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
CACHE_BASE="${CACHE_BASE:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_prediction_cache}"
PREPARE_CACHE="${PREPARE_CACHE:-1}"
REBUILD_FORWARD_PREDICTIONS="${REBUILD_FORWARD_PREDICTIONS:-1}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
HISTORICAL_OUTPUT_MODE="${HISTORICAL_OUTPUT_MODE:-compact}"
FORWARD_OUTPUT_MODE="${FORWARD_OUTPUT_MODE:-compact}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-5,10,15,20,25}"
SELL_RANK_LIST="${SELL_RANK_LIST:-75,100,150,200,250,300}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
FORCE_HISTORICAL_GRID="${FORCE_HISTORICAL_GRID:-${FORCE:-0}}"
RESET_FORWARD_RESULTS="${RESET_FORWARD_RESULTS:-0}"
REUSE_HISTORICAL_ROOT="${REUSE_HISTORICAL_ROOT:-}"

case "$TARGET_COL" in
  r01_fwd)
    REBALANCE_EVERY="${REBALANCE_EVERY:-1}"
    OFFSET_MODE="${OFFSET_MODE:-zero}"
    TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4,5}"
    ;;
  r05_fwd)
    REBALANCE_EVERY="${REBALANCE_EVERY:-5}"
    OFFSET_MODE="${OFFSET_MODE:-full}"
    TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4,5}"
    ;;
  r21_fwd)
    REBALANCE_EVERY="${REBALANCE_EVERY:-21}"
    OFFSET_MODE="${OFFSET_MODE:-full}"
    TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4}"
    ;;
  *) echo "[ERROR] unsupported TARGET_COL=$TARGET_COL" >&2; exit 2 ;;
esac

case "$SIGNAL_KIND" in
  all5)
    FIXED_SIGNAL_SPEC="ensemble_all5_mean:0,1,2,3,4:mean"
    GRID_SCRIPT="$PWD/scripts/run_as1455_close_auction_grid_fixed_all5_ensemble.py"
    MARKER_SCRIPT="scripts/add_as1455_all5_rebalance_markers_to_global_plots.py"
    SIGNAL_LABEL="top-five mean ensemble"
    ;;
  first3)
    FIXED_SIGNAL_SPEC="ensemble_first3_mean:0,1,2:mean"
    GRID_SCRIPT="$PWD/scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py"
    MARKER_SCRIPT="scripts/add_as1455_rebalance_markers_to_global_plots.py"
    SIGNAL_LABEL="top-three mean ensemble"
    ;;
  best)
    FIXED_SIGNAL_SPEC="model_0:0:single"
    GRID_SCRIPT="$PWD/scripts/run_as1455_close_auction_grid_fixed_best_model.py"
    MARKER_SCRIPT="scripts/add_as1455_best_model_rebalance_markers_to_global_plots.py"
    SIGNAL_LABEL="best single model"
    ;;
  *) echo "[ERROR] unsupported SIGNAL_KIND=$SIGNAL_KIND" >&2; exit 2 ;;
esac
FINALIZER="scripts/finalize_as1455_dynamic_global_fold_forward_results.py"

first_fold="${TARGET_FOLDS%%,*}"
last_fold="${TARGET_FOLDS##*,}"
fold_label="fold${first_fold}_${last_fold}"
CACHE_ROOT="${CACHE_ROOT:-$CACHE_BASE/${FEATURE_PRESET}_${TARGET_COL}_top${TOP_N}}"
HISTORICAL_CACHE_ROOT="${HISTORICAL_CACHE_ROOT:-$CACHE_ROOT/historical_${fold_label}}"
FORWARD_CACHE_ROOT="${FORWARD_CACHE_ROOT:-$CACHE_ROOT/fold0_forward_latest}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_NAME="${TARGET_COL}_${SIGNAL_KIND}_reb${REBALANCE_EVERY}_${fold_label}_forward_${RUN_STAMP}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/$DEFAULT_NAME}"
LOCAL_HISTORICAL_ROOT="$OUT_ROOT/historical_fold_selection"
FORWARD_ROOT="$OUT_ROOT/strict_oos_forward"

if [[ "$FORCE_HISTORICAL_GRID" == "1" ]]; then
  REUSE_HISTORICAL_ROOT=""
fi
if [[ -n "$REUSE_HISTORICAL_ROOT" ]]; then
  HISTORICAL_ROOT="$($PYTHON_BIN - "$REUSE_HISTORICAL_ROOT" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
  [[ -d "$HISTORICAL_ROOT" ]] || { echo "[ERROR] missing reused historical root: $HISTORICAL_ROOT" >&2; exit 1; }
  HISTORICAL_REUSED=1
else
  HISTORICAL_ROOT="$LOCAL_HISTORICAL_ROOT"
  HISTORICAL_REUSED=0
fi

[[ "$CAPACITY_MODE" == "none" ]] || { echo "[ERROR] this protocol requires CAPACITY_MODE=none" >&2; exit 1; }
[[ -s "$HISTORICAL_MODEL_DATA" ]] || { echo "[ERROR] missing historical model_data: $HISTORICAL_MODEL_DATA" >&2; exit 1; }
[[ -s "$FORWARD_MODEL_DATA" ]] || { echo "[ERROR] missing forward model_data: $FORWARD_MODEL_DATA" >&2; exit 1; }
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_target_one_lag_backtest.py \
  scripts/run_as1455_fold0_forward_backtest.py \
  "$GRID_SCRIPT" "$FINALIZER" "$MARKER_SCRIPT"

if [[ "$PREPARE_CACHE" == "1" ]]; then
  cache_env=(
    PYTHON_BIN="$PYTHON_BIN"
    FEATURE_PRESET="$FEATURE_PRESET"
    TARGET_COL="$TARGET_COL"
    TARGET_FOLDS="$TARGET_FOLDS"
    TOP_N="$TOP_N"
    REBALANCE_EVERY="$REBALANCE_EVERY"
    HISTORICAL_MODEL_DATA="$HISTORICAL_MODEL_DATA"
    FORWARD_MODEL_DATA="$FORWARD_MODEL_DATA"
    SOURCE_DIR="$SOURCE_DIR"
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR"
    CACHE_BASE="$CACHE_BASE"
    CACHE_ROOT="$CACHE_ROOT"
    HISTORICAL_CACHE_ROOT="$HISTORICAL_CACHE_ROOT"
    FORWARD_CACHE_ROOT="$FORWARD_CACHE_ROOT"
    FORCE_HISTORICAL_PREDICTIONS="${FORCE_HISTORICAL_PREDICTIONS:-0}"
    REBUILD_FORWARD_PREDICTIONS="$REBUILD_FORWARD_PREDICTIONS"
  )
  [[ -n "${NESTED_PREDICTION_SOURCE_ROOT+x}" ]] && cache_env+=(NESTED_PREDICTION_SOURCE_ROOT="$NESTED_PREDICTION_SOURCE_ROOT")
  [[ -n "${FOLD_DIR_TEMPLATE+x}" ]] && cache_env+=(FOLD_DIR_TEMPLATE="$FOLD_DIR_TEMPLATE")
  env "${cache_env[@]}" bash scripts/prepare_as1455_global_fixed_signal_prediction_cache.sh
fi

historical_cache_pred="$HISTORICAL_CACHE_ROOT/00_predictions/test_preds.h5"
forward_cache_pred="$FORWARD_CACHE_ROOT/00_predictions/fold0_forward_preds.h5"
segments_cache="$HISTORICAL_CACHE_ROOT/00_predictions/prediction_segments.csv"
[[ -s "$forward_cache_pred" ]] || { echo "[ERROR] missing forward cache: $forward_cache_pred" >&2; exit 1; }
if [[ "$HISTORICAL_REUSED" != "1" ]]; then
  [[ -s "$historical_cache_pred" ]] || { echo "[ERROR] missing historical cache: $historical_cache_pred" >&2; exit 1; }
  [[ -s "$segments_cache" ]] || { echo "[ERROR] missing segment cache: $segments_cache" >&2; exit 1; }
fi

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_ROOT" --min-free-gb "$MIN_FREE_GB" \
  --label "${TARGET_COL}-${SIGNAL_KIND}-reb${REBALANCE_EVERY}-global"

case "$OFFSET_MODE" in
  zero) GRID_COUNT=$((5 * 6)) ;;
  full) GRID_COUNT=$((5 * 6 * REBALANCE_EVERY)) ;;
  *) echo "[ERROR] unsupported OFFSET_MODE=$OFFSET_MODE" >&2; exit 2 ;;
esac

printf '%s\n' \
  "[MODE] target=$TARGET_COL" \
  "[MODE] target_folds=$TARGET_FOLDS" \
  "[MODE] rebalance_every=$REBALANCE_EVERY offset_mode=$OFFSET_MODE" \
  "[MODE] signal=$FIXED_SIGNAL_SPEC ($SIGNAL_LABEL)" \
  "[MODE] historical trading grid count=$GRID_COUNT" \
  "[MODE] historical_reused=$HISTORICAL_REUSED" \
  "[MODE] historical_root=$HISTORICAL_ROOT" \
  "[MODE] latest forward cache=$forward_cache_pred" \
  "[MODE] out_root=$OUT_ROOT"

if [[ "$HISTORICAL_REUSED" == "1" ]]; then
  echo "[REUSE] validated historical grid: $HISTORICAL_ROOT"
else
  "$PYTHON_BIN" - "$HISTORICAL_CACHE_ROOT/00_predictions" "$HISTORICAL_ROOT/00_predictions" <<'PY'
import os
import shutil
import sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
dst.mkdir(parents=True, exist_ok=True)
for source in src.iterdir():
    if not source.is_file():
        continue
    target = dst / source.name
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
print(f"[OK] materialized prediction metadata: {src} -> {dst}")
PY

  local_historical_pred="$HISTORICAL_ROOT/00_predictions/test_preds.h5"
  historical_args=(
    "$PYTHON_BIN" scripts/run_as1455_target_one_lag_backtest.py
    --feature-preset "$FEATURE_PRESET"
    --target-col "$TARGET_COL"
    --target-folds "$TARGET_FOLDS"
    --rebalance-every "$REBALANCE_EVERY"
    --offset-mode "$OFFSET_MODE"
    --top-n "$TOP_N"
    --out-root "$HISTORICAL_ROOT"
    --grid-script "$GRID_SCRIPT"
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
    --capacity-mode "$CAPACITY_MODE"
    --output-mode "$HISTORICAL_OUTPUT_MODE"
    --max-positions-list "$MAX_POSITIONS_LIST"
    --sell-rank-list "$SELL_RANK_LIST"
    --python-bin "$PYTHON_BIN"
    --model-data "$HISTORICAL_MODEL_DATA"
    --skip-predictions
    --prediction-file "$local_historical_pred"
  )
  [[ "$FORCE_HISTORICAL_GRID" == "1" ]] && historical_args+=(--force-grid)
  "${historical_args[@]}"
fi

if [[ "$RESET_FORWARD_RESULTS" == "1" && -d "$FORWARD_ROOT" ]]; then
  echo "[RESET] remove previous strict-forward result: $FORWARD_ROOT"
  rm -rf "$FORWARD_ROOT"
fi

forward_args=(
  "$PYTHON_BIN" scripts/run_as1455_fold0_forward_backtest.py
  --feature-preset "$FEATURE_PRESET"
  --target-col "$TARGET_COL"
  --rebalance-every "$REBALANCE_EVERY"
  --offset-mode "$OFFSET_MODE"
  --model-selection-mode strict_oos
  --selection-backtest-root "$HISTORICAL_ROOT"
  --selection-rank-metric sharpe
  --out-root "$FORWARD_ROOT"
  --grid-script "$GRID_SCRIPT"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --capacity-mode "$CAPACITY_MODE"
  --output-mode "$FORWARD_OUTPUT_MODE"
  --python-bin "$PYTHON_BIN"
  --model-data "$FORWARD_MODEL_DATA"
  --skip-predictions
  --prediction-file "$forward_cache_pred"
  --force-grid
)
[[ -n "${START_DATE:-}" ]] && forward_args+=(--start-date "$START_DATE")
[[ -n "${END_DATE:-}" ]] && forward_args+=(--end-date "$END_DATE")
"${forward_args[@]}"

"$PYTHON_BIN" scripts/plot_as1455_backtest_return_curves.py \
  --backtest-root "$FORWARD_ROOT" \
  --label "Strict forward: $TARGET_COL $SIGNAL_LABEL rebalance=$REBALANCE_EVERY" \
  --rank-metric sharpe \
  --out-dir "$OUT_ROOT/plots" \
  --title-prefix "AS1455 $TARGET_COL $SIGNAL_KIND rebalance=$REBALANCE_EVERY strict forward" \
  --show-selected

"$PYTHON_BIN" "$FINALIZER" \
  --signal-kind "$SIGNAL_KIND" \
  --out-root "$OUT_ROOT" \
  --historical-root "$HISTORICAL_ROOT" \
  --prediction-source-root "shared_prediction_cache:$CACHE_ROOT"
"$PYTHON_BIN" "$MARKER_SCRIPT" --out-root "$OUT_ROOT"

"$PYTHON_BIN" - "$OUT_ROOT" "$TARGET_COL" "$SIGNAL_KIND" "$FIXED_SIGNAL_SPEC" "$REBALANCE_EVERY" "$OFFSET_MODE" "$TARGET_FOLDS" "$GRID_COUNT" "$CACHE_ROOT" "$HISTORICAL_CACHE_ROOT" "$HISTORICAL_ROOT" "$HISTORICAL_REUSED" <<'PY'
import json
import sys
from pathlib import Path
(
    out_root, target_col, signal_kind, signal_spec, rebalance_every,
    offset_mode, target_folds, grid_count, cache_root, historical_cache_root,
    historical_root, historical_reused,
) = sys.argv[1:]
root = Path(out_root)
manifest_file = root / 'global_fold0_to_fold5_forward_manifest.json'
payload = json.loads(manifest_file.read_text(encoding='utf-8'))
folds = [int(v) for v in target_folds.split(',') if v.strip()]
payload.update({
    'target_col': target_col,
    'fixed_signal_kind': signal_kind,
    'fixed_signal_spec': signal_spec,
    'rebalance_every': int(rebalance_every),
    'offset_mode': offset_mode,
    'target_folds': folds,
    'expected_historical_grid_count': int(grid_count),
    'prediction_cache_root': cache_root,
    'historical_prediction_cache_root': historical_cache_root,
    'historical_result_root': historical_root,
    'historical_result_reused': historical_reused == '1',
    'forward_prediction_cache_is_latest_model_data_inference': True,
    'target_fold5_skipped': 5 not in folds,
})
manifest_file.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
    encoding='utf-8',
)
print(f'[OK] experiment manifest={manifest_file}')
PY

echo "[PASS] fixed-signal global experiment finished"
echo "[PASS] output=$OUT_ROOT"
