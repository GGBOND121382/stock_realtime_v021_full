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
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
FORWARD_MODEL_DIR="${FORWARD_MODEL_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"
FORWARD_MODEL_DATA="${FORWARD_MODEL_DATA:-$FORWARD_MODEL_DIR/model_data_as1455.h5}"
CACHE_BASE="${CACHE_BASE:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_prediction_cache}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"
SKIP_DATA_REFRESH="${SKIP_DATA_REFRESH:-0}"
FORCE_HISTORICAL_GRID="${FORCE_HISTORICAL_GRID:-0}"
FORCE_HISTORICAL_PREDICTIONS="${FORCE_HISTORICAL_PREDICTIONS:-0}"
RESET_FORWARD_RESULTS="${RESET_FORWARD_RESULTS:-1}"
TOP_N="${TOP_N:-5}"

mkdir -p "$MATRIX_ROOT"
PLAN_JSON="$MATRIX_ROOT/fold_availability_plan.json"

"$PYTHON_BIN" -m py_compile \
  scripts/resolve_as1455_fixed_signal_matrix_folds.py \
  scripts/run_as1455_close_auction_grid_fixed_all5_ensemble.py \
  scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py \
  scripts/run_as1455_close_auction_grid_fixed_best_model.py \
  scripts/finalize_as1455_dynamic_global_fold_forward_results.py \
  scripts/reuse_as1455_nested_predictions_for_global_grid.py

# Resolve folds before the long data refresh.  fold0..4 are mandatory; fold5 is
# added only when source_fold6 contains at least TOP_N valid saved checkpoints.
eval "$(
  "$PYTHON_BIN" scripts/resolve_as1455_fixed_signal_matrix_folds.py \
    --top-n "$TOP_N" \
    --output-json "$PLAN_JSON" \
    --format shell
)"

if [[ "$SKIP_DATA_REFRESH" != "1" ]]; then
  echo "===== 0/5 refresh market data and forward model_data once ====="
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
[[ -s "$FORWARD_MODEL_DATA" ]] || { echo "[ERROR] missing forward model_data: $FORWARD_MODEL_DATA" >&2; exit 1; }

prepare_target_cache() {
  local target_col="$1"
  local rebalance_every="$2"
  local target_folds="$3"
  local nested_root="${4:-}"
  local args=(
    PYTHON_BIN="$PYTHON_BIN"
    FEATURE_PRESET=rotation_addon_onehot
    TARGET_COL="$target_col"
    TARGET_FOLDS="$target_folds"
    TOP_N="$TOP_N"
    REBALANCE_EVERY="$rebalance_every"
    HISTORICAL_MODEL_DATA="$HISTORICAL_MODEL_DATA"
    FORWARD_MODEL_DATA="$FORWARD_MODEL_DATA"
    SOURCE_DIR="$SOURCE_DIR"
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR"
    CACHE_BASE="$CACHE_BASE"
    FORCE_HISTORICAL_PREDICTIONS="$FORCE_HISTORICAL_PREDICTIONS"
    REBUILD_FORWARD_PREDICTIONS=1
  )
  if [[ -n "$nested_root" && -d "$nested_root" ]]; then
    args+=(NESTED_PREDICTION_SOURCE_ROOT="$nested_root")
  fi
  echo "===== cache target=$target_col folds=$target_folds ====="
  env "${args[@]}" bash scripts/prepare_as1455_global_fixed_signal_prediction_cache.sh
}

experiment_names=()
run_experiment() {
  local target_col="$1"
  local rebalance_every="$2"
  local target_folds="$3"
  local fold_label="$4"
  local signal_kind="$5"
  local out_name="${target_col%%_*}_${signal_kind}_reb${rebalance_every}_${fold_label}_forward"
  experiment_names+=("$out_name")
  echo "===== experiment target=$target_col rebalance=$rebalance_every folds=$target_folds signal=$signal_kind ====="
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    FEATURE_PRESET=rotation_addon_onehot \
    TARGET_COL="$target_col" \
    TARGET_FOLDS="$target_folds" \
    SIGNAL_KIND="$signal_kind" \
    TOP_N="$TOP_N" \
    REBALANCE_EVERY="$rebalance_every" \
    PREPARE_CACHE=0 \
    HISTORICAL_MODEL_DATA="$HISTORICAL_MODEL_DATA" \
    FORWARD_MODEL_DATA="$FORWARD_MODEL_DATA" \
    SOURCE_DIR="$SOURCE_DIR" \
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
    CACHE_BASE="$CACHE_BASE" \
    OUT_ROOT="$MATRIX_ROOT/$out_name" \
    FORCE_HISTORICAL_GRID="$FORCE_HISTORICAL_GRID" \
    RESET_FORWARD_RESULTS="$RESET_FORWARD_RESULTS" \
    bash scripts/run_as1455_global_fixed_signal_experiment.sh
}

run_target_matrix() {
  local target_col="$1"
  local rebalance_every="$2"
  local target_folds="$3"
  local fold_label="$4"
  local nested_root="${5:-}"
  prepare_target_cache "$target_col" "$rebalance_every" "$target_folds" "$nested_root"
  run_experiment "$target_col" "$rebalance_every" "$target_folds" "$fold_label" all5
  run_experiment "$target_col" "$rebalance_every" "$target_folds" "$fold_label" first3
  run_experiment "$target_col" "$rebalance_every" "$target_folds" "$fold_label" best
}

run_target_matrix r01_fwd 1 "$TARGET_FOLDS_R01" "$FOLD_LABEL_R01"
run_target_matrix \
  r05_fwd 5 "$TARGET_FOLDS_R05" "$FOLD_LABEL_R05" \
  "saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1"
run_target_matrix r21_fwd 21 "$TARGET_FOLDS_R21" "$FOLD_LABEL_R21"

EXPERIMENT_LIST="$MATRIX_ROOT/expected_experiments.txt"
printf '%s\n' "${experiment_names[@]}" > "$EXPERIMENT_LIST"

"$PYTHON_BIN" - "$MATRIX_ROOT" "$PLAN_JSON" "$EXPERIMENT_LIST" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
plan_file = Path(sys.argv[2])
experiment_file = Path(sys.argv[3])
expected = [line.strip() for line in experiment_file.read_text().splitlines() if line.strip()]
plan = json.loads(plan_file.read_text(encoding='utf-8'))
rows = []
missing = []
for experiment in expected:
    result_file = root / experiment / 'strict_forward_result.csv'
    if not result_file.is_file():
        missing.append(str(result_file))
        continue
    frame = pd.read_csv(result_file)
    if frame.empty:
        missing.append(f'empty: {result_file}')
        continue
    row = frame.iloc[0].to_dict()
    row['experiment'] = experiment
    row['result_file'] = str(result_file)
    manifest_file = result_file.parent / 'global_fold0_to_fold5_forward_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    for key in (
        'target_col',
        'fixed_signal_kind',
        'fixed_signal_spec',
        'rebalance_every',
        'offset_mode',
        'expected_historical_grid_count',
        'historical_target_folds',
        'historical_source_folds',
        'target_fold5_skipped',
        'strict_forward_start',
        'strict_forward_end',
    ):
        row[key] = manifest.get(key)
    rows.append(row)
if missing:
    raise RuntimeError('matrix results are incomplete:\n' + '\n'.join(missing))
if len(rows) != 9:
    raise RuntimeError(f'expected 9 completed experiments, got {len(rows)}')

summary = pd.DataFrame(rows)
preferred = [
    'experiment', 'target_col', 'rebalance_every', 'fixed_signal_kind',
    'historical_target_folds', 'total_return', 'annual_return', 'sharpe',
    'max_drawdown', 'forward_start', 'forward_end', 'result_file',
]
ordered = [column for column in preferred if column in summary.columns]
ordered += [column for column in summary.columns if column not in ordered]
summary = summary[ordered]
summary_file = root / 'fixed_signal_matrix_summary.csv'
summary.to_csv(summary_file, index=False, encoding='utf-8-sig')

manifest = {
    'status': 'ok',
    'experiment_count': len(rows),
    'experiments': expected,
    'fold_availability_plan': plan,
    'summary_file': str(summary_file),
    'semantics': {
        'signals': ['all5', 'first3', 'best'],
        'rebalance_targets': {'r01_fwd': 1, 'r05_fwd': 5, 'r21_fwd': 21},
        'fold_rule': 'use target_fold0..5 when source_fold6 has top-5 checkpoints; otherwise use target_fold0..4',
        'historical_grid_selection_metric': 'sharpe',
        'forward_used_for_selection': False,
    },
}
(root / 'fixed_signal_matrix_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
    encoding='utf-8',
)
show = [
    column for column in (
        'experiment', 'target_col', 'fixed_signal_kind', 'rebalance_every',
        'historical_target_folds', 'total_return', 'sharpe', 'max_drawdown',
        'forward_start', 'forward_end',
    ) if column in summary.columns
]
print(summary[show].to_string(index=False))
print(f'[OK] nine-experiment summary={summary_file}')
PY

echo "[PASS] refreshed all nine fixed-signal backtests"
echo "[PASS] output=$MATRIX_ROOT"
