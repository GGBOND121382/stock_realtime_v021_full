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
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1}"
SKIP_DATA_REFRESH="${SKIP_DATA_REFRESH:-0}"
TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4,5}"

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_close_auction_grid_fixed_all5_ensemble.py \
  scripts/run_as1455_close_auction_grid_fixed_first3_ensemble.py \
  scripts/run_as1455_close_auction_grid_fixed_best_model.py \
  scripts/finalize_as1455_all5_global_fold_forward_results.py \
  scripts/add_as1455_all5_rebalance_markers_to_global_plots.py

if [[ "$SKIP_DATA_REFRESH" != "1" ]]; then
  echo "===== 0/4 refresh shared data once ====="
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
  echo "[SKIP] shared data refresh disabled; reuse $FORWARD_MODEL_DATA"
fi
[[ -s "$FORWARD_MODEL_DATA" ]] || { echo "[ERROR] missing forward model_data: $FORWARD_MODEL_DATA" >&2; exit 1; }

prepare_target_cache() {
  local target_col="$1"
  local rebalance_every="$2"
  local nested_root="${3:-}"
  echo "===== prepare cache: target=$target_col rebalance=$rebalance_every ====="
  local args=(
    PYTHON_BIN="$PYTHON_BIN"
    FEATURE_PRESET=rotation_addon_onehot
    TARGET_COL="$target_col"
    TARGET_FOLDS="$TARGET_FOLDS"
    TOP_N=5
    REBALANCE_EVERY="$rebalance_every"
    HISTORICAL_MODEL_DATA="$HISTORICAL_MODEL_DATA"
    FORWARD_MODEL_DATA="$FORWARD_MODEL_DATA"
    SOURCE_DIR="$SOURCE_DIR"
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR"
    CACHE_BASE="$CACHE_BASE"
    FORCE_HISTORICAL_PREDICTIONS="${FORCE_HISTORICAL_PREDICTIONS:-0}"
    REBUILD_FORWARD_PREDICTIONS=1
  )
  if [[ -n "$nested_root" ]]; then
    args+=(NESTED_PREDICTION_SOURCE_ROOT="$nested_root")
  fi
  env "${args[@]}" bash scripts/prepare_as1455_global_fixed_signal_prediction_cache.sh
}

run_experiment() {
  local target_col="$1"
  local rebalance_every="$2"
  local signal_kind="$3"
  local out_name="$4"
  echo "===== experiment: target=$target_col rebalance=$rebalance_every signal=$signal_kind ====="
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    FEATURE_PRESET=rotation_addon_onehot \
    TARGET_COL="$target_col" \
    TARGET_FOLDS="$TARGET_FOLDS" \
    SIGNAL_KIND="$signal_kind" \
    REBALANCE_EVERY="$rebalance_every" \
    PREPARE_CACHE=0 \
    HISTORICAL_MODEL_DATA="$HISTORICAL_MODEL_DATA" \
    FORWARD_MODEL_DATA="$FORWARD_MODEL_DATA" \
    SOURCE_DIR="$SOURCE_DIR" \
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" \
    CACHE_BASE="$CACHE_BASE" \
    OUT_ROOT="$MATRIX_ROOT/$out_name" \
    FORCE="${FORCE:-0}" \
    bash scripts/run_as1455_global_fixed_signal_experiment.sh
}

mkdir -p "$MATRIX_ROOT"

prepare_target_cache \
  r05_fwd 5 \
  "saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1"
run_experiment r05_fwd 5 all5 r05_all5_reb5_fold0_5_forward

prepare_target_cache r01_fwd 1
run_experiment r01_fwd 1 all5   r01_all5_reb1_fold0_5_forward
run_experiment r01_fwd 1 first3 r01_first3_reb1_fold0_5_forward
run_experiment r01_fwd 1 best   r01_best_reb1_fold0_5_forward

# r21 target_fold5 strictly requires source_fold6; do not silently omit it.
prepare_target_cache r21_fwd 21
run_experiment r21_fwd 21 all5   r21_all5_reb21_fold0_5_forward
run_experiment r21_fwd 21 first3 r21_first3_reb21_fold0_5_forward
run_experiment r21_fwd 21 best   r21_best_reb21_fold0_5_forward

"$PYTHON_BIN" - "$MATRIX_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
rows = []
for result_file in sorted(root.glob("*/strict_forward_result.csv")):
    df = pd.read_csv(result_file)
    if df.empty:
        continue
    row = df.iloc[0].to_dict()
    row["experiment"] = result_file.parent.name
    row["result_file"] = str(result_file)
    manifest_file = result_file.parent / "global_fold0_to_fold5_forward_manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for key in (
            "target_col",
            "fixed_signal_kind",
            "fixed_signal_spec",
            "rebalance_every",
            "expected_historical_grid_count",
        ):
            row[key] = manifest.get(key)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(root / "fixed_signal_matrix_summary.csv", index=False, encoding="utf-8-sig")
(root / "fixed_signal_matrix_manifest.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "experiment_count": len(rows),
            "experiments": [row["experiment"] for row in rows],
            "summary_file": str(root / "fixed_signal_matrix_summary.csv"),
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)
columns = [
    c for c in (
        "experiment",
        "target_col",
        "fixed_signal_kind",
        "rebalance_every",
        "total_return",
        "sharpe",
        "max_drawdown",
        "forward_start",
        "forward_end",
    ) if c in summary.columns
]
print(summary[columns].to_string(index=False))
print(f"[OK] matrix summary={root / 'fixed_signal_matrix_summary.csv'}")
PY

echo "[PASS] requested seven-experiment matrix finished"
echo "[PASS] output=$MATRIX_ROOT"
