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
HISTORICAL_OUTPUT_MODE="${HISTORICAL_OUTPUT_MODE:-summary}"
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
  scripts/plot_as1455_backtest_return_curves.py

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

"$PYTHON_BIN" - "$OUT_ROOT" "$HISTORICAL_ROOT" "$FORWARD_ROOT" "$PREDICTION_SOURCE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

project = Path.cwd()
sys.path.insert(0, str(project))
from utils.as1455_model_selection import find_summary_file, read_csv_auto, select_historical_signal, successful_rows

out_root, history_root, forward_root = map(lambda value: Path(value).resolve(), sys.argv[1:4])
prediction_source = sys.argv[4] or None
selection = select_historical_signal(backtest_root=history_root, rank_metric="sharpe")
if selection.signal_spec != "ensemble_first3_mean:0,1,2:mean":
    raise RuntimeError(f"unexpected selected signal: {selection.signal_spec}")
summary_file, _ = find_summary_file(history_root)
grid = successful_rows(read_csv_auto(summary_file))
grid["sharpe"] = pd.to_numeric(grid["sharpe"], errors="coerce")
grid.sort_values("sharpe", ascending=False).head(20).to_csv(
    out_root / "historical_grid_top20.csv", index=False, encoding="utf-8-sig"
)
strict_file = forward_root / "01_close_auction_grid" / "strict_oos_manifest.json"
strict = json.loads(strict_file.read_text(encoding="utf-8"))
run_name = strict["retained_run_name"]
run_dir = forward_root / "01_close_auction_grid" / "01_runs" / run_name
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
row = {
    "signal_spec": selection.signal_spec,
    "historical_rank_metric": selection.rank_metric,
    "historical_rank_metric_value": selection.rank_metric_value,
    "max_positions": selection.historical_max_positions,
    "sell_rank": selection.historical_sell_rank,
    "historical_offset": selection.historical_rebalance_offset,
    "effective_forward_offset": config.get("rebalance_offset"),
    **summary,
}
pd.DataFrame([row]).to_csv(out_root / "strict_forward_result.csv", index=False, encoding="utf-8-sig")
manifest = {
    "protocol": "global_fold0_to_fold5_fixed_first3_ensemble_then_strict_forward",
    "fixed_signal_spec": "ensemble_first3_mean:0,1,2:mean",
    "historical_target_folds": [5, 4, 3, 2, 1, 0],
    "historical_source_folds": [6, 5, 4, 3, 2, 1],
    "historical_grid_count": int(len(grid)),
    "forward_grid_count": 0,
    "forward_fixed_backtest_count": 1,
    "historical_folds_used_for_selection": True,
    "forward_results_used_for_selection": False,
    "pure_nested_historical_evaluation": False,
    "forward_account_starts_empty": True,
    "prediction_source_root": prediction_source,
    "selection": selection.to_dict(),
    "strict_oos_manifest": strict,
    "strict_forward_run_dir": str(run_dir),
    "strict_forward_summary": summary,
    "model_training": False,
    "data_refresh": False,
    "model_data_rebuild": False,
}
(out_root / "global_fold0_to_fold5_forward_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({"status": "ok", "selection": selection.to_dict(), "forward_summary": summary}, ensure_ascii=False, indent=2))
PY

echo "[PASS] fixed first3 ensemble global fold0..5 -> strict forward finished"
echo "[PASS] output=$OUT_ROOT"
