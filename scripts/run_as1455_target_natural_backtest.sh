#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGET_COL="${TARGET_COL:-r05_fwd}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
# Full parameter searches retain only per-run JSON summaries.  The best row is
# then re-run once in compact/full mode so plotting does not require thousands
# of duplicate NAV and drawdown files.
OUTPUT_MODE="${OUTPUT_MODE:-summary}"
MATERIALIZE_BEST="${MATERIALIZE_BEST:-1}"
MATERIALIZED_OUTPUT_MODE="${MATERIALIZED_OUTPUT_MODE:-compact}"
RANK_METRIC="${RANK_METRIC:-sharpe}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-5,10,15,20,25}"
SELL_RANK_LIST="${SELL_RANK_LIST:-75,100,150,200,250,300}"
TOP_N="${TOP_N:-5}"
OUT_BASE="${OUT_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"
FORCE_GRID="${FORCE_GRID:-1}"
SMOKE="${SMOKE:-0}"
PARITY_CHECK_ONLY="${PARITY_CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_PREDICTION_CSV="${KEEP_PREDICTION_CSV:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_BASE" \
  --min-free-gb "$MIN_FREE_GB" \
  --label historical-target-grid

read -r REBALANCE_EVERY OFFSET_MODE <<<"$($PYTHON_BIN - "$TARGET_COL" <<'PY'
import sys
from utils.as1455_ch17_common import target_spec
spec = target_spec(sys.argv[1])
print(spec.rebalance_every, spec.offset_mode)
PY
)"

case "$TARGET_COL" in
  r01_fwd|r05_fwd)
    DEFAULT_TARGET_FOLDS="0,1,2,3,4,5"
    ;;
  r21_fwd)
    # Current data has no source fold6, so target fold5 is excluded.
    DEFAULT_TARGET_FOLDS="0,1,2,3,4"
    ;;
  *)
    echo "[ERROR] unsupported TARGET_COL=$TARGET_COL" >&2
    exit 2
    ;;
esac
TARGET_FOLDS="${TARGET_FOLDS:-$DEFAULT_TARGET_FOLDS}"

for preset in $FEATURE_PRESETS; do
  out_root="$OUT_BASE/${preset}_${TARGET_COL}_reb${REBALANCE_EVERY}_${RUN_STAMP}"
  echo "===== backtest preset=${preset} target=${TARGET_COL} rebalance_every=${REBALANCE_EVERY} target_folds=${TARGET_FOLDS} output_mode=${OUTPUT_MODE} ====="
  args=(
    scripts/run_as1455_target_one_lag_backtest.py
    --feature-preset "$preset"
    --target-col "$TARGET_COL"
    --rebalance-every "$REBALANCE_EVERY"
    --offset-mode "$OFFSET_MODE"
    --model-data "$MODEL_DATA"
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
    --out-root "$out_root"
    --target-folds "$TARGET_FOLDS"
    --top-n "$TOP_N"
    --sector-encoding onehot
    --dropna-mode target_only
    --capacity-mode "$CAPACITY_MODE"
    --output-mode "$OUTPUT_MODE"
    --max-positions-list "$MAX_POSITIONS_LIST"
    --sell-rank-list "$SELL_RANK_LIST"
  )
  [[ "$FORCE_GRID" == "1" ]] && args+=(--force-grid)
  [[ "$SMOKE" == "1" ]] && args+=(--smoke)
  [[ "$PARITY_CHECK_ONLY" == "1" ]] && args+=(--parity-check-only)
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)

  "$PYTHON_BIN" "${args[@]}"

  if [[ "$KEEP_PREDICTION_CSV" != "1" && "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" scripts/compact_as1455_prediction_artifacts.py \
      --prediction-dir "$out_root/00_predictions"
  fi

  if [[ "$MATERIALIZE_BEST" == "1" \
        && "$OUTPUT_MODE" == "summary" \
        && "$PARITY_CHECK_ONLY" != "1" \
        && "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" scripts/materialize_as1455_best_run.py \
      --backtest-root "$out_root" \
      --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
      --rank-metric "$RANK_METRIC" \
      --capacity-mode "$CAPACITY_MODE" \
      --output-mode "$MATERIALIZED_OUTPUT_MODE" \
      --force
  fi

  echo "Output root: $out_root"
  echo "Key metrics:"
  echo "  $out_root/01_close_auction_grid/02_summary/grid_summary_compact.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_sharpe.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_calmar.csv"
  echo "Materialized best:"
  echo "  $out_root/materialized_best_run.json"
  ls -lh "$out_root/01_close_auction_grid/02_summary" || true
done

echo "[DONE] natural-frequency backtests finished: target=$TARGET_COL"
