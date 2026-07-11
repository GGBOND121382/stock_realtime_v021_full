#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGET_COL="${TARGET_COL:-r21_fwd}"
REBALANCE_EVERY="${REBALANCE_EVERY:-21}"
OFFSET_MODE="${OFFSET_MODE:-full}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-5,10,15,20,25}"
SELL_RANK_LIST="${SELL_RANK_LIST:-75,100,150,200,250,300}"
# r21_fwd does not have enough valid dates for source fold6 in the current AS1455 dataset.
# Therefore the default one-fold-lag backtest uses target folds 0..4:
# source fold5 -> target fold4, ..., source fold1 -> target fold0.
# If fold6_search is available in a future dataset, override with TARGET_FOLDS="0,1,2,3,4,5".
TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4}"
TOP_N="${TOP_N:-5}"
OUT_BASE="${OUT_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"
FORCE_GRID="${FORCE_GRID:-1}"
SMOKE="${SMOKE:-0}"
PARITY_CHECK_ONLY="${PARITY_CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

for preset in $FEATURE_PRESETS; do
  out_root="$OUT_BASE/${preset}_${TARGET_COL}_reb${REBALANCE_EVERY}_$(date +%Y%m%d)"
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
  if [[ "$FORCE_GRID" == "1" ]]; then
    args+=(--force-grid)
  fi
  if [[ "$SMOKE" == "1" ]]; then
    args+=(--smoke)
  fi
  if [[ "$PARITY_CHECK_ONLY" == "1" ]]; then
    args+=(--parity-check-only)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  "$PYTHON_BIN" "${args[@]}"
  echo "Output root: $out_root"
  echo "Key metrics:"
  echo "  $out_root/01_close_auction_grid/02_summary/grid_summary_compact.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_sharpe.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_calmar.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_max_drawdown.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_trade_win_rate.csv"
  echo "  $out_root/01_close_auction_grid/02_summary/leaderboard_by_fee_efficiency.csv"
  ls -lh "$out_root/01_close_auction_grid/02_summary" || true
done

echo "[DONE] r21 natural-frequency backtests finished."
