#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_rotation_addon_one_lag_daily_backtest_$(date +%Y%m%d)}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
# Do not put the literal {fold} placeholder inside ${VAR:-...}; bash may
# terminate the parameter expansion at the inner }, corrupting the template.
if [[ -z "${FOLD_DIR_TEMPLATE:-}" ]]; then
  FOLD_DIR_TEMPLATE='saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold{fold}_search'
fi
CAPACITY_MODE="${CAPACITY_MODE:-none}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-5,10,15,20,25}"
SELL_RANK_LIST="${SELL_RANK_LIST:-75,100,150,200,250,300}"
TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4,5}"
TOP_N="${TOP_N:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py \
  --model-data "$MODEL_DATA" \
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
  --fold-dir-template "$FOLD_DIR_TEMPLATE" \
  --out-root "$OUT_ROOT" \
  --target-folds "$TARGET_FOLDS" \
  --top-n "$TOP_N" \
  --sector-encoding onehot \
  --dropna-mode r01_only \
  --capacity-mode "$CAPACITY_MODE" \
  --output-mode "$OUTPUT_MODE" \
  --max-positions-list "$MAX_POSITIONS_LIST" \
  --sell-rank-list "$SELL_RANK_LIST" \
  --force-grid

echo
echo "Output root: $OUT_ROOT"
echo "Key metric files:"
echo "  $OUT_ROOT/01_close_auction_daily_grid/02_summary/grid_summary_compact.csv"
echo "  $OUT_ROOT/01_close_auction_daily_grid/02_summary/leaderboard_by_sharpe.csv"
echo "  $OUT_ROOT/01_close_auction_daily_grid/02_summary/leaderboard_by_calmar.csv"
echo "  $OUT_ROOT/01_close_auction_daily_grid/02_summary/leaderboard_by_max_drawdown.csv"
echo "  $OUT_ROOT/01_close_auction_daily_grid/02_summary/leaderboard_by_trade_win_rate.csv"
echo "  $OUT_ROOT/01_close_auction_daily_grid/02_summary/leaderboard_by_fee_efficiency.csv"
ls -lh "$OUT_ROOT/01_close_auction_daily_grid/02_summary" || true
