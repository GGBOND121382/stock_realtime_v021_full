#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PREDICTIONS="${PREDICTIONS:-saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_models_smoke}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"

python3 code/backtest/run_as1455_close_auction_grid_v1.py \
  --smoke \
  --force \
  --out-root "$OUT_ROOT" \
  --predictions "$PREDICTIONS" \
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
  --profile close_auction_skip_limit \
  --capacity-mode "$CAPACITY_MODE" \
  --run-output-mode "$OUTPUT_MODE"

echo
echo "Output mode: $OUTPUT_MODE"
echo "Smoke output: $OUT_ROOT"
ls -lh "$OUT_ROOT/02_summary" || true
echo
echo "Smoke summary:"
OUT_ROOT="$OUT_ROOT" python3 - <<'PY'
from pathlib import Path
import os
import pandas as pd
p = Path(os.environ['OUT_ROOT']) / '02_summary' / 'grid_summary_compact.csv'
if p.exists():
    cols = ['run_name','status','signal_name','max_positions','sell_rank','rebalance_every','final_nav','total_return','sharpe','max_drawdown','gross_trade_amount','total_fee','trade_win_rate']
    df = pd.read_csv(p)
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
else:
    print(f'[WARN] summary not found: {p}')
PY
