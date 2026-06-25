#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PREDICTIONS="${PREDICTIONS:-saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_smoke}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"

python3 code/backtest/run_as1455_close_auction_grid_v1.py \
  --smoke \
  --force \
  --out-root "$OUT_ROOT" \
  --predictions "$PREDICTIONS" \
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
  --profile close_auction_skip_limit \
  --capacity-mode "$CAPACITY_MODE"

echo
ls -lh "$OUT_ROOT/02_summary" || true
echo
echo "Smoke summary:"
python3 - <<'PY'
from pathlib import Path
import pandas as pd
p = Path('saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_smoke/02_summary/grid_summary_compact.csv')
if p.exists():
    cols = ['run_name','status','max_positions','sell_rank','rebalance_every','final_equity','total_return','sharpe','max_drawdown','gross_trade_amount','total_fee','trade_win_rate']
    df = pd.read_csv(p)
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
else:
    print(f'[WARN] summary not found: {p}')
PY
