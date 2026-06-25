#!/usr/bin/env bash
set -euo pipefail

# Weekly rolling retrain + top-5 Sharpe strategy full backtests.
# Run from repository root: bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh

MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY="${RAW_DAILY:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
RAW_5M="${RAW_5M:-saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache}"
START_DATE="${START_DATE:-2024-07-17}"
END_DATE="${END_DATE:-2026-05-15}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_top5_full_${RUN_TAG}}"
TRAIN_OUT="${TRAIN_OUT:-${OUT_ROOT}/00_weekly_retrain}"
BT_ROOT="${BT_ROOT:-${OUT_ROOT}/01_top5_full_backtests}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
CAPACITY_MISSING_POLICY="${CAPACITY_MISSING_POLICY:-reject}"
VERBOSE="${VERBOSE:-0}"
MAX_UPDATES="${MAX_UPDATES:-}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_ROOT" "$BT_ROOT"

TRAIN_CMD=(
  python3 code/training/run_as1455_weekly_retrain_predict_v1.py
  --model-data "$MODEL_DATA"
  --out-dir "$TRAIN_OUT"
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --verbose "$VERBOSE"
)
if [[ -n "$MAX_UPDATES" ]]; then
  TRAIN_CMD+=(--max-updates "$MAX_UPDATES")
fi
if [[ "$FORCE" == "1" ]]; then
  TRAIN_CMD+=(--force)
fi

echo "[INFO] weekly retrain command: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}" 2>&1 | tee "$OUT_ROOT/weekly_retrain.log"

PRED="$TRAIN_OUT/results/weekly_predictions.h5"
PARAMS="$TRAIN_OUT/results/best_params.csv"
SCHEDULE="$TRAIN_OUT/results/model_update_schedule.csv"
PRED_SHA=$(python3 - <<PY
import hashlib
p = "$PRED"
h = hashlib.sha256()
with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1024*1024), b''):
        h.update(b)
print(h.hexdigest())
PY
)

COMMON_ARGS=(
  --predictions "$PRED"
  --prediction-file-sha256 "$PRED_SHA"
  --model-family "ML4T Ch17 NN weekly rolling retrain"
  --model-run "weekly_retrain_${RUN_TAG}"
  --model-params-file "$PARAMS"
  --raw-daily-cache-dir "$RAW_DAILY"
  --profile close_auction_skip_limit
  --capacity-mode "$CAPACITY_MODE"
  --capacity-missing-policy "$CAPACITY_MISSING_POLICY"
  --initial-cash 200000
  --commission-rate 0.000085
  --min-commission 5
  --stamp-tax-rate 0.0005
  --transfer-fee-rate 0.00001
  --slippage-bps 0
  --output-mode full
)
if [[ "$CAPACITY_MODE" != "none" ]]; then
  COMMON_ARGS+=(--raw-5m-cache-dir "$RAW_5M")
fi

run_bt() {
  local run_name="$1"; shift
  local out_dir="$BT_ROOT/$run_name"
  mkdir -p "$out_dir"
  echo "[INFO] backtest $run_name"
  python3 code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py \
    "${COMMON_ARGS[@]}" \
    --out-dir "$out_dir" \
    "$@" 2>&1 | tee "$out_dir/backtest.log"
  cp "$SCHEDULE" "$out_dir/model_update_schedule.csv"
}

# Sharpe top-5 from the previous 1050-run static-prediction grid.
run_bt "01_all5_max15_sell300_reb3" \
  --signal-name ensemble_all5_mean --signal-cols 0,1,2,3,4 --signal-mode mean \
  --max-positions 15 --sell-rank 300 --buy-candidate-rank 300 --rebalance-every 3 --rebalance-offset 0

run_bt "02_all5_max10_sell300_reb3" \
  --signal-name ensemble_all5_mean --signal-cols 0,1,2,3,4 --signal-mode mean \
  --max-positions 10 --sell-rank 300 --buy-candidate-rank 300 --rebalance-every 3 --rebalance-offset 0

run_bt "03_all5_max10_sell150_reb3" \
  --signal-name ensemble_all5_mean --signal-cols 0,1,2,3,4 --signal-mode mean \
  --max-positions 10 --sell-rank 150 --buy-candidate-rank 150 --rebalance-every 3 --rebalance-offset 0

run_bt "04_all5_max15_sell250_reb3" \
  --signal-name ensemble_all5_mean --signal-cols 0,1,2,3,4 --signal-mode mean \
  --max-positions 15 --sell-rank 250 --buy-candidate-rank 250 --rebalance-every 3 --rebalance-offset 0

run_bt "05_first3_max5_sell300_reb2" \
  --signal-name ensemble_first3_mean --signal-cols 0,1,2 --signal-mode mean \
  --max-positions 5 --sell-rank 300 --buy-candidate-rank 300 --rebalance-every 2 --rebalance-offset 0

python3 - <<PY
import json, pathlib, pandas as pd
root = pathlib.Path("$BT_ROOT")
rows = []
for p in sorted(root.glob("*/summary.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    row = {"run_name": p.parent.name, "run_dir": str(p.parent)}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[k] = v
        elif k == "rejection_reason_counts" and isinstance(v, dict):
            for rk, rv in v.items():
                row[f"reject_{rk}"] = rv
    rows.append(row)
out = pathlib.Path("$OUT_ROOT") / "02_summary"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "top5_weekly_retrain_full_summary.csv", index=False, encoding="utf-8-sig")
if not df.empty and "sharpe" in df.columns:
    df.sort_values("sharpe", ascending=False).to_csv(out / "leaderboard_by_sharpe.csv", index=False, encoding="utf-8-sig")
print(df[[c for c in ["run_name","final_nav","total_return","annual_return","sharpe","max_drawdown","monthly_win_rate","trade_win_rate","gross_trade_amount","total_fee"] if c in df.columns]].to_string(index=False))
PY

echo "[DONE] $OUT_ROOT"
du -sh "$OUT_ROOT" || true
