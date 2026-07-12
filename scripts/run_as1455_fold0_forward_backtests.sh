#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
OUTPUT_MODE="${OUTPUT_MODE:-full}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-5,10,15,20,25}"
SELL_RANK_LIST="${SELL_RANK_LIST:-75,100,150,200,250,300}"
TOP_N="${TOP_N:-5}"
OUT_BASE="${OUT_BASE:-saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
FORCE_GRID="${FORCE_GRID:-1}"
PARITY_CHECK_ONLY="${PARITY_CHECK_ONLY:-0}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

for target in $TARGETS; do
  case "$target" in
    r01_fwd)
      rebalance_every=1
      offset_mode=zero
      ;;
    r05_fwd)
      rebalance_every=5
      offset_mode=full
      ;;
    r21_fwd)
      rebalance_every=21
      offset_mode=full
      ;;
    *)
      echo "unsupported target: $target" >&2
      exit 2
      ;;
  esac

  for preset in $FEATURE_PRESETS; do
    out_root="$OUT_BASE/${preset}_${target}_reb${rebalance_every}_$(date +%Y%m%d)"
    echo "===== fold0 forward preset=${preset} target=${target} rebalance_every=${rebalance_every} output_mode=${OUTPUT_MODE} ====="
    args=(
      scripts/run_as1455_fold0_forward_backtest.py
      --feature-preset "$preset"
      --target-col "$target"
      --rebalance-every "$rebalance_every"
      --offset-mode "$offset_mode"
      --model-data "$MODEL_DATA"
      --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
      --out-root "$out_root"
      --top-n "$TOP_N"
      --sector-encoding onehot
      --dropna-mode target_only
      --capacity-mode "$CAPACITY_MODE"
      --output-mode "$OUTPUT_MODE"
      --max-positions-list "$MAX_POSITIONS_LIST"
      --sell-rank-list "$SELL_RANK_LIST"
    )
    if [[ -n "$START_DATE" ]]; then
      args+=(--start-date "$START_DATE")
    fi
    if [[ -n "$END_DATE" ]]; then
      args+=(--end-date "$END_DATE")
    fi
    if [[ "$FORCE_GRID" == "1" ]]; then
      args+=(--force-grid)
    fi
    if [[ "$PARITY_CHECK_ONLY" == "1" ]]; then
      args+=(--parity-check-only)
    fi
    if [[ "$SMOKE" == "1" ]]; then
      args+=(--smoke)
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      args+=(--dry-run)
    fi
    "$PYTHON_BIN" "${args[@]}"
    echo "Output root: $out_root"
  done
done

echo "[DONE] fold0 forward backtests finished."
