#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
FORWARD_MODEL_DIR="${FORWARD_MODEL_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"
REFRESH_DATA="${REFRESH_DATA:-1}"
MODEL_DATA="${MODEL_DATA:-$FORWARD_MODEL_DIR/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
OUTPUT_MODE="${OUTPUT_MODE:-full}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-5,10,15,20,25}"
SELL_RANK_LIST="${SELL_RANK_LIST:-75,100,150,200,250,300}"

# Default protocol: select the best historical model signal from the corresponding
# ch17_as1455_target_backtest directory, then use the same checkpoint rank or
# ensemble definition with fold0 checkpoints on later dates.
MODEL_SELECTION_MODE="${MODEL_SELECTION_MODE:-historical_best}"
SELECTION_RANK_METRIC="${SELECTION_RANK_METRIC:-sharpe}"
TARGET_BACKTEST_BASE="${TARGET_BACKTEST_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"
# Optional explicit root; normally leave empty so each preset/target resolves its
# latest matching completed historical backtest independently.
SELECTION_BACKTEST_ROOT="${SELECTION_BACKTEST_ROOT:-}"
# Used only when MODEL_SELECTION_MODE=all_top_n. In historical_best mode Python
# derives the minimum required checkpoint count from the selected signal columns.
TOP_N="${TOP_N:-5}"

OUT_BASE="${OUT_BASE:-saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"
FORCE_GRID="${FORCE_GRID:-1}"
PARITY_CHECK_ONLY="${PARITY_CHECK_ONLY:-0}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$REFRESH_DATA" == "1" ]]; then
  echo "===== refresh latest AS1455 history and rebuild forward model_data ====="
  refresh_env=(
    "PYTHON_BIN=$PYTHON_BIN"
    "TRADE_DATE=$TRADE_DATE"
    "HISTORY_END_DATE=$HISTORY_END_DATE"
    "TIMEZONE=$TIMEZONE"
    "UNIVERSE=$UNIVERSE"
    "SOURCE_DIR=$SOURCE_DIR"
    "FORWARD_MODEL_DIR=$FORWARD_MODEL_DIR"
  )
  [[ -n "$MAX_SYMBOLS" ]] && refresh_env+=("MAX_SYMBOLS=$MAX_SYMBOLS")
  env "${refresh_env[@]}" bash scripts/refresh_as1455_forward_model_data.sh
fi

[[ -s "$MODEL_DATA" ]] || {
  echo "[ERROR] forward model_data not found: $MODEL_DATA" >&2
  echo "Run with REFRESH_DATA=1 or set MODEL_DATA explicitly." >&2
  exit 1
}

if [[ -n "$SELECTION_BACKTEST_ROOT" ]]; then
  target_count=$(wc -w <<<"$TARGETS")
  preset_count=$(wc -w <<<"$FEATURE_PRESETS")
  if [[ "$target_count" -ne 1 || "$preset_count" -ne 1 ]]; then
    echo "[ERROR] SELECTION_BACKTEST_ROOT may be used only with one TARGETS value and one FEATURE_PRESETS value." >&2
    exit 2
  fi
fi

for target in $TARGETS; do
  read -r rebalance_every offset_mode <<<"$($PYTHON_BIN - "$target" <<'PY'
import sys
from utils.as1455_ch17_common import target_spec
spec = target_spec(sys.argv[1])
print(spec.rebalance_every, spec.offset_mode)
PY
)"

  for preset in $FEATURE_PRESETS; do
    out_root="$OUT_BASE/${preset}_${target}_reb${rebalance_every}_$(date +%Y%m%d)"
    echo "===== fold0 forward preset=${preset} target=${target} rebalance_every=${rebalance_every} selection_mode=${MODEL_SELECTION_MODE} selection_metric=${SELECTION_RANK_METRIC} output_mode=${OUTPUT_MODE} ====="
    args=(
      scripts/run_as1455_fold0_forward_backtest.py
      --feature-preset "$preset"
      --target-col "$target"
      --rebalance-every "$rebalance_every"
      --offset-mode "$offset_mode"
      --model-data "$MODEL_DATA"
      --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
      --out-root "$out_root"
      --model-selection-mode "$MODEL_SELECTION_MODE"
      --selection-rank-metric "$SELECTION_RANK_METRIC"
      --selection-backtest-base "$TARGET_BACKTEST_BASE"
      --top-n "$TOP_N"
      --sector-encoding onehot
      --dropna-mode target_only
      --capacity-mode "$CAPACITY_MODE"
      --output-mode "$OUTPUT_MODE"
      --max-positions-list "$MAX_POSITIONS_LIST"
      --sell-rank-list "$SELL_RANK_LIST"
    )
    [[ -n "$SELECTION_BACKTEST_ROOT" ]] && args+=(--selection-backtest-root "$SELECTION_BACKTEST_ROOT")
    [[ -n "$START_DATE" ]] && args+=(--start-date "$START_DATE")
    [[ -n "$END_DATE" ]] && args+=(--end-date "$END_DATE")
    [[ "$FORCE_GRID" == "1" ]] && args+=(--force-grid)
    [[ "$PARITY_CHECK_ONLY" == "1" ]] && args+=(--parity-check-only)
    [[ "$SMOKE" == "1" ]] && args+=(--smoke)
    [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)

    "$PYTHON_BIN" "${args[@]}"
    echo "Output root: $out_root"
  done
done

echo "[DONE] fold0 forward backtests finished."
