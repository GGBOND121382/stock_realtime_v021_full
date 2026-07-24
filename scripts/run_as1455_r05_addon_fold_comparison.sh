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
HIST_BASE="${HIST_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"
FWD_BASE="${FWD_BASE:-saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-$SOURCE_DIR/baostock_5m_cache}"
INITIAL_CASH="${INITIAL_CASH:-200000}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/$RUN_STAMP}"
PAIR_JSON="$OUT_ROOT/existing_result_pair.json"
PAIR_TSV="$OUT_ROOT/existing_result_pair.tsv"

mkdir -p "$OUT_ROOT"
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || {
  echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2
  exit 1
}
[[ -d "$HIST_BASE" ]] || {
  echo "[ERROR] missing historical result base: $HIST_BASE" >&2
  exit 1
}
[[ -d "$FWD_BASE" ]] || {
  echo "[ERROR] missing strict-OOS result base: $FWD_BASE" >&2
  exit 1
}

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_r05_addon_fold_comparison.py \
  scripts/run_as1455_r05_addon_fold_comparison_v2.py

printf '%s\n' \
  "[MODE] r05_fwd rotation_addon_onehot complete fold comparison" \
  "[MODE] independent_historical_backtests=6" \
  "[MODE] forward_strict_oos=reuse_retained_result" \
  "[MODE] historical_cross_fold=reuse_materialized_result" \
  "[MODE] historical_plus_forward=continuous_state_rerun" \
  "[MODE] execution_panel_builds=1 continuous_engine_calls=2" \
  "[MODE] prediction_generation=false grid=false training=false data_refresh=false" \
  "[MODE] initial_cash=$INITIAL_CASH output_mode=$OUTPUT_MODE"

"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_ROOT" \
  --min-free-gb "$MIN_FREE_GB" \
  --label r05-addon-fold-comparison

"$PYTHON_BIN" scripts/resolve_as1455_existing_result_pairs.py \
  --historical-base "$HIST_BASE" \
  --forward-base "$FWD_BASE" \
  --feature-presets rotation_addon_onehot \
  --targets r05_fwd \
  --json-out "$PAIR_JSON" \
  --tsv-out "$PAIR_TSV"

args=(
  "$PYTHON_BIN"
  scripts/run_as1455_r05_addon_fold_comparison_v2.py
  --pair-manifest "$PAIR_JSON"
  --out-root "$OUT_ROOT"
  --initial-cash "$INITIAL_CASH"
  --output-mode "$OUTPUT_MODE"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
)

# The v2 Python runner disables raw-5m scanning automatically when both frozen
# historical and forward configs use capacity_mode=none.
[[ -d "$RAW_5M_CACHE_DIR" ]] && args+=(--raw-5m-cache-dir "$RAW_5M_CACHE_DIR")
[[ -n "${LAST5_PANEL:-}" ]] && args+=(--last5-panel "$LAST5_PANEL")
[[ -n "${UNIVERSE:-}" ]] && args+=(--universe "$UNIVERSE")
[[ -n "${ST_SYMBOLS:-}" ]] && args+=(--st-symbols "$ST_SYMBOLS")
[[ -n "${ST_STATUS:-}" ]] && args+=(--st-status "$ST_STATUS")
[[ -n "${CORPORATE_ACTIONS:-}" ]] && args+=(--corporate-actions "$CORPORATE_ACTIONS")

"${args[@]}"

echo "[PASS] r05_fwd rotation_addon_onehot complete comparison finished"
echo "[PASS] output=$OUT_ROOT"
