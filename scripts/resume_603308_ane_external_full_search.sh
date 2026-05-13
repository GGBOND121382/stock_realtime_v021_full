#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

SAMPLES="${SAMPLES:-saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment_live_board_v2/training_samples_with_aero_nuclear_equipment_external.csv}"
INTRADAY="${INTRADAY:-saved_data/603308_pipeline_out/00_base/603308_5m.csv}"
OUT_ROOT="${OUT_ROOT:-saved_data/603308_pipeline_out}"
SUMMARY_OUT="${SUMMARY_OUT:-saved_data/603308_pipeline_out/99_summary_ane_live_board_v2_external_full}"

FEATURE_GROUPS="${FEATURE_GROUPS:-stock_external,external,base_reversal_regime_external,reversal_fundamental_regime_external,reversal_fundamental_regime_sector_external,all_no_ak}"
MODELS="${MODELS:-xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_300_lr004_mcw3,xgb_d3_600_lr002_mcw3,xgb_d3_800_lr0015_mcw5,xgb_d4_700_lr002_mcw2,xgb_d4_500_lr002_mcw5,xgb_d5_900_lr002_mcw1,extra_trees_600_d3,random_forest_600_d4,lgbm_leaves7_400,lgbm_leaves15_700}"

TRAIN_ROWS="${TRAIN_ROWS:-756}"
VALID_ROWS="${VALID_ROWS:-126}"
TEST_ROWS="${TEST_ROWS:-63}"
MIN_VALID_TRADES="${MIN_VALID_TRADES:-8}"
MIN_TRAIN_ENTRIES="${MIN_TRAIN_ENTRIES:-80}"
ROUND_TRIP_COST_BPS="${ROUND_TRIP_COST_BPS:-1.7}"
ENTRY_VWAP_PREMIUM_BPS="${ENTRY_VWAP_PREMIUM_BPS:-50}"
QUANTILES="${QUANTILES:-0.5,0.6,0.7,0.8}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

check_inputs() {
  test -f "$SAMPLES" || { echo "[ERROR] missing samples: $SAMPLES" >&2; exit 2; }
  test -f "$INTRADAY" || { echo "[ERROR] missing intraday bars: $INTRADAY" >&2; exit 2; }
  "$PYTHON" - <<PY
from types import SimpleNamespace
from model_training.search_walk_forward_model_complexity import make_dataset
args = SimpleNamespace(
    samples="${SAMPLES}",
    intraday_bars="${INTRADAY}",
    round_trip_cost_bps=float("${ROUND_TRIP_COST_BPS}"),
    target_hit_bps=50.0,
    label_mode="close_profit",
    entry_policy="all_days",
    entry_vwap_premium_bps=float("${ENTRY_VWAP_PREMIUM_BPS}"),
    max_missing=0.35,
)
df, groups = make_dataset(args)
requested = [x.strip() for x in "${FEATURE_GROUPS}".split(',') if x.strip()]
print('dataset rows:', len(df))
print('requested feature groups:')
for g in requested:
    print(f'  {g}: {len(groups.get(g, []))}')
missing = [g for g in requested if not groups.get(g)]
if missing:
    raise SystemExit('[ERROR] empty or missing feature groups: ' + ','.join(missing))
PY
}

search_done() {
  local out_dir="$1"
  local bps="$2"
  local f="$out_dir/summary_${bps}bps.csv"
  [[ -s "$f" ]]
}

run_search() {
  local name="$1"
  local out_dir="$2"
  local label_mode="$3"
  local target_bps="$4"
  local entry_policy="$5"

  if search_done "$out_dir" "$target_bps"; then
    log "SKIP already done: $name -> $out_dir/summary_${target_bps}bps.csv"
    return 0
  fi

  log "RUN $name"
  mkdir -p "$out_dir"

  local cmd=(
    "$PYTHON" model_training/search_walk_forward_model_complexity.py
    --samples "$SAMPLES"
    --intraday-bars "$INTRADAY"
    --out-dir "$out_dir"
    --round-trip-cost-bps "$ROUND_TRIP_COST_BPS"
    --target-hit-bps "$target_bps"
    --label-mode "$label_mode"
    --entry-policy "$entry_policy"
    --groups "$FEATURE_GROUPS"
    --models "$MODELS"
    --quantiles "$QUANTILES"
    --train-rows "$TRAIN_ROWS"
    --valid-rows "$VALID_ROWS"
    --test-rows "$TEST_ROWS"
    --min-valid-trades "$MIN_VALID_TRADES"
    --min-train-entries "$MIN_TRAIN_ENTRIES"
  )

  if [[ "$entry_policy" == "vwap_low" ]]; then
    cmd+=(--entry-vwap-premium-bps "$ENTRY_VWAP_PREMIUM_BPS")
  fi

  "${cmd[@]}"
}

make_summary() {
  log "Generate combined summary"
  mkdir -p "$SUMMARY_OUT"
  local dirs=(
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_close50_all_days"
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_close50_vwap_low"
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_hit80_all_days"
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_hit80_vwap_low"
  )

  local args=()
  for d in "${dirs[@]}"; do
    if [[ -d "$d" ]]; then
      args+=(--search-dir "$d")
    else
      log "WARN missing search dir, not included in summary: $d"
    fi
  done

  "$PYTHON" model_training/summarize_nextday_search_results.py \
    "${args[@]}" \
    --out-dir "$SUMMARY_OUT" \
    --excel || \
  "$PYTHON" model_training/summarize_nextday_search_results.py \
    "${args[@]}" \
    --out-dir "$SUMMARY_OUT"

  log "Top candidates:"
  "$PYTHON" - <<PY
import pandas as pd
from pathlib import Path
p = Path("${SUMMARY_OUT}") / "final_leaderboard.csv"
if not p.exists():
    raise SystemExit(f"missing {p}")
df = pd.read_csv(p)
cols = [c for c in [
    'entry_policy','label_mode','target_hit_bps','feature_group','model_name',
    'trades','win_rate','avg_return','median_return','compound_return','max_drawdown','profit_factor'
] if c in df.columns]
print(df[cols].head(30).to_string(index=False))
PY
}

main() {
  log "Check inputs"
  check_inputs

  # If system reboot happened after close50_all_days, this will skip it and continue.
  run_search "close_profit 50bps all_days" \
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_close50_all_days" \
    close_profit 50 all_days

  run_search "close_profit 50bps vwap_low" \
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_close50_vwap_low" \
    close_profit 50 vwap_low

  run_search "hit 80bps all_days" \
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_hit80_all_days" \
    hit 80 all_days

  run_search "hit 80bps vwap_low" \
    "$OUT_ROOT/05_search_ane_live_board_v2_external_full_hit80_vwap_low" \
    hit 80 vwap_low

  make_summary

  log "DONE"
  echo "Summary:"
  echo "  ${SUMMARY_OUT}/final_leaderboard.csv"
  echo "  ${SUMMARY_OUT}/best_by_target_top5.csv"
  echo "  ${SUMMARY_OUT}/final_leaderboard.xlsx"
}

main "$@"
