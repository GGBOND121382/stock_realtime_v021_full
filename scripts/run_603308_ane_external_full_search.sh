#!/usr/bin/env bash
set -euo pipefail

# 603308 应流股份 ane_live_board_v2 external-only model search.
# Purpose: test whether the rebuilt external features bring incremental value.
# This script intentionally DOES NOT run non-external-only feature groups.

PYTHON="${PYTHON:-python3}"
STOCK_CODE="${STOCK_CODE:-603308.SH}"
PIPELINE_OUT="${PIPELINE_OUT:-saved_data/603308_pipeline_out}"
EXT_OUT="${EXT_OUT:-${PIPELINE_OUT}/04_external/aero_nuclear_equipment_live_board_v2}"
SAMPLES="${SAMPLES:-${EXT_OUT}/training_samples_with_aero_nuclear_equipment_external.csv}"
INTRADAY="${INTRADAY:-${PIPELINE_OUT}/00_base/603308_5m.csv}"
OUT_ROOT="${OUT_ROOT:-${PIPELINE_OUT}}"
SUMMARY_OUT="${SUMMARY_OUT:-${PIPELINE_OUT}/99_summary_ane_live_board_v2_external_full}"

# Only external-related groups. No base/reversal/regime/sector-only groups here.
# Do NOT name this variable GROUPS: Bash has a special $GROUPS array for Unix group ids.
FEATURE_GROUPS="${FEATURE_GROUPS:-stock_external,external,base_reversal_regime_external,reversal_fundamental_regime_external,reversal_fundamental_regime_sector_external,all_no_ak}"
MODELS="${MODELS:-xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_300_lr004_mcw3,xgb_d3_600_lr002_mcw3,xgb_d3_800_lr0015_mcw5,xgb_d4_700_lr002_mcw2,xgb_d4_500_lr002_mcw5,xgb_d5_900_lr002_mcw1,extra_trees_600_d3,random_forest_600_d4,lgbm_leaves7_400,lgbm_leaves15_700}"
QUANTILES="${QUANTILES:-0.5,0.6,0.7,0.8}"
TRAIN_ROWS="${TRAIN_ROWS:-756}"
VALID_ROWS="${VALID_ROWS:-126}"
TEST_ROWS="${TEST_ROWS:-63}"
MIN_VALID_TRADES="${MIN_VALID_TRADES:-8}"
MIN_TRAIN_ENTRIES="${MIN_TRAIN_ENTRIES:-80}"
ROUND_TRIP_COST_BPS="${ROUND_TRIP_COST_BPS:-1.7}"
ENTRY_VWAP_PREMIUM_BPS="${ENTRY_VWAP_PREMIUM_BPS:-50}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SUMMARY_OUT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/external_full_search_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "[603308 ANE LIVE BOARD V2 EXTERNAL FULL SEARCH]"
echo "Purpose: test NEW external features only."
echo "Started: $(date '+%F %T')"
echo "PYTHON=$PYTHON"
echo "STOCK_CODE=$STOCK_CODE"
echo "SAMPLES=$SAMPLES"
echo "INTRADAY=$INTRADAY"
echo "FEATURE_GROUPS=$FEATURE_GROUPS"
echo "MODELS=$MODELS"
echo "SUMMARY_OUT=$SUMMARY_OUT"
echo "LOG_FILE=$LOG_FILE"
echo "============================================================"

echo "[1/6] Check input files"
if [[ "$FEATURE_GROUPS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] FEATURE_GROUPS looks numeric: $FEATURE_GROUPS"
  echo "        This usually means an old script used Bash special variable GROUPS."
  exit 2
fi
test -f "$SAMPLES" || { echo "[ERROR] missing samples: $SAMPLES"; exit 2; }
test -f "$INTRADAY" || { echo "[ERROR] missing intraday bars: $INTRADAY"; exit 2; }

if [[ -f "${EXT_OUT}/validation_report.json" ]]; then
  echo "[INFO] external validation report quick view"
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("${EXT_OUT}/validation_report.json")
j = json.loads(p.read_text(encoding="utf-8"))
print("profile=", j.get("profile"))
print("sample_rows=", j.get("sample_rows"), "merged_rows=", j.get("merged_rows"), "feature_cols=", j.get("feature_cols"))
print("used.boards=", (j.get("source_meta", {}) or {}).get("used", {}).get("boards"))
print("errors=", j.get("errors", {}))
PY
else
  echo "[WARN] validation_report.json not found: ${EXT_OUT}/validation_report.json"
fi

echo "[2/6] Check selected external feature groups exist and are non-empty"
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
print("dataset rows:", len(df))
requested = [x.strip() for x in "${FEATURE_GROUPS}".split(",") if x.strip()]
for g in requested:
    n = len(groups.get(g, []))
    print(f"{g}: {n}")
    if n <= 0:
        raise SystemExit(f"[ERROR] feature group is empty or missing: {g}")
PY

run_search() {
  local label_mode="$1"
  local target_bps="$2"
  local entry_policy="$3"
  local out_dir="$4"

  echo "------------------------------------------------------------"
  echo "[RUN] label=${label_mode}, target=${target_bps}bps, entry=${entry_policy}"
  echo "out_dir=${out_dir}"
  mkdir -p "$out_dir"

  local extra_args=()
  if [[ "$entry_policy" == "vwap_low" ]]; then
    extra_args+=(--entry-vwap-premium-bps "$ENTRY_VWAP_PREMIUM_BPS")
  fi

  "$PYTHON" model_training/search_walk_forward_model_complexity.py \
    --samples "$SAMPLES" \
    --intraday-bars "$INTRADAY" \
    --out-dir "$out_dir" \
    --round-trip-cost-bps "$ROUND_TRIP_COST_BPS" \
    --target-hit-bps "$target_bps" \
    --label-mode "$label_mode" \
    --entry-policy "$entry_policy" \
    "${extra_args[@]}" \
    --groups "$FEATURE_GROUPS" \
    --models "$MODELS" \
    --quantiles "$QUANTILES" \
    --train-rows "$TRAIN_ROWS" \
    --valid-rows "$VALID_ROWS" \
    --test-rows "$TEST_ROWS" \
    --min-valid-trades "$MIN_VALID_TRADES" \
    --min-train-entries "$MIN_TRAIN_ENTRIES"
}

SEARCH_DIRS=(
  "${OUT_ROOT}/05_search_ane_live_board_v2_external_full_close50_all_days"
  "${OUT_ROOT}/05_search_ane_live_board_v2_external_full_close50_vwap_low"
  "${OUT_ROOT}/05_search_ane_live_board_v2_external_full_hit80_all_days"
  "${OUT_ROOT}/05_search_ane_live_board_v2_external_full_hit80_vwap_low"
)

echo "[3/6] Run external-only full searches"
run_search close_profit 50 all_days "${SEARCH_DIRS[0]}"
run_search close_profit 50 vwap_low "${SEARCH_DIRS[1]}"
run_search hit 80 all_days "${SEARCH_DIRS[2]}"
run_search hit 80 vwap_low "${SEARCH_DIRS[3]}"

echo "[4/6] Summarize results"
mkdir -p "$SUMMARY_OUT"
SUMMARY_ARGS=()
for d in "${SEARCH_DIRS[@]}"; do
  SUMMARY_ARGS+=(--search-dir "$d")
done

if "$PYTHON" model_training/summarize_nextday_search_results.py "${SUMMARY_ARGS[@]}" --out-dir "$SUMMARY_OUT" --excel; then
  echo "[OK] summary with Excel completed"
else
  echo "[WARN] Excel summary failed; retry CSV summary"
  "$PYTHON" model_training/summarize_nextday_search_results.py "${SUMMARY_ARGS[@]}" --out-dir "$SUMMARY_OUT"
fi

echo "[5/6] Print top external candidates"
"$PYTHON" scripts/print_603308_ane_external_top.py \
  --leaderboard "${SUMMARY_OUT}/final_leaderboard.csv" \
  --top 30 || true

echo "[6/6] Write save commands for top candidates"
"$PYTHON" scripts/make_603308_ane_external_save_commands.py \
  --leaderboard "${SUMMARY_OUT}/final_leaderboard.csv" \
  --samples "$SAMPLES" \
  --intraday-bars "$INTRADAY" \
  --stock-code "$STOCK_CODE" \
  --out "${SUMMARY_OUT}/save_top_artifacts_commands.sh" \
  --top 5
chmod +x "${SUMMARY_OUT}/save_top_artifacts_commands.sh"

echo "============================================================"
echo "[DONE] $(date '+%F %T')"
echo "Summary: ${SUMMARY_OUT}/final_leaderboard.csv"
echo "Best top5: ${SUMMARY_OUT}/best_by_target_top5.csv"
echo "Excel: ${SUMMARY_OUT}/final_leaderboard.xlsx"
echo "Log: ${LOG_FILE}"
echo "Save commands: ${SUMMARY_OUT}/save_top_artifacts_commands.sh"
echo "Review the leaderboard first. Then, if acceptable:"
echo "  bash ${SUMMARY_OUT}/save_top_artifacts_commands.sh"
echo "============================================================"
