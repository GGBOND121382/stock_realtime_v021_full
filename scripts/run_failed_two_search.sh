#!/usr/bin/env bash
# run_failed_two_search.sh
# 只补跑上次行业名失败的 002311 和 600176。

set -uo pipefail

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"

COMMON_ARGS=(
  --start-date 2018-01-01
  --end-date "$END_DATE"
  --feature-pipeline fundamental,sector
  --search-targets hit50,hit80,close_profit
  --entry-policies vwap_low,all_days
  --groups reversal_fundamental_regime,reversal_fundamental_regime_sector,all_no_ak
  --models xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_600_lr002_mcw3,xgb_d4_500_lr002_mcw5,lgbm_leaves7_400,lgbm_leaves15_700,extra_trees_600_d3,random_forest_600_d4
  --quantiles 0.5,0.6,0.7,0.8
  --train-rows 756
  --valid-rows 126
  --test-rows 63
  --min-valid-trades 8
  --min-train-entries 80
  --resume
  --excel
)

run_one() {
  local symbol="$1"
  local sector="$2"
  local external="${3:-}"

  echo "============================================================"
  echo "[START] $(date '+%F %T') symbol=${symbol}, sector=${sector}, external=${external:-none}"
  echo "============================================================"

  if [[ -n "$external" ]]; then
    "$PYTHON" pipelines/run_nextday_pipeline.py \
      --symbol "$symbol" \
      --sector-symbol "$sector" \
      --external "$external" \
      "${COMMON_ARGS[@]}" || {
        echo "[WARN] external failed for ${symbol}, fallback to baseline without external"
        "$PYTHON" pipelines/run_nextday_pipeline.py \
          --symbol "$symbol" \
          --sector-symbol "$sector" \
          "${COMMON_ARGS[@]}"
      }
  else
    "$PYTHON" pipelines/run_nextday_pipeline.py \
      --symbol "$symbol" \
      --sector-symbol "$sector" \
      "${COMMON_ARGS[@]}"
  fi
}

run_one 002311.SZ 农产品加工 "feed,hog"
run_one 600176.SH 建筑材料

echo "[ALL DONE] $(date '+%F %T')"
