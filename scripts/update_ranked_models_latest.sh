#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/update_ranked_models_latest.sh
#
# Safe replacement:
#   - scans existing saved_data/*_pipeline_out*/99_summary/final_leaderboard.csv
#   - saves selected models into saved_models
#   - does not rerun pipelines
#   - does not remove pipeline folders
#
# Usage:
#   SKIP_PIPELINE=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/update_ranked_models_latest.sh

PYTHON="${PYTHON:-python3}"
END_DATE="${END_DATE:-$(date +%F)}"
SAVED_DATA_DIR="${SAVED_DATA_DIR:-saved_data}"
MODELS_DIR="${MODELS_DIR:-saved_models}"
LOG_ROOT="${LOG_ROOT:-saved_data/model_update_logs/ranked_safe_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
MAX_PER_STOCK="${MAX_PER_STOCK:-2}"
MIN_RANK_SCORE="${MIN_RANK_SCORE:-0.0}"
MIN_TRADES="${MIN_TRADES:-80}"
MIN_AVG_RETURN="${MIN_AVG_RETURN:-0.002}"
MIN_PROFIT_FACTOR="${MIN_PROFIT_FACTOR:-1.35}"
MAX_DRAWDOWN_FLOOR="${MAX_DRAWDOWN_FLOOR:--0.35}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-safe_${END_DATE//-/}}"
SKIP_PIPELINE="${SKIP_PIPELINE:-1}"
CLEAN_PIPELINE="${CLEAN_PIPELINE:-0}"

if [[ "$CLEAN_PIPELINE" != "0" ]]; then
  echo "[REFUSE] CLEAN_PIPELINE=$CLEAN_PIPELINE is not allowed in this safe script."
  exit 3
fi

if [[ "$SKIP_PIPELINE" != "1" ]]; then
  echo "[REFUSE] SKIP_PIPELINE must be 1. Use rebuild_603308_pipeline_safe.sh for 603308-only rebuild."
  exit 4
fi

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "[SAFE RANKED MODEL UPDATE]"
echo "SAVED_DATA_DIR=$SAVED_DATA_DIR"
echo "MODELS_DIR=$MODELS_DIR"
echo "LOG_ROOT=$LOG_ROOT"
echo "DRY_RUN=$DRY_RUN"
echo "ONLY=$ONLY"
echo "MAX_PER_STOCK=$MAX_PER_STOCK"
echo "ARTIFACT_SUFFIX=$ARTIFACT_SUFFIX"
echo "Existing pipeline folders are read-only inputs."
echo "============================================================"

cmd=(
  "$PYTHON" model_saving/auto_update_ranked_models_safe.py
  --saved-data-dir "$SAVED_DATA_DIR"
  --models-dir "$MODELS_DIR"
  --out-dir "$LOG_ROOT"
  --artifact-suffix "$ARTIFACT_SUFFIX"
  --max-per-stock "$MAX_PER_STOCK"
  --min-rank-score "$MIN_RANK_SCORE"
  --min-trades "$MIN_TRADES"
  --min-avg-return "$MIN_AVG_RETURN"
  --min-profit-factor "$MIN_PROFIT_FACTOR"
  --max-drawdown-floor "$MAX_DRAWDOWN_FLOOR"
)

if [[ -n "$ONLY" ]]; then
  cmd+=(--only "$ONLY")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf '[RUN]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$LOG_ROOT/auto_update_ranked_models_safe.log"
rc=${PIPESTATUS[0]}
echo "[RETURN_CODE] $rc"
exit "$rc"
