#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/retrain_existing_saved_models_safe.sh
#
# Retrain current saved_models from their own metadata.
# Default: dry-run, quality-filtered, creates new artifact names with suffix.
#
# Dry run:
#   PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
#
# Real run, create refreshed artifacts:
#   APPLY=1 PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
#
# Real run, replace existing artifact names safely:
#   APPLY=1 REPLACE_EXISTING=1 PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
#
# Limit stocks:
#   ONLY=603308.SH,600312.SH APPLY=1 PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh

PYTHON="${PYTHON:-python3}"
APPLY="${APPLY:-0}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
INCLUDE_POOR="${INCLUDE_POOR:-0}"
ONLY="${ONLY:-}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-refresh_$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-saved_data/model_update_logs/retrain_existing_$(date +%Y%m%d_%H%M%S)}"

cmd=(
  "$PYTHON" model_saving/retrain_existing_models_safe.py
  --out-dir "$OUT_DIR"
  --artifact-suffix "$ARTIFACT_SUFFIX"
)

if [[ -n "$ONLY" ]]; then
  cmd+=(--only "$ONLY")
fi
if [[ "$APPLY" != "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "$REPLACE_EXISTING" == "1" ]]; then
  cmd+=(--replace-existing)
fi
if [[ "$INCLUDE_POOR" == "1" ]]; then
  cmd+=(--include-poor)
fi

printf '[RUN]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
