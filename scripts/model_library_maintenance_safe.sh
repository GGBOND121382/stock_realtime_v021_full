#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/model_library_maintenance_safe.sh
#
# Unified entry for model-library maintenance.
#
# Commands:
#   inspect
#   retrain-existing
#   train-selected
#   cleanup-preview
#   cleanup-apply
#   restore-preview
#   restore-apply
#
# Environment:
#   PYTHON=python3
#   ONLY=603308.SH,600312.SH
#   APPLY=1                 used by retrain-existing/train-selected wrappers
#   REPLACE_EXISTING=1      for retrain-existing; old artifact moved to cleanup_trash
#   INCLUDE_VETTED_NEW=1    for train-selected
#   INCLUDE_EXTERNAL_FULL=1 EXTERNAL_FULL_SAMPLES=... for train-selected
#
# No command deletes files. Cleanup/restore move directories only.

PYTHON="${PYTHON:-python3}"
CMD="${1:-help}"
shift || true

case "$CMD" in
  inspect)
    "$PYTHON" model_saving/inspect_saved_model_features.py "$@"
    ;;
  retrain-existing)
    APPLY="${APPLY:-0}"
    REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
    ONLY="${ONLY:-}"
    ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-refresh_$(date +%Y%m%d)}"
    OUT_DIR="${OUT_DIR:-saved_data/model_update_logs/retrain_existing_$(date +%Y%m%d_%H%M%S)}"
    cmd=("$PYTHON" model_saving/retrain_existing_models_safe.py --out-dir "$OUT_DIR" --artifact-suffix "$ARTIFACT_SUFFIX")
    [[ -n "$ONLY" ]] && cmd+=(--only "$ONLY")
    [[ "$APPLY" != "1" ]] && cmd+=(--dry-run)
    [[ "$REPLACE_EXISTING" == "1" ]] && cmd+=(--replace-existing)
    printf '[RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
    ;;
  train-selected)
    bash scripts/train_selected_models_safe.sh "$@"
    ;;
  cleanup-preview)
    "$PYTHON" model_saving/prune_saved_models_keep_good.py "$@"
    ;;
  cleanup-apply)
    "$PYTHON" model_saving/prune_saved_models_keep_good.py --apply "$@"
    ;;
  restore-preview)
    "$PYTHON" model_saving/restore_keep_good_models_from_trash.py "$@"
    ;;
  restore-apply)
    "$PYTHON" model_saving/restore_keep_good_models_from_trash.py --apply "$@"
    ;;
  help|*)
    cat <<'EOF'
Usage:
  bash scripts/model_library_maintenance_safe.sh inspect
  APPLY=1 REPLACE_EXISTING=1 bash scripts/model_library_maintenance_safe.sh retrain-existing
  bash scripts/model_library_maintenance_safe.sh cleanup-preview
  bash scripts/model_library_maintenance_safe.sh cleanup-apply
  bash scripts/model_library_maintenance_safe.sh restore-preview
  bash scripts/model_library_maintenance_safe.sh restore-apply
  INCLUDE_VETTED_NEW=1 APPLY=1 bash scripts/model_library_maintenance_safe.sh train-selected

Recommended:
  1) inspect
  2) restore-preview / restore-apply if needed
  3) cleanup-preview, inspect report
  4) cleanup-apply only after checking report
  5) retrain-existing with APPLY=1 REPLACE_EXISTING=1 after data refresh
EOF
    ;;
esac
