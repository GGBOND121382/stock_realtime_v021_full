#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON="${PYTHON:-python3}"
APPLY="${APPLY:-0}"
ONLY="${ONLY:-}"
cmd=("$PYTHON" model_saving/cleanup_saved_models_safe.py)
if [[ -n "$ONLY" ]]; then
  cmd+=(--only "$ONLY")
fi
if [[ "$APPLY" == "1" ]]; then
  cmd+=(--apply)
else
  echo "[DRY-RUN] Set APPLY=1 to move candidates to cleanup_trash."
fi
printf '[RUN]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
