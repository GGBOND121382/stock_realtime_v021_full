#!/usr/bin/env bash
set -Eeuo pipefail

# Apply no-search 603308 rebuild patch.
# It only replaces scripts/rebuild_603308_pipeline_safe.sh.
# No saved_data folder is removed.

PYTHON="${PYTHON:-python3}"
BACKUP_ROOT="${BACKUP_ROOT:-saved_data/patch_backups/603308_no_search_patch_$(date +%Y%m%d_%H%M%S)}"
PAYLOAD="patches/big_safe_payload/scripts/rebuild_603308_pipeline_safe.sh"

die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -f "$PAYLOAD" ]] || die "missing payload: $PAYLOAD"

bash -n "$PAYLOAD"

if grep -nE 'rm[[:space:]]+-rf|search-targets|search_targets|10_search' "$PAYLOAD" >/tmp/no_search_patch_grep.$$ 2>/dev/null; then
  cat /tmp/no_search_patch_grep.$$ >&2 || true
  rm -f /tmp/no_search_patch_grep.$$
  die "payload contains forbidden destructive/search pattern"
fi
rm -f /tmp/no_search_patch_grep.$$ || true

mkdir -p "$BACKUP_ROOT/scripts"
if [[ -f scripts/rebuild_603308_pipeline_safe.sh ]]; then
  cp -a scripts/rebuild_603308_pipeline_safe.sh "$BACKUP_ROOT/scripts/rebuild_603308_pipeline_safe.sh.bak"
  echo "[BACKUP] scripts/rebuild_603308_pipeline_safe.sh -> $BACKUP_ROOT/scripts/rebuild_603308_pipeline_safe.sh.bak"
fi

cp -a "$PAYLOAD" scripts/rebuild_603308_pipeline_safe.sh
chmod +x scripts/rebuild_603308_pipeline_safe.sh

bash -n scripts/rebuild_603308_pipeline_safe.sh

echo "[DONE] Installed no-search 603308 rebuild script."
echo "[RUN]"
echo "  PYTHON=python3 END_DATE=2026-05-15 bash scripts/rebuild_603308_pipeline_safe.sh"
