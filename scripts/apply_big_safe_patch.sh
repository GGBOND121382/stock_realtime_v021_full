#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/apply_big_safe_patch.sh
#
# Applies a safe patch bundle:
#   - safe ranked model update from existing leaderboards
#   - 603308-only rebuild/search/model script
#   - realtime context TOML append with validation
#   - realtime scoring/bar patches
#
# No saved_data pipeline folder is removed by this bundle.

PYTHON="${PYTHON:-python3}"
BACKUP_ROOT="${BACKUP_ROOT:-saved_data/patch_backups/big_safe_patch_$(date +%Y%m%d_%H%M%S)}"
PAYLOAD_ROOT="${PAYLOAD_ROOT:-patches/big_safe_payload}"

die() {
  echo "[FATAL] $*" >&2
  exit 1
}

backup_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$f")"
    cp -a "$f" "$BACKUP_ROOT/$f.bak"
    echo "[BACKUP] $f -> $BACKUP_ROOT/$f.bak"
  fi
}

install_file() {
  local src="$1"
  local dst="$2"
  [[ -f "$src" ]] || die "payload missing: $src"
  backup_file "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "[INSTALLED] $dst"
}

validate_toml_file() {
  "$PYTHON" - "$1" <<'PY'
import sys, pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
p = pathlib.Path(sys.argv[1])
tomllib.loads(p.read_text(encoding="utf-8"))
print(f"[TOML OK] {p}")
PY
}

refuse_bad_patterns() {
  local f="$1"
  if grep -nE 'rm[[:space:]]+-rf|CLEAN_PIPELINE="\$\{CLEAN_PIPELINE:-1\}"' "$f" >/tmp/big_safe_patch_grep.$$ 2>/dev/null; then
    cat /tmp/big_safe_patch_grep.$$ >&2 || true
    rm -f /tmp/big_safe_patch_grep.$$
    die "unsafe pattern found in $f"
  fi
  rm -f /tmp/big_safe_patch_grep.$$ || true
}

echo "============================================================"
echo "[BIG SAFE PATCH]"
echo "PAYLOAD_ROOT=$PAYLOAD_ROOT"
echo "BACKUP_ROOT=$BACKUP_ROOT"
echo "PYTHON=$PYTHON"
echo "============================================================"

[[ -d "$PAYLOAD_ROOT" ]] || die "payload root not found: $PAYLOAD_ROOT"

for f in \
  "$PAYLOAD_ROOT/scripts/update_ranked_models_latest.sh" \
  "$PAYLOAD_ROOT/scripts/rebuild_603308_pipeline_safe.sh" \
  "$PAYLOAD_ROOT/scripts/apply_safe_realtime_code_patches.sh"
do
  bash -n "$f"
  refuse_bad_patterns "$f"
done

"$PYTHON" -m py_compile \
  "$PAYLOAD_ROOT/model_saving/auto_update_ranked_models_safe.py" \
  "$PAYLOAD_ROOT/tools/fix_5m_ohlcv_from_snapshots.py"

install_file "$PAYLOAD_ROOT/scripts/update_ranked_models_latest.sh" "scripts/update_ranked_models_latest.sh"
install_file "$PAYLOAD_ROOT/scripts/rebuild_603308_pipeline_safe.sh" "scripts/rebuild_603308_pipeline_safe.sh"
install_file "$PAYLOAD_ROOT/scripts/apply_safe_realtime_code_patches.sh" "scripts/apply_safe_realtime_code_patches.sh"
install_file "$PAYLOAD_ROOT/model_saving/auto_update_ranked_models_safe.py" "model_saving/auto_update_ranked_models_safe.py"
install_file "$PAYLOAD_ROOT/tools/fix_5m_ohlcv_from_snapshots.py" "tools/fix_5m_ohlcv_from_snapshots.py"
chmod +x scripts/update_ranked_models_latest.sh scripts/rebuild_603308_pipeline_safe.sh scripts/apply_safe_realtime_code_patches.sh tools/fix_5m_ohlcv_from_snapshots.py

TOML="configs/realtime_context_sources.toml"
APPEND="$PAYLOAD_ROOT/patches/realtime_context_ranked_models_append.toml"
if [[ -f "$TOML" && -f "$APPEND" ]]; then
  validate_toml_file "$TOML"
  if grep -q '^\[stocks."600522.SH"\]' "$TOML" || grep -q '^\[contexts.ocg_stocks\]' "$TOML"; then
    echo "[SKIP] realtime_context ocg/sp entries appear already present"
  else
    backup_file "$TOML"
    tmp="$(mktemp)"
    cat "$TOML" "$APPEND" > "$tmp"
    validate_toml_file "$tmp"
    cp "$tmp" "$TOML"
    rm -f "$tmp"
    echo "[PATCHED] $TOML with ocg/sp context entries"
  fi
else
  echo "[WARN] TOML or append payload missing; skip TOML append"
fi

bash scripts/apply_safe_realtime_code_patches.sh

bash -n scripts/update_ranked_models_latest.sh
bash -n scripts/rebuild_603308_pipeline_safe.sh
bash -n scripts/apply_safe_realtime_code_patches.sh
"$PYTHON" -m py_compile \
  model_saving/auto_update_ranked_models_safe.py \
  tools/fix_5m_ohlcv_from_snapshots.py \
  pipelines/run_intraday_nextday_signals.py \
  pipelines/run_trading_day_signal_pipeline.py \
  data_collection/collect_akshare_l1_cache.py

refuse_bad_patterns scripts/update_ranked_models_latest.sh
refuse_bad_patterns scripts/rebuild_603308_pipeline_safe.sh

echo "============================================================"
echo "[DONE] Big safe patch applied."
echo "[BACKUPS] $BACKUP_ROOT"
echo ""
echo "Safe model update from existing leaderboards:"
echo "  SKIP_PIPELINE=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/update_ranked_models_latest.sh"
echo ""
echo "Rebuild only 603308 pipeline/search/model:"
echo "  PYTHON=python3 END_DATE=2026-05-15 bash scripts/rebuild_603308_pipeline_safe.sh"
echo "============================================================"
