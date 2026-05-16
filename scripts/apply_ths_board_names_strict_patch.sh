#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/ths_board_names_strict_${TS}"

FILES=(
  "feature_building/build_stock_external_features.py"
  "configs/realtime_context_sources.toml"
)

if [[ -f "scripts/run_new27_v2_full_pipelines.sh" ]]; then
  FILES+=("scripts/run_new27_v2_full_pipelines.sh")
fi
if [[ -f "README_RUN_NEW27_V2_FULL_PIPELINES.md" ]]; then
  FILES+=("README_RUN_NEW27_V2_FULL_PIPELINES.md")
fi

mkdir -p "$BACKUP_DIR"
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    exit 2
  fi
  mkdir -p "$BACKUP_DIR/$(dirname "$f")"
  cp -p "$f" "$BACKUP_DIR/$f"
  echo "[BACKUP] $f -> $BACKUP_DIR/$f"
done

"$PYTHON" scripts/patch_ths_board_names_strict.py

"$PYTHON" -m py_compile feature_building/build_stock_external_features.py

echo "[CHECK] invalid board names should be absent from active external config"
BAD_RE='公用事业|玻璃玻纤|基础化工|工程建设|煤炭行业|饮料乳品'
if grep -R -n -E "$BAD_RE" \
  feature_building/build_stock_external_features.py \
  configs/realtime_context_sources.toml \
  scripts/run_new27_v2_full_pipelines.sh \
  README_RUN_NEW27_V2_FULL_PIPELINES.md 2>/dev/null; then
  echo "[ERROR] invalid THS board names remain in active files" >&2
  exit 3
fi

echo "[DONE] THS board names strict patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
