#!/usr/bin/env bash
set -euo pipefail

# Apply the BaoStock-compatible realtime 5m bar timestamp patch.
# Run from project root.

PYTHON="${PYTHON:-python3}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-saved_data/patch_backups/realtime_5m_bar_patch_$(date +%Y%m%d_%H%M%S)}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$BACKUP_DIR"

TARGET="tools/fix_5m_ohlcv_from_snapshots.py"
SOURCE="$PATCH_ROOT/tools/fix_5m_ohlcv_from_snapshots.py"

if [[ ! -f "$SOURCE" ]]; then
  echo "[ERROR] patch source not found: $SOURCE" >&2
  exit 2
fi

$PYTHON -m py_compile "$SOURCE"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "[VALIDATE_ONLY] source syntax ok: $SOURCE"
  exit 0
fi

if [[ -f "$TARGET" ]]; then
  cp -a "$TARGET" "$BACKUP_DIR/fix_5m_ohlcv_from_snapshots.py.bak"
  echo "[BACKUP] $TARGET -> $BACKUP_DIR/fix_5m_ohlcv_from_snapshots.py.bak"
fi

mkdir -p tools
cp "$SOURCE" "$TARGET"
chmod +x "$TARGET"
$PYTHON -m py_compile "$TARGET"

echo "[DONE] installed $TARGET"
echo "[BACKUP_DIR] $BACKUP_DIR"
echo
cat <<'EOF'
Next validation after a trading-day collection:
  python3 tools/fix_5m_ohlcv_from_snapshots.py \
    --date YYYYMMDD \
    --cache-dir saved_data/akshare_realtime_cache \
    --symbols-file saved_data/intraday_nextday_signals/YYYYMMDD/effective_watchlist.txt \
    --cutoff-time 14:55 \
    --dry-run

Then run without --dry-run if the summary is ok.
EOF
