#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/backtest_point_in_time_risk_history_${TS}"
TARGET="portfolio_decision/backtest_historical_score_portfolio.py"

if [[ ! -f "$TARGET" ]]; then
  echo "[ERROR] missing $TARGET. Apply the historical backtest patch first." >&2
  exit 2
fi

mkdir -p "$BACKUP_DIR/$(dirname "$TARGET")"
cp -p "$TARGET" "$BACKUP_DIR/$TARGET"
echo "[BACKUP] $TARGET -> $BACKUP_DIR/$TARGET"

"$PYTHON" scripts/patch_backtest_point_in_time_risk_history.py
"$PYTHON" -m py_compile "$TARGET"

echo "[CHECK] point-in-time risk history patch markers"
grep -n 'write_point_in_time_risk_history' "$TARGET"
grep -n 'risk_history_path' "$TARGET"
grep -n 'risk_history_mode' "$TARGET" || true

echo "[DONE] backtest point-in-time risk history patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
