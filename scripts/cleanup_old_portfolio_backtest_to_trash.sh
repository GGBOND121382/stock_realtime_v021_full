#!/usr/bin/env bash
set -Eeuo pipefail
TS="$(date +%Y%m%d_%H%M%S)"
BACKTEST_DIR="${BACKTEST_DIR:-portfolio_reports/backtests/historical_score_portfolio}"
TRASH_ROOT="${TRASH_ROOT:-cleanup_trash}"
DEST="$TRASH_ROOT/portfolio_backtest_old_${TS}"
mkdir -p "$DEST"
if [[ -e "$BACKTEST_DIR" ]]; then
  mkdir -p "$DEST/$(dirname "$BACKTEST_DIR")"
  mv "$BACKTEST_DIR" "$DEST/$BACKTEST_DIR"
  echo "[MOVED] $BACKTEST_DIR -> $DEST/$BACKTEST_DIR"
else
  echo "[SKIP] not found: $BACKTEST_DIR"
fi
echo "[DONE] old portfolio backtest data moved to: $DEST"
