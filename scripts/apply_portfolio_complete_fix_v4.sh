#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKTEST_DIR="${BACKTEST_DIR:-portfolio_reports/backtests/historical_score_portfolio}"
TRASH_ROOT="${TRASH_ROOT:-cleanup_trash}"

"$PYTHON" scripts/patch_portfolio_complete_fix_v4.py

if [[ "${CLEAN_OLD_BACKTEST:-1}" == "1" && -e "$BACKTEST_DIR" ]]; then
  DEST="${TRASH_ROOT}/portfolio_backtest_old_${TS}"
  mkdir -p "$DEST/$(dirname "$BACKTEST_DIR")"
  mv "$BACKTEST_DIR" "$DEST/$BACKTEST_DIR"
  echo "[MOVED] $BACKTEST_DIR -> $DEST/$BACKTEST_DIR"
fi
