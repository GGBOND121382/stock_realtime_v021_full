#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON="${PYTHON:-python3}"
CLEAN_OLD_BACKTEST="${CLEAN_OLD_BACKTEST:-1}"
BACKTEST_DIR="${BACKTEST_DIR:-portfolio_reports/backtests/historical_score_portfolio}"
if [[ "$CLEAN_OLD_BACKTEST" == "1" && -e "$BACKTEST_DIR" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  DEST="cleanup_trash/portfolio_backtest_old_${TS}"
  mkdir -p "$DEST/$(dirname "$BACKTEST_DIR")"
  mv "$BACKTEST_DIR" "$DEST/$BACKTEST_DIR"
  echo "[MOVED] $BACKTEST_DIR -> $DEST/$BACKTEST_DIR"
fi
"$PYTHON" scripts/patch_portfolio_complete_fix_v3.py
