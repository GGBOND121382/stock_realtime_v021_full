#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_complete_fix_${TS}"
FILES=(
  "portfolio_decision/backtest_historical_score_portfolio.py"
  "portfolio_decision/portfolio_confirm_from_buy_signals.py"
  "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
)
mkdir -p "$BACKUP_DIR"
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then echo "[ERROR] missing required file: $f" >&2; exit 2; fi
  mkdir -p "$BACKUP_DIR/$(dirname "$f")"; cp -p "$f" "$BACKUP_DIR/$f"; echo "[BACKUP] $f -> $BACKUP_DIR/$f"
done
if [[ "${CLEAN_OLD_BACKTEST:-1}" == "1" ]]; then bash scripts/cleanup_old_portfolio_backtest_to_trash.sh; else echo "[SKIP] cleanup disabled by CLEAN_OLD_BACKTEST=0"; fi
"$PYTHON" scripts/patch_portfolio_complete_fix.py
"$PYTHON" scripts/validate_portfolio_deep_state.py --max-artifacts "${VALIDATE_N:-20}"
echo "[DONE] complete portfolio fix applied and validated"
echo "[BACKUP_DIR] $BACKUP_DIR"
