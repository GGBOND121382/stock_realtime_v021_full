#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_fix_cleanup_${TS}"

FILES=(
  "portfolio_decision/backtest_historical_score_portfolio.py"
  "portfolio_decision/portfolio_confirm_from_buy_signals.py"
  "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
)

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

if [[ "${CLEAN_OLD_BACKTEST:-1}" == "1" ]]; then
  bash scripts/cleanup_old_portfolio_backtest_to_trash.sh
else
  echo "[SKIP] cleanup disabled by CLEAN_OLD_BACKTEST=0"
fi

"$PYTHON" scripts/patch_portfolio_bugfix_cleanup.py

"$PYTHON" -m py_compile \
  scripts/settle_portfolio_backtest_consistently.py \
  portfolio_decision/backtest_historical_score_portfolio.py \
  portfolio_decision/portfolio_confirm_from_buy_signals.py \
  portfolio_decision/daily_portfolio_confirm_pyscipopt.py

echo "[CHECK] backtest recomputes trade returns"
grep -n 'add_trade_returns\|trade_net_close_return\|trade_target_or_close_return' portfolio_decision/backtest_historical_score_portfolio.py | head -40

echo "[CHECK] adapter/optimizer propagate artifact usage fields"
grep -n 'expected_return_col\|entry_vwap_premium_bps\|samples' portfolio_decision/portfolio_confirm_from_buy_signals.py | head -30
grep -n 'expected_return_col\|entry_policy: str\|entry_vwap_premium_bps' portfolio_decision/daily_portfolio_confirm_pyscipopt.py | head -30

echo "[DONE] portfolio cleanup + bug fix patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
