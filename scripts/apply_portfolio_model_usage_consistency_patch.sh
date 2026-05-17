#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_model_usage_consistency_${TS}"

FILES=(
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

"$PYTHON" scripts/patch_portfolio_model_usage_consistency.py

"$PYTHON" -m py_compile \
  scripts/settle_portfolio_backtest_consistently.py \
  portfolio_decision/portfolio_confirm_from_buy_signals.py \
  portfolio_decision/daily_portfolio_confirm_pyscipopt.py

echo "[CHECK] adapter propagated fields"
grep -n 'expected_return_col\|entry_vwap_premium_bps\|samples' portfolio_decision/portfolio_confirm_from_buy_signals.py | head -30

echo "[CHECK] optimizer propagated fields"
grep -n 'expected_return_col\|entry_policy: str\|entry_vwap_premium_bps' portfolio_decision/daily_portfolio_confirm_pyscipopt.py | head -30

echo "[DONE] portfolio model usage consistency patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
