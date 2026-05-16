#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_cov_live_risk_${TS}"

FILES=(
  "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
  "scripts/run_portfolio_confirm_from_signals.sh"
)

if [[ -f "portfolio_decision/backtest_historical_score_portfolio.py" ]]; then
  FILES+=("portfolio_decision/backtest_historical_score_portfolio.py")
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

"$PYTHON" scripts/patch_portfolio_cov_live_risk.py

"$PYTHON" -m py_compile \
  scripts/build_portfolio_risk_history.py \
  portfolio_decision/daily_portfolio_confirm_pyscipopt.py

if [[ -f "portfolio_decision/backtest_historical_score_portfolio.py" ]]; then
  "$PYTHON" -m py_compile portfolio_decision/backtest_historical_score_portfolio.py
fi

bash -n scripts/run_portfolio_confirm_from_signals.sh

echo "[CHECK] live risk history integration"
grep -n 'build_portfolio_risk_history.py\|AUTO_RISK_HISTORY\|RISK_HISTORY_DIR' scripts/run_portfolio_confirm_from_signals.sh

echo "[CHECK] covariance linear fix"
grep -n 'cov_linear_penalty_bps\|covariance_penalty_mode\|cov_linear_self_weight' portfolio_decision/daily_portfolio_confirm_pyscipopt.py
if grep -n '[^#]*amount\[i\].*amount\[j\]\|[^#]*amount\[j\].*amount\[i\]' portfolio_decision/daily_portfolio_confirm_pyscipopt.py; then
  echo "[ERROR] nonlinear amount_i times amount_j objective still exists" >&2
  exit 3
fi

echo "[DONE] portfolio covariance + live risk history patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
