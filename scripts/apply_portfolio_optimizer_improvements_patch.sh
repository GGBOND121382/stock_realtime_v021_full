#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_optimizer_improvements_${TS}"

FILES=(
  "portfolio_decision/portfolio_confirm_from_buy_signals.py"
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

mkdir -p configs
if [[ ! -f configs/portfolio_model_overrides.csv ]]; then
  cp configs/portfolio_model_overrides.template.csv configs/portfolio_model_overrides.csv
  echo "[INIT] configs/portfolio_model_overrides.csv created from template"
else
  echo "[KEEP] configs/portfolio_model_overrides.csv already exists"
fi

"$PYTHON" scripts/patch_portfolio_optimizer_improvements.py

"$PYTHON" -m py_compile \
  portfolio_decision/portfolio_confirm_from_buy_signals.py \
  portfolio_decision/daily_portfolio_confirm_pyscipopt.py

if [[ -f "portfolio_decision/backtest_historical_score_portfolio.py" ]]; then
  "$PYTHON" -m py_compile portfolio_decision/backtest_historical_score_portfolio.py
fi

bash -n scripts/run_portfolio_confirm_from_signals.sh

echo "[DONE] portfolio optimizer improvements patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
