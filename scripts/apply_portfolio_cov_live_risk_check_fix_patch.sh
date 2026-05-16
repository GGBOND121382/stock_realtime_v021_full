#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_cov_live_risk_check_fix_${TS}"

FILES=(
  "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
)

if [[ -f "scripts/apply_portfolio_cov_live_risk_patch.sh" ]]; then
  FILES+=("scripts/apply_portfolio_cov_live_risk_patch.sh")
fi
if [[ -f "scripts/patch_portfolio_cov_live_risk.py" ]]; then
  FILES+=("scripts/patch_portfolio_cov_live_risk.py")
fi

mkdir -p "$BACKUP_DIR"
for f in "${FILES[@]}"; do
  mkdir -p "$BACKUP_DIR/$(dirname "$f")"
  cp -p "$f" "$BACKUP_DIR/$f"
  echo "[BACKUP] $f -> $BACKUP_DIR/$f"
done

"$PYTHON" scripts/patch_portfolio_cov_live_risk_check_fix.py

"$PYTHON" -m py_compile portfolio_decision/daily_portfolio_confirm_pyscipopt.py

if [[ -f "scripts/apply_portfolio_cov_live_risk_patch.sh" ]]; then
  bash -n scripts/apply_portfolio_cov_live_risk_patch.sh
fi

echo "[CHECK] no real nonlinear amount[i] * amount[j] expression should remain"
if grep -nE '[^#]*amount\[i\].*amount\[j\]|[^#]*amount\[j\].*amount\[i\]' \
  portfolio_decision/daily_portfolio_confirm_pyscipopt.py; then
  echo "[ERROR] real nonlinear amount[i] * amount[j] expression still exists" >&2
  exit 3
fi

echo "[DONE] false-positive check fixed"
echo "[BACKUP_DIR] $BACKUP_DIR"
