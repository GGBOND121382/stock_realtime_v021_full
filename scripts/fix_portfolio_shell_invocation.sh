#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TARGET="scripts/run_trading_day_signal_and_portfolio_all_models.sh"

echo "============================================================"
echo "[PATCH] fix portfolio shell script invocation"
echo "TARGET = ${TARGET}"
echo "============================================================"

if [[ ! -f "$TARGET" ]]; then
  echo "[ERROR] file not found: $TARGET"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "backups/fix_portfolio_shell_invocation_${TS}"
cp "$TARGET" "backups/fix_portfolio_shell_invocation_${TS}/run_trading_day_signal_and_portfolio_all_models.sh"

"$PYTHON" - <<'PY'
from pathlib import Path

p = Path("scripts/run_trading_day_signal_and_portfolio_all_models.sh")
s = p.read_text(encoding="utf-8")

old = 'DATE_DASH="$DATE_DASH" DATE_COMPACT="$DATE_COMPACT" ACCOUNT="$ACCOUNT" HISTORY="$HISTORY" \\\n  "$PYTHON" scripts/run_portfolio_confirm_from_signals.sh\n'
new = 'DATE_DASH="$DATE_DASH" DATE_COMPACT="$DATE_COMPACT" ACCOUNT="$ACCOUNT" HISTORY="$HISTORY" \\\n  bash scripts/run_portfolio_confirm_from_signals.sh\n'

if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print("[OK] replaced python invocation with bash invocation")
elif 'bash scripts/run_portfolio_confirm_from_signals.sh' in s:
    print("[OK] already patched")
else:
    s2 = s.replace('"$PYTHON" scripts/run_portfolio_confirm_from_signals.sh', 'bash scripts/run_portfolio_confirm_from_signals.sh')
    if s2 == s:
        raise SystemExit("[ERROR] expected invocation line not found")
    p.write_text(s2, encoding="utf-8")
    print("[OK] replaced invocation with fallback pattern")
PY

echo
echo "[CHECK] shell syntax"
bash -n "$TARGET"
bash -n scripts/run_portfolio_confirm_from_signals.sh

echo
echo "[DONE]"
echo "Now rerun:"
echo "  bash scripts/run_trading_day_signal_and_portfolio_all_models.sh"
