#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/patch_portfolio_full_fix_selftest.py

if [[ -f scripts/validate_portfolio_deep_state.py ]]; then
  "$PYTHON" scripts/validate_portfolio_deep_state.py --max-artifacts "${VALIDATE_N:-20}" || {
    echo "[ERROR] deep validator failed after self-test patch" >&2
    exit 3
  }
else
  echo "[WARN] scripts/validate_portfolio_deep_state.py not found; skipped optional deep validator"
fi
