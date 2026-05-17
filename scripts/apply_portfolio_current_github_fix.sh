#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
"$PYTHON" scripts/patch_portfolio_current_github_fix.py
