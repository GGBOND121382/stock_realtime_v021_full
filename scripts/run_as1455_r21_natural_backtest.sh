#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_COL="${TARGET_COL:-r21_fwd}" \
  bash scripts/run_as1455_target_natural_backtest.sh
