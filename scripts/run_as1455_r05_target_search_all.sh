#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_COL="${TARGET_COL:-r05_fwd}" \
  bash scripts/run_as1455_target_search_all.sh
