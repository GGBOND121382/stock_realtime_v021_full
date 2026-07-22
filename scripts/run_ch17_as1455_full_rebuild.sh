#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
  existing-results|plot-existing|backtest|backtest-only)
    exec bash scripts/run_ch17_as1455_backtest_only.sh
    ;;
  all|preflight|data|selfcheck|training|historical|forward|plots|audit|status|"")
    echo "[BLOCKED] The full-rebuild entry is disabled on this branch." >&2
    echo "This task only reads complete existing historical and strict-OOS NAV results." >&2
    echo "Run:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh existing-results" >&2
    exit 2
    ;;
  *)
    echo "[ERROR] unsupported mode: $mode" >&2
    echo "Usage: bash scripts/run_ch17_as1455_full_rebuild.sh existing-results" >&2
    exit 2
    ;;
esac
