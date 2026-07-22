#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
  backtest|backtest-only)
    exec bash scripts/run_ch17_as1455_backtest_only.sh
    ;;
  all|preflight|data|selfcheck|training|historical|forward|plots|audit|status|"")
    echo "[BLOCKED] The full-rebuild entry is disabled on this branch." >&2
    echo "It may refresh data, rebuild model_data, or train models, which is not required for this task." >&2
    echo "Run the existing-model workflow instead:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only" >&2
    exit 2
    ;;
  *)
    echo "[ERROR] unsupported mode: $mode" >&2
    echo "Usage: bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only" >&2
    exit 2
    ;;
esac
