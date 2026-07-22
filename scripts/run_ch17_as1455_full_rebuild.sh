#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
  independent-folds|backtest|backtest-only)
    exec bash scripts/run_ch17_as1455_backtest_only.sh
    ;;
  existing-results|plots)
    exec bash scripts/run_ch17_as1455_existing_results.sh
    ;;
  all|preflight|data|selfcheck|training|historical|forward|audit|status|"")
    echo "[BLOCKED] The full-rebuild entry is disabled on this branch." >&2
    echo "This task must not refresh data, rebuild model_data, train models, or sweep grids." >&2
    echo "Run independent fold backtests with frozen existing configurations:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds" >&2
    echo "Or only replot the old continuous NAV results:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh existing-results" >&2
    exit 2
    ;;
  *)
    echo "[ERROR] unsupported mode: $mode" >&2
    echo "Usage:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh existing-results" >&2
    exit 2
    ;;
esac
