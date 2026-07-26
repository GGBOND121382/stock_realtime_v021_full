#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
  independent-folds|backtest|backtest-only)
    exec bash scripts/run_ch17_as1455_backtest_only.sh
    ;;
  r05-addon-comparison|r05-addon-folds|r05-addon-nested)
    exec bash scripts/run_as1455_r05_addon_fold_comparison.sh
    ;;
  existing-results|plots)
    exec bash scripts/run_ch17_as1455_existing_results.sh
    ;;
  all|preflight|data|selfcheck|training|historical|forward|audit|status|"")
    echo "[BLOCKED] The unrestricted full-rebuild entry is disabled on this branch." >&2
    echo "Use the corrected nested r05_fwd workflow; it reuses existing checkpoints" >&2
    echo "but runs one validation grid per source fold:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison" >&2
    echo "Run all independent frozen-config fold backtests:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds" >&2
    echo "Or only replot old results:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh existing-results" >&2
    exit 2
    ;;
  *)
    echo "[ERROR] unsupported mode: $mode" >&2
    echo "Usage:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh existing-results" >&2
    exit 2
    ;;
esac
