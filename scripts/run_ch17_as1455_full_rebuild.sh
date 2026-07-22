#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
  backtest|backtest-only)
    exec bash scripts/run_ch17_as1455_backtest_only.sh
    ;;
  all)
    echo "[BLOCKED] The aligned full-rebuild experiment is disabled because it would rebuild data and retrain models." >&2
    echo "Use: bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only" >&2
    exit 2
    ;;
  *)
    exec bash scripts/run_ch17_as1455_full_rebuild_aligned.sh "$@"
    ;;
esac
