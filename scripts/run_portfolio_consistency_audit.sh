#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON="${PYTHON:-python3}"
BACKTEST_DIR="${BACKTEST_DIR:-portfolio_reports/backtests/historical_score_portfolio}"
SAVED_MODELS="${SAVED_MODELS:-saved_models}"
INITIAL_CASH="${INITIAL_CASH:-200000}"
STRICT="${STRICT:-0}"
CMD=("$PYTHON" scripts/settle_portfolio_backtest_consistently.py --backtest-dir "$BACKTEST_DIR" --saved-models "$SAVED_MODELS" --initial-cash "$INITIAL_CASH")
if [[ "$STRICT" == "1" ]]; then CMD+=(--strict); fi
echo "[RUN]"; printf ' %q' "${CMD[@]}"; echo
exec "${CMD[@]}"
