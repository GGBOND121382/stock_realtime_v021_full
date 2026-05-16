#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"

CMD=(
  "$PYTHON" portfolio_decision/backtest_historical_score_portfolio.py
  --models-dir "${MODELS_DIR:-saved_models}"
  --saved-data-dir "${SAVED_DATA_DIR:-saved_data}"
  --context-config "${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
  --config "${CONFIG:-configs/portfolio_confirm_config.json}"
  --out-dir "${OUT_DIR:-portfolio_reports/backtests/historical_score_portfolio}"
  --model-policy "${MODEL_POLICY:-all}"
  --initial-cash "${INITIAL_CASH:-200000}"
  --hold-days "${HOLD_DAYS:-1}"
  --min-amount-yuan "${MIN_AMOUNT_YUAN:-50000000}"
)

# HISTORY is optional.  The script no longer expects history_close.csv.
if [[ -n "${HISTORY:-}" ]]; then CMD+=(--history "$HISTORY"); fi
if [[ -n "${WATCHLIST:-}" ]]; then CMD+=(--watchlist "$WATCHLIST"); fi
if [[ -n "${START_DATE:-}" ]]; then CMD+=(--start-date "$START_DATE"); fi
if [[ -n "${END_DATE:-}" ]]; then CMD+=(--end-date "$END_DATE"); fi
if [[ -n "${RESTORE_END_DATE:-}" ]]; then CMD+=(--restore-end-date "$RESTORE_END_DATE"); fi
if [[ -n "${GENERATED_SIGNAL_ROOT:-}" ]]; then CMD+=(--generated-signal-root "$GENERATED_SIGNAL_ROOT"); fi
if [[ "${CLOSE_OPEN_AT_END:-0}" == "1" ]]; then CMD+=(--close-open-at-end); fi
if [[ "${USE_COVARIANCE_PENALTY:-0}" == "1" ]]; then CMD+=(--use-covariance-penalty); fi
if [[ -n "${COV_RISK_AVERSION:-}" ]]; then CMD+=(--cov-risk-aversion "$COV_RISK_AVERSION"); fi
if [[ -n "${TIME_LIMIT_SEC:-}" ]]; then CMD+=(--time-limit-sec "$TIME_LIMIT_SEC"); fi
if [[ "${SCORE_ONLY:-0}" == "1" ]]; then CMD+=(--score-only); fi

printf '[RUN]'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
