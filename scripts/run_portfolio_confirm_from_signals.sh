#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE_DASH="${DATE_DASH:-$(date +%F)}"
DATE_COMPACT="${DATE_COMPACT:-$(date +%Y%m%d)}"

SIGNAL_DIR="${SIGNAL_DIR:-saved_data/intraday_nextday_signals/${DATE_COMPACT}}"
ACCOUNT="${ACCOUNT:-account.json}"
HISTORY="${HISTORY:-history_close.csv}"
SAVED_MODELS="${SAVED_MODELS:-saved_models}"
CONFIG="${CONFIG:-configs/portfolio_confirm_config.json}"
OUT_DIR="${OUT_DIR:-portfolio_reports}"

EXTRA_ARGS=()
if [[ "${USE_COVARIANCE_PENALTY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use-covariance-penalty)
  EXTRA_ARGS+=(--cov-risk-aversion "${COV_RISK_AVERSION:-3.0}")
fi

if [[ ! -f "$ACCOUNT" ]]; then
  echo "[ERROR] account file not found: $ACCOUNT"
  echo "Create it from template:"
  echo "  cp configs/account_template.json account.json"
  exit 2
fi

if [[ ! -d "$SIGNAL_DIR" ]]; then
  echo "[ERROR] signal dir not found: $SIGNAL_DIR"
  exit 2
fi

CMD=(
  "$PYTHON" portfolio_decision/portfolio_confirm_from_buy_signals.py
  --date "$DATE_DASH"
  --signal-dir "$SIGNAL_DIR"
  --account "$ACCOUNT"
  --saved-models "$SAVED_MODELS"
  --config "$CONFIG"
  --out-dir "$OUT_DIR"
)

if [[ -f "$HISTORY" ]]; then
  CMD+=(--history "$HISTORY")
else
  echo "[WARN] history file not found: $HISTORY; risk model will use conservative fallbacks."
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[RUN]"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
