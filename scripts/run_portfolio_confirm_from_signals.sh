#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATE_DASH="${DATE_DASH:-$(date +%F)}"
DATE_COMPACT="${DATE_COMPACT:-$(date +%Y%m%d)}"

SIGNAL_DIR="${SIGNAL_DIR:-saved_data/intraday_nextday_signals/${DATE_COMPACT}}"
ACCOUNT="${ACCOUNT:-account.json}"
HISTORY="${HISTORY:-}"
SAVED_MODELS="${SAVED_MODELS:-saved_models}"
SAVED_DATA_DIR="${SAVED_DATA_DIR:-saved_data}"
CONFIG="${CONFIG:-configs/portfolio_confirm_config.json}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
MODEL_OVERRIDES="${MODEL_OVERRIDES:-configs/portfolio_model_overrides.csv}"
RECENT_PERF="${RECENT_PERF:-}"
OUT_DIR="${OUT_DIR:-portfolio_reports}"
AUTO_RISK_HISTORY="${AUTO_RISK_HISTORY:-1}"
RISK_HISTORY_DIR="${RISK_HISTORY_DIR:-${OUT_DIR}/risk_history}"

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


if [[ "${AUTO_RISK_HISTORY:-1}" == "1" && ( -z "${HISTORY:-}" || ! -f "$HISTORY" ) ]]; then
  mkdir -p "$RISK_HISTORY_DIR"
  AUTO_HISTORY="${RISK_HISTORY_DIR}/risk_history_for_portfolio_${DATE_COMPACT}.csv"
  SIGNAL_FILE="$SIGNAL_DIR/buy_signals.csv"
  if [[ ! -f "$SIGNAL_FILE" ]]; then
    SIGNAL_FILE="$SIGNAL_DIR/all_scores.csv"
  fi

  echo "[RISK_HISTORY] building point-in-time live risk history: $AUTO_HISTORY"
  "$PYTHON" scripts/build_portfolio_risk_history.py \
    --saved-models "$SAVED_MODELS" \
    --saved-data-dir "$SAVED_DATA_DIR" \
    --signals "$SIGNAL_FILE" \
    --account "$ACCOUNT" \
    --date "$DATE_DASH" \
    --out "$AUTO_HISTORY"

  if [[ -f "$AUTO_HISTORY" ]]; then
    "$PYTHON" -c "import pandas as pd, sys; p=sys.argv[1]; df=pd.read_csv(p); ok=df.shape[0] >= 20 and df.shape[1] >= 2; print(f'[RISK_HISTORY] shape={df.shape} path={p}'); sys.exit(0 if ok else 3)" "$AUTO_HISTORY"
    HISTORY="$AUTO_HISTORY"
  fi
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

if [[ -f "$CONTEXT_CONFIG" ]]; then
  CMD+=(--context-config "$CONTEXT_CONFIG")
fi

if [[ -f "$MODEL_OVERRIDES" ]]; then
  CMD+=(--model-overrides "$MODEL_OVERRIDES")
fi

if [[ -n "$RECENT_PERF" && -f "$RECENT_PERF" ]]; then
  CMD+=(--recent-perf "$RECENT_PERF")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[RUN]"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
