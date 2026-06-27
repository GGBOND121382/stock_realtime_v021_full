#!/usr/bin/env bash
# Collect AS1455 live row and finalize prediction features under the 14:55 time budget.
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
FEATURE_COLUMNS="${FEATURE_COLUMNS:-}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-980}"
MAX_FINALIZE_SECONDS="${MAX_FINALIZE_SECONDS:-40}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
WARN_ONLY_TIME="${WARN_ONLY_TIME:-0}"

live_date() {
  "${PYTHON}" - <<PY
from datetime import datetime
s = "${TRADE_DATE}"
if s.lower() == "today":
    print(datetime.now().strftime("%Y%m%d"))
else:
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
}
LIVE_DATE="$(live_date)"
LIVE_DIR="${OUT_ROOT}/${LIVE_DATE}"

echo "[CONFIG]"
echo "  MODE=postfast"
echo "  TRADE_DATE=${TRADE_DATE}"
echo "  LIVE_DATE=${LIVE_DATE}"
echo "  LIVE_DIR=${LIVE_DIR}"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  FEATURE_COLUMNS=${FEATURE_COLUMNS:-<default>}"
echo "  MIN_FEATURE_ROWS=${MIN_FEATURE_ROWS}"
echo "  MAX_FINALIZE_SECONDS=${MAX_FINALIZE_SECONDS}"
echo "  SKIP_COLLECT=${SKIP_COLLECT}"

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  bash scripts/run_as1455_live_data_feature_pipeline.sh collect
fi

args=(
  features/finalize_as1455_live_features_fast.py
  --trade-date "${TRADE_DATE}"
  --out-root "${OUT_ROOT}"
  --min-feature-rows "${MIN_FEATURE_ROWS}"
  --max-elapsed-seconds "${MAX_FINALIZE_SECONDS}"
)
if [[ -n "${FEATURE_COLUMNS}" ]]; then
  args+=(--training-feature-columns "${FEATURE_COLUMNS}")
fi
if [[ "${WARN_ONLY_TIME}" == "1" ]]; then
  args+=(--warn-only-time)
fi

"${PYTHON}" "${args[@]}"
"${PYTHON}" - "${LIVE_DIR}/12_feature_build_report.json" "${LIVE_DIR}/13_live_feature_strict_validation_report.json" <<'PY'
import json, sys
r12 = json.load(open(sys.argv[1], encoding='utf-8'))
r13 = json.load(open(sys.argv[2], encoding='utf-8'))
if r12.get('feature_passed') is not True:
    raise SystemExit(f"feature_passed is not true: {sys.argv[1]}: {r12}")
if r13.get('passed') is not True:
    raise SystemExit(f"strict validation is not true: {sys.argv[2]}: {r13}")
print(f"[OK] postfast prediction features ready: {r13.get('prediction_file')}")
print(f"[OK] elapsed_seconds={r13.get('elapsed_seconds')} usable_rows={r13.get('feature_rows_usable')}")
PY
