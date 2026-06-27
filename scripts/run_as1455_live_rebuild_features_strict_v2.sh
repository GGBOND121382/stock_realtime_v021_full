#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
LIVE_DATE="$(${PYTHON} - <<PY
from datetime import datetime
s = "${TRADE_DATE}"
if s.lower() == "today":
    print(datetime.now().strftime("%Y%m%d"))
else:
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
)"
LIVE_DIR="${LIVE_DIR:-${OUT_ROOT}/${LIVE_DATE}}"
FEATURE_COLUMNS="${FEATURE_COLUMNS:-}"
SECTOR_REFERENCE="${SECTOR_REFERENCE:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-980}"
ALLOW_SECTOR_FALLBACK="${ALLOW_SECTOR_FALLBACK:-0}"

echo "[CONFIG]"
echo "  TRADE_DATE=${TRADE_DATE}"
echo "  LIVE_DATE=${LIVE_DATE}"
echo "  LIVE_DIR=${LIVE_DIR}"
echo "  SECTOR_REFERENCE=${SECTOR_REFERENCE}"
echo "  FEATURE_COLUMNS=${FEATURE_COLUMNS:-<default>}"
echo "  MIN_FEATURE_ROWS=${MIN_FEATURE_ROWS}"
echo "  ALLOW_SECTOR_FALLBACK=${ALLOW_SECTOR_FALLBACK}"

args=(
  features/build_as1455_live_features.py
  --trade-date "${TRADE_DATE}"
  --live-dir "${LIVE_DIR}"
  --min-feature-rows "${MIN_FEATURE_ROWS}"
  --sector-reference "${SECTOR_REFERENCE}"
)

if [[ -n "${FEATURE_COLUMNS}" ]]; then
  args+=(--training-feature-columns "${FEATURE_COLUMNS}")
fi
if [[ "${ALLOW_SECTOR_FALLBACK}" == "1" ]]; then
  args+=(--allow-sector-fallback)
fi

"${PYTHON}" "${args[@]}"

"${PYTHON}" tools/validate_as1455_live_model_features_v2.py \
  --live-dir "${LIVE_DIR}" \
  --trade-date "${TRADE_DATE}" \
  --sector-reference "${SECTOR_REFERENCE}" \
  --min-feature-rows "${MIN_FEATURE_ROWS}"

echo "[OK] live model features are aligned and usable for model prediction:"
echo "     ${LIVE_DIR}/11_live_model_features_for_prediction.csv"
echo "     ${LIVE_DIR}/11_live_model_features_usable.csv"
echo "     audit full file: ${LIVE_DIR}/11_live_model_features.csv"
echo "     ${LIVE_DIR}/12_feature_build_report.json"
echo "     ${LIVE_DIR}/13_live_feature_strict_validation_report.json"
