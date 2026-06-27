#!/usr/bin/env bash
# Build AS1455 fast live feature state before 14:55.
# This is production pre-work. It may be run after the normal pre stage.
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
SECTOR_REFERENCE="${SECTOR_REFERENCE:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
FEATURE_COLUMNS="${FEATURE_COLUMNS:-}"
TAIL_DAYS="${TAIL_DAYS:-252}"
RUN_PRE_STAGE="${RUN_PRE_STAGE:-1}"

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
echo "  MODE=prefast"
echo "  TRADE_DATE=${TRADE_DATE}"
echo "  LIVE_DATE=${LIVE_DATE}"
echo "  LIVE_DIR=${LIVE_DIR}"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  SECTOR_REFERENCE=${SECTOR_REFERENCE}"
echo "  FEATURE_COLUMNS=${FEATURE_COLUMNS:-<default>}"
echo "  TAIL_DAYS=${TAIL_DAYS}"
echo "  RUN_PRE_STAGE=${RUN_PRE_STAGE}"

if [[ "${RUN_PRE_STAGE}" == "1" ]]; then
  bash scripts/run_as1455_live_data_feature_pipeline.sh pre
fi

args=(
  features/build_as1455_live_feature_state_fast.py
  --trade-date "${TRADE_DATE}"
  --out-root "${OUT_ROOT}"
  --sector-reference "${SECTOR_REFERENCE}"
  --tail-days "${TAIL_DAYS}"
)
if [[ -n "${FEATURE_COLUMNS}" ]]; then
  args+=(--training-feature-columns "${FEATURE_COLUMNS}")
fi

"${PYTHON}" "${args[@]}"
"${PYTHON}" - "${LIVE_DIR}/06_live_feature_state_fast_report.json" <<'PY'
import json, sys
p = sys.argv[1]
obj = json.load(open(p, encoding='utf-8'))
if obj.get('prefast_passed') is not True:
    raise SystemExit(f"prefast failed: {p}: {obj}")
print(f"[OK] prefast state ready: {obj.get('state_file')}")
PY
