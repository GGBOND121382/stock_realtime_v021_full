#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2026-06-26}"

OUT_DIR="${OUT_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
BAR_CACHE_DIR="${BAR_CACHE_DIR:-${OUT_DIR}/baostock_5m_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-${OUT_DIR}/as1455_daily_cache}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-${OUT_DIR}/baostock_raw_daily_cache}"
QFQ_DAILY_CACHE_DIR="${QFQ_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_reproduce/baostock_qfq_daily_cache}"

# Default is strict offline rebuild. Set FETCH_MISSING_RAW_DAILY=1 only if you intentionally want BaoStock network fetch.
FETCH_MISSING_BAOSTOCK="${FETCH_MISSING_BAOSTOCK:-0}"
FETCH_MISSING_RAW_DAILY="${FETCH_MISSING_RAW_DAILY:-0}"
FETCH_MISSING_QFQ_DAILY="${FETCH_MISSING_QFQ_DAILY:-0}"
QFQ5M_AUDIT_SAMPLES="${QFQ5M_AUDIT_SAMPLES:-0}"

fetch_baostock_arg="--no-fetch-missing-baostock"
fetch_raw_daily_arg="--no-fetch-missing-raw-daily"
fetch_qfq_daily_arg="--no-fetch-missing-qfq-daily"
[[ "${FETCH_MISSING_BAOSTOCK}" == "1" ]] && fetch_baostock_arg="--fetch-missing-baostock"
[[ "${FETCH_MISSING_RAW_DAILY}" == "1" ]] && fetch_raw_daily_arg="--fetch-missing-raw-daily"
[[ "${FETCH_MISSING_QFQ_DAILY}" == "1" ]] && fetch_qfq_daily_arg="--fetch-missing-qfq-daily"

echo "[CONFIG]"
echo "  OUT_DIR=${OUT_DIR}"
echo "  UNIVERSE=${UNIVERSE}"
echo "  BAR_CACHE_DIR=${BAR_CACHE_DIR}"
echo "  AS1455_DAILY_CACHE_DIR=${AS1455_DAILY_CACHE_DIR}"
echo "  RAW_DAILY_CACHE_DIR=${RAW_DAILY_CACHE_DIR}"
echo "  QFQ_DAILY_CACHE_DIR=${QFQ_DAILY_CACHE_DIR}"
echo "  START_DATE=${START_DATE}"
echo "  END_DATE=${END_DATE}"
echo "  FETCH_MISSING_BAOSTOCK=${FETCH_MISSING_BAOSTOCK}"
echo "  FETCH_MISSING_RAW_DAILY=${FETCH_MISSING_RAW_DAILY}"
echo "  FETCH_MISSING_QFQ_DAILY=${FETCH_MISSING_QFQ_DAILY}"

[[ -d "${BAR_CACHE_DIR}" ]] || { echo "[ERROR] missing BAR_CACHE_DIR=${BAR_CACHE_DIR}" >&2; exit 1; }
[[ -d "${RAW_DAILY_CACHE_DIR}" ]] || { echo "[ERROR] missing RAW_DAILY_CACHE_DIR=${RAW_DAILY_CACHE_DIR}" >&2; exit 1; }
[[ -f "${UNIVERSE}" ]] || { echo "[ERROR] missing UNIVERSE=${UNIVERSE}" >&2; exit 1; }

mkdir -p "${OUT_DIR}" "${AS1455_DAILY_CACHE_DIR}"

"${PYTHON}" scripts/build_ashare_ch12_as1455_model_data.py \
  --out-dir "${OUT_DIR}" \
  --universe "${UNIVERSE}" \
  --bar-root "${BAR_CACHE_DIR}" \
  --bar-glob "*_5m_raw.csv" \
  --baostock-5m-cache-dir "${BAR_CACHE_DIR}" \
  --as1455-daily-cache-dir "${AS1455_DAILY_CACHE_DIR}" \
  --raw-daily-cache-dir "${RAW_DAILY_CACHE_DIR}" \
  --qfq-daily-cache-dir "${QFQ_DAILY_CACHE_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --rebuild-as1455-daily-cache \
  --adjust-factor-mode raw_preclose \
  --qfq5m-audit-samples "${QFQ5M_AUDIT_SAMPLES}" \
  "${fetch_baostock_arg}" \
  "${fetch_raw_daily_arg}" \
  "${fetch_qfq_daily_arg}" \
  --profile-memory

"${PYTHON}" tools/validate_as1455_model_data_contract.py \
  --model-data "${OUT_DIR}/model_data_as1455.h5" \
  --write-contract \
  --require-adjusted-artifacts

echo "[DONE] clean adjusted AS1455 model_data rebuilt: ${OUT_DIR}/model_data_as1455.h5"
echo "[DONE] contract: ${OUT_DIR}/model_data_contract.json"
