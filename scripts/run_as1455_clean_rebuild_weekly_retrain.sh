#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"

# Rebuild range for the training HDF.
REBUILD_START_DATE="${REBUILD_START_DATE:-2020-01-01}"
REBUILD_END_DATE="${REBUILD_END_DATE:-2026-06-26}"

# Weekly retrain/backtest range.
START_DATE="${START_DATE:-2026-05-16}"
END_DATE="${END_DATE:-${REBUILD_END_DATE}}"

MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY="${RAW_DAILY:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
RAW_5M="${RAW_5M:-saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_${START_DATE}_to_${END_DATE}_clean_adj}"
FORCE="${FORCE:-1}"

echo "[1/3] clean adjusted model_data rebuild"
START_DATE="${REBUILD_START_DATE}" \
END_DATE="${REBUILD_END_DATE}" \
bash scripts/rebuild_as1455_model_data_clean_adj.sh

echo "[2/3] validate rebuilt model_data contract"
"${PYTHON}" tools/validate_as1455_model_data_contract.py \
  --model-data "${MODEL_DATA}" \
  --require-contract \
  --require-adjusted-artifacts

echo "[3/3] weekly retrain/backtest"
MODEL_DATA="${MODEL_DATA}" \
RAW_DAILY="${RAW_DAILY}" \
RAW_5M="${RAW_5M}" \
START_DATE="${START_DATE}" \
END_DATE="${END_DATE}" \
OUT_ROOT="${OUT_ROOT}" \
FORCE="${FORCE}" \
bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh

echo "[DONE] OUT_ROOT=${OUT_ROOT}"
