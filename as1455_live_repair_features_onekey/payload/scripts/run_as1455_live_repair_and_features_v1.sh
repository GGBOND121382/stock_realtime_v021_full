#!/usr/bin/env bash
# Repair AS1455 live 08 collection status and build 09/10/11/12 feature outputs
# from already-collected data. This does NOT re-fetch realtime quotes.
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
LIVE_DIR="${LIVE_DIR:-}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55:00}"
MIN_VALID_RATE="${MIN_VALID_RATE:-0.98}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-980}"
FEATURE_COLUMNS="${FEATURE_COLUMNS:-}"
NO_FEATURES="${NO_FEATURES:-0}"

args=(
  tools/repair_as1455_live_collect_and_features_v1.py
  --repo-root .
  --trade-date "${TRADE_DATE}"
  --out-root "${OUT_ROOT}"
  --cutoff-time "${CUTOFF_TIME}"
  --min-valid-rate "${MIN_VALID_RATE}"
  --min-feature-rows "${MIN_FEATURE_ROWS}"
)
if [[ -n "${LIVE_DIR}" ]]; then
  args+=(--live-dir "${LIVE_DIR}")
fi
if [[ -n "${FEATURE_COLUMNS}" ]]; then
  args+=(--feature-columns "${FEATURE_COLUMNS}")
fi
if [[ "${NO_FEATURES}" == "1" ]]; then
  args+=(--no-features)
fi

"${PYTHON}" "${args[@]}"
