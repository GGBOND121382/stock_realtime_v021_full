#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
DATE="${DATE:-$(date +%Y%m%d)}"
CONTEXT_DIR="${CONTEXT_DIR:-saved_data/realtime_context}"
CUTOFF_TIME="${CUTOFF_TIME:-14:55}"
"$PYTHON" scripts/fill_sector_range_z20_from_history.py \
  --date "$DATE" \
  --context-dir "$CONTEXT_DIR" \
  --cutoff-time "$CUTOFF_TIME"
