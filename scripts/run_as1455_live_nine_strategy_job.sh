#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# Compatibility entry point retained for dashboard buttons and existing cron.
# 09:35 pre still prepares data and maintains all nine tracking accounts.
# 14:50 post now uses the latency-critical production strategy path only.
exec bash scripts/run_as1455_live_production_job.sh "$@"
