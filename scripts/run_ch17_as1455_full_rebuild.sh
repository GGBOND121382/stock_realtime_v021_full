#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_ch17_as1455_full_rebuild_aligned.sh "$@"
