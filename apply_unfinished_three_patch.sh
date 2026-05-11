#!/usr/bin/env bash
# apply_unfinished_three_patch.sh
set -euo pipefail

mkdir -p configs scripts

cp configs/realtime_context_sources.toml ./configs/realtime_context_sources.toml
cp scripts/run_unfinished_three_search.sh ./run_unfinished_three_search.sh
chmod +x ./run_unfinished_three_search.sh

echo "[OK] patched configs/realtime_context_sources.toml"
echo "[OK] installed ./run_unfinished_three_search.sh"
echo
echo "Run:"
echo "  PYTHON=python3 ./run_unfinished_three_search.sh"
