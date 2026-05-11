#!/usr/bin/env bash
set -euo pipefail

mkdir -p configs scripts
cp configs/realtime_context_sources.toml ./configs/realtime_context_sources.toml
cp scripts/run_muyuan_hog_only_search.sh ./run_muyuan_hog_only_search.sh
chmod +x ./run_muyuan_hog_only_search.sh

echo "[OK] updated configs/realtime_context_sources.toml"
echo "[OK] installed ./run_muyuan_hog_only_search.sh"
echo
echo "Run:"
echo "  PYTHON=python3 ./run_muyuan_hog_only_search.sh"
