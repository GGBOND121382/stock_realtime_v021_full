#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-saved_data/ashare_ml4t/ch17_as1455_backtest_plots/default_ab_$(date +%Y%m%d_%H%M%S)}"
RANK_METRIC="${RANK_METRIC:-sharpe}"
FREQUENCIES="${FREQUENCIES:-daily,weekly,monthly}"

# Optional override. Use comma-separated lists with the same length, for example:
# BACKTEST_ROOTS="dir_a,dir_b,dir_c" LABELS="A,B,C" bash scripts/plot_as1455_default_ab_nav_curves.sh
BACKTEST_ROOTS="${BACKTEST_ROOTS:-}"
LABELS="${LABELS:-}"

args=(
  scripts/plot_as1455_backtest_return_curves.py
  --rank-metric "$RANK_METRIC"
  --frequencies "$FREQUENCIES"
  --out-dir "$OUT_DIR"
  --show-selected
)

if [[ -n "$BACKTEST_ROOTS" ]]; then
  IFS=',' read -r -a roots <<< "$BACKTEST_ROOTS"
  for root in "${roots[@]}"; do
    args+=(--backtest-root "$root")
  done
fi

if [[ -n "$LABELS" ]]; then
  IFS=',' read -r -a labels <<< "$LABELS"
  for label in "${labels[@]}"; do
    args+=(--label "$label")
  done
fi

"$PYTHON_BIN" "${args[@]}"

echo
echo "Output dir: $OUT_DIR"
echo "Generated files:"
find "$OUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.csv' -o -name '*.json' \) -print | sort
