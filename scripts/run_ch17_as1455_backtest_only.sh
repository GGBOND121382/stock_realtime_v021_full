#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="${BASE_PYTHON:-python3}"
  fi
fi

FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
HIST_BASE="${HIST_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"
FWD_BASE="${FWD_BASE:-saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest}"
PLOT_DIR="${PLOT_DIR:-saved_data/ashare_ml4t/ch17_as1455_backtest_plots/existing_results_${RUN_STAMP}}"
REPORT_DIR="${REPORT_DIR:-saved_data/ashare_ml4t/ch17_as1455_existing_results/${RUN_STAMP}}"
PAIR_JSON="$REPORT_DIR/existing_result_pairs.json"
PAIR_TSV="$REPORT_DIR/existing_result_pairs.tsv"
START_EPOCH="$(date +%s)"

mkdir -p "$PLOT_DIR" "$REPORT_DIR"

echo "[MODE] existing results only"
echo "[MODE] prediction=false backtest=false grid=false training=false data_refresh=false"

"$PYTHON_BIN" scripts/resolve_as1455_existing_result_pairs.py \
  --historical-base "$HIST_BASE" \
  --forward-base "$FWD_BASE" \
  --feature-presets "$FEATURE_PRESETS" \
  --targets "$TARGETS" \
  --json-out "$PAIR_JSON" \
  --tsv-out "$PAIR_TSV"

roots=""
labels=""
pair_count=0
sequence_args=(
  "$PYTHON_BIN" scripts/plot_as1455_fold_sequence_curves.py
  --rank-metric sharpe
  --out-dir "$PLOT_DIR/fold_sequence"
)

while IFS=$'\t' read -r label historical_root forward_root; do
  [[ -n "$label" && -n "$historical_root" && -n "$forward_root" ]] || continue
  roots+="${roots:+,}$forward_root"
  labels+="${labels:+,}$label"
  sequence_args+=(
    --historical-root "$historical_root"
    --forward-root "$forward_root"
    --label "$label"
  )
  pair_count=$((pair_count + 1))
done < "$PAIR_TSV"

expected_pairs=$(( $(wc -w <<<"$FEATURE_PRESETS") * $(wc -w <<<"$TARGETS") ))
if [[ "$pair_count" -ne "$expected_pairs" ]]; then
  echo "[ERROR] result pair count mismatch: expected=$expected_pairs actual=$pair_count" >&2
  exit 1
fi

env \
  PYTHON_BIN="$PYTHON_BIN" \
  BACKTEST_ROOTS="$roots" \
  LABELS="$labels" \
  OUT_DIR="$PLOT_DIR" \
  RANK_METRIC=sharpe \
  bash scripts/plot_as1455_default_ab_nav_curves.sh

"${sequence_args[@]}"

END_EPOCH="$(date +%s)"
DURATION_SECONDS=$((END_EPOCH - START_EPOCH))
"$PYTHON_BIN" - "$RUN_STAMP" "$PLOT_DIR" "$REPORT_DIR" "$PAIR_JSON" "$DURATION_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

stamp, plot_text, report_text, pair_text, duration_text = sys.argv[1:]
plot_dir = Path(plot_text)
report_dir = Path(report_text)
pair_file = Path(pair_text)
pairs = json.loads(pair_file.read_text(encoding="utf-8"))
fold_pngs = sorted((plot_dir / "fold_sequence").glob("fold*/return_curve_*.png"))
forward_pngs = [plot_dir / f"return_curve_{frequency}.png" for frequency in ("daily", "weekly", "monthly")]
report = {
    "run_stamp": stamp,
    "mode": "existing_results_plot_only",
    "prediction": False,
    "backtest": False,
    "grid": False,
    "training": False,
    "data_refresh": False,
    "pair_count": int(pairs.get("pair_count", 0)),
    "fold_plot_expected": 21,
    "fold_plot_ok": len(fold_pngs),
    "forward_plot_expected": 3,
    "forward_plot_ok": sum(path.is_file() for path in forward_pngs),
    "duration_seconds": int(duration_text),
    "pair_manifest": str(pair_file),
    "plot_dir": str(plot_dir),
}
report["all_ok"] = (
    report["pair_count"] == 6
    and report["fold_plot_ok"] == report["fold_plot_expected"]
    and report["forward_plot_ok"] == report["forward_plot_expected"]
)
(report_dir / "existing_results_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["all_ok"]:
    raise SystemExit(1)
PY

echo "[PASS] existing results plotted without prediction, backtest, grid, training, or data refresh"
echo "[PASS] plots=$PLOT_DIR"
echo "[PASS] report=$REPORT_DIR/existing_results_report.json"
