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

SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-$SOURCE_DIR/baostock_5m_cache}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
INITIAL_CASH="${INITIAL_CASH:-200000}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
HIST_BASE="${HIST_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"
FWD_BASE="${FWD_BASE:-saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_independent_folds/$RUN_STAMP}"
PAIR_JSON="$OUT_ROOT/existing_result_pairs.json"
PAIR_TSV="$OUT_ROOT/existing_result_pairs.tsv"

mkdir -p "$OUT_ROOT"
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || {
  echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2
  exit 1
}
[[ -d "$HIST_BASE" ]] || {
  echo "[ERROR] missing historical result base: $HIST_BASE" >&2
  exit 1
}
[[ -d "$FWD_BASE" ]] || {
  echo "[ERROR] missing strict-OOS result base: $FWD_BASE" >&2
  exit 1
}

echo "[MODE] independent folds with frozen configurations"
echo "[MODE] prediction_generation=false grid=false training=false data_refresh=false"
echo "[MODE] backtest=true initial_state=empty_positions_and_initial_cash"
echo "[MODE] expected_backtests=40 initial_cash=$INITIAL_CASH"

"$PYTHON_BIN" scripts/check_as1455_disk_space.py \
  --path "$OUT_ROOT" \
  --min-free-gb "$MIN_FREE_GB" \
  --label independent-fold-frozen-config

"$PYTHON_BIN" scripts/resolve_as1455_existing_result_pairs.py \
  --historical-base "$HIST_BASE" \
  --forward-base "$FWD_BASE" \
  --feature-presets "$FEATURE_PRESETS" \
  --targets "$TARGETS" \
  --json-out "$PAIR_JSON" \
  --tsv-out "$PAIR_TSV"

args=(
  "$PYTHON_BIN"
  scripts/run_as1455_independent_fold_backtests.py
  --pair-manifest "$PAIR_JSON"
  --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
  --out-root "$OUT_ROOT"
  --initial-cash "$INITIAL_CASH"
  --output-mode "$OUTPUT_MODE"
  --frequencies daily,weekly,monthly
)

if [[ -d "$RAW_5M_CACHE_DIR" ]]; then
  args+=(--raw-5m-cache-dir "$RAW_5M_CACHE_DIR")
fi

"${args[@]}"

"$PYTHON_BIN" - "$OUT_ROOT/independent_fold_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
report = {
    "mode": payload.get("mode"),
    "prediction_generation": payload.get("prediction_generation"),
    "backtest": True,
    "parameter_grid": payload.get("parameter_grid"),
    "training": payload.get("training"),
    "data_refresh": payload.get("data_refresh"),
    "initial_state": payload.get("initial_state"),
    "initial_cash": payload.get("initial_cash"),
    "expected_backtests": payload.get("expected_backtests"),
    "backtest_count": payload.get("backtest_count"),
    "expected_plots": payload.get("expected_plots"),
    "plot_count": payload.get("plot_count"),
    "duration_seconds": payload.get("duration_seconds"),
    "all_ok": payload.get("all_ok"),
    "manifest": str(path),
}
report_path = path.parent / "independent_fold_report.json"
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["all_ok"]:
    raise SystemExit(1)
PY

echo "[PASS] 40 independent frozen-config fold backtests completed"
echo "[PASS] no model training, prediction generation, parameter grid, or data refresh"
echo "[PASS] output=$OUT_ROOT"
