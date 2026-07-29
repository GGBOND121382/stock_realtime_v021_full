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

FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
TARGET_COL="${TARGET_COL:?TARGET_COL is required: r01_fwd, r05_fwd, or r21_fwd}"
TARGET_FOLDS="${TARGET_FOLDS:-0,1,2,3,4,5}"
TOP_N="${TOP_N:-5}"
HISTORICAL_MODEL_DATA="${HISTORICAL_MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
FORWARD_MODEL_DATA="${FORWARD_MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5}"
SOURCE_DIR="${SOURCE_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-$SOURCE_DIR/baostock_raw_daily_cache}"
CACHE_BASE="${CACHE_BASE:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_prediction_cache}"
CACHE_ROOT="${CACHE_ROOT:-$CACHE_BASE/${FEATURE_PRESET}_${TARGET_COL}_top${TOP_N}}"
HISTORICAL_CACHE_ROOT="$CACHE_ROOT/historical_fold0_to_fold5"
FORWARD_CACHE_ROOT="$CACHE_ROOT/fold0_forward_latest"
FORCE_HISTORICAL_PREDICTIONS="${FORCE_HISTORICAL_PREDICTIONS:-0}"
REBUILD_FORWARD_PREDICTIONS="${REBUILD_FORWARD_PREDICTIONS:-1}"

case "$TARGET_COL" in
  r01_fwd) REBALANCE_EVERY="${REBALANCE_EVERY:-1}" ;;
  r05_fwd) REBALANCE_EVERY="${REBALANCE_EVERY:-5}" ;;
  r21_fwd) REBALANCE_EVERY="${REBALANCE_EVERY:-21}" ;;
  *) echo "[ERROR] unsupported TARGET_COL=$TARGET_COL" >&2; exit 2 ;;
esac

FOLD_DIR_TEMPLATE="${FOLD_DIR_TEMPLATE:-$($PYTHON_BIN - "$FEATURE_PRESET" "$TARGET_COL" <<'PY'
import sys
from utils.as1455_ch17_common import default_fold_dir_template
print(default_fold_dir_template(sys.argv[1], sys.argv[2]))
PY
)}"

if [[ -z "${NESTED_PREDICTION_SOURCE_ROOT+x}" ]]; then
  if [[ "$TARGET_COL" == "r05_fwd" && -d "saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1" ]]; then
    NESTED_PREDICTION_SOURCE_ROOT="saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1"
  else
    NESTED_PREDICTION_SOURCE_ROOT=""
  fi
fi

[[ -s "$HISTORICAL_MODEL_DATA" ]] || { echo "[ERROR] missing historical model_data: $HISTORICAL_MODEL_DATA" >&2; exit 1; }
[[ -s "$FORWARD_MODEL_DATA" ]] || { echo "[ERROR] missing forward model_data: $FORWARD_MODEL_DATA" >&2; exit 1; }
[[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache: $RAW_DAILY_CACHE_DIR" >&2; exit 1; }

"$PYTHON_BIN" -m py_compile \
  scripts/run_as1455_target_one_lag_backtest.py \
  scripts/run_as1455_fold0_forward_backtest.py \
  scripts/reuse_as1455_nested_predictions_for_global_grid.py

mkdir -p "$CACHE_ROOT"

check_source_folds() {
  "$PYTHON_BIN" - "$FOLD_DIR_TEMPLATE" "$TARGET_FOLDS" <<'PY'
import sys
from pathlib import Path

template, target_folds = sys.argv[1:]
missing = []
for token in target_folds.split(','):
    token = token.strip()
    if not token:
        continue
    target_fold = int(token)
    source_fold = target_fold + 1
    path = Path(template.format(fold=source_fold)).expanduser().resolve()
    if not path.is_dir():
        missing.append((source_fold, path))
if missing:
    for fold, path in missing:
        print(f"[MISSING] source_fold{fold}: {path}", file=sys.stderr)
    raise SystemExit(3)
PY
}

ensure_prediction_segments() {
  "$PYTHON_BIN" - "$HISTORICAL_CACHE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
pred_dir = root / "00_predictions"
pred_file = pred_dir / "test_preds.h5"
segments_file = pred_dir / "prediction_segments.csv"
if segments_file.exists():
    print(f"[RESUME] prediction segments={segments_file}")
    raise SystemExit(0)
if not pred_file.exists():
    raise FileNotFoundError(pred_file)
manifest_file = pred_dir / "one_lag_prediction_manifest.json"
if not manifest_file.exists():
    raise FileNotFoundError(
        f"missing {segments_file} and cannot derive it without {manifest_file}"
    )
manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
mapping = manifest.get("fold_mapping") or []
df = pd.read_hdf(pred_file, "predictions")
dates = pd.DatetimeIndex(df.index.get_level_values("date")).normalize()
rows = []
for item in mapping:
    start = pd.Timestamp(item["target_test_start"]).normalize()
    end = pd.Timestamp(item["target_test_end"]).normalize()
    mask = (dates >= start) & (dates <= end)
    selected = dates[mask]
    rows.append({
        "source_fold": int(item["source_fold"]),
        "target_fold": int(item["target_fold"]),
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "n_days": int(selected.nunique()),
        "n_rows": int(mask.sum()),
    })
segments = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
if len(segments) != 6:
    raise RuntimeError(f"expected six target-fold segments, got {len(segments)}")
segments.to_csv(segments_file, index=False, encoding="utf-8-sig")
print(f"[OK] prediction segments={segments_file}")
PY
}

historical_pred="$HISTORICAL_CACHE_ROOT/00_predictions/test_preds.h5"
segments_file="$HISTORICAL_CACHE_ROOT/00_predictions/prediction_segments.csv"
if [[ "$FORCE_HISTORICAL_PREDICTIONS" == "1" || ! -s "$historical_pred" ]]; then
  rm -rf "$HISTORICAL_CACHE_ROOT"
  mkdir -p "$HISTORICAL_CACHE_ROOT"
  if [[ -n "$NESTED_PREDICTION_SOURCE_ROOT" && -d "$NESTED_PREDICTION_SOURCE_ROOT" ]]; then
    echo "[CACHE] reuse nested historical predictions from $NESTED_PREDICTION_SOURCE_ROOT"
    "$PYTHON_BIN" scripts/reuse_as1455_nested_predictions_for_global_grid.py \
      --nested-root "$NESTED_PREDICTION_SOURCE_ROOT" \
      --out-root "$HISTORICAL_CACHE_ROOT" \
      --force
  else
    echo "[CACHE] generate historical one-fold-lag predictions: target=$TARGET_COL folds=$TARGET_FOLDS"
    if ! check_source_folds; then
      if [[ "$TARGET_COL" == "r21_fwd" ]]; then
        echo "[BLOCKED] r21 fold0..5 requires a source_fold6 checkpoint directory." >&2
        echo "Try training it explicitly (only if the updated dataset now supports fold6):" >&2
        echo "  TARGET_COL=r21_fwd FEATURE_PRESETS=rotation_addon_onehot FOLDS=6 bash scripts/run_as1455_target_search_all.sh" >&2
      fi
      exit 3
    fi
    "$PYTHON_BIN" scripts/run_as1455_target_one_lag_backtest.py \
      --feature-preset "$FEATURE_PRESET" \
      --target-col "$TARGET_COL" \
      --rebalance-every "$REBALANCE_EVERY" \
      --target-folds "$TARGET_FOLDS" \
      --top-n "$TOP_N" \
      --model-data "$HISTORICAL_MODEL_DATA" \
      --fold-dir-template "$FOLD_DIR_TEMPLATE" \
      --out-root "$HISTORICAL_CACHE_ROOT" \
      --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
      --skip-grid
  fi
else
  echo "[RESUME] historical prediction cache=$historical_pred"
fi
ensure_prediction_segments
[[ -s "$historical_pred" && -s "$segments_file" ]] || { echo "[ERROR] incomplete historical prediction cache" >&2; exit 1; }

forward_pred="$FORWARD_CACHE_ROOT/00_predictions/fold0_forward_preds.h5"
if [[ "$REBUILD_FORWARD_PREDICTIONS" == "1" || ! -s "$forward_pred" ]]; then
  fold0_dir="${FOLD0_DIR:-${FOLD_DIR_TEMPLATE//\{fold\}/0}}"
  [[ -d "$fold0_dir" ]] || { echo "[ERROR] missing fold0 checkpoint directory: $fold0_dir" >&2; exit 1; }
  rm -rf "$FORWARD_CACHE_ROOT"
  mkdir -p "$FORWARD_CACHE_ROOT"
  echo "[CACHE] rebuild latest fold0 forward predictions: target=$TARGET_COL top_n=$TOP_N"
  "$PYTHON_BIN" scripts/run_as1455_fold0_forward_backtest.py \
    --feature-preset "$FEATURE_PRESET" \
    --target-col "$TARGET_COL" \
    --rebalance-every "$REBALANCE_EVERY" \
    --fold0-dir "$fold0_dir" \
    --model-data "$FORWARD_MODEL_DATA" \
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR" \
    --out-root "$FORWARD_CACHE_ROOT" \
    --model-selection-mode all_top_n \
    --top-n "$TOP_N" \
    --skip-grid
else
  echo "[RESUME] forward prediction cache=$forward_pred"
fi
[[ -s "$forward_pred" ]] || { echo "[ERROR] missing forward prediction cache: $forward_pred" >&2; exit 1; }

"$PYTHON_BIN" - "$historical_pred" "$forward_pred" <<'PY'
import sys
from pathlib import Path
import pandas as pd
for label, value in (("historical", sys.argv[1]), ("forward", sys.argv[2])):
    path = Path(value)
    df = pd.read_hdf(path, "predictions")
    cols = [str(c) for c in df.columns]
    if not all(str(i) in cols for i in range(5)):
        raise RuntimeError(f"{label} cache does not contain model columns 0..4: {cols}")
    dates = pd.DatetimeIndex(df.index.get_level_values("date"))
    print(
        f"[CACHE OK] {label}: rows={len(df)} dates={dates.nunique()} "
        f"range={dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d} columns={cols}"
    )
PY

echo "[PASS] prediction cache ready: $CACHE_ROOT"
