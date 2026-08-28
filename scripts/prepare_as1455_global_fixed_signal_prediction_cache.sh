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
FORCE_HISTORICAL_PREDICTIONS="${FORCE_HISTORICAL_PREDICTIONS:-0}"
REBUILD_FORWARD_PREDICTIONS="${REBUILD_FORWARD_PREDICTIONS:-1}"

case "$TARGET_COL" in
  r01_fwd) REBALANCE_EVERY="${REBALANCE_EVERY:-1}" ;;
  r05_fwd) REBALANCE_EVERY="${REBALANCE_EVERY:-5}" ;;
  r21_fwd) REBALANCE_EVERY="${REBALANCE_EVERY:-21}" ;;
  *) echo "[ERROR] unsupported TARGET_COL=$TARGET_COL" >&2; exit 2 ;;
esac

first_fold="${TARGET_FOLDS%%,*}"
last_fold="${TARGET_FOLDS##*,}"
fold_label="fold${first_fold}_${last_fold}"
HISTORICAL_CACHE_ROOT="${HISTORICAL_CACHE_ROOT:-$CACHE_ROOT/historical_${fold_label}}"
FORWARD_CACHE_ROOT="${FORWARD_CACHE_ROOT:-$CACHE_ROOT/fold0_forward_latest}"

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
  "$PYTHON_BIN" - "$FOLD_DIR_TEMPLATE" "$TARGET_FOLDS" "$TOP_N" <<'PY'
import sys
from pathlib import Path
import pandas as pd

template, target_folds, top_n_text = sys.argv[1:]
top_n = int(top_n_text)
problems = []
for token in target_folds.split(','):
    token = token.strip()
    if not token:
        continue
    target_fold = int(token)
    source_fold = target_fold + 1
    root = Path(template.format(fold=source_fold)).expanduser().resolve()
    table_file = root / 'search_best_checkpoints.csv'
    if not root.is_dir():
        problems.append(f'source_fold{source_fold}: missing directory {root}')
        continue
    if not table_file.is_file():
        problems.append(f'source_fold{source_fold}: missing {table_file}')
        continue
    table = pd.read_csv(table_file)
    if 'checkpoint_saved' in table.columns:
        mask = table['checkpoint_saved'].astype(str).str.strip().str.lower().isin(
            ['true', '1', 'yes']
        )
        table = table.loc[mask]
    if len(table) < top_n:
        problems.append(
            f'source_fold{source_fold}: need {top_n} saved checkpoints, got {len(table)}'
        )
if problems:
    for problem in problems:
        print(f'[MISSING] {problem}', file=sys.stderr)
    raise SystemExit(3)
PY
}

ensure_prediction_segments() {
  "$PYTHON_BIN" - "$HISTORICAL_CACHE_ROOT" "$TARGET_FOLDS" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
expected_folds = [
    int(token.strip()) for token in sys.argv[2].split(',') if token.strip()
]
if not expected_folds:
    raise RuntimeError('TARGET_FOLDS is empty')
if len(expected_folds) != len(set(expected_folds)):
    raise RuntimeError(f'duplicate TARGET_FOLDS: {expected_folds}')

pred_dir = root / '00_predictions'
pred_file = pred_dir / 'test_preds.h5'
segments_file = pred_dir / 'prediction_segments.csv'
manifest_file = pred_dir / 'one_lag_prediction_manifest.json'
if not pred_file.exists():
    raise FileNotFoundError(pred_file)
if not manifest_file.exists():
    raise FileNotFoundError(manifest_file)

manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
mapping = manifest.get('fold_mapping') or []
if not mapping and manifest.get('segments'):
    mapping = [
        {
            'source_fold': item['source_fold'],
            'target_fold': item['target_fold'],
            'target_test_start': item['start'],
            'target_test_end': item['end'],
        }
        for item in manifest['segments']
    ]
by_target = {int(item['target_fold']): item for item in mapping}
missing = sorted(set(expected_folds) - set(by_target))
if missing:
    raise RuntimeError(
        f'prediction manifest does not cover requested target folds: {missing}'
    )

df = pd.read_hdf(pred_file, 'predictions')
dates = pd.DatetimeIndex(df.index.get_level_values('date')).normalize()
rows = []
for target_fold in expected_folds:
    item = by_target[target_fold]
    source_fold = int(item['source_fold'])
    if source_fold != target_fold + 1:
        raise RuntimeError(
            f'bad one-fold-lag mapping: source={source_fold} target={target_fold}'
        )
    start = pd.Timestamp(item['target_test_start']).normalize()
    end = pd.Timestamp(item['target_test_end']).normalize()
    mask = (dates >= start) & (dates <= end)
    selected = dates[mask]
    if not mask.any():
        raise RuntimeError(
            f'no prediction rows for target_fold{target_fold}: '
            f'{start:%Y-%m-%d}..{end:%Y-%m-%d}'
        )
    rows.append({
        'source_fold': source_fold,
        'target_fold': target_fold,
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'n_days': int(selected.nunique()),
        'n_rows': int(mask.sum()),
    })
segments = pd.DataFrame(rows).sort_values('start').reset_index(drop=True)
actual = sorted(segments['target_fold'].astype(int).tolist())
if actual != sorted(expected_folds):
    raise RuntimeError(
        f'segment fold mismatch: expected={sorted(expected_folds)} actual={actual}'
    )
segments.to_csv(segments_file, index=False, encoding='utf-8-sig')
print(
    f"[OK] prediction segments={segments_file} "
    f"target_folds={segments['target_fold'].astype(int).tolist()}"
)
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
      --target-folds "$TARGET_FOLDS" \
      --top-n "$TOP_N" \
      --force
  else
    echo "[CACHE] generate historical one-fold-lag predictions: target=$TARGET_COL folds=$TARGET_FOLDS"
    if ! check_source_folds; then
      echo "[BLOCKED] requested folds require unavailable source-fold checkpoints." >&2
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

"$PYTHON_BIN" - "$historical_pred" "$forward_pred" "$TARGET_FOLDS" "$TOP_N" <<'PY'
import sys
from pathlib import Path
import pandas as pd

historical, forward, target_folds, top_n_text = sys.argv[1:]
top_n = int(top_n_text)
for label, value in (('historical', historical), ('forward', forward)):
    path = Path(value)
    df = pd.read_hdf(path, 'predictions')
    normalized = {int(column) for column in df.columns}
    missing = sorted(set(range(top_n)) - normalized)
    if missing:
        raise RuntimeError(
            f'{label} cache lacks rank slots 0..{top_n - 1}: missing={missing}'
        )
    dates = pd.DatetimeIndex(df.index.get_level_values('date'))
    print(
        f'[CACHE OK] {label}: rows={len(df)} dates={dates.nunique()} '
        f'range={dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d} '
        f'columns={sorted(normalized)}'
    )
print(f'[CACHE OK] requested target folds={target_folds}')
PY

echo "[PASS] prediction cache ready: $CACHE_ROOT historical=$HISTORICAL_CACHE_ROOT"
