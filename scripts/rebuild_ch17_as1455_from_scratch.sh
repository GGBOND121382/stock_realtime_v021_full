#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-all}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_as1455}"
INSTALL_MISSING_DEPS="${INSTALL_MISSING_DEPS:-1}"
HISTORY_START_DATE="${HISTORY_START_DATE:-2020-01-02}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
TRADE_DATE="${TRADE_DATE:-today}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
CH12_DIR="${CH12_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_5M_DIR="${RAW_5M_DIR:-$CH12_DIR/baostock_5m_cache}"
RAW_DAILY_DIR="${RAW_DAILY_DIR:-$CH12_DIR/baostock_raw_daily_cache}"
AS1455_DAILY_DIR="${AS1455_DAILY_DIR:-$CH12_DIR/as1455_daily_cache}"
MODEL_DATA="${MODEL_DATA:-$CH12_DIR/model_data_as1455.h5}"
REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"
STATE_DIR="${STATE_DIR:-$REBUILD_ROOT/state}"
LOG_DIR="${LOG_DIR:-$REBUILD_ROOT/logs}"
HISTORY_REPORT_ROOT="${HISTORY_REPORT_ROOT:-$REBUILD_ROOT/history_reports}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
EPOCHS="${EPOCHS:-20}"
BEST_N="${BEST_N:-5}"
SEED="${SEED:-42}"
MAX_HISTORY_PASSES="${MAX_HISTORY_PASSES:-5}"
HISTORY_SLEEP_SECONDS="${HISTORY_SLEEP_SECONDS:-0.05}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
MIN_INITIAL_FREE_GB="${MIN_INITIAL_FREE_GB:-20}"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$HISTORY_REPORT_ROOT"

if [[ -s "$STATE_DIR/run_stamp.txt" ]]; then
  RUN_STAMP="$(tr -d '\r\n' < "$STATE_DIR/run_stamp.txt")"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  printf '%s\n' "$RUN_STAMP" > "$STATE_DIR/run_stamp.txt"
fi

CURRENT_STAGE="startup"
trap 'rc=$?; printf "[ERROR] stage=%s rc=%s line=%s command=%q\n" "$CURRENT_STAGE" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

info() {
  printf '[INFO] %s\n' "$*"
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

marker() {
  printf '%s/%s.done\n' "$STATE_DIR" "$1"
}

is_done() {
  [[ -s "$(marker "$1")" ]]
}

mark_done() {
  {
    printf 'stage=%s\n' "$1"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'run_stamp=%s\n' "$RUN_STAMP"
  } > "$(marker "$1")"
}

run_logged() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  info "running $name; log=$log"
  "$@" 2>&1 | tee -a "$log"
}

check_free_gb() {
  local required="$1"
  local label="$2"
  "$PY" scripts/check_as1455_disk_space.py \
    --path saved_data/ashare_ml4t \
    --min-free-gb "$required" \
    --label "$label"
}

resolve_python() {
  CURRENT_STAGE="environment"
  command -v "$BASE_PYTHON" >/dev/null 2>&1 || fail "python not found: $BASE_PYTHON"

  if [[ "$USE_VENV" == "1" ]]; then
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
      "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
    fi
    PY="$VENV_DIR/bin/python"
  else
    PY="$BASE_PYTHON"
  fi

  if ! "$PY" - <<'PY'
mods = ['numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'baostock', 'tables', 'talib', 'tensorflow', 'matplotlib']
for name in mods:
    __import__(name)
PY
  then
    [[ "$INSTALL_MISSING_DEPS" == "1" ]] || fail "python dependencies are incomplete and INSTALL_MISSING_DEPS=0"
    "$PY" -m pip install --upgrade pip setuptools wheel
    "$PY" -m pip install numpy pandas scipy scikit-learn joblib baostock tables TA-Lib tensorflow matplotlib psutil
  fi

  "$PY" - <<'PY'
import baostock, joblib, matplotlib, numpy, pandas, scipy, sklearn, tables, talib, tensorflow
print('[OK] python dependency imports passed')
PY
}

preflight() {
  CURRENT_STAGE="preflight"
  [[ -f "$UNIVERSE" ]] || fail "missing universe: $UNIVERSE"
  for path in \
    pipelines/as1455_update_history_to_prevday_fast_v4.py \
    scripts/build_ashare_ch12_as1455_model_data.py \
    scripts/run_as1455_target_fold_param_search.py \
    scripts/run_as1455_target_natural_backtest.sh \
    scripts/run_as1455_fold0_forward_backtests.sh \
    scripts/check_ch17_as1455_refactor.sh; do
    [[ -f "$path" ]] || fail "missing required file: $path"
  done

  resolve_python
  check_free_gb "$MIN_INITIAL_FREE_GB" "from-scratch-initial"
  "$PY" -m compileall -q pipelines scripts utils code/backtest
  bash -n scripts/run_as1455_target_natural_backtest.sh
  bash -n scripts/run_as1455_fold0_forward_backtests.sh

  "$PY" - "$UNIVERSE" <<'PY'
import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p, dtype={'code': str}, encoding='utf-8-sig')
if len(df) != 1000:
    raise SystemExit(f'expected 1000 universe rows, got {len(df)}')
if df['code'].astype(str).str.zfill(6).nunique() != 1000:
    raise SystemExit('universe code count is not 1000 unique symbols')
print('[OK] static universe has 1000 unique symbols')
PY
}

history_report_path() {
  "$PY" - "$TRADE_DATE" "$HISTORY_REPORT_ROOT" <<'PY'
from datetime import datetime
from pathlib import Path
import sys
trade_date, root = sys.argv[1:]
if trade_date.lower() == 'today':
    value = datetime.now().strftime('%Y%m%d')
else:
    value = trade_date.replace('-', '')
print(Path(root) / value / '00_history_update_report.json')
PY
}

history_report_ok() {
  local report
  report="$(history_report_path)"
  [[ -s "$report" ]] || return 1
  "$PY" - "$report" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], encoding='utf-8'))
assert int(obj.get('n_symbols', 0)) == 1000, obj
assert int(obj.get('errors', -1)) == 0, obj
print('[OK] history report:', sys.argv[1])
PY
}

validate_cache_contracts() {
  "$PY" - "$RAW_5M_DIR" "$RAW_DAILY_DIR" "$AS1455_DAILY_DIR" <<'PY'
from pathlib import Path
import csv, sys

raw5, rawd, asd = map(Path, sys.argv[1:])
expected = {
    'raw5': ['symbol','trade_date','datetime','open','high','low','close','volume','amount','source','bar_freq','bar_label'],
    'rawd': ['date','code','open','high','low','close','preclose','volume','amount','adjustflag','pctChg','tradestatus','isST','symbol','turn'],
    'asd': ['symbol','date','raw_open_as1455','raw_high_as1455','raw_low_as1455','raw_close_as1455','raw_volume_as1455','raw_amount_as1455','max_datetime_used','has_14_55_bar','last_bar_time','used_after_cutoff','source_path'],
}
groups = [
    ('raw5', raw5, '*_5m_raw.csv'),
    ('rawd', rawd, '*_daily_raw.csv'),
    ('asd', asd, '*_as1455_daily.csv'),
]
for name, root, pattern in groups:
    files = sorted(root.glob(pattern))
    if len(files) != 1000:
        raise SystemExit(f'{name}: expected 1000 files, got {len(files)} in {root}')
    variants = {}
    for path in files:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            header = tuple(next(csv.reader(f)))
        variants[header] = variants.get(header, 0) + 1
    if len(variants) != 1:
        raise SystemExit(f'{name}: multiple header variants: {variants}')
    header = list(next(iter(variants)))
    if header != expected[name]:
        raise SystemExit(f'{name}: unexpected header: {header}')
    print(f'[OK] {name}: files=1000 header={header}')
PY
}

build_history() {
  CURRENT_STAGE="history"
  if is_done history; then
    info "history already complete"
    return
  fi

  mkdir -p "$RAW_5M_DIR" "$RAW_DAILY_DIR" "$AS1455_DAILY_DIR"
  local pass
  for pass in $(seq 1 "$MAX_HISTORY_PASSES"); do
    info "history pass $pass/$MAX_HISTORY_PASSES"
    run_logged "history_pass_${pass}" \
      "$PY" pipelines/as1455_update_history_to_prevday_fast_v4.py \
        --trade-date "$TRADE_DATE" \
        --history-end-date "$HISTORY_END_DATE" \
        --history-start-date "$HISTORY_START_DATE" \
        --universe "$UNIVERSE" \
        --raw-5m-cache-dir "$RAW_5M_DIR" \
        --raw-daily-cache-dir "$RAW_DAILY_DIR" \
        --as1455-daily-cache-dir "$AS1455_DAILY_DIR" \
        --out-root "$HISTORY_REPORT_ROOT" \
        --sleep-seconds "$HISTORY_SLEEP_SECONDS"
    if history_report_ok; then
      break
    fi
    [[ "$pass" -lt "$MAX_HISTORY_PASSES" ]] || fail "history still has errors after $MAX_HISTORY_PASSES passes"
  done

  validate_cache_contracts
  mark_done history
}

validate_model_data() {
  "$PY" - "$MODEL_DATA" <<'PY'
import pandas as pd, sys
path = sys.argv[1]
expected = [
    'dollar_vol','dollar_vol_rank','rsi','bb_high','bb_low','NATR','ATR','PPO','MACD','sector',
    'r01','r05','r10','r21','r42','r63','r01dec','r05dec','r10dec','r21dec','r42dec','r63dec',
    'r01q_sector','r05q_sector','r10q_sector','r21q_sector','r42q_sector','r63q_sector',
    'r01_fwd','r05_fwd','r21_fwd','year','month','weekday'
]
df = pd.read_hdf(path, 'model_data')
if list(df.index.names) != ['symbol', 'date']:
    raise SystemExit(f'bad index names: {df.index.names}')
if list(df.columns) != expected:
    raise SystemExit(f'bad model columns: {list(df.columns)}')
if df.empty:
    raise SystemExit('model_data is empty')
if df.index.get_level_values('symbol').nunique() < 900:
    raise SystemExit('too few symbols in model_data')
print({
    'rows': int(len(df)),
    'symbols': int(df.index.get_level_values('symbol').nunique()),
    'date_min': str(df.index.get_level_values('date').min().date()),
    'date_max': str(df.index.get_level_values('date').max().date()),
    'columns': int(df.shape[1]),
})
PY
}

build_model_data() {
  CURRENT_STAGE="model_data"
  if is_done model_data; then
    info "model_data already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "model-data-build"
  run_logged model_data_build \
    "$PY" scripts/build_ashare_ch12_as1455_model_data.py \
      --universe "$UNIVERSE" \
      --out-dir "$CH12_DIR" \
      --bar-root "$RAW_5M_DIR" \
      --bar-glob '*_5m_raw.csv' \
      --baostock-5m-cache-dir "$RAW_5M_DIR" \
      --as1455-daily-cache-dir "$AS1455_DAILY_DIR" \
      --raw-daily-cache-dir "$RAW_DAILY_DIR" \
      --start-date "$HISTORY_START_DATE" \
      --no-fetch-missing-baostock \
      --no-fetch-missing-raw-daily \
      --qfq5m-audit-samples 0 \
      --profile-memory
  validate_model_data
  mark_done model_data
}

selfcheck() {
  CURRENT_STAGE="selfcheck"
  if is_done selfcheck; then
    info "selfcheck already complete"
    return
  fi
  run_logged ch17_selfcheck env PYTHON_BIN="$PY" bash scripts/check_ch17_as1455_refactor.sh
  mark_done selfcheck
}

folds_for_target() {
  case "$1" in
    r01_fwd|r05_fwd) printf '0 1 2 3 4 5 6\n' ;;
    r21_fwd) printf '0 1 2 3 4 5\n' ;;
    *) fail "unsupported target: $1" ;;
  esac
}

training_dir() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import sys
from utils.as1455_ch17_common import default_fold_dir_template, fold_dir_from_template
preset, target, fold = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(fold_dir_from_template(default_fold_dir_template(preset, target), fold))
PY
}

training_complete() {
  local dir="$1"
  "$PY" - "$dir" "$BEST_N" <<'PY' >/dev/null 2>&1
from pathlib import Path
import pandas as pd, sys
root = Path(sys.argv[1]); best_n = int(sys.argv[2])
required = [
    root / 'search_best_checkpoints.csv',
    root / 'scores_summary.csv',
    root / 'best_params.csv',
    root / 'fold_report.json',
    root / 'preprocess' / 'scaler.pkl',
    root / 'preprocess' / 'feature_manifest.json',
]
assert all(p.is_file() and p.stat().st_size > 0 for p in required)
best = pd.read_csv(root / 'search_best_checkpoints.csv')
assert len(best) >= best_n
assert int(best.get('checkpoint_saved', pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) >= 1
assert any((root / 'search_checkpoints').glob('*.keras'))
PY
}

train_models() {
  CURRENT_STAGE="training"
  if is_done training; then
    info "training already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "training"

  local target preset fold dir backup
  for target in $TARGETS; do
    for preset in $FEATURE_PRESETS; do
      for fold in $(folds_for_target "$target"); do
        dir="$(training_dir "$preset" "$target" "$fold")"
        if training_complete "$dir"; then
          info "training complete; skip preset=$preset target=$target fold=$fold"
          continue
        fi
        if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
          backup="${dir}.incomplete.${RUN_STAMP}.$(date +%H%M%S)"
          info "preserving incomplete training directory: $dir -> $backup"
          mv "$dir" "$backup"
        fi
        run_logged "train_${preset}_${target}_fold${fold}" \
          "$PY" scripts/run_as1455_target_fold_param_search.py \
            --feature-preset "$preset" \
            --target-col "$target" \
            --model-data "$MODEL_DATA" \
            --fold-index "$fold" \
            --sector-encoding onehot \
            --dropna-mode target_only \
            --epochs "$EPOCHS" \
            --best-n "$BEST_N" \
            --seed "$SEED"
        training_complete "$dir" || fail "training validation failed: $dir"
      done
    done
  done
  mark_done training
}

rebalance_for_target() {
  case "$1" in
    r01_fwd) printf '1\n' ;;
    r05_fwd) printf '5\n' ;;
    r21_fwd) printf '21\n' ;;
    *) fail "unsupported target: $1" ;;
  esac
}

historical_root() {
  local preset="$1" target="$2" rebalance
  rebalance="$(rebalance_for_target "$target")"
  printf 'saved_data/ashare_ml4t/ch17_as1455_target_backtest/%s_%s_reb%s_%s\n' "$preset" "$target" "$rebalance" "$RUN_STAMP"
}

historical_complete() {
  local root="$1"
  [[ -s "$root/materialized_best_run.json" ]] && \
  [[ -s "$root/01_close_auction_grid/02_summary/grid_summary_compact.csv" ]] && \
  [[ -s "$root/01_close_auction_grid/02_summary/leaderboard_by_sharpe.csv" ]]
}

historical_backtests() {
  CURRENT_STAGE="historical_backtests"
  if is_done historical_backtests; then
    info "historical backtests already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "historical-backtests"

  local target preset root
  for target in $TARGETS; do
    for preset in $FEATURE_PRESETS; do
      root="$(historical_root "$preset" "$target")"
      if historical_complete "$root"; then
        info "historical backtest complete; skip preset=$preset target=$target"
        continue
      fi
      run_logged "historical_${preset}_${target}" \
        env PYTHON_BIN="$PY" \
          MODEL_DATA="$MODEL_DATA" \
          RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" \
          FEATURE_PRESETS="$preset" \
          TARGET_COL="$target" \
          OUTPUT_MODE=summary \
          MATERIALIZE_BEST=1 \
          MATERIALIZED_OUTPUT_MODE=compact \
          RANK_METRIC=sharpe \
          FORCE_GRID=1 \
          MIN_FREE_GB="$MIN_FREE_GB" \
          RUN_STAMP="$RUN_STAMP" \
          bash scripts/run_as1455_target_natural_backtest.sh
      historical_complete "$root" || fail "historical backtest validation failed: $root"
    done
  done
  mark_done historical_backtests
}

forward_root() {
  local preset="$1" target="$2" rebalance
  rebalance="$(rebalance_for_target "$target")"
  printf 'saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/%s_%s_reb%s_%s\n' "$preset" "$target" "$rebalance" "$RUN_STAMP"
}

forward_complete() {
  local root="$1"
  [[ -s "$root/strict_oos_manifest.json" ]] && \
  [[ -s "$root/00_predictions/fold0_forward_preds.h5" ]]
}

forward_backtests() {
  CURRENT_STAGE="fold0_forward"
  if is_done fold0_forward; then
    info "fold0 forward already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "fold0-forward"

  run_logged fold0_forward \
    env PYTHON_BIN="$PY" \
      REFRESH_DATA=0 \
      MODEL_DATA="$MODEL_DATA" \
      SOURCE_DIR="$CH12_DIR" \
      RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" \
      TARGETS="$TARGETS" \
      FEATURE_PRESETS="$FEATURE_PRESETS" \
      MODEL_SELECTION_MODE=strict_oos \
      SELECTION_RANK_METRIC=sharpe \
      OUTPUT_MODE=compact \
      FORCE_GRID=1 \
      MIN_FREE_GB="$MIN_FREE_GB" \
      RUN_STAMP="$RUN_STAMP" \
      bash scripts/run_as1455_fold0_forward_backtests.sh

  local target preset root
  for target in $TARGETS; do
    for preset in $FEATURE_PRESETS; do
      root="$(forward_root "$preset" "$target")"
      forward_complete "$root" || fail "forward validation failed: $root"
    done
  done
  mark_done fold0_forward
}

final_audit() {
  CURRENT_STAGE="final_audit"
  validate_cache_contracts
  validate_model_data
  "$PY" - "$REBUILD_ROOT" "$RUN_STAMP" "$FEATURE_PRESETS" "$TARGETS" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json, sys

root = Path(sys.argv[1])
stamp = sys.argv[2]
presets = sys.argv[3].split()
targets = sys.argv[4].split()
reb = {'r01_fwd': 1, 'r05_fwd': 5, 'r21_fwd': 21}
training = []
for target in targets:
    folds = range(7) if target in {'r01_fwd', 'r05_fwd'} else range(6)
    for preset in presets:
        for fold in folds:
            if target == 'r01_fwd':
                p = Path('saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold%d_search' % fold) if preset == 'rotation_onehot' else Path('saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold%d_search' % fold)
            else:
                p = Path('saved_data/ashare_ml4t/ch17_as1455_target_search') / preset / target / f'fold{fold}_search'
            training.append({'preset': preset, 'target': target, 'fold': fold, 'path': str(p), 'ok': (p/'search_best_checkpoints.csv').is_file()})
historical = []
forward = []
for target in targets:
    for preset in presets:
        h = Path('saved_data/ashare_ml4t/ch17_as1455_target_backtest') / f'{preset}_{target}_reb{reb[target]}_{stamp}'
        f = Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest') / f'{preset}_{target}_reb{reb[target]}_{stamp}'
        historical.append({'preset': preset, 'target': target, 'path': str(h), 'ok': (h/'materialized_best_run.json').is_file()})
        forward.append({'preset': preset, 'target': target, 'path': str(f), 'ok': (f/'strict_oos_manifest.json').is_file()})
report = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'run_stamp': stamp,
    'training_expected': len(training),
    'training_ok': sum(x['ok'] for x in training),
    'historical_expected': len(historical),
    'historical_ok': sum(x['ok'] for x in historical),
    'forward_expected': len(forward),
    'forward_ok': sum(x['ok'] for x in forward),
    'training': training,
    'historical': historical,
    'forward': forward,
}
report['all_ok'] = (
    report['training_ok'] == report['training_expected'] and
    report['historical_ok'] == report['historical_expected'] and
    report['forward_ok'] == report['forward_expected']
)
root.mkdir(parents=True, exist_ok=True)
path = root / 'final_report.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ['run_stamp','training_expected','training_ok','historical_expected','historical_ok','forward_expected','forward_ok','all_ok']}, ensure_ascii=False, indent=2))
if not report['all_ok']:
    raise SystemExit(f'final audit failed; see {path}')
print(f'[PASS] full Ch17 AS1455 rebuild completed; report={path}')
PY
  mark_done final_audit
}

status() {
  printf 'run_stamp=%s\n' "$RUN_STAMP"
  printf 'python=%s\n' "${PY:-<not-resolved>}"
  local stage
  for stage in preflight history model_data selfcheck training historical_backtests fold0_forward final_audit; do
    if is_done "$stage"; then
      printf '[DONE] %s\n' "$stage"
    else
      printf '[PENDING] %s\n' "$stage"
    fi
  done
  [[ -s "$REBUILD_ROOT/final_report.json" ]] && cat "$REBUILD_ROOT/final_report.json"
}

run_all() {
  if ! is_done preflight; then
    preflight
    mark_done preflight
  else
    resolve_python
  fi
  build_history
  build_model_data
  selfcheck
  train_models
  historical_backtests
  forward_backtests
  final_audit
}

case "$MODE" in
  all) run_all ;;
  status) resolve_python; status ;;
  preflight) preflight; mark_done preflight ;;
  history) resolve_python; build_history ;;
  model_data) resolve_python; build_model_data ;;
  selfcheck) resolve_python; selfcheck ;;
  training) resolve_python; train_models ;;
  historical) resolve_python; historical_backtests ;;
  forward) resolve_python; forward_backtests ;;
  audit) resolve_python; final_audit ;;
  *)
    cat >&2 <<'EOF'
Usage:
  bash scripts/rebuild_ch17_as1455_from_scratch.sh all
  bash scripts/rebuild_ch17_as1455_from_scratch.sh status
  bash scripts/rebuild_ch17_as1455_from_scratch.sh preflight|history|model_data|selfcheck|training|historical|forward|audit
EOF
    exit 2
    ;;
esac
