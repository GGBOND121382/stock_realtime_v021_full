#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY="$PYTHON_BIN"

HISTORY_START_DATE="${HISTORY_START_DATE:-2020-01-02}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
TRADE_DATE="${TRADE_DATE:-today}"
TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
CH12_DIR="${CH12_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_5M_DIR="${RAW_5M_DIR:-$CH12_DIR/baostock_5m_cache}"
RAW_DAILY_DIR="${RAW_DAILY_DIR:-$CH12_DIR/baostock_raw_daily_cache}"
AS1455_DAILY_DIR="${AS1455_DAILY_DIR:-$CH12_DIR/as1455_daily_cache}"
MODEL_DATA="${MODEL_DATA:-$CH12_DIR/model_data_as1455.h5}"
FORWARD_MODEL_DIR="${FORWARD_MODEL_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"
FORWARD_MODEL_DATA="${FORWARD_MODEL_DATA:-$FORWARD_MODEL_DIR/model_data_as1455.h5}"

REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"
STATE_DIR="${STATE_DIR:-$REBUILD_ROOT/state}"
LOG_DIR="${LOG_DIR:-$REBUILD_ROOT/logs}"
PLOT_BASE="${PLOT_BASE:-saved_data/ashare_ml4t/ch17_as1455_backtest_plots}"

FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
EPOCHS="${EPOCHS:-20}"
BEST_N="${BEST_N:-5}"
SEED="${SEED:-42}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
MIN_INITIAL_FREE_GB="${MIN_INITIAL_FREE_GB:-20}"
MAX_DATA_PASSES="${MAX_DATA_PASSES:-5}"
BAOSTOCK_FETCH_RETRIES="${BAOSTOCK_FETCH_RETRIES:-3}"
BAOSTOCK_FETCH_SLEEP="${BAOSTOCK_FETCH_SLEEP:-0.05}"
BAOSTOCK_QUERY_TIMEOUT="${BAOSTOCK_QUERY_TIMEOUT:-180}"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$PLOT_BASE"

if [[ -s "$STATE_DIR/run_stamp.txt" ]]; then
  RUN_STAMP="$(tr -d '\r\n' < "$STATE_DIR/run_stamp.txt")"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  printf '%s\n' "$RUN_STAMP" > "$STATE_DIR/run_stamp.txt"
fi

CURRENT_STAGE="startup"
trap 'rc=$?; printf "[ERROR] stage=%s rc=%s line=%s command=%q\n" "$CURRENT_STAGE" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

info() { printf '[INFO] %s\n' "$*"; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
marker() { printf '%s/%s.done\n' "$STATE_DIR" "$1"; }
is_done() { [[ -s "$(marker "$1")" ]]; }
mark_done() {
  {
    printf 'stage=%s\n' "$1"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'run_stamp=%s\n' "$RUN_STAMP"
  } > "$(marker "$1")"
}
run_logged() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  info "running $name; log=$log"
  "$@" 2>&1 | tee -a "$log"
}

check_free_gb() {
  local required="$1" label="$2"
  "$PY" scripts/check_as1455_disk_space.py \
    --path saved_data/ashare_ml4t \
    --min-free-gb "$required" \
    --label "$label"
}

resolved_history_end() {
  if [[ -s "$STATE_DIR/history_end_date.txt" ]]; then
    tr -d '\r\n' < "$STATE_DIR/history_end_date.txt"
    return
  fi
  local value
  value=$(TZ="$TIMEZONE" "$PY" - "$TRADE_DATE" "$HISTORY_END_DATE" <<'PY'
import sys
from features.as1455_live_common import parse_trade_date, yyyymmdd_to_dash
from pipelines.as1455_update_history_to_prevday import resolve_history_end_date
trade_date = parse_trade_date(sys.argv[1])
history_end = resolve_history_end_date(trade_date, sys.argv[2], True)
print(yyyymmdd_to_dash(history_end))
PY
)
  printf '%s\n' "$value" > "$STATE_DIR/history_end_date.txt"
  printf '%s\n' "$value"
}

preflight() {
  CURRENT_STAGE="preflight"
  command -v "$PY" >/dev/null 2>&1 || fail "python executable not found: $PY"
  [[ -f "$UNIVERSE" ]] || fail "missing universe: $UNIVERSE"

  local required=(
    CH17_AS1455_DEVELOPMENT_OUTLINE.md
    README_AS1455_R1_R5_R21.md
    scripts/build_ashare_ch12_as1455_model_data.py
    scripts/run_as1455_target_search_all.sh
    scripts/run_as1455_r05_target_search_all.sh
    scripts/run_as1455_r21_target_search_all.sh
    scripts/run_as1455_target_natural_backtest.sh
    scripts/run_as1455_r05_natural_backtest.sh
    scripts/run_as1455_r21_natural_backtest.sh
    scripts/refresh_as1455_forward_model_data.sh
    scripts/run_as1455_fold0_forward_backtests.sh
    scripts/plot_as1455_default_ab_nav_curves.sh
    scripts/check_ch17_as1455_refactor.sh
    scripts/check_as1455_disk_space.py
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || fail "missing required existing project entry: $path"
  done

  check_free_gb "$MIN_INITIAL_FREE_GB" "from-scratch-initial"
  "$PY" -m compileall -q pipelines scripts utils code/backtest
  for path in \
    scripts/run_as1455_target_search_all.sh \
    scripts/run_as1455_r05_target_search_all.sh \
    scripts/run_as1455_r21_target_search_all.sh \
    scripts/run_as1455_target_natural_backtest.sh \
    scripts/run_as1455_r05_natural_backtest.sh \
    scripts/run_as1455_r21_natural_backtest.sh \
    scripts/refresh_as1455_forward_model_data.sh \
    scripts/run_as1455_fold0_forward_backtests.sh \
    scripts/plot_as1455_default_ab_nav_curves.sh; do
    bash -n "$path"
  done

  "$PY" - "$UNIVERSE" <<'PY'
import pandas as pd, sys
path = sys.argv[1]
df = pd.read_csv(path, dtype={'code': str}, encoding='utf-8-sig')
if len(df) != 1000:
    raise SystemExit(f'expected 1000 universe rows, got {len(df)}')
if df['code'].astype(str).str.zfill(6).nunique() != 1000:
    raise SystemExit('universe does not contain 1000 unique stock codes')
print('[OK] static universe contains 1000 unique symbols')
PY

  info "history_end_date=$(resolved_history_end)"
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
for name, root, pattern in [
    ('raw5', raw5, '*_5m_raw.csv'),
    ('rawd', rawd, '*_daily_raw.csv'),
    ('asd', asd, '*_as1455_daily.csv'),
]:
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
    print(f'[OK] {name}: files=1000')
PY
}

validate_model_data() {
  local path="${1:-$MODEL_DATA}"
  "$PY" - "$path" <<'PY'
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
dates = pd.DatetimeIndex(df.index.get_level_values('date'))
print({'path': path, 'rows': len(df), 'symbols': df.index.get_level_values('symbol').nunique(), 'date_min': str(dates.min().date()), 'date_max': str(dates.max().date()), 'columns': df.shape[1]})
PY
}

build_data() {
  CURRENT_STAGE="data_bootstrap"
  if is_done data_bootstrap; then
    info "data bootstrap already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "data-bootstrap"
  mkdir -p "$RAW_5M_DIR" "$RAW_DAILY_DIR" "$AS1455_DAILY_DIR"
  local end_date pass
  end_date="$(resolved_history_end)"

  # Canonical empty-cache batch entry introduced by the original AS1455 data
  # construction. First build/fill raw 5m + AS1455 daily caches without loading
  # the full feature panel. Re-running resumes from files already present.
  if ! is_done as1455_daily_cache; then
    for pass in $(seq 1 "$MAX_DATA_PASSES"); do
      run_logged "data_daily_cache_pass_${pass}" \
        "$PY" scripts/build_ashare_ch12_as1455_model_data.py \
          --universe "$UNIVERSE" \
          --out-dir "$CH12_DIR" \
          --bar-root "$RAW_5M_DIR" \
          --bar-glob '*_5m_raw.csv' \
          --baostock-5m-cache-dir "$RAW_5M_DIR" \
          --as1455-daily-cache-dir "$AS1455_DAILY_DIR" \
          --raw-daily-cache-dir "$RAW_DAILY_DIR" \
          --start-date "$HISTORY_START_DATE" \
          --end-date "$end_date" \
          --daily-cache-only \
          --baostock-fetch-retries "$BAOSTOCK_FETCH_RETRIES" \
          --baostock-fetch-sleep "$BAOSTOCK_FETCH_SLEEP" \
          --baostock-query-timeout "$BAOSTOCK_QUERY_TIMEOUT" \
          --qfq5m-audit-samples 0
      local n5 na
      n5=$(find "$RAW_5M_DIR" -maxdepth 1 -type f -name '*_5m_raw.csv' | wc -l)
      na=$(find "$AS1455_DAILY_DIR" -maxdepth 1 -type f -name '*_as1455_daily.csv' | wc -l)
      if [[ "$n5" -eq 1000 && "$na" -eq 1000 ]]; then
        mark_done as1455_daily_cache
        break
      fi
      [[ "$pass" -lt "$MAX_DATA_PASSES" ]] || fail "AS1455 daily cache incomplete after $MAX_DATA_PASSES passes: raw5=$n5 as1455=$na"
    done
  fi

  # Same existing batch builder now fills raw-daily factors and constructs the
  # canonical 34-column model_data. It explicitly supports rerunning to resume.
  for pass in $(seq 1 "$MAX_DATA_PASSES"); do
    run_logged "data_model_pass_${pass}" \
      "$PY" scripts/build_ashare_ch12_as1455_model_data.py \
        --universe "$UNIVERSE" \
        --out-dir "$CH12_DIR" \
        --bar-root "$RAW_5M_DIR" \
        --bar-glob '*_5m_raw.csv' \
        --baostock-5m-cache-dir "$RAW_5M_DIR" \
        --as1455-daily-cache-dir "$AS1455_DAILY_DIR" \
        --raw-daily-cache-dir "$RAW_DAILY_DIR" \
        --start-date "$HISTORY_START_DATE" \
        --end-date "$end_date" \
        --baostock-fetch-retries "$BAOSTOCK_FETCH_RETRIES" \
        --baostock-fetch-sleep "$BAOSTOCK_FETCH_SLEEP" \
        --baostock-query-timeout "$BAOSTOCK_QUERY_TIMEOUT" \
        --qfq5m-audit-samples 0 \
        --profile-memory && break
    [[ "$pass" -lt "$MAX_DATA_PASSES" ]] || fail "model-data batch build failed after $MAX_DATA_PASSES passes"
  done

  validate_cache_contracts
  validate_model_data "$MODEL_DATA"
  mark_done data_bootstrap
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
    root/'search_best_checkpoints.csv', root/'scores_summary.csv', root/'best_params.csv',
    root/'fold_report.json', root/'preprocess'/'scaler.pkl', root/'preprocess'/'feature_manifest.json'
]
assert all(p.is_file() and p.stat().st_size > 0 for p in required)
best = pd.read_csv(root/'search_best_checkpoints.csv')
assert len(best) >= best_n
assert any((root/'search_checkpoints').glob('*.keras'))
PY
}

training_wrapper() {
  case "$1" in
    r01_fwd) printf 'scripts/run_as1455_target_search_all.sh\n' ;;
    r05_fwd) printf 'scripts/run_as1455_r05_target_search_all.sh\n' ;;
    r21_fwd) printf 'scripts/run_as1455_r21_target_search_all.sh\n' ;;
    *) fail "unsupported target: $1" ;;
  esac
}

train_models() {
  CURRENT_STAGE="training"
  if is_done training; then
    info "training already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "training"
  local target preset fold dir backup missing wrapper
  for target in $TARGETS; do
    wrapper="$(training_wrapper "$target")"
    for preset in $FEATURE_PRESETS; do
      missing=""
      for fold in $(folds_for_target "$target"); do
        dir="$(training_dir "$preset" "$target" "$fold")"
        if training_complete "$dir"; then
          info "training complete; skip preset=$preset target=$target fold=$fold"
          continue
        fi
        if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
          backup="${dir}.incomplete.${RUN_STAMP}.$(date +%H%M%S)"
          info "preserve incomplete training directory: $dir -> $backup"
          mv "$dir" "$backup"
        fi
        missing+=" $fold"
      done
      missing="${missing# }"
      [[ -n "$missing" ]] || continue
      run_logged "train_${preset}_${target}" \
        env PYTHON_BIN="$PY" MODEL_DATA="$MODEL_DATA" FEATURE_PRESETS="$preset" TARGET_COL="$target" \
          FOLDS="$missing" EPOCHS="$EPOCHS" BEST_N="$BEST_N" SEED="$SEED" FORCE=0 RETRAIN_BEST=0 \
          bash "$wrapper"
      for fold in $missing; do
        dir="$(training_dir "$preset" "$target" "$fold")"
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
  printf 'saved_data/ashare_ml4t/ch17_as1455_target_backtest/%s_%s_reb%s_%s\n' "$1" "$2" "$(rebalance_for_target "$2")" "$RUN_STAMP"
}
historical_complete() {
  local root="$1"
  [[ -s "$root/materialized_best_run.json" ]] && \
  [[ -s "$root/01_close_auction_grid/02_summary/grid_summary_compact.csv" ]] && \
  [[ -s "$root/01_close_auction_grid/02_summary/leaderboard_by_sharpe.csv" ]]
}
historical_wrapper() {
  case "$1" in
    r01_fwd) printf 'scripts/run_as1455_target_natural_backtest.sh\n' ;;
    r05_fwd) printf 'scripts/run_as1455_r05_natural_backtest.sh\n' ;;
    r21_fwd) printf 'scripts/run_as1455_r21_natural_backtest.sh\n' ;;
    *) fail "unsupported target: $1" ;;
  esac
}

historical_backtests() {
  CURRENT_STAGE="historical_backtests"
  if is_done historical_backtests; then
    info "historical backtests already complete"
    return
  fi
  check_free_gb "$MIN_FREE_GB" "historical-backtests"
  local target preset root missing wrapper
  for target in $TARGETS; do
    missing=""
    for preset in $FEATURE_PRESETS; do
      root="$(historical_root "$preset" "$target")"
      if historical_complete "$root"; then
        info "historical complete; skip preset=$preset target=$target"
      else
        missing+=" $preset"
      fi
    done
    missing="${missing# }"
    [[ -n "$missing" ]] || continue
    wrapper="$(historical_wrapper "$target")"
    run_logged "historical_${target}" \
      env PYTHON_BIN="$PY" MODEL_DATA="$MODEL_DATA" RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" \
        FEATURE_PRESETS="$missing" TARGET_COL="$target" OUTPUT_MODE=summary MATERIALIZE_BEST=1 \
        MATERIALIZED_OUTPUT_MODE=compact RANK_METRIC=sharpe FORCE_GRID=1 \
        MIN_FREE_GB="$MIN_FREE_GB" RUN_STAMP="$RUN_STAMP" \
        bash "$wrapper"
    for preset in $missing; do
      root="$(historical_root "$preset" "$target")"
      historical_complete "$root" || fail "historical validation failed: $root"
    done
  done
  mark_done historical_backtests
}

refresh_forward_data() {
  CURRENT_STAGE="forward_data"
  if is_done forward_data; then
    info "forward model data already refreshed"
    return
  fi
  run_logged forward_data_refresh \
    env PYTHON_BIN="$PY" TRADE_DATE="$TRADE_DATE" HISTORY_END_DATE="$HISTORY_END_DATE" TIMEZONE="$TIMEZONE" \
      UNIVERSE="$UNIVERSE" SOURCE_DIR="$CH12_DIR" FORWARD_MODEL_DIR="$FORWARD_MODEL_DIR" \
      RAW_5M_CACHE_DIR="$RAW_5M_DIR" RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" \
      AS1455_DAILY_CACHE_DIR="$AS1455_DAILY_DIR" MIN_FREE_GB="$MIN_FREE_GB" \
      FORWARD_ARTIFACT_MODE=model_only FORWARD_REPORT_MODE=compact \
      bash scripts/refresh_as1455_forward_model_data.sh
  validate_model_data "$FORWARD_MODEL_DATA"
  mark_done forward_data
}

forward_root() {
  printf 'saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/%s_%s_reb%s_%s\n' "$1" "$2" "$(rebalance_for_target "$2")" "$RUN_STAMP"
}
forward_complete() {
  local root="$1"
  [[ -s "$root/00_predictions/fold0_forward_preds.h5" ]] || return 1
  [[ -s "$root/01_close_auction_grid/strict_oos_manifest.json" ]] || return 1
  "$PY" - "$root/01_close_auction_grid/strict_oos_manifest.json" <<'PY' >/dev/null 2>&1
import json, sys
obj = json.load(open(sys.argv[1], encoding='utf-8'))
assert obj['evaluation_mode'] == 'strict_oos'
assert obj['historical_trading_parameters_reused'] is True
assert obj['historical_rebalance_phase_reused'] is True
assert obj['generated_config_count'] == 1
assert obj['retained_config_count'] == 1
PY
}

forward_backtests() {
  CURRENT_STAGE="fold0_forward"
  if is_done fold0_forward; then
    info "fold0 forward already complete"
    return
  fi
  refresh_forward_data
  local target preset root
  for target in $TARGETS; do
    for preset in $FEATURE_PRESETS; do
      root="$(forward_root "$preset" "$target")"
      if forward_complete "$root"; then
        info "forward complete; skip preset=$preset target=$target"
        continue
      fi
      run_logged "forward_${preset}_${target}" \
        env PYTHON_BIN="$PY" REFRESH_DATA=0 MODEL_DATA="$FORWARD_MODEL_DATA" SOURCE_DIR="$CH12_DIR" \
          RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" TARGETS="$target" FEATURE_PRESETS="$preset" \
          MODEL_SELECTION_MODE=strict_oos SELECTION_RANK_METRIC=sharpe OUTPUT_MODE=compact \
          FORCE_GRID=1 MIN_FREE_GB="$MIN_FREE_GB" RUN_STAMP="$RUN_STAMP" \
          bash scripts/run_as1455_fold0_forward_backtests.sh
      forward_complete "$root" || fail "forward validation failed: $root"
    done
  done
  mark_done fold0_forward
}

plot_results() {
  CURRENT_STAGE="plots"
  if is_done plots; then
    info "plots already complete"
    return
  fi
  local target a b out labels
  for target in $TARGETS; do
    a="$(forward_root rotation_onehot "$target")"
    b="$(forward_root rotation_addon_onehot "$target")"
    forward_complete "$a" || fail "cannot plot incomplete result: $a"
    forward_complete "$b" || fail "cannot plot incomplete result: $b"
    out="$PLOT_BASE/${target}_${RUN_STAMP}"
    case "$target" in
      r01_fwd) labels='r1-A-fold0-forward,r1-B-fold0-forward' ;;
      r05_fwd) labels='r5-A-fold0-forward,r5-B-fold0-forward' ;;
      r21_fwd) labels='r21-A-fold0-forward,r21-B-fold0-forward' ;;
    esac
    run_logged "plot_${target}" \
      env BACKTEST_ROOTS="$a,$b" LABELS="$labels" RANK_METRIC=sharpe OUT_DIR="$out" PYTHON_BIN="$PY" \
        bash scripts/plot_as1455_default_ab_nav_curves.sh
    find "$out" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q . || fail "plot PNG missing: $out"
  done
  mark_done plots
}

final_audit() {
  CURRENT_STAGE="final_audit"
  validate_cache_contracts
  validate_model_data "$MODEL_DATA"
  validate_model_data "$FORWARD_MODEL_DATA"
  "$PY" - "$REBUILD_ROOT" "$RUN_STAMP" "$FEATURE_PRESETS" "$TARGETS" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json, sys
root = Path(sys.argv[1]); stamp = sys.argv[2]; presets = sys.argv[3].split(); targets = sys.argv[4].split()
reb = {'r01_fwd': 1, 'r05_fwd': 5, 'r21_fwd': 21}
training=[]
for target in targets:
    folds = range(7) if target in {'r01_fwd','r05_fwd'} else range(6)
    for preset in presets:
        for fold in folds:
            if target == 'r01_fwd':
                p = Path('saved_data/ashare_ml4t') / (f'ch17_as1455_sector_rotation_onehot_fold{fold}_search' if preset == 'rotation_onehot' else f'ch17_as1455_full_rotation_plus_first_batch_compact_fold{fold}_search')
            else:
                p = Path('saved_data/ashare_ml4t/ch17_as1455_target_search')/preset/target/f'fold{fold}_search'
            training.append({'preset':preset,'target':target,'fold':fold,'path':str(p),'ok':(p/'search_best_checkpoints.csv').is_file() and (p/'preprocess/scaler.pkl').is_file()})
historical=[]; forward=[]; plots=[]
for target in targets:
    for preset in presets:
        h=Path('saved_data/ashare_ml4t/ch17_as1455_target_backtest')/f'{preset}_{target}_reb{reb[target]}_{stamp}'
        f=Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')/f'{preset}_{target}_reb{reb[target]}_{stamp}'
        historical.append({'preset':preset,'target':target,'path':str(h),'ok':(h/'materialized_best_run.json').is_file()})
        forward.append({'preset':preset,'target':target,'path':str(f),'ok':(f/'01_close_auction_grid/strict_oos_manifest.json').is_file()})
    p=Path('saved_data/ashare_ml4t/ch17_as1455_backtest_plots')/f'{target}_{stamp}'
    plots.append({'target':target,'path':str(p),'ok':any(p.glob('*.png'))})
report={
    'created_at_utc':datetime.now(timezone.utc).isoformat(), 'run_stamp':stamp,
    'training_expected':len(training),'training_ok':sum(x['ok'] for x in training),
    'historical_expected':len(historical),'historical_ok':sum(x['ok'] for x in historical),
    'forward_expected':len(forward),'forward_ok':sum(x['ok'] for x in forward),
    'plots_expected':len(plots),'plots_ok':sum(x['ok'] for x in plots),
    'training':training,'historical':historical,'forward':forward,'plots':plots,
}
report['all_ok']=(report['training_ok']==report['training_expected'] and report['historical_ok']==report['historical_expected'] and report['forward_ok']==report['forward_expected'] and report['plots_ok']==report['plots_expected'])
root.mkdir(parents=True, exist_ok=True)
path=root/'final_report.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['run_stamp','training_expected','training_ok','historical_expected','historical_ok','forward_expected','forward_ok','plots_expected','plots_ok','all_ok']},ensure_ascii=False,indent=2))
if not report['all_ok']:
    raise SystemExit(f'final audit failed; see {path}')
print(f'[PASS] full Ch17 AS1455 rebuild completed; report={path}')
PY
  mark_done final_audit
}

status() {
  printf 'run_stamp=%s\n' "$RUN_STAMP"
  local stage
  for stage in preflight as1455_daily_cache data_bootstrap selfcheck training historical_backtests forward_data fold0_forward plots final_audit; do
    if is_done "$stage"; then printf '[DONE] %s\n' "$stage"; else printf '[PENDING] %s\n' "$stage"; fi
  done
  [[ -s "$STATE_DIR/history_end_date.txt" ]] && printf 'history_end_date=%s\n' "$(cat "$STATE_DIR/history_end_date.txt")"
  [[ -s "$REBUILD_ROOT/final_report.json" ]] && cat "$REBUILD_ROOT/final_report.json"
}

run_all() {
  if ! is_done preflight; then preflight; mark_done preflight; fi
  build_data
  selfcheck
  train_models
  historical_backtests
  forward_backtests
  plot_results
  final_audit
}

case "$MODE" in
  all) run_all ;;
  status) status ;;
  preflight) preflight; mark_done preflight ;;
  data|history|model_data) build_data ;;
  selfcheck) selfcheck ;;
  training) train_models ;;
  historical) historical_backtests ;;
  forward_data) refresh_forward_data ;;
  forward) forward_backtests ;;
  plots) plot_results ;;
  audit) final_audit ;;
  *)
    cat >&2 <<'EOF'
Usage:
  bash scripts/rebuild_ch17_as1455_from_scratch.sh all|status
  bash scripts/rebuild_ch17_as1455_from_scratch.sh preflight|data|selfcheck|training|historical|forward_data|forward|plots|audit
EOF
    exit 2
    ;;
esac
