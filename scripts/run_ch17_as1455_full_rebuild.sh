#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"; BASE_PYTHON="${BASE_PYTHON:-python3}"; VENV_DIR="${VENV_DIR:-.venv_as1455}"
HISTORY_START_DATE="${HISTORY_START_DATE:-2020-01-02}"; HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"; TRADE_DATE="${TRADE_DATE:-today}"; TIMEZONE="${TIMEZONE:-Asia/Shanghai}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
CH12_DIR="${CH12_DIR:-saved_data/ashare_ml4t/ch12_as1455}"; RAW_5M_DIR="${RAW_5M_DIR:-$CH12_DIR/baostock_5m_cache}"; RAW_DAILY_DIR="${RAW_DAILY_DIR:-$CH12_DIR/baostock_raw_daily_cache}"; AS1455_DAILY_DIR="${AS1455_DAILY_DIR:-$CH12_DIR/as1455_daily_cache}"; MODEL_DATA="$CH12_DIR/model_data_as1455.h5"
FORWARD_DIR="${FORWARD_DIR:-saved_data/ashare_ml4t/ch12_as1455_forward_latest}"; FORWARD_DATA="$FORWARD_DIR/model_data_as1455.h5"
REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"; STATE_DIR="$REBUILD_ROOT/state_v2"; LOG_DIR="$REBUILD_ROOT/logs_v2"; SEARCH_LOG_DIR="$REBUILD_ROOT/search_logs_v2"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"; TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
EPOCHS="${EPOCHS:-20}"; BEST_N="${BEST_N:-5}"; SEED="${SEED:-42}"; MIN_FREE_GB="${MIN_FREE_GB:-5}"; MIN_INITIAL_FREE_GB="${MIN_INITIAL_FREE_GB:-20}"
BOOTSTRAP_BATCH_SIZE="${BOOTSTRAP_BATCH_SIZE:-250}"; MAX_BOOTSTRAP_PASSES="${MAX_BOOTSTRAP_PASSES:-8}"; BAOSTOCK_FETCH_SLEEP="${BAOSTOCK_FETCH_SLEEP:-0.05}"
CPU_THREADS="${CPU_THREADS:-2}"; export OMP_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS" TF_NUM_INTRAOP_THREADS="$CPU_THREADS" TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2
mkdir -p "$STATE_DIR" "$LOG_DIR" "$SEARCH_LOG_DIR"; [[ -s "$STATE_DIR/run_stamp.txt" ]] || date +%Y%m%d_%H%M%S > "$STATE_DIR/run_stamp.txt"; RUN_STAMP="$(tr -d '\r\n' < "$STATE_DIR/run_stamp.txt")"; PLOT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/full_rebuild_$RUN_STAMP"
CURRENT_STAGE=startup; trap 'rc=$?; echo "[ERROR] stage=$CURRENT_STAGE rc=$rc line=$LINENO command=$BASH_COMMAND" >&2; exit $rc' ERR

info(){ echo "[INFO] $*"; }; fail(){ echo "[ERROR] $*" >&2; exit 1; }; mark(){ echo "$STATE_DIR/$1.done"; }; donep(){ [[ -s "$(mark "$1")" ]]; }; finish(){ printf 'stage=%s\ncompleted_at=%s\n' "$1" "$(date -Is)" > "$(mark "$1")"; }
run(){ local n="$1"; shift; info "$n; log=$LOG_DIR/$n.log"; "$@" 2>&1 | tee -a "$LOG_DIR/$n.log"; }
count(){ find "$1" -maxdepth 1 -type f -name "$2" 2>/dev/null | wc -l; }

setup_python(){
  command -v "$BASE_PYTHON" >/dev/null || fail "missing $BASE_PYTHON"; [[ -x "$VENV_DIR/bin/python" ]] || "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"; REAL_PYTHON="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")/bin/python"
  "$REAL_PYTHON" - <<'PY' || "$REAL_PYTHON" -m pip install -r requirements.txt numpy pandas scipy matplotlib psutil
for x in ['numpy','pandas','scipy','sklearn','joblib','baostock','tables','talib','tensorflow','matplotlib','psutil']: __import__(x)
print('[OK] dependencies')
PY
  bash -n scripts/as1455_python_memory_guard.sh; PY="$STATE_DIR/python_with_memory_guard"; install -m 700 scripts/as1455_python_memory_guard.sh "$PY"; export AS1455_REAL_PYTHON="$REAL_PYTHON"
}
disk(){ "$PY" scripts/check_as1455_disk_space.py --path saved_data/ashare_ml4t --min-free-gb "$1" --label "$2"; }
end_date(){
  [[ -s "$STATE_DIR/history_end.txt" ]] || "$PY" - "$TRADE_DATE" "$HISTORY_END_DATE" <<'PY' > "$STATE_DIR/history_end.txt"
import sys
from features.as1455_live_common import parse_trade_date,yyyymmdd_to_dash
from pipelines.as1455_update_history_to_prevday import resolve_history_end_date
print(yyyymmdd_to_dash(resolve_history_end_date(parse_trade_date(sys.argv[1]),sys.argv[2],True)))
PY
  tr -d '\r\n' < "$STATE_DIR/history_end.txt"
}

preflight(){
  CURRENT_STAGE=preflight
  for f in CH17_AS1455_DEVELOPMENT_OUTLINE.md README_AS1455_R1_R5_R21.md requirements.txt scripts/as1455_python_memory_guard.sh scripts/build_ashare_ch12_as1455_model_data.py scripts/build_ashare_ch12_as1455_lowmem.sh scripts/run_as1455_live_data_feature_pipeline.sh scripts/run_as1455_target_search_all.sh scripts/run_as1455_r05_target_search_all.sh scripts/run_as1455_r21_target_search_all.sh scripts/run_as1455_target_natural_backtest.sh scripts/run_as1455_r05_natural_backtest.sh scripts/run_as1455_r21_natural_backtest.sh scripts/refresh_as1455_forward_model_data.sh scripts/run_as1455_fold0_forward_backtests.sh scripts/plot_as1455_default_ab_nav_curves.sh scripts/check_ch17_as1455_refactor.sh "$UNIVERSE"; do [[ -f "$f" ]] || fail "missing $f"; done
  bash -n scripts/run_ch17_as1455_full_rebuild.sh; "$PY" - "$UNIVERSE" <<'PY'
import pandas as pd,sys
d=pd.read_csv(sys.argv[1],dtype={'code':str},encoding='utf-8-sig'); assert len(d)==1000 and d.code.astype(str).str.zfill(6).nunique()==1000
print('[OK] universe=1000')
PY
  disk "$MIN_INITIAL_FREE_GB" from-scratch; info "history_end=$(end_date)"
}

validate_model(){ "$PY" - "$1" <<'PY'
import pandas as pd,sys
cols=['dollar_vol','dollar_vol_rank','rsi','bb_high','bb_low','NATR','ATR','PPO','MACD','sector','r01','r05','r10','r21','r42','r63','r01dec','r05dec','r10dec','r21dec','r42dec','r63dec','r01q_sector','r05q_sector','r10q_sector','r21q_sector','r42q_sector','r63q_sector','r01_fwd','r05_fwd','r21_fwd','year','month','weekday']
with pd.HDFStore(sys.argv[1]) as s: n=s.get_storer('model_data').nrows; x=s.select('model_data',start=0,stop=1)
assert n and list(x.index.names)==['symbol','date'] and list(x.columns)==cols; print('[OK]',sys.argv[1],n)
PY
}

data(){
  CURRENT_STAGE=data; disk "$MIN_FREE_GB" data; mkdir -p "$RAW_5M_DIR" "$RAW_DAILY_DIR" "$AS1455_DAILY_DIR"
  local p n5 na
  for p in $(seq 1 "$MAX_BOOTSTRAP_PASSES"); do n5="$(count "$RAW_5M_DIR" '*_5m_raw.csv')"; na="$(count "$AS1455_DAILY_DIR" '*_as1455_daily.csv')"; info "bootstrap $p/$MAX_BOOTSTRAP_PASSES raw5=$n5 as1455=$na"; [[ "$n5" == 1000 && "$na" == 1000 ]] && break
    run "bootstrap_$p" "$PY" scripts/build_ashare_ch12_as1455_model_data.py --universe "$UNIVERSE" --out-dir "$CH12_DIR" --bar-root "$RAW_5M_DIR" --bar-glob '*_5m_raw.csv' --baostock-5m-cache-dir "$RAW_5M_DIR" --as1455-daily-cache-dir "$AS1455_DAILY_DIR" --raw-daily-cache-dir "$RAW_DAILY_DIR" --start-date "$HISTORY_START_DATE" --end-date "$(end_date)" --daily-cache-only --allow-partial-coverage --baostock-fetch-limit "$BOOTSTRAP_BATCH_SIZE" --baostock-fetch-retries 3 --baostock-fetch-sleep "$BAOSTOCK_FETCH_SLEEP" --baostock-query-timeout 180 --qfq5m-audit-samples 0 --profile-memory || true
  done
  [[ "$(count "$RAW_5M_DIR" '*_5m_raw.csv')" == 1000 ]] || fail "5m cache incomplete"
  run history env PYTHON="$PY" TRADE_DATE="$TRADE_DATE" HISTORY_END_DATE="$(end_date)" HISTORY_START_DATE="$HISTORY_START_DATE" UNIVERSE="$UNIVERSE" RAW_5M_CACHE_DIR="$RAW_5M_DIR" RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" AS1455_DAILY_CACHE_DIR="$AS1455_DAILY_DIR" bash scripts/run_as1455_live_data_feature_pipeline.sh history
  [[ "$(count "$RAW_DAILY_DIR" '*_daily_raw.csv')" == 1000 && "$(count "$AS1455_DAILY_DIR" '*_as1455_daily.csv')" == 1000 ]] || fail "daily cache incomplete"
  run model_data env PYTHON="$PY" OUT_DIR="$CH12_DIR" BAR_CACHE_DIR="$RAW_5M_DIR" DAILY_CACHE_DIR="$AS1455_DAILY_DIR" bash scripts/build_ashare_ch12_as1455_lowmem.sh --universe "$UNIVERSE" --start-date "$HISTORY_START_DATE" --end-date "$(end_date)" --raw-daily-cache-dir "$RAW_DAILY_DIR" --no-fetch-missing-raw-daily --qfq5m-audit-samples 0
  validate_model "$MODEL_DATA"
}

missing_folds(){ "$PY" - "$1" "$2" "$BEST_N" "$RUN_STAMP" <<'PY'
from datetime import datetime
import pandas as pd,sys
from utils.as1455_ch17_common import default_fold_dir_template,fold_dir_from_template
p,t,n,stamp=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]; out=[]
for f in (range(7) if t in {'r01_fwd','r05_fwd'} else range(6)):
 r=fold_dir_from_template(default_fold_dir_template(p,t),f); ok=(r/'search_best_checkpoints.csv').is_file() and (r/'preprocess/scaler.pkl').is_file() and (r/'preprocess/feature_manifest.json').is_file() and any((r/'search_checkpoints').glob('*.keras'))
 try: ok=ok and len(pd.read_csv(r/'search_best_checkpoints.csv'))>=n
 except: ok=False
 if not ok:
  if r.is_dir() and any(r.iterdir()): r.rename(r.with_name(f'{r.name}.incomplete.{stamp}.{datetime.now():%H%M%S}'))
  out.append(str(f))
print(' '.join(out))
PY
}
selfcheck(){ CURRENT_STAGE=selfcheck; run selfcheck env PYTHON_BIN="$PY" bash scripts/check_ch17_as1455_refactor.sh; }

train_wrapper(){ case "$1" in r01_fwd) echo scripts/run_as1455_target_search_all.sh;; r05_fwd) echo scripts/run_as1455_r05_target_search_all.sh;; r21_fwd) echo scripts/run_as1455_r21_target_search_all.sh;; esac; }
training(){ CURRENT_STAGE=training; local t p f w; for t in $TARGETS; do w="$(train_wrapper "$t")"; for p in $FEATURE_PRESETS; do f="$(missing_folds "$p" "$t")"; [[ -z "$f" ]] || run "train_${p}_${t}" env PYTHON_BIN="$PY" MODEL_DATA="$MODEL_DATA" FEATURE_PRESETS="$p" TARGET_COL="$t" FOLDS="$f" EPOCHS="$EPOCHS" BEST_N="$BEST_N" SEED="$SEED" LOG_DIR="$SEARCH_LOG_DIR" bash "$w"; done; done; }

hist_wrapper(){ case "$1" in r01_fwd) echo scripts/run_as1455_target_natural_backtest.sh;; r05_fwd) echo scripts/run_as1455_r05_natural_backtest.sh;; r21_fwd) echo scripts/run_as1455_r21_natural_backtest.sh;; esac; }
historical(){ CURRENT_STAGE=historical; local t; for t in $TARGETS; do donep "historical_$t" && continue; run "historical_$t" env PYTHON_BIN="$PY" MODEL_DATA="$MODEL_DATA" RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" FEATURE_PRESETS="$FEATURE_PRESETS" TARGET_COL="$t" OUTPUT_MODE=summary MATERIALIZE_BEST=1 MATERIALIZED_OUTPUT_MODE=compact RANK_METRIC=sharpe RUN_STAMP="$RUN_STAMP" MIN_FREE_GB="$MIN_FREE_GB" bash "$(hist_wrapper "$t")"; finish "historical_$t"; done; }

reb(){ "$PY" - "$1" <<'PY'
import sys
from utils.as1455_ch17_common import target_spec
print(target_spec(sys.argv[1]).rebalance_every)
PY
}
forward(){
  CURRENT_STAGE=forward; run forward_refresh env PYTHON_BIN="$PY" TRADE_DATE="$TRADE_DATE" HISTORY_END_DATE="$(end_date)" TIMEZONE="$TIMEZONE" UNIVERSE="$UNIVERSE" SOURCE_DIR="$CH12_DIR" FORWARD_MODEL_DIR="$FORWARD_DIR" RAW_5M_CACHE_DIR="$RAW_5M_DIR" RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" AS1455_DAILY_CACHE_DIR="$AS1455_DAILY_DIR" MIN_FREE_GB="$MIN_FREE_GB" FORWARD_ARTIFACT_MODE=model_only FORWARD_REPORT_MODE=compact bash scripts/refresh_as1455_forward_model_data.sh; validate_model "$FORWARD_DATA"
  local t; for t in $TARGETS; do donep "forward_$t" && continue; run "forward_$t" env PYTHON_BIN="$PY" REFRESH_DATA=0 MODEL_DATA="$FORWARD_DATA" SOURCE_DIR="$CH12_DIR" RAW_DAILY_CACHE_DIR="$RAW_DAILY_DIR" TARGETS="$t" FEATURE_PRESETS="$FEATURE_PRESETS" MODEL_SELECTION_MODE=strict_oos SELECTION_RANK_METRIC=sharpe OUTPUT_MODE=compact RUN_STAMP="$RUN_STAMP" MIN_FREE_GB="$MIN_FREE_GB" bash scripts/run_as1455_fold0_forward_backtests.sh; finish "forward_$t"; done
}

plots(){ CURRENT_STAGE=plots; local roots='' labels='' t p r x; for t in $TARGETS; do r="$(reb "$t")"; for p in $FEATURE_PRESETS; do x="saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/${p}_${t}_reb${r}_${RUN_STAMP}"; [[ -s "$x/01_close_auction_grid/strict_oos_manifest.json" ]] || fail "missing strict result $x"; roots+="${roots:+,}$x"; labels+="${labels:+,}${t}_${p}"; done; done; run plots env PYTHON_BIN="$PY" BACKTEST_ROOTS="$roots" LABELS="$labels" OUT_DIR="$PLOT_DIR" RANK_METRIC=sharpe bash scripts/plot_as1455_default_ab_nav_curves.sh; }

audit(){ CURRENT_STAGE=audit; "$PY" - "$RUN_STAMP" "$FEATURE_PRESETS" "$TARGETS" "$PLOT_DIR" "$REBUILD_ROOT" <<'PY'
from pathlib import Path
import json,sys
from utils.as1455_ch17_common import target_spec,default_fold_dir_template,fold_dir_from_template
s,ps,ts,plot,root=sys.argv[1],sys.argv[2].split(),sys.argv[3].split(),Path(sys.argv[4]),Path(sys.argv[5]); train=h=f=0
for t in ts:
 for p in ps:
  for fold in (range(7) if t in {'r01_fwd','r05_fwd'} else range(6)):
   d=fold_dir_from_template(default_fold_dir_template(p,t),fold); train+=int((d/'search_best_checkpoints.csv').is_file() and (d/'preprocess/scaler.pkl').is_file() and any((d/'search_checkpoints').glob('*.keras')))
  r=target_spec(t).rebalance_every; h+=int((Path('saved_data/ashare_ml4t/ch17_as1455_target_backtest')/f'{p}_{t}_reb{r}_{s}'/'materialized_best_run.json').is_file())
  m=Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')/f'{p}_{t}_reb{r}_{s}'/'01_close_auction_grid/strict_oos_manifest.json'
  try: o=json.loads(m.read_text()); f+=int(o['evaluation_mode']=='strict_oos' and o['historical_trading_parameters_reused'] is True and o['historical_rebalance_phase_reused'] is True and o['generated_config_count']==1 and o['retained_config_count']==1)
  except: pass
report={'training_expected':40,'training_ok':train,'historical_expected':6,'historical_ok':h,'forward_expected':6,'forward_ok':f,'plots_ok':any(plot.glob('*.png'))}; report['all_ok']=train==40 and h==6 and f==6 and report['plots_ok']; root.mkdir(parents=True,exist_ok=True); q=root/'final_report.json'; q.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); assert report['all_ok']; print('[PASS] full Ch17 AS1455 rebuild completed')
PY
}

run_stage(){ donep "$1" && { info "skip $1"; return; }; "$2"; finish "$1"; }
status(){ echo "run_stamp=$RUN_STAMP"; for s in preflight data selfcheck training historical forward plots audit; do donep "$s" && echo "[DONE] $s" || echo "[PENDING] $s"; done; [[ -s "$REBUILD_ROOT/final_report.json" ]] && cat "$REBUILD_ROOT/final_report.json"; }
[[ "$MODE" == status ]] && { status; exit; }; setup_python
case "$MODE" in
 all) run_stage preflight preflight; run_stage data data; run_stage selfcheck selfcheck; run_stage training training; run_stage historical historical; run_stage forward forward; run_stage plots plots; run_stage audit audit;;
 preflight|data|training|historical|forward|plots|audit) run_stage "$MODE" "$MODE";; selfcheck) run_stage selfcheck selfcheck;;
 *) echo 'Usage: ... all|status|preflight|data|selfcheck|training|historical|forward|plots|audit' >&2; exit 2;; esac
