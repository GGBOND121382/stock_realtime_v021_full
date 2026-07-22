#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv_as1455/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="${BASE_PYTHON:-python3}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"; TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"; RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
HIST_BASE="${HIST_BASE:-saved_data/ashare_ml4t/ch17_as1455_target_backtest}"; FWD_BASE="${FWD_BASE:-saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest}"
PLOT_DIR="${PLOT_DIR:-saved_data/ashare_ml4t/ch17_as1455_backtest_plots/backtest_only_$RUN_STAMP}"; REPORT_DIR="${REPORT_DIR:-saved_data/ashare_ml4t/ch17_as1455_backtest_only/$RUN_STAMP}"
mkdir -p "$PLOT_DIR" "$REPORT_DIR"
[[ -s "$MODEL_DATA" ]] || { echo "[ERROR] missing model_data: $MODEL_DATA" >&2; exit 1; }; [[ -d "$RAW_DAILY_CACHE_DIR" ]] || { echo "[ERROR] missing raw daily cache" >&2; exit 1; }
"$PYTHON_BIN" scripts/check_as1455_disk_space.py --path saved_data/ashare_ml4t --min-free-gb "$MIN_FREE_GB" --label existing-model-backtest-only
"$PYTHON_BIN" - "$FEATURE_PRESETS" "$TARGETS" <<'PY'
import sys
from utils.as1455_ch17_common import default_fold_dir_template,fold_dir_from_template
missing=[]
for t in sys.argv[2].split():
 for p in sys.argv[1].split():
  for f in (range(7) if t in {'r01_fwd','r05_fwd'} else range(6)):
   d=fold_dir_from_template(default_fold_dir_template(p,t),f); ok=(d/'search_best_checkpoints.csv').is_file() and (d/'preprocess/scaler.pkl').is_file() and any((d/'search_checkpoints').glob('*.keras'))
   if not ok: missing.append(str(d))
if missing: raise SystemExit('[ERROR] trained fold artifacts missing:\n'+'\n'.join(missing))
print('[OK] existing models found; training=false; model directories are read-only')
PY
before="$(du -sb "$HIST_BASE" "$FWD_BASE" "$PLOT_DIR" "$REPORT_DIR" 2>/dev/null | awk '{s+=$1} END{print s+0}')"
for target in $TARGETS; do
 case "$target" in r01_fwd) w=scripts/run_as1455_target_natural_backtest.sh; tf=0,1,2,3,4,5;; r05_fwd) w=scripts/run_as1455_r05_natural_backtest.sh; tf=0,1,2,3,4,5;; r21_fwd) w=scripts/run_as1455_r21_natural_backtest.sh; tf=0,1,2,3,4;; *) exit 2;; esac
 env TARGET_FOLDS="$tf" PYTHON_BIN="$PYTHON_BIN" MODEL_DATA="$MODEL_DATA" RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" FEATURE_PRESETS="$FEATURE_PRESETS" TARGET_COL="$target" OUT_BASE="$HIST_BASE" OUTPUT_MODE=summary MATERIALIZE_BEST=1 MATERIALIZED_OUTPUT_MODE=compact RANK_METRIC=sharpe MIN_FREE_GB="$MIN_FREE_GB" RUN_STAMP="$RUN_STAMP" bash "$w"
done
COMMON_START="$("$PYTHON_BIN" scripts/resolve_as1455_common_forward_start.py --model-data "$MODEL_DATA" --feature-presets "$FEATURE_PRESETS" --targets "$TARGETS")"; echo "[INFO] common forward start=$COMMON_START"
env PYTHON_BIN="$PYTHON_BIN" REFRESH_DATA=0 MODEL_DATA="$MODEL_DATA" SOURCE_DIR="$(dirname "$MODEL_DATA")" RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" FEATURE_PRESETS="$FEATURE_PRESETS" TARGETS="$TARGETS" TARGET_BACKTEST_BASE="$HIST_BASE" OUT_BASE="$FWD_BASE" START_DATE="$COMMON_START" MODEL_SELECTION_MODE=strict_oos SELECTION_RANK_METRIC=sharpe OUTPUT_MODE=compact MIN_FREE_GB="$MIN_FREE_GB" RUN_STAMP="$RUN_STAMP" bash scripts/run_as1455_fold0_forward_backtests.sh
roots=""; labels=""; seq=("$PYTHON_BIN" scripts/plot_as1455_fold_sequence_curves.py --rank-metric sharpe --out-dir "$PLOT_DIR/fold_sequence")
for target in $TARGETS; do
 reb="$("$PYTHON_BIN" - "$target" <<'PY'
import sys
from utils.as1455_ch17_common import target_spec
print(target_spec(sys.argv[1]).rebalance_every)
PY
)"
 for preset in $FEATURE_PRESETS; do
  h="$HIST_BASE/${preset}_${target}_reb${reb}_${RUN_STAMP}"; f="$FWD_BASE/${preset}_${target}_reb${reb}_${RUN_STAMP}"
  [[ -s "$h/materialized_best_run.json" ]] || { echo "[ERROR] missing $h" >&2; exit 1; }; [[ -s "$f/01_close_auction_grid/strict_oos_manifest.json" ]] || { echo "[ERROR] missing $f" >&2; exit 1; }
  roots+="${roots:+,}$f"; labels+="${labels:+,}${target}_${preset}"; seq+=(--historical-root "$h" --forward-root "$f" --label "${target}_${preset}")
 done
done
env PYTHON_BIN="$PYTHON_BIN" BACKTEST_ROOTS="$roots" LABELS="$labels" OUT_DIR="$PLOT_DIR" RANK_METRIC=sharpe bash scripts/plot_as1455_default_ab_nav_curves.sh
"${seq[@]}"
after="$(du -sb "$HIST_BASE" "$FWD_BASE" "$PLOT_DIR" "$REPORT_DIR" 2>/dev/null | awk '{s+=$1} END{print s+0}')"; delta=$((after-before))
"$PYTHON_BIN" - "$RUN_STAMP" "$COMMON_START" "$PLOT_DIR" "$REPORT_DIR" "$delta" <<'PY'
import json,sys
from pathlib import Path
stamp,start,plot,rd,delta=sys.argv[1],sys.argv[2],Path(sys.argv[3]),Path(sys.argv[4]),int(sys.argv[5]); n=len(list((plot/'fold_sequence').glob('fold*/return_curve_*.png'))); fw=all((plot/f'return_curve_{x}.png').is_file() for x in ('daily','weekly','monthly'))
o={'run_stamp':stamp,'mode':'existing_models_backtest_only','data_refresh':False,'model_data_rebuild':False,'training':False,'common_forward_start':start,'fold_plot_expected':21,'fold_plot_ok':n,'forward_plot_ok':fw,'new_bytes_in_result_roots':max(0,delta)}; o['all_ok']=n==21 and fw; (rd/'backtest_only_report.json').write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(o,ensure_ascii=False,indent=2)); assert o['all_ok']
PY
echo "[PASS] backtest-only completed; plots=$PLOT_DIR"
