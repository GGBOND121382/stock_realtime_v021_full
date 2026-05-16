#!/usr/bin/env bash
set -uo pipefail

# scripts/update_ranked_models_latest.sh
#
# Purpose:
#   根据最新 pipeline 数据，更新/新增前面榜单中建议进入 saved_models 的模型。
#
# Default core update set:
#   1) 603308.SH sector d4 close_profit
#   2) 600312.SH vwap_low d3_600 close_profit
#   3) 600312.SH all_days d3_600 close_profit
#   4) 603308.SH external aero_nuclear_equipment v2
#   5) 601899.SH zijin extra_trees close_profit
#   6) 600522.SH optical_cable_grid close_profit
#   7) 600487.SH optical_cable_grid close_profit
#
# Optional:
#   INCLUDE_600487_HIT=1   保存 600487 hit80 辅助模型
#   INCLUDE_002518=1       保存 002518 storage_power 高收益高回撤模型
#   INCLUDE_BORDERLINE=1   也更新 600096/002311/600276/601985/002714 等边缘模型
#
# Example:
#   chmod +x scripts/update_ranked_models_latest.sh
#   PYTHON=python3 END_DATE=2026-05-15 ./scripts/update_ranked_models_latest.sh
#
# Dry run:
#   DRY_RUN=1 ./scripts/update_ranked_models_latest.sh
#
# Only selected:
#   ONLY=603308.SH,600522.SH ./scripts/update_ranked_models_latest.sh
#
# Reuse existing saved_data without rerun pipeline:
#   SKIP_PIPELINE=1 ./scripts/update_ranked_models_latest.sh

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
JOB_TIMEOUT="${JOB_TIMEOUT:-8h}"

MODEL_OUT_DIR="${MODEL_OUT_DIR:-saved_models}"
LOG_DIR="${LOG_DIR:-saved_data/model_search_queue_logs/update_ranked_models_$(date +%Y%m%d_%H%M%S)}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_SAVE="${SKIP_SAVE:-0}"
CLEAN_PIPELINE="${CLEAN_PIPELINE:-1}"

INCLUDE_600487_HIT="${INCLUDE_600487_HIT:-0}"
INCLUDE_002518="${INCLUDE_002518:-0}"
INCLUDE_BORDERLINE="${INCLUDE_BORDERLINE:-0}"

# 是否删除被 v2 替代的旧 external artifact。
PRUNE_SUPERSEDED="${PRUNE_SUPERSEDED:-1}"

ONLY="${ONLY:-}"

mkdir -p "$LOG_DIR" "$MODEL_OUT_DIR"

PIPELINE_SUMMARY="$LOG_DIR/pipeline_summary.csv"
SAVE_SUMMARY="$LOG_DIR/save_summary.csv"
VERIFY_SUMMARY="$LOG_DIR/verify_summary.csv"

echo "symbol,sector,external,status,returncode,start_time,end_time,log_file" > "$PIPELINE_SUMMARY"
echo "symbol,artifact,status,returncode,samples,intraday_bars,artifact_dir,start_time,end_time,log_file" > "$SAVE_SUMMARY"
echo "symbol,artifact,date_max,trades,win_rate,avg_return,median_return,max_drawdown,profit_factor,threshold,metadata" > "$VERIFY_SUMMARY"

COMMON_PIPELINE_ARGS=(
  --start-date "$START_DATE"
  --end-date "$END_DATE"
  --feature-pipeline fundamental,sector
  --search-targets hit50,hit80,close_profit
  --entry-policies vwap_low,all_days
  --groups reversal_fundamental_regime,reversal_fundamental_regime_sector,reversal_fundamental_regime_sector_external,all_no_ak
  --models xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_600_lr002_mcw3,xgb_d4_500_lr002_mcw5,lgbm_leaves7_400,lgbm_leaves15_700,extra_trees_600_d3,random_forest_600_d4
  --quantiles 0.5,0.6,0.7,0.8
  --train-rows 756
  --valid-rows 126
  --test-rows 63
  --min-valid-trades 8
  --min-train-entries 80
  --external-lag-days 1
  --stock-external-domestic-lag-days 0
  --stock-external-future-lag-days 1
  --stock-external-us-lag-days 1
  --resume
)

SAVE_COMMON_ARGS=(
  --entry-vwap-premium-bps 50
  --round-trip-cost-bps 1.7
  --valid-rows 252
  --min-train-entries 80
  --min-valid-trades 8
  --quantiles 0.5,0.6,0.7,0.8
)

contains_only() {
  local symbol="$1"
  if [[ -z "$ONLY" ]]; then
    return 0
  fi

  local raw="${symbol%%.*}"
  local sym_upper="${symbol^^}"

  IFS=',' read -ra arr <<< "$ONLY"
  for x in "${arr[@]}"; do
    x="$(echo "$x" | tr '[:lower:]' '[:upper:]' | xargs)"
    if [[ "$x" == "$sym_upper" || "$x" == "$raw" ]]; then
      return 0
    fi
  done
  return 1
}

print_cmd() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
}

run_pipeline() {
  local symbol="$1"
  local sector="$2"
  local external="$3"

  if ! contains_only "$symbol"; then
    echo "[SKIP] pipeline $symbol not selected by ONLY=$ONLY"
    return 0
  fi

  if [[ "$SKIP_PIPELINE" == "1" ]]; then
    echo "[SKIP] pipeline $symbol because SKIP_PIPELINE=1"
    return 0
  fi

  local raw="${symbol%%.*}"
  local safe_symbol="${symbol//./_}"
  local ext_label="${external:-none}"
  local out_root="saved_data/${raw}_pipeline_out"
  local log_file="$LOG_DIR/pipeline_${safe_symbol}_${ext_label}.log"
  local external_args=()
  local start_time end_time rc

  if [[ -n "$external" ]]; then
    external_args+=(--external "$external")
  fi

  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[PIPELINE START] $start_time symbol=$symbol sector=$sector external=$ext_label"
  echo "[OUT_ROOT] $out_root"
  echo "============================================================"

  if [[ "$CLEAN_PIPELINE" == "1" ]]; then
    echo "[CLEAN] removing pipeline output: $out_root"
    if [[ "$DRY_RUN" != "1" ]]; then
      rm -rf "$out_root"
    fi
  fi

  local cmd=(
    timeout --foreground "$JOB_TIMEOUT" "$PYTHON" pipelines/run_nextday_pipeline.py
    --symbol "$symbol"
    --sector-symbol "$sector"
    --out-root "$out_root"
    "${external_args[@]}"
    "${COMMON_PIPELINE_ARGS[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}" | tee -a "$log_file"
    rc=0
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"

  if [[ "$rc" -eq 0 ]]; then
    echo "[PIPELINE DONE] $symbol"
    echo "${symbol},${sector},${ext_label},ok,${rc},${start_time},${end_time},${log_file}" >> "$PIPELINE_SUMMARY"
    return 0
  elif [[ "$rc" -eq 124 ]]; then
    echo "[PIPELINE TIMEOUT] $symbol timeout=$JOB_TIMEOUT"
    echo "${symbol},${sector},${ext_label},timeout,${rc},${start_time},${end_time},${log_file}" >> "$PIPELINE_SUMMARY"
    return "$rc"
  else
    echo "[PIPELINE FAIL] $symbol returncode=$rc"
    echo "${symbol},${sector},${ext_label},failed,${rc},${start_time},${end_time},${log_file}" >> "$PIPELINE_SUMMARY"
    return "$rc"
  fi
}

verify_artifact() {
  local symbol="$1"
  local artifact="$2"
  local metadata="$MODEL_OUT_DIR/$symbol/$artifact/metadata.json"

  if [[ ! -f "$metadata" ]]; then
    echo "[VERIFY WARN] metadata not found: $metadata"
    echo "${symbol},${artifact},,,,,,,,,,${metadata}" >> "$VERIFY_SUMMARY"
    return 1
  fi

  "$PYTHON" - "$symbol" "$artifact" "$metadata" "$VERIFY_SUMMARY" <<'PY'
import csv
import json
import sys
from pathlib import Path

symbol, artifact, metadata, out_csv = sys.argv[1:5]
p = Path(metadata)
data = json.loads(p.read_text(encoding="utf-8"))
m = data.get("validation_tail_trade_metrics", {}) or {}

row = {
    "symbol": symbol,
    "artifact": artifact,
    "date_max": data.get("date_max", ""),
    "trades": m.get("trades", ""),
    "win_rate": m.get("win_rate", ""),
    "avg_return": m.get("avg_return", ""),
    "median_return": m.get("median_return", ""),
    "max_drawdown": m.get("max_drawdown", ""),
    "profit_factor": m.get("profit_factor", ""),
    "threshold": data.get("threshold", ""),
    "metadata": str(p),
}

with open(out_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[
        "symbol", "artifact", "date_max", "trades", "win_rate",
        "avg_return", "median_return", "max_drawdown",
        "profit_factor", "threshold", "metadata"
    ])
    w.writerow(row)

print(
    f"[VERIFY] {symbol} {artifact} "
    f"date_max={row['date_max']} trades={row['trades']} "
    f"win_rate={row['win_rate']} avg_return={row['avg_return']} "
    f"mdd={row['max_drawdown']} pf={row['profit_factor']}"
)
PY
}

save_model() {
  local symbol="$1"
  local artifact="$2"
  local samples="$3"
  local intraday="$4"
  local feature_group="$5"
  local model_name="$6"
  local label_mode="$7"
  local entry_policy="$8"
  local target_hit_bps="${9:-50}"

  if ! contains_only "$symbol"; then
    echo "[SKIP] save $symbol not selected by ONLY=$ONLY"
    return 0
  fi

  if [[ "$SKIP_SAVE" == "1" ]]; then
    echo "[SKIP] save $symbol because SKIP_SAVE=1"
    return 0
  fi

  local artifact_dir="$MODEL_OUT_DIR/$symbol/$artifact"
  local safe_symbol="${symbol//./_}"
  local log_file="$LOG_DIR/save_${safe_symbol}_${artifact}.log"
  local start_time end_time rc

  start_time="$(date '+%F %T')"

  echo
  echo "============================================================"
  echo "[SAVE START] $start_time $symbol -> $artifact"
  echo "[ARTIFACT_DIR] $artifact_dir"
  echo "[SAMPLES] $samples"
  echo "[INTRADAY] $intraday"
  echo "============================================================"

  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -f "$samples" ]]; then
      echo "[SAVE ERROR] missing samples: $samples"
      end_time="$(date '+%F %T')"
      echo "${symbol},${artifact},missing_samples,2,${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
      return 2
    fi
    if [[ ! -f "$intraday" ]]; then
      echo "[SAVE ERROR] missing intraday bars: $intraday"
      end_time="$(date '+%F %T')"
      echo "${symbol},${artifact},missing_intraday,2,${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
      return 2
    fi

    if [[ -d "$artifact_dir" ]]; then
      echo "[OVERWRITE] removing old artifact: $artifact_dir"
      rm -rf "$artifact_dir"
    fi
  else
    print_cmd rm -rf "$artifact_dir"
  fi

  local cmd=(
    "$PYTHON" model_saving/save_nextday_model.py
    --stock-code "$symbol"
    --artifact-name "$artifact"
    --samples "$samples"
    --intraday-bars "$intraday"
    --out-dir "$MODEL_OUT_DIR"
    --feature-group "$feature_group"
    --model-name "$model_name"
    --label-mode "$label_mode"
    --entry-policy "$entry_policy"
    --target-hit-bps "$target_hit_bps"
    "${SAVE_COMMON_ARGS[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}" | tee -a "$log_file"
    rc=0
  else
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
  fi

  end_time="$(date '+%F %T')"

  if [[ "$rc" -eq 0 ]]; then
    echo "[SAVE DONE] $symbol -> $artifact"
    echo "${symbol},${artifact},ok,${rc},${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
    if [[ "$DRY_RUN" != "1" ]]; then
      verify_artifact "$symbol" "$artifact" || true
    fi
    return 0
  else
    echo "[SAVE FAIL] $symbol -> $artifact returncode=$rc"
    echo "${symbol},${artifact},failed,${rc},${samples},${intraday},${artifact_dir},${start_time},${end_time},${log_file}" >> "$SAVE_SUMMARY"
    return "$rc"
  fi
}

prune_superseded() {
  local dir="$1"

  if [[ "$PRUNE_SUPERSEDED" != "1" ]]; then
    return 0
  fi

  if [[ -d "$dir" ]]; then
    echo "[PRUNE] removing superseded artifact: $dir"
    if [[ "$DRY_RUN" == "1" ]]; then
      print_cmd rm -rf "$dir"
    else
      rm -rf "$dir"
    fi
  fi
}

PIPE_FAILED=0
SAVE_FAILED=0

echo "============================================================"
echo "[CONFIG]"
echo "PYTHON=$PYTHON"
echo "START_DATE=$START_DATE"
echo "END_DATE=$END_DATE"
echo "JOB_TIMEOUT=$JOB_TIMEOUT"
echo "MODEL_OUT_DIR=$MODEL_OUT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "DRY_RUN=$DRY_RUN"
echo "SKIP_PIPELINE=$SKIP_PIPELINE"
echo "SKIP_SAVE=$SKIP_SAVE"
echo "CLEAN_PIPELINE=$CLEAN_PIPELINE"
echo "INCLUDE_600487_HIT=$INCLUDE_600487_HIT"
echo "INCLUDE_002518=$INCLUDE_002518"
echo "INCLUDE_BORDERLINE=$INCLUDE_BORDERLINE"
echo "ONLY=$ONLY"
echo "============================================================"

# ============================================================
# 1) Run latest pipelines
# ============================================================

run_pipeline 603308.SH 通用设备       "aero_nuclear_equipment" || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 600312.SH 电网设备       ""                       || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 601899.SH 贵金属         "zijin_external"         || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 600522.SH 光通信设备     "optical_cable_grid"     || PIPE_FAILED=$((PIPE_FAILED + 1))
run_pipeline 600487.SH 光通信设备     "optical_cable_grid"     || PIPE_FAILED=$((PIPE_FAILED + 1))

if [[ "$INCLUDE_002518" == "1" ]]; then
  run_pipeline 002518.SZ 储能电池     "storage_power"          || PIPE_FAILED=$((PIPE_FAILED + 1))
fi

if [[ "$INCLUDE_BORDERLINE" == "1" ]]; then
  run_pipeline 600096.SH 农化制品     "fertilizer"             || PIPE_FAILED=$((PIPE_FAILED + 1))
  run_pipeline 002311.SZ 农产品加工   "feed,hog"               || PIPE_FAILED=$((PIPE_FAILED + 1))
  run_pipeline 600276.SH 化学制药     ""                       || PIPE_FAILED=$((PIPE_FAILED + 1))
  run_pipeline 601985.SH 电力         ""                       || PIPE_FAILED=$((PIPE_FAILED + 1))
  run_pipeline 002714.SZ 养殖业       "hog,muyuan_hk"          || PIPE_FAILED=$((PIPE_FAILED + 1))
fi

# ============================================================
# 2) Save / overwrite selected models
# ============================================================

# ---------- 603308.SH 应流股份：当前最强 sector 模型 ----------
save_model \
  603308.SH \
  nextday_all_days_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
  "saved_data/603308_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/603308_pipeline_out/00_base/603308_5m.csv" \
  reversal_fundamental_regime_sector \
  xgb_d4_500_lr002_mcw5 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# ---------- 603308.SH 应流股份：external v2 ----------
# 旧 v1 如果不删，会继续被 model-policy=all 扫到，可能继续出现 realtime_context partial。
prune_superseded "saved_models/603308.SH/nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_external_aero_nuclear_v1"

save_model \
  603308.SH \
  nextday_all_days_close_profit_xgb_d3_400_reversal_fundamental_regime_sector_external_aero_nuclear_v2 \
  "saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment/training_samples_with_aero_nuclear_equipment_external.csv" \
  "saved_data/603308_pipeline_out/00_base/603308_5m.csv" \
  reversal_fundamental_regime_sector_external \
  xgb_d3_400_lr003_mcw3 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# ---------- 600312.SH 平高电气：两个强模型 ----------
save_model \
  600312.SH \
  nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_v1 \
  "saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/600312_pipeline_out/00_base/600312_5m.csv" \
  reversal_fundamental_regime \
  xgb_d3_600_lr002_mcw3 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

save_model \
  600312.SH \
  nextday_all_days_close_profit_xgb_d3_600_reversal_fundamental_regime_v1 \
  "saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv" \
  "saved_data/600312_pipeline_out/00_base/600312_5m.csv" \
  reversal_fundamental_regime \
  xgb_d3_600_lr002_mcw3 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# ---------- 601899.SH 紫金矿业 ----------
save_model \
  601899.SH \
  nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_zijin_v1 \
  "saved_data/601899_pipeline_out/04_external/zijin_external/training_samples_with_zijin_external.csv" \
  "saved_data/601899_pipeline_out/00_base/601899_5m.csv" \
  reversal_fundamental_regime_sector \
  extra_trees_600_d3 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# ---------- 600522.SH：新增，榜单未保存中最值得补 ----------
save_model \
  600522.SH \
  nextday_all_days_close_profit_random_forest_reversal_fundamental_regime_sector_external_optical_cable_grid_v1 \
  "saved_data/600522_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
  "saved_data/600522_pipeline_out/00_base/600522_5m.csv" \
  reversal_fundamental_regime_sector_external \
  random_forest_600_d4 \
  close_profit \
  all_days \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# ---------- 600487.SH：新增 close_profit ----------
save_model \
  600487.SH \
  nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_external_optical_cable_grid_v1 \
  "saved_data/600487_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
  "saved_data/600487_pipeline_out/00_base/600487_5m.csv" \
  reversal_fundamental_regime_sector_external \
  extra_trees_600_d3 \
  close_profit \
  vwap_low \
  50 || SAVE_FAILED=$((SAVE_FAILED + 1))

# ---------- 600487.SH：可选 hit80 辅助 ----------
if [[ "$INCLUDE_600487_HIT" == "1" ]]; then
  save_model \
    600487.SH \
    nextday_all_days_hit80_xgb_d2_reversal_fundamental_regime_optical_cable_grid_v1 \
    "saved_data/600487_pipeline_out/04_external/optical_cable_grid/training_samples_with_optical_cable_grid_external.csv" \
    "saved_data/600487_pipeline_out/00_base/600487_5m.csv" \
    reversal_fundamental_regime \
    xgb_d2_200_lr003_mcw5 \
    hit \
    all_days \
    80 || SAVE_FAILED=$((SAVE_FAILED + 1))
fi

# ---------- 002518.SZ：可选，高收益但高回撤，默认不保存 ----------
if [[ "$INCLUDE_002518" == "1" ]]; then
  save_model \
    002518.SZ \
    nextday_all_days_close_profit_xgb_d2_200_reversal_fundamental_regime_sector_storage_power_v1 \
    "saved_data/002518_pipeline_out/04_external/storage_power/training_samples_with_storage_power_external.csv" \
    "saved_data/002518_pipeline_out/00_base/002518_5m.csv" \
    reversal_fundamental_regime_sector \
    xgb_d2_200_lr003_mcw5 \
    close_profit \
    all_days \
    50 || SAVE_FAILED=$((SAVE_FAILED + 1))
fi

# ---------- 边缘模型：默认不更新 ----------
if [[ "$INCLUDE_BORDERLINE" == "1" ]]; then
  save_model \
    600096.SH \
    nextday_vwap_low_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
    "saved_data/600096_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/600096_pipeline_out/00_base/600096_5m.csv" \
    reversal_fundamental_regime_sector \
    xgb_d4_500_lr002_mcw5 \
    close_profit \
    vwap_low \
    50 || SAVE_FAILED=$((SAVE_FAILED + 1))

  save_model \
    600096.SH \
    nextday_all_days_close_profit_xgb_d4_reversal_fundamental_regime_sector_v1 \
    "saved_data/600096_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/600096_pipeline_out/00_base/600096_5m.csv" \
    reversal_fundamental_regime_sector \
    xgb_d4_500_lr002_mcw5 \
    close_profit \
    all_days \
    50 || SAVE_FAILED=$((SAVE_FAILED + 1))

  save_model \
    002311.SZ \
    nextday_vwap_low_close_profit_xgb_d3_600_reversal_fundamental_regime_sector_v1 \
    "saved_data/002311_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/002311_pipeline_out/00_base/002311_5m.csv" \
    reversal_fundamental_regime_sector \
    xgb_d3_600_lr002_mcw3 \
    close_profit \
    vwap_low \
    50 || SAVE_FAILED=$((SAVE_FAILED + 1))

  save_model \
    600276.SH \
    nextday_all_days_hit80_extra_trees_reversal_fundamental_regime_sector_v1 \
    "saved_data/600276_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/600276_pipeline_out/00_base/600276_5m.csv" \
    reversal_fundamental_regime_sector \
    extra_trees_600_d3 \
    hit \
    all_days \
    80 || SAVE_FAILED=$((SAVE_FAILED + 1))

  save_model \
    601985.SH \
    nextday_all_days_close_profit_extra_trees_reversal_fundamental_regime_v1 \
    "saved_data/601985_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/601985_pipeline_out/00_base/601985_5m.csv" \
    reversal_fundamental_regime \
    extra_trees_600_d3 \
    close_profit \
    all_days \
    50 || SAVE_FAILED=$((SAVE_FAILED + 1))

  save_model \
    002714.SZ \
    nextday_vwap_low_close_profit_random_forest_reversal_fundamental_regime_sector_v1 \
    "saved_data/002714_pipeline_out/03_sector/training_samples_with_sector.csv" \
    "saved_data/002714_pipeline_out/00_base/002714_5m.csv" \
    reversal_fundamental_regime_sector \
    random_forest_600_d4 \
    close_profit \
    vwap_low \
    50 || SAVE_FAILED=$((SAVE_FAILED + 1))
fi

echo
echo "============================================================"
echo "[ALL DONE] $(date '+%F %T')"
echo "[MODEL_OUT_DIR] $MODEL_OUT_DIR"
echo "[PIPELINE_FAILED] $PIPE_FAILED"
echo "[SAVE_FAILED] $SAVE_FAILED"
echo "[PIPELINE_SUMMARY] $PIPELINE_SUMMARY"
echo "[SAVE_SUMMARY] $SAVE_SUMMARY"
echo "[VERIFY_SUMMARY] $VERIFY_SUMMARY"
echo "============================================================"

if [[ "$PIPE_FAILED" -gt 0 || "$SAVE_FAILED" -gt 0 ]]; then
  exit 1
fi

exit 0