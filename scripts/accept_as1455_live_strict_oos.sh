#!/usr/bin/env bash
# Validate the clean AS1455 strict-OOS live monitor.
#
# static: syntax, unit and structural checks; no market/model artifacts required.
# replay: isolated completed-date replay using saved 03/05/06/08 live artifacts.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-static}"
PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-}"
TARGET_COL="${TARGET_COL:-r05_fwd}"
FEATURE_PRESET="${FEATURE_PRESET:-rotation_addon_onehot}"
SOURCE_OUT_ROOT="${SOURCE_OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
ACCEPT_OUT_ROOT="${ACCEPT_OUT_ROOT:-saved_data/ashare_ml4t/live_as1455_acceptance}"
POSITIONS_FILE="${POSITIONS_FILE:-}"
CASH="${CASH:-}"
CASH_FILE="${CASH_FILE:-}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
SELECTION_BACKTEST_ROOT="${SELECTION_BACKTEST_ROOT:-}"
FOLD0_DIR="${FOLD0_DIR:-}"
MIN_FEATURE_ROWS="${MIN_FEATURE_ROWS:-980}"
MAX_FINALIZE_SECONDS="${MAX_FINALIZE_SECONDS:-120}"
ALLOW_INDICATOR_FALLBACK="${ALLOW_INDICATOR_FALLBACK:-0}"
ALLOW_MISSING_BUY_DATE="${ALLOW_MISSING_BUY_DATE:-0}"

info() { echo "[INFO] $*"; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

static_check() {
  info "shell and Python syntax"
  bash -n scripts/run_as1455_live_strict_oos_pipeline.sh
  bash -n scripts/accept_as1455_live_strict_oos.sh
  "$PYTHON" -m py_compile \
    pipelines/as1455_live_prepare.py \
    data_collection/collect_as1455_live_quotes_as1455.py \
    features/build_as1455_live_feature_state_fast.py \
    features/finalize_as1455_live_features_fast.py \
    tools/build_as1455_live_execution_sidecar_v1.py \
    utils/as1455_live_inference.py \
    scripts/run_as1455_live_strict_oos_monitor.py \
    code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py

  info "unit and clean-tree regression"
  "$PYTHON" -m pytest -q tests/test_as1455_live_strict_oos_helpers.py
  bash scripts/check_ch17_as1455_refactor.sh
  bash scripts/run_as1455_live_data_feature_pipeline.sh check
  UNIVERSE="$UNIVERSE" bash scripts/run_as1455_live_strict_oos_pipeline.sh check
  pass "AS1455 live strict-OOS static acceptance"
}

normalize_date() {
  "$PYTHON" - "$1" <<'PY'
import re, sys
from datetime import datetime
s=re.sub(r"\D", "", sys.argv[1])
if len(s) != 8:
    raise SystemExit("TRADE_DATE must be YYYYMMDD")
datetime.strptime(s, "%Y%m%d")
print(s)
PY
}

validate_replay() {
  "$PYTHON" - "$1" "$2" "$TARGET_COL" "$FEATURE_PRESET" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(sys.argv[1]); date=sys.argv[2]; target=sys.argv[3]; preset=sys.argv[4]
required=[
 "12_feature_build_report.json", "13_live_feature_strict_validation_report.json",
 "08_live_execution_sidecar_report.json", "17_live_strict_oos_manifest.json",
 "11_live_model_features_for_prediction.csv", "14_live_predictions.csv",
 "15_live_rank.csv", "16_live_nav.csv", "16_live_orders.csv",
 "16_live_rejections.csv", "16_live_positions_after_plan.csv",
 "16_live_target_portfolio.csv",
]
missing=[name for name in required if not (root/name).is_file()]
if missing: raise SystemExit(f"missing outputs: {missing}")

def load_json(name): return json.loads((root/name).read_text(encoding="utf-8"))
def read_csv(name):
    try: return pd.read_csv(root/name)
    except pd.errors.EmptyDataError: return pd.DataFrame()

f12=load_json("12_feature_build_report.json")
f13=load_json("13_live_feature_strict_validation_report.json")
side=load_json("08_live_execution_sidecar_report.json")
man=load_json("17_live_strict_oos_manifest.json")
features=read_csv("11_live_model_features_for_prediction.csv")
preds=read_csv("14_live_predictions.csv")
rank=read_csv("15_live_rank.csv")
nav=read_csv("16_live_nav.csv")
orders=read_csv("16_live_orders.csv")
portfolio=read_csv("16_live_target_portfolio.csv")

checks={
 "feature_build_passed": f12.get("feature_passed") is True,
 "feature_contract_passed": f13.get("passed") is True,
 "execution_sidecar_passed": side.get("passed") is True,
 "manifest_passed": man.get("passed") is True,
 "protocol": man.get("protocol") == "as1455_clean_live_strict_oos_v2_v7_single_engine",
 "date": man.get("trade_date", "").replace("-", "") == date,
 "target": man.get("target_col") == target,
 "preset": man.get("feature_preset") == preset,
 "canonical_v7": str(man.get("trade_engine", "")).endswith("run_as1455_close_auction_backtest_v7_maxpos_grid.py"),
 "no_broker_submission": man.get("broker_orders_submitted") is False,
 "no_account_truth_writeback": man.get("planned_orders_persisted_as_account_truth") is False,
 "capacity_fail_closed": man.get("capacity_mode_effective") == "none",
 "single_nav_row": len(nav) == 1,
 "nonempty_features": len(features) > 0,
 "feature_prediction_rows": len(features) == len(preds),
 "prediction_rank_rows": len(preds) == len(rank),
 "finite_scores": (not rank.empty and np.isfinite(pd.to_numeric(rank["pred_score"], errors="coerce")).all()),
 "phase_complete": (man.get("rebalance_phase") or {}).get("forward_date_coverage_complete") is True,
 "order_count": int(man.get("planned_order_count", -1)) == len(orders),
 "portfolio_count": int((man.get("planned_state") or {}).get("n_positions", len(portfolio))) == len(portfolio),
 "cash_nonnegative": float((man.get("planned_state") or {}).get("cash", 0)) >= -1e-6,
}
if not orders.empty:
    checks["orders_not_submitted"] = set(orders["order_status"].astype(str)) == {"planned_not_submitted"}
    lot=int((man.get("trade_config") or {}).get("lot_size", 100))
    shares=pd.to_numeric(orders.get("shares", orders.get("order_shares")), errors="coerce")
    checks["orders_positive"] = shares.notna().all() and shares.gt(0).all()
    checks["orders_lot_multiple"] = ((shares.astype(int) % lot) == 0).all()
failed=[name for name, ok in checks.items() if not bool(ok)]
report={
 "passed": not failed, "trade_date": date, "target_col": target,
 "feature_preset": preset, "feature_rows": len(features),
 "prediction_rows": len(preds), "planned_orders": len(orders),
 "target_positions": len(portfolio), "checks": checks, "failed": failed,
}
(root/"18_live_acceptance_report.json").write_text(
 json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if failed: raise SystemExit(f"acceptance failed: {failed}")
print(f"[PASS] acceptance report: {root/'18_live_acceptance_report.json'}")
PY
}

replay() {
  [[ -n "$TRADE_DATE" ]] || fail "set TRADE_DATE=YYYYMMDD"
  [[ -n "$POSITIONS_FILE" && -f "$POSITIONS_FILE" ]] || fail "set POSITIONS_FILE to holdings before the replay date"
  [[ -n "$CASH" || -n "$CASH_FILE" ]] || fail "set CASH or CASH_FILE"
  [[ -z "$CASH_FILE" || -f "$CASH_FILE" ]] || fail "cash file not found: $CASH_FILE"
  [[ -f "$MODEL_DATA" ]] || fail "model data not found: $MODEL_DATA"
  [[ -f "$UNIVERSE" ]] || fail "universe not found: $UNIVERSE"

  local date source_dir accept_dir name
  date="$(normalize_date "$TRADE_DATE")"
  source_dir="${SOURCE_LIVE_DIR:-$SOURCE_OUT_ROOT/$date}"
  accept_dir="${ACCEPT_LIVE_DIR:-$ACCEPT_OUT_ROOT/$date}"
  [[ "$(realpath -m "$source_dir")" != "$(realpath -m "$accept_dir")" ]] || fail "source and acceptance directories must differ"
  for name in 03_adjustment_events.csv 05_execution_calendar.csv 06_live_feature_state_fast.npz 08_live_raw_row_as1455.csv; do
    [[ -f "$source_dir/$name" ]] || fail "missing completed-date artifact: $source_dir/$name"
  done
  rm -rf "$accept_dir"; mkdir -p "$accept_dir"
  for name in 03_adjustment_events.csv 05_execution_calendar.csv 06_live_feature_state_fast.npz 08_live_raw_row_as1455.csv; do
    cp -f "$source_dir/$name" "$accept_dir/$name"
  done

  local env_args=(
    TRADE_DATE="$date" TARGET_COL="$TARGET_COL" FEATURE_PRESET="$FEATURE_PRESET"
    OUT_ROOT="$ACCEPT_OUT_ROOT" UNIVERSE="$UNIVERSE"
    RAW_DAILY_CACHE_DIR="$RAW_DAILY_CACHE_DIR" MODEL_DATA="$MODEL_DATA"
    POSITIONS_FILE="$POSITIONS_FILE" SKIP_COLLECT=1 CAPACITY_MODE=none
    MIN_FEATURE_ROWS="$MIN_FEATURE_ROWS" MAX_FINALIZE_SECONDS="$MAX_FINALIZE_SECONDS"
    ALLOW_INDICATOR_FALLBACK="$ALLOW_INDICATOR_FALLBACK"
    ALLOW_MISSING_BUY_DATE="$ALLOW_MISSING_BUY_DATE"
  )
  [[ -z "$CASH" ]] || env_args+=(CASH="$CASH")
  [[ -z "$CASH_FILE" ]] || env_args+=(CASH_FILE="$CASH_FILE")
  [[ -z "$SELECTION_BACKTEST_ROOT" ]] || env_args+=(SELECTION_BACKTEST_ROOT="$SELECTION_BACKTEST_ROOT")
  [[ -z "$FOLD0_DIR" ]] || env_args+=(FOLD0_DIR="$FOLD0_DIR")

  static_check
  env "${env_args[@]}" bash scripts/run_as1455_live_strict_oos_pipeline.sh post
  validate_replay "$accept_dir" "$date"
  pass "AS1455 completed-date replay acceptance"
}

case "$MODE" in
  static) static_check ;;
  replay) replay ;;
  *)
    cat <<'EOF'
Usage:
  bash scripts/accept_as1455_live_strict_oos.sh static

  TRADE_DATE=20260722 \
  POSITIONS_FILE=/secure/positions_before_20260722.csv \
  CASH_FILE=/secure/cash_before_20260722.txt \
  bash scripts/accept_as1455_live_strict_oos.sh replay

The completed-date source directory must contain:
  03_adjustment_events.csv
  05_execution_calendar.csv
  06_live_feature_state_fast.npz
  08_live_raw_row_as1455.csv
EOF
    exit 2
    ;;
esac
