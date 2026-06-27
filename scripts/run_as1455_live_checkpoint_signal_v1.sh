#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
TZ="${TIMEZONE:-Asia/Shanghai}"
export TZ

TRADE_DATE="${TRADE_DATE:-today}"
if [[ "$TRADE_DATE" == "today" ]]; then
  LIVE_DATE="$(date +%Y%m%d)"
else
  LIVE_DATE="$TRADE_DATE"
fi

OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
LIVE_DIR="${LIVE_DIR:-$OUT_ROOT/$LIVE_DATE}"

TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7}"
MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
DEPLOY_DIR="${DEPLOY_DIR:-saved_data/ashare_ml4t/ch17_as1455_deploy/sharpe1_checkpoint_ensemble_all5_v1}"

MODEL_ROWS="${MODEL_ROWS:-0,1,2,3,4}"
FOLDS="${FOLDS:-0,1,2,3,4,5,6}"
FOLD_MODE="${FOLD_MODE:-mean_all_folds}"
SINGLE_FOLD="${SINGLE_FOLD:-0}"
FORCE_REBALANCE="${FORCE_REBALANCE:-0}"
SIGNAL_CASH="${SIGNAL_CASH:-}"
PORTFOLIO_VALUE="${PORTFOLIO_VALUE:-}"
BUY_CASH_PER_POSITION="${BUY_CASH_PER_POSITION:-}"
CASH_BUFFER_PCT="${CASH_BUFFER_PCT:-0}"
LOT_SIZE="${LOT_SIZE:-100}"
REBALANCE_EVERY="${REBALANCE_EVERY:-3}"
REBALANCE_OFFSET="${REBALANCE_OFFSET:-0}"
DAY_INDEX="${DAY_INDEX:-}"
REBALANCE_CALENDAR="${REBALANCE_CALENDAR:-}"
CALENDAR_UNKNOWN_POLICY="${CALENDAR_UNKNOWN_POLICY:-force}"
UNKNOWN_BUY_DATE_POLICY="${UNKNOWN_BUY_DATE_POLICY:-allow}"
SKIP_REBALANCE="${SKIP_REBALANCE:-0}"
COMMISSION_RATE="${COMMISSION_RATE:-0.000085}"
STAMP_TAX_RATE="${STAMP_TAX_RATE:-0.0005}"
TRANSFER_FEE_RATE="${TRANSFER_FEE_RATE:-0.00001}"
MIN_COMMISSION="${MIN_COMMISSION:-5}"
SLIPPAGE_BPS="${SLIPPAGE_BPS:-0}"
PROFILE="${PROFILE:-close_auction_skip_limit}"
CAPACITY_MODE="${CAPACITY_MODE:-none}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-0.05}"
MAINBOARD_ONLY="${MAINBOARD_ONLY:-1}"
EXCLUDE_ST="${EXCLUDE_ST:-0}"
MIN_PRICE="${MIN_PRICE:-0}"
LIMIT_EPS="${LIMIT_EPS:-0.000001}"
PRICE_FILE="${PRICE_FILE:-}"
PRICE_COLUMN="${PRICE_COLUMN:-}"
POSITIONS_FILE="${POSITIONS_FILE:-}"
AUTO_STATE="${AUTO_STATE:-1}"
STATE_FILE="${STATE_FILE:-$OUT_ROOT/checkpoint_signal_paper_state.json}"
STATE_POSITIONS_CSV="${STATE_POSITIONS_CSV:-$OUT_ROOT/current_positions.csv}"
STATE_CASH_FILE="${STATE_CASH_FILE:-$OUT_ROOT/current_cash.txt}"
INITIAL_CASH="${INITIAL_CASH:-200000}"
RESET_STATE="${RESET_STATE:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Recreate by default so MODEL_ROWS/FOLDS/FOLD_MODE changes cannot be ignored by a stale manifest.
RECREATE_DEPLOY="${RECREATE_DEPLOY:-1}"

echo "[CONFIG]"
echo "  LIVE_DATE=$LIVE_DATE"
echo "  LIVE_DIR=$LIVE_DIR"
echo "  TRAIN_RUN_DIR=$TRAIN_RUN_DIR"
echo "  MODEL_DATA=$MODEL_DATA"
echo "  DEPLOY_DIR=$DEPLOY_DIR"
echo "  MODEL_ROWS=$MODEL_ROWS"
echo "  FOLDS=$FOLDS"
echo "  FOLD_MODE=$FOLD_MODE"
echo "  SINGLE_FOLD=$SINGLE_FOLD"
echo "  FORCE_REBALANCE=$FORCE_REBALANCE"
echo "  SIGNAL_CASH=${SIGNAL_CASH:-<none>}"
echo "  PORTFOLIO_VALUE=${PORTFOLIO_VALUE:-<none>}"
echo "  BUY_CASH_PER_POSITION=${BUY_CASH_PER_POSITION:-<none>}"
echo "  CASH_BUFFER_PCT=$CASH_BUFFER_PCT"
echo "  LOT_SIZE=$LOT_SIZE"
echo "  REBALANCE_EVERY=$REBALANCE_EVERY"
echo "  REBALANCE_OFFSET=$REBALANCE_OFFSET"
echo "  DAY_INDEX=${DAY_INDEX:-<none>}"
echo "  REBALANCE_CALENDAR=${REBALANCE_CALENDAR:-<none>}"
echo "  CALENDAR_UNKNOWN_POLICY=$CALENDAR_UNKNOWN_POLICY"
echo "  UNKNOWN_BUY_DATE_POLICY=$UNKNOWN_BUY_DATE_POLICY"
echo "  SKIP_REBALANCE=$SKIP_REBALANCE"
echo "  COMMISSION_RATE=$COMMISSION_RATE"
echo "  STAMP_TAX_RATE=$STAMP_TAX_RATE"
echo "  TRANSFER_FEE_RATE=$TRANSFER_FEE_RATE"
echo "  MIN_COMMISSION=$MIN_COMMISSION"
echo "  SLIPPAGE_BPS=$SLIPPAGE_BPS"
echo "  PROFILE=$PROFILE"
echo "  CAPACITY_MODE=$CAPACITY_MODE"
echo "  PARTICIPATION_RATE=$PARTICIPATION_RATE"
echo "  MAINBOARD_ONLY=$MAINBOARD_ONLY"
echo "  EXCLUDE_ST=$EXCLUDE_ST"
echo "  MIN_PRICE=$MIN_PRICE"
echo "  LIMIT_EPS=$LIMIT_EPS"
echo "  PRICE_FILE=${PRICE_FILE:-<auto>}"
echo "  PRICE_COLUMN=${PRICE_COLUMN:-<auto>}"
echo "  POSITIONS_FILE=${POSITIONS_FILE:-<default>}"
echo "  AUTO_STATE=$AUTO_STATE"
echo "  STATE_FILE=$STATE_FILE"
echo "  STATE_POSITIONS_CSV=$STATE_POSITIONS_CSV"
echo "  STATE_CASH_FILE=$STATE_CASH_FILE"
echo "  INITIAL_CASH=$INITIAL_CASH"
echo "  RESET_STATE=$RESET_STATE"
echo "  DRY_RUN=$DRY_RUN"
echo "  RECREATE_DEPLOY=$RECREATE_DEPLOY"

if [[ "$RECREATE_DEPLOY" == "1" || ! -f "$DEPLOY_DIR/manifest.json" ]]; then
  echo "[INFO] creating checkpoint deploy bundle"
  "$PYTHON" tools/create_as1455_sharpe1_checkpoint_bundle_v1.py \
    --train-run-dir "$TRAIN_RUN_DIR" \
    --out-dir "$DEPLOY_DIR" \
    --model-data "$MODEL_DATA" \
    --model-rows "$MODEL_ROWS" \
    --folds "$FOLDS" \
    --fold-mode "$FOLD_MODE" \
    --single-fold "$SINGLE_FOLD" \
    --force
else
  echo "[INFO] using existing deploy bundle: $DEPLOY_DIR/manifest.json"
fi

INFER_ARGS=(
  --live-dir "$LIVE_DIR"
  --deploy-dir "$DEPLOY_DIR"
  --model-data "$MODEL_DATA"
  --fold-mode "$FOLD_MODE"
)

if [[ "$FOLD_MODE" == "single_fold" ]]; then
  INFER_ARGS+=(--single-fold "$SINGLE_FOLD")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  INFER_ARGS+=(--dry-run)
fi

echo "[INFO] running checkpoint ensemble inference"
"$PYTHON" prediction/run_as1455_live_checkpoint_ensemble_inference_v1.py "${INFER_ARGS[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] dry-run done; skip rank/signal"
  exit 0
fi

echo "[INFO] ranking predictions"
"$PYTHON" trading/rank_as1455_live_predictions_v1.py \
  --live-dir "$LIVE_DIR"

# Automatic planned-state management for multi-day replay/live dry-run.
# It stores a paper state separately from broker truth. The state is updated
# only after signal generation succeeds and assumes planned orders are filled.
STATE_POSITIONS_BEFORE="$LIVE_DIR/00_state_positions_before_signal.csv"
STATE_BEFORE_REPORT="$LIVE_DIR/00_state_before_signal.json"
if [[ "$AUTO_STATE" == "1" ]]; then
  mkdir -p "$LIVE_DIR" "$OUT_ROOT"
  echo "[INFO] preparing auto state"
  AUTO_STATE_JSON=$(STATE_FILE="$STATE_FILE" \
    STATE_POSITIONS_BEFORE="$STATE_POSITIONS_BEFORE" \
    STATE_BEFORE_REPORT="$STATE_BEFORE_REPORT" \
    SIGNAL_CASH_ENV="${SIGNAL_CASH:-}" \
    INITIAL_CASH="$INITIAL_CASH" \
    RESET_STATE="$RESET_STATE" \
    MANUAL_POSITIONS_FILE="${POSITIONS_FILE:-}" \
    "$PYTHON" - <<'PY_AUTOSTATE'
import json, os, re
from pathlib import Path
import pandas as pd

def normalize_symbol(value):
    s=str(value).strip()
    if not s or s.lower()=="nan": return ""
    s=s.replace(".XSHE",".SZ").replace(".XSHG",".SH")
    m=re.search(r"(\d{6})", s)
    if m: code=m.group(1)
    elif re.fullmatch(r"\d{1,6}", s): code=s.zfill(6)
    else: return s.upper()
    return f"{code}.SH" if code.startswith(("6","9")) else f"{code}.SZ"

def clean_pos(rows):
    df=pd.DataFrame(rows or [])
    if df.empty:
        return pd.DataFrame(columns=["symbol","shares","buy_date","avg_entry_price"])
    if "symbol" not in df.columns:
        if "code" in df.columns: df["symbol"]=df["code"]
        else: return pd.DataFrame(columns=["symbol","shares","buy_date","avg_entry_price"])
    if "shares" not in df.columns: df["shares"]=0
    df["symbol"]=df["symbol"].map(normalize_symbol)
    df["shares"]=pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    if "buy_date" not in df.columns:
        for c in ["entry_date","date_bought","open_date","date"]:
            if c in df.columns:
                df["buy_date"]=df[c]; break
    if "buy_date" not in df.columns: df["buy_date"]=""
    if "avg_entry_price" not in df.columns:
        for c in ["cost_price","entry_price","price","last_price"]:
            if c in df.columns:
                df["avg_entry_price"]=df[c]; break
    if "avg_entry_price" not in df.columns: df["avg_entry_price"]=pd.NA
    df["avg_entry_price"]=pd.to_numeric(df["avg_entry_price"], errors="coerce")
    df=df[(df["symbol"].astype(str).str.len()>0)&(df["shares"]>0)]
    return df[["symbol","shares","buy_date","avg_entry_price"]].drop_duplicates("symbol", keep="last")

state_file=Path(os.environ["STATE_FILE"])
pos_before=Path(os.environ["STATE_POSITIONS_BEFORE"])
report_file=Path(os.environ["STATE_BEFORE_REPORT"])
manual_pos=os.environ.get("MANUAL_POSITIONS_FILE","")
reset=os.environ.get("RESET_STATE","0")=="1"
signal_cash_env=os.environ.get("SIGNAL_CASH_ENV","")
initial_cash=float(os.environ.get("INITIAL_CASH","200000"))
if reset:
    positions=clean_pos([])
    cash=initial_cash
    source="reset_initial_cash"
elif state_file.exists():
    state=json.loads(state_file.read_text(encoding="utf-8"))
    positions=clean_pos(state.get("positions", []))
    cash=float(state.get("cash", initial_cash))
    source="state_file"
elif manual_pos and Path(manual_pos).exists():
    positions=clean_pos(pd.read_csv(manual_pos).to_dict("records"))
    cash=initial_cash
    source="manual_positions_initial_cash"
else:
    positions=clean_pos([])
    cash=initial_cash
    source="new_initial_cash"
if signal_cash_env.strip():
    cash=float(signal_cash_env)
    source += "+signal_cash_override"
pos_before.parent.mkdir(parents=True, exist_ok=True)
positions.to_csv(pos_before, index=False, encoding="utf-8-sig")
report={"passed": True, "state_source": source, "state_file": str(state_file), "positions_file_for_signal": str(pos_before), "cash_for_signal": cash, "n_positions": int(len(positions)), "reset_state": reset}
report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"positions_file": str(pos_before), "cash": cash, "state_source": source}, ensure_ascii=False))
PY_AUTOSTATE
  )
  echo "$AUTO_STATE_JSON"
  POSITIONS_FILE="$STATE_POSITIONS_BEFORE"
  SIGNAL_CASH=$(AUTO_STATE_JSON="$AUTO_STATE_JSON" "$PYTHON" - <<'PY_CASH'
import json, os
x=json.loads(os.environ["AUTO_STATE_JSON"])
print(x["cash"])
PY_CASH
)
fi

SIGNAL_ARGS=(
  --live-dir "$LIVE_DIR"
  --cash-buffer-pct "$CASH_BUFFER_PCT"
  --lot-size "$LOT_SIZE"
  --rebalance-every "$REBALANCE_EVERY"
  --rebalance-offset "$REBALANCE_OFFSET"
  --calendar-unknown-policy "$CALENDAR_UNKNOWN_POLICY"
  --unknown-buy-date-policy "$UNKNOWN_BUY_DATE_POLICY"
  --commission-rate "$COMMISSION_RATE"
  --stamp-tax-rate "$STAMP_TAX_RATE"
  --transfer-fee-rate "$TRANSFER_FEE_RATE"
  --min-commission "$MIN_COMMISSION"
  --slippage-bps "$SLIPPAGE_BPS"
  --profile "$PROFILE"
  --capacity-mode "$CAPACITY_MODE"
  --participation-rate "$PARTICIPATION_RATE"
  --min-price "$MIN_PRICE"
  --limit-eps "$LIMIT_EPS"
)
if [[ "$FORCE_REBALANCE" == "1" ]]; then
  SIGNAL_ARGS+=(--force-rebalance)
fi
if [[ "$SKIP_REBALANCE" == "1" ]]; then
  SIGNAL_ARGS+=(--skip-rebalance)
fi
if [[ -n "$DAY_INDEX" ]]; then
  SIGNAL_ARGS+=(--day-index "$DAY_INDEX")
fi
if [[ -n "$REBALANCE_CALENDAR" ]]; then
  SIGNAL_ARGS+=(--rebalance-calendar "$REBALANCE_CALENDAR")
fi
if [[ "$MAINBOARD_ONLY" == "1" ]]; then
  SIGNAL_ARGS+=(--mainboard-only)
else
  SIGNAL_ARGS+=(--no-mainboard-only)
fi
if [[ "$EXCLUDE_ST" == "1" ]]; then
  SIGNAL_ARGS+=(--exclude-st)
else
  SIGNAL_ARGS+=(--no-exclude-st)
fi
if [[ -n "$SIGNAL_CASH" ]]; then
  SIGNAL_ARGS+=(--cash "$SIGNAL_CASH")
fi
if [[ -n "$PORTFOLIO_VALUE" ]]; then
  SIGNAL_ARGS+=(--portfolio-value "$PORTFOLIO_VALUE")
fi
if [[ -n "$BUY_CASH_PER_POSITION" ]]; then
  SIGNAL_ARGS+=(--buy-cash-per-position "$BUY_CASH_PER_POSITION")
fi
if [[ -n "$PRICE_FILE" ]]; then
  SIGNAL_ARGS+=(--price-file "$PRICE_FILE")
fi
if [[ -n "$PRICE_COLUMN" ]]; then
  SIGNAL_ARGS+=(--price-column "$PRICE_COLUMN")
fi
if [[ -n "$POSITIONS_FILE" ]]; then
  SIGNAL_ARGS+=(--positions-file "$POSITIONS_FILE")
fi

echo "[INFO] generating trade signal"
"$PYTHON" trading/generate_as1455_live_trade_signal_v1.py "${SIGNAL_ARGS[@]}"

if [[ "$AUTO_STATE" == "1" ]]; then
  echo "[INFO] updating auto state from planned signal"
  "$PYTHON" trading/update_as1455_live_state_from_signal_v1.py \
    --state-file "$STATE_FILE" \
    --positions-before "$STATE_POSITIONS_BEFORE" \
    --signal-file "$LIVE_DIR/16_live_trade_signal.csv" \
    --signal-report "$LIVE_DIR/16_live_trade_signal_report.json" \
    --out-positions-csv "$STATE_POSITIONS_CSV" \
    --out-cash-file "$STATE_CASH_FILE"
fi

REPORT="$LIVE_DIR/17_live_signal_pipeline_report.json"
"$PYTHON" - <<PY
import json
from pathlib import Path
live = Path("$LIVE_DIR")
parts = {}
for name in ["14_live_predictions_report.json", "15_live_rank_report.json", "16_live_trade_signal_report.json"]:
    p = live / name
    parts[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"passed": False, "missing": str(p)}
passed = all(bool(v.get("passed")) for v in parts.values())
out = {"passed": passed, "live_dir": str(live), "parts": parts}
(live / "17_live_signal_pipeline_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"passed": passed, "report": str(live / "17_live_signal_pipeline_report.json")}, ensure_ascii=False, indent=2))
PY

echo "[OK] checkpoint signal pipeline finished"
ls -lh "$LIVE_DIR"/14_live_predictions.csv "$LIVE_DIR"/15_live_rank.csv "$LIVE_DIR"/16_live_trade_signal.csv "$REPORT"
