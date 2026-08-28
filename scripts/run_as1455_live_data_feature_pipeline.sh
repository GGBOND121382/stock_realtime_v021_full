#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-help}"
PYTHON="${PYTHON:-python3}"
TRADE_DATE="${TRADE_DATE:-today}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
HISTORY_START_DATE="${HISTORY_START_DATE:-2020-01-01}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/live_as1455}"
RAW_5M_CACHE_DIR="${RAW_5M_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
HISTORY_WORKERS="${HISTORY_WORKERS:-3}"
SYMBOL_RETRIES="${SYMBOL_RETRIES:-2}"
DRY_RUN="${DRY_RUN:-0}"
NO_BAOSTOCK_CALENDAR="${NO_BAOSTOCK_CALENDAR:-0}"
SKIP_AS1455_AGGREGATE="${SKIP_AS1455_AGGREGATE:-0}"
LOG_ROOT="${LOG_ROOT:-logs}"
SCRIPT_COMMON="features/as1455_live_common.py"
SCRIPT_BASE="pipelines/as1455_update_history_to_prevday.py"
SCRIPT_HISTORY="pipelines/as1455_update_history_to_prevday_fast_v4.py"
SCRIPT_DISPATCH="pipelines/as1455_history_parallel_dispatch.py"
PROJECT_PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_ROOT"

fail() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }

live_date() {
  PYTHONPATH="$PROJECT_PYTHONPATH" "$PYTHON" - "$TRADE_DATE" <<'PY'
import sys
from datetime import datetime
s = sys.argv[1]
if s.lower() == "today":
    print(datetime.now().strftime("%Y%m%d"))
else:
    s = s.replace("-", "")
    datetime.strptime(s, "%Y%m%d")
    print(s)
PY
}

check_files() {
  require_file "$SCRIPT_COMMON"
  require_file "$SCRIPT_BASE"
  require_file "$SCRIPT_HISTORY"
  require_file "$SCRIPT_DISPATCH"
  require_file "$UNIVERSE"
  PYTHONPATH="$PROJECT_PYTHONPATH" "$PYTHON" -m py_compile "$SCRIPT_COMMON" "$SCRIPT_BASE" "$SCRIPT_HISTORY" "$SCRIPT_DISPATCH"
  PYTHONPATH="$PROJECT_PYTHONPATH" "$PYTHON" - <<'PY'
import features.as1455_live_common
import pipelines.as1455_update_history_to_prevday
import pipelines.as1455_update_history_to_prevday_fast_v4
import pipelines.as1455_history_parallel_dispatch
print('[PASS] AS1455 Python imports resolved from repository root')
PY
}

print_context() {
  cat <<EOF
[CONFIG]
  MODE=$MODE
  PYTHON=$PYTHON
  PYTHONPATH=$PROJECT_PYTHONPATH
  TRADE_DATE=$TRADE_DATE
  HISTORY_END_DATE=$HISTORY_END_DATE
  HISTORY_START_DATE=$HISTORY_START_DATE
  UNIVERSE=$UNIVERSE
  RAW_5M_CACHE_DIR=$RAW_5M_CACHE_DIR
  RAW_DAILY_CACHE_DIR=$RAW_DAILY_CACHE_DIR
  AS1455_DAILY_CACHE_DIR=$AS1455_DAILY_CACHE_DIR
  MAX_SYMBOLS=${MAX_SYMBOLS:-<none>}
  HISTORY_WORKERS=$HISTORY_WORKERS
  SYMBOL_RETRIES=$SYMBOL_RETRIES
EOF
}

validate_history_report() {
  local report="$1"
  PYTHONPATH="$PROJECT_PYTHONPATH" "$PYTHON" - \
    "$report" "$RAW_DAILY_CACHE_DIR" "$AS1455_DAILY_CACHE_DIR" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

report_path = Path(sys.argv[1])
raw_daily_dir = Path(sys.argv[2])
as1455_dir = Path(sys.argv[3])
obj = json.loads(report_path.read_text(encoding="utf-8"))
history_end = pd.Timestamp(obj["history_end_date"]).normalize()
by_symbol_path = Path(obj.get("by_symbol_report", ""))
if not by_symbol_path.exists():
    raise SystemExit(f"missing by-symbol history report: {by_symbol_path}")

by_symbol = pd.read_csv(by_symbol_path, dtype={"symbol": str}, encoding="utf-8-sig", low_memory=False)
unresolved = [str(value) for value in obj.get("unresolved_symbols", [])]
accepted = []
remaining = []


def code_of(symbol: str) -> str:
    return "".join(ch for ch in symbol if ch.isdigit())[:6].zfill(6)


def latest_date(path: Path) -> pd.Timestamp | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        dates = pd.to_datetime(
            pd.read_csv(path, usecols=["date"], encoding="utf-8-sig")["date"],
            errors="coerce",
        ).dropna()
    except Exception:
        return None
    return pd.Timestamp(dates.max()).normalize() if not dates.empty else None


def raw_daily_path(code: str) -> Path | None:
    candidates = [
        raw_daily_dir / f"{code}_daily_raw.csv",
        raw_daily_dir / f"{code}_raw_daily.csv",
        raw_daily_dir / f"{code}.csv",
    ]
    return next((path for path in candidates if path.exists() and path.stat().st_size > 0), None)


def has_active_daily_after(path: Path, last_as1455: pd.Timestamp) -> bool:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    columns = [name for name in ["date", "tradestatus", "open", "high", "low", "close"] if name in header]
    if "date" not in columns:
        return True
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig", low_memory=False)
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    active = dates.notna() & dates.gt(last_as1455) & dates.le(history_end)
    if "tradestatus" in frame.columns:
        active &= pd.to_numeric(frame["tradestatus"], errors="coerce").eq(1)
    else:
        price_columns = [name for name in ["open", "high", "low", "close"] if name in frame.columns]
        if not price_columns:
            return True
        prices = frame[price_columns].apply(pd.to_numeric, errors="coerce")
        active &= prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
    return bool(active.any())


for symbol in unresolved:
    code = code_of(symbol)
    as_path = as1455_dir / f"{code}_as1455_daily.csv"
    last_as1455 = latest_date(as_path)
    daily_path = raw_daily_path(code)
    if last_as1455 is None or daily_path is None:
        remaining.append(symbol)
        continue
    try:
        later_active = has_active_daily_after(daily_path, last_as1455)
    except Exception:
        remaining.append(symbol)
        continue
    if later_active:
        remaining.append(symbol)
        continue

    accepted.append(symbol)
    mask = by_symbol["symbol"].astype(str).eq(symbol)
    by_symbol.loc[mask, "as1455_initial_status"] = by_symbol.loc[mask, "as1455_status"].astype(str)
    by_symbol.loc[mask, "as1455_status"] = "cached_no_new_trading_bars"
    by_symbol.loc[mask, "as1455_no_new_bars_reason"] = (
        "no tradestatus=1 raw-daily row after last AS1455 date"
    )
    if "raw_5m_status" in by_symbol.columns:
        empty_mask = mask & by_symbol["raw_5m_status"].astype(str).eq("empty")
        by_symbol.loc[empty_mask, "raw_5m_status"] = "empty_no_new_trading_bars"
    by_symbol.loc[mask, "as1455_error"] = ""
    by_symbol.loc[mask, "error"] = ""

by_symbol.to_csv(by_symbol_path, index=False, encoding="utf-8-sig")
obj["active_trading_day_validation"] = True
obj["accepted_no_new_trading_bars_symbols"] = accepted
obj["unresolved_symbols"] = remaining
obj["errors"] = len(remaining)
obj["raw_5m_status_counts"] = by_symbol.get("raw_5m_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
obj["raw_daily_status_counts"] = by_symbol.get("raw_daily_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
obj["as1455_status_counts"] = by_symbol.get("as1455_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
report_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

if accepted:
    print("[PASS] accepted no-new-trading-bar symbols: " + ",".join(accepted))

missing = [str(value) for value in obj.get("missing_as1455_symbols", [])]
if remaining or missing:
    raise SystemExit(
        f"history update incomplete: errors={len(remaining)} "
        f"unresolved={','.join(remaining) or '<none>'} "
        f"missing_as1455={','.join(missing) or '<none>'}; see {report_path}"
    )

print(
    f"[PASS] history caches updated through {obj.get('history_end_date')} "
    f"for {obj.get('n_symbols')} symbols with workers={obj.get('workers', 1)}; "
    f"AS1455 full-scan repair={obj.get('as1455_full_scan_recovered', 0)}/"
    f"{obj.get('as1455_full_scan_repair_candidates', 0)}"
)
PY
}

run_history() {
  check_files
  [[ "$HISTORY_WORKERS" =~ ^[1-8]$ ]] || fail "HISTORY_WORKERS must be an integer from 1 to 8"
  [[ "$SYMBOL_RETRIES" =~ ^[0-9]+$ ]] || fail "SYMBOL_RETRIES must be a non-negative integer"
  local args=(
    "$SCRIPT_DISPATCH"
    --trade-date "$TRADE_DATE"
    --history-end-date "$HISTORY_END_DATE"
    --history-start-date "$HISTORY_START_DATE"
    --universe "$UNIVERSE"
    --raw-5m-cache-dir "$RAW_5M_CACHE_DIR"
    --raw-daily-cache-dir "$RAW_DAILY_CACHE_DIR"
    --as1455-daily-cache-dir "$AS1455_DAILY_CACHE_DIR"
    --out-root "$OUT_ROOT"
    --sleep-seconds "$SLEEP_SECONDS"
    --workers "$HISTORY_WORKERS"
    --symbol-retries "$SYMBOL_RETRIES"
  )
  [[ -n "$MAX_SYMBOLS" ]] && args+=(--max-symbols "$MAX_SYMBOLS")
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  [[ "$NO_BAOSTOCK_CALENDAR" == "1" ]] && args+=(--no-baostock-calendar)
  [[ "$SKIP_AS1455_AGGREGATE" == "1" ]] && args+=(--skip-as1455-aggregate)

  local stamp log report
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_ROOT/as1455_history_${stamp}.log"
  print_context
  PYTHONPATH="$PROJECT_PYTHONPATH" "$PYTHON" "${args[@]}" 2>&1 | tee "$log"

  if [[ "$DRY_RUN" != "1" ]]; then
    report="$OUT_ROOT/$(live_date)/00_history_update_report.json"
    require_file "$report"
    validate_history_report "$report"
  fi
}

status() {
  local report="$OUT_ROOT/$(live_date)/00_history_update_report.json"
  print_context
  if [[ -s "$report" ]]; then
    cat "$report"
  else
    echo "[MISSING] $report"
  fi
}

case "$MODE" in
  history) run_history ;;
  check) check_files; echo "[PASS] AS1455 history pipeline check passed" ;;
  status) status ;;
  help|-h|--help)
    cat <<'EOF'
Usage:
  bash scripts/run_as1455_live_data_feature_pipeline.sh history
  bash scripts/run_as1455_live_data_feature_pipeline.sh check
  bash scripts/run_as1455_live_data_feature_pipeline.sh status

Parallel defaults:
  HISTORY_WORKERS=3 SYMBOL_RETRIES=2 bash scripts/run_as1455_live_data_feature_pipeline.sh history
EOF
    ;;
  *) fail "unknown mode: $MODE" ;;
esac
