#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
HISTORY_WORKERS="${HISTORY_WORKERS:-2}"
MAX_HISTORY_PASSES="${MAX_HISTORY_PASSES:-5}"
WORKER_START_DELAY_SECONDS="${WORKER_START_DELAY_SECONDS:-2}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
HISTORY_SLEEP_SECONDS="${HISTORY_SLEEP_SECONDS:-0.02}"
HISTORY_START_DATE="${HISTORY_START_DATE:-2020-01-02}"
HISTORY_END_DATE="${HISTORY_END_DATE:-auto}"
TRADE_DATE="${TRADE_DATE:-today}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
CH12_DIR="${CH12_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
RAW_5M_DIR="${RAW_5M_DIR:-$CH12_DIR/baostock_5m_cache}"
RAW_DAILY_DIR="${RAW_DAILY_DIR:-$CH12_DIR/baostock_raw_daily_cache}"
AS1455_DAILY_DIR="${AS1455_DAILY_DIR:-$CH12_DIR/as1455_daily_cache}"
REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"
STATE_DIR="${STATE_DIR:-$REBUILD_ROOT/state}"
LOG_DIR="${LOG_DIR:-$REBUILD_ROOT/logs}"
HISTORY_REPORT_ROOT="${HISTORY_REPORT_ROOT:-$REBUILD_ROOT/history_reports}"
SHARD_ROOT="${SHARD_ROOT:-$REBUILD_ROOT/history_shards}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

[[ "$HISTORY_WORKERS" =~ ^[1-4]$ ]] || fail "HISTORY_WORKERS must be 1..4, got $HISTORY_WORKERS"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "python is unavailable: $PYTHON_BIN"
[[ -f "$UNIVERSE" ]] || fail "missing universe: $UNIVERSE"
[[ -f pipelines/as1455_update_history_to_prevday_fast_v4.py ]] || fail "missing fast_v4 updater"

mkdir -p \
  "$RAW_5M_DIR" \
  "$RAW_DAILY_DIR" \
  "$AS1455_DAILY_DIR" \
  "$STATE_DIR" \
  "$LOG_DIR" \
  "$HISTORY_REPORT_ROOT" \
  "$SHARD_ROOT"

TRADE_DATE_KEY="$($PYTHON_BIN - "$TRADE_DATE" <<'PY'
from datetime import datetime
import sys
value = sys.argv[1]
if value.lower() == 'today':
    print(datetime.now().strftime('%Y%m%d'))
else:
    value = value.replace('-', '')
    datetime.strptime(value, '%Y%m%d')
    print(value)
PY
)"

split_universe() {
  local pass_dir="$1"
  mkdir -p "$pass_dir/universe"
  "$PYTHON_BIN" - "$UNIVERSE" "$pass_dir/universe" "$HISTORY_WORKERS" <<'PY'
from pathlib import Path
import pandas as pd
import sys

source = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
workers = int(sys.argv[3])
df = pd.read_csv(source, dtype={'code': str}, encoding='utf-8-sig')
if len(df) != 1000:
    raise SystemExit(f'expected 1000 universe rows, got {len(df)}')
if df['code'].astype(str).str.zfill(6).nunique() != 1000:
    raise SystemExit('universe codes are not 1000 unique values')
for worker in range(workers):
    shard = df.iloc[worker::workers].copy()
    path = out_dir / f'worker_{worker:02d}.csv'
    shard.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'[SHARD] worker={worker} rows={len(shard)} path={path}', flush=True)
PY
}

aggregate_reports() {
  local pass_dir="$1"
  "$PYTHON_BIN" - \
    "$UNIVERSE" \
    "$pass_dir" \
    "$HISTORY_REPORT_ROOT" \
    "$TRADE_DATE_KEY" \
    "$HISTORY_WORKERS" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import sys

universe_path = Path(sys.argv[1])
pass_dir = Path(sys.argv[2])
central_root = Path(sys.argv[3])
trade_date = sys.argv[4]
workers = int(sys.argv[5])
central = central_root / trade_date
central.mkdir(parents=True, exist_ok=True)
universe = pd.read_csv(universe_path, dtype={'code': str}, encoding='utf-8-sig')
universe.to_csv(central / '01_universe.csv', index=False, encoding='utf-8-sig')

parts = []
missing_worker_reports = []
for worker in range(workers):
    report = pass_dir / f'worker_{worker:02d}' / trade_date / '00_history_update_by_symbol.csv'
    if report.is_file() and report.stat().st_size > 0:
        part = pd.read_csv(report, dtype={'symbol': str}, encoding='utf-8-sig')
        part['parallel_worker'] = worker
        parts.append(part)
    else:
        missing_worker_reports.append(worker)

if parts:
    frame = pd.concat(parts, ignore_index=True, sort=False)
    if 'symbol' in frame:
        frame['symbol'] = frame['symbol'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(6).str[-6:]
        frame = frame.drop_duplicates('symbol', keep='last')
else:
    frame = pd.DataFrame()

expected_symbols = universe['code'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(6).str[-6:]
seen = set(frame['symbol']) if 'symbol' in frame else set()
missing_symbols = sorted(set(expected_symbols) - seen)
if missing_symbols:
    synthetic = pd.DataFrame({
        'symbol': missing_symbols,
        'error': ['missing worker report or symbol result'] * len(missing_symbols),
        'raw_5m_status': ['error'] * len(missing_symbols),
        'raw_daily_status': ['error'] * len(missing_symbols),
        'as1455_status': ['error'] * len(missing_symbols),
    })
    frame = pd.concat([frame, synthetic], ignore_index=True, sort=False)

if 'error' not in frame:
    frame['error'] = ''
frame['error'] = frame['error'].fillna('').astype(str)
frame = frame.sort_values('symbol').reset_index(drop=True)
frame.to_csv(central / '00_history_update_by_symbol.csv', index=False, encoding='utf-8-sig')

summary = {
    'trade_date': trade_date,
    'parallel_sharded_rebuild': True,
    'history_workers': workers,
    'n_symbols': int(len(frame)),
    'errors': int(frame['error'].str.strip().ne('').sum()),
    'missing_worker_reports': missing_worker_reports,
    'missing_symbol_results': len(missing_symbols),
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
}
for column in ['raw_5m_status', 'raw_daily_status', 'as1455_status']:
    if column in frame:
        summary[f'{column}_counts'] = frame[column].fillna('').astype(str).value_counts(dropna=False).to_dict()
(central / '00_history_update_report.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

validate_complete() {
  "$PYTHON_BIN" - \
    "$HISTORY_REPORT_ROOT/$TRADE_DATE_KEY/00_history_update_report.json" \
    "$RAW_5M_DIR" \
    "$RAW_DAILY_DIR" \
    "$AS1455_DAILY_DIR" <<'PY'
from pathlib import Path
import csv
import json
import sys

report_path = Path(sys.argv[1])
raw5, rawd, asd = map(Path, sys.argv[2:])
report = json.loads(report_path.read_text(encoding='utf-8'))
if int(report.get('n_symbols', 0)) != 1000 or int(report.get('errors', -1)) != 0:
    raise SystemExit(f'history report incomplete: {report}')
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
        raise SystemExit(f'{name}: expected 1000 files, got {len(files)}')
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
    print(f'[OK] {name}: files=1000', flush=True)
PY
}

for pass in $(seq 1 "$MAX_HISTORY_PASSES"); do
  pass_dir="$SHARD_ROOT/pass_${pass}"
  mkdir -p "$pass_dir"
  split_universe "$pass_dir"

  printf '[INFO] parallel history pass %s/%s workers=%s\n' \
    "$pass" "$MAX_HISTORY_PASSES" "$HISTORY_WORKERS"

  pids=()
  for worker in $(seq 0 $((HISTORY_WORKERS - 1))); do
    worker_id="$(printf '%02d' "$worker")"
    worker_root="$pass_dir/worker_${worker_id}"
    worker_universe="$pass_dir/universe/worker_${worker_id}.csv"
    worker_log="$LOG_DIR/history_pass_${pass}_worker_${worker_id}.log"
    mkdir -p "$worker_root"
    printf '[START] pass=%s worker=%s universe=%s log=%s\n' \
      "$pass" "$worker" "$worker_universe" "$worker_log"
    (
      "$PYTHON_BIN" pipelines/as1455_update_history_to_prevday_fast_v4.py \
        --trade-date "$TRADE_DATE" \
        --history-end-date "$HISTORY_END_DATE" \
        --history-start-date "$HISTORY_START_DATE" \
        --universe "$worker_universe" \
        --raw-5m-cache-dir "$RAW_5M_DIR" \
        --raw-daily-cache-dir "$RAW_DAILY_DIR" \
        --as1455-daily-cache-dir "$AS1455_DAILY_DIR" \
        --out-root "$worker_root" \
        --sleep-seconds "$HISTORY_SLEEP_SECONDS"
    ) >"$worker_log" 2>&1 &
    pids+=("$!")
    sleep "$WORKER_START_DELAY_SECONDS"
  done

  while :; do
    running=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
    raw5_count="$(find "$RAW_5M_DIR" -maxdepth 1 -type f -name '*_5m_raw.csv' 2>/dev/null | wc -l)"
    rawd_count="$(find "$RAW_DAILY_DIR" -maxdepth 1 -type f -name '*_daily_raw.csv' 2>/dev/null | wc -l)"
    asd_count="$(find "$AS1455_DAILY_DIR" -maxdepth 1 -type f -name '*_as1455_daily.csv' 2>/dev/null | wc -l)"
    printf '[PROGRESS] pass=%s running_workers=%s raw5=%s raw_daily=%s as1455_daily=%s time=%s\n' \
      "$pass" "$running" "$raw5_count" "$rawd_count" "$asd_count" "$(date -Is)"
    (( running == 0 )) && break
    sleep "$PROGRESS_INTERVAL_SECONDS"
  done

  worker_failures=0
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    if wait "$pid"; then
      printf '[DONE] pass=%s worker=%s rc=0\n' "$pass" "$index"
    else
      rc=$?
      worker_failures=$((worker_failures + 1))
      printf '[WARN] pass=%s worker=%s rc=%s\n' "$pass" "$index" "$rc" >&2
    fi
  done

  aggregate_reports "$pass_dir"
  if validate_complete; then
    {
      printf 'stage=history\n'
      printf 'completed_at=%s\n' "$(date -Is)"
      printf 'history_workers=%s\n' "$HISTORY_WORKERS"
      printf 'parallel_sharded_rebuild=true\n'
    } > "$STATE_DIR/history.done"
    printf '[PASS] parallel AS1455 history rebuild completed\n'
    exit 0
  fi

  printf '[WARN] history pass %s incomplete; worker_failures=%s; retrying cached tails\n' \
    "$pass" "$worker_failures" >&2
  [[ "$pass" -lt "$MAX_HISTORY_PASSES" ]] || fail "history incomplete after $MAX_HISTORY_PASSES passes"
done
