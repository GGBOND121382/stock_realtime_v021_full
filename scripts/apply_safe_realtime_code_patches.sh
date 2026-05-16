#!/usr/bin/env bash
set -Eeuo pipefail

# scripts/apply_safe_realtime_code_patches.sh
#
# Applies code-only safe patches:
#   - score-now prefers 5min over 1min
#   - build-bars command forces --freqs 5min
#   - default realtime spot source removes disabled em
#
# Does not delete data.

PYTHON="${PYTHON:-python3}"
BACKUP_ROOT="${BACKUP_ROOT:-saved_data/patch_backups/realtime_code_safe_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BACKUP_ROOT"

"$PYTHON" - "$BACKUP_ROOT" <<'PY'
from pathlib import Path
import sys
backup_root = Path(sys.argv[1])

def backup(p: Path):
    if p.exists():
        dst = backup_root / (str(p) + ".bak")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
        print(f"[BACKUP] {p} -> {dst}")

p = Path("pipelines/run_intraday_nextday_signals.py")
if p.exists():
    s = p.read_text(encoding="utf-8")
    old = 'candidates = [base / "minute_bars_1min.csv", base / "minute_bars_5min.csv"]'
    new = 'candidates = [base / "minute_bars_5min.csv", base / "minute_bars_1min.csv"]'
    if old in s:
        backup(p); p.write_text(s.replace(old, new), encoding="utf-8")
        print("[PATCHED] score-now prefers minute_bars_5min.csv")
    elif new in s:
        print("[SKIP] score-now already prefers 5min")
    else:
        print("[WARN] score-now candidate pattern not found")

p = Path("pipelines/run_trading_day_signal_pipeline.py")
if p.exists():
    s = p.read_text(encoding="utf-8")
    if '"--freqs", "5min"' in s or "'--freqs', '5min'" in s:
        print("[SKIP] build-bars already has --freqs 5min")
    elif '"build-bars",' in s:
        backup(p)
        s = s.replace('"build-bars",\n', '"build-bars",\n            "--freqs", "5min",\n', 1)
        p.write_text(s, encoding="utf-8")
        print("[PATCHED] build-bars command adds --freqs 5min")
    else:
        print("[WARN] build-bars pattern not found")

p = Path("data_collection/collect_akshare_l1_cache.py")
if p.exists():
    s = p.read_text(encoding="utf-8")
    changed = False
    for old, new in [
        ('default="sina,ths,em,xq"', 'default="sina_batch,ths_etf,xq"'),
        ('${SPOT_SOURCE_PRIORITY:-sina,ths,em,xq}', '${SPOT_SOURCE_PRIORITY:-sina_batch,ths_etf,xq}'),
    ]:
        if old in s:
            s = s.replace(old, new)
            changed = True
    if changed:
        backup(p); p.write_text(s, encoding="utf-8")
        print("[PATCHED] realtime spot default source")
    else:
        print("[SKIP/WARN] spot default pattern absent or already safe")
PY

"$PYTHON" -m py_compile \
  pipelines/run_intraday_nextday_signals.py \
  pipelines/run_trading_day_signal_pipeline.py \
  data_collection/collect_akshare_l1_cache.py

echo "[DONE] realtime safe code patches"
