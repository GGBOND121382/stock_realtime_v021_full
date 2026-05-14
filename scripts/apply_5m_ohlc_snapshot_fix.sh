#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TARGET="pipelines/run_trading_day_signal_pipeline.py"

echo "============================================================"
echo "[PATCH] rebuild local 5m OHLCV from snapshots"
echo "============================================================"

test -f tools/fix_5m_ohlcv_from_snapshots.py || {
  echo "[ERROR] missing tools/fix_5m_ohlcv_from_snapshots.py"
  echo "Did you unzip the patch at project root?"
  exit 2
}

if [[ ! -f "$TARGET" ]]; then
  echo "[ERROR] missing $TARGET"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "backups/5m_ohlc_snapshot_fix_${TS}"
cp "$TARGET" "backups/5m_ohlc_snapshot_fix_${TS}/run_trading_day_signal_pipeline.py"

"$PYTHON" - <<'PY'
from pathlib import Path

p = Path("pipelines/run_trading_day_signal_pipeline.py")
s = p.read_text(encoding="utf-8")

# Replace older cumulative-volume fix command if it exists.
s2 = s.replace("tools/fix_5m_cumulative_volume_amount.py", "tools/fix_5m_ohlcv_from_snapshots.py")
s2 = s2.replace("fix_5m_cumulative_volume_amount", "fix_5m_ohlcv_from_snapshots")
s2 = s2.replace("Fix 5m bars built from cumulative snapshot volume/amount.", "Rebuild local 5m OHLCV bars from snapshots.")

if "tools/fix_5m_ohlcv_from_snapshots.py" not in s2:
    old = '                mark("build_bars", "ok")\n'
    insert = '''                mark("build_bars", "ok")

                # Rebuild local 5m OHLCV bars from snapshots.
                # Snapshot high/low can be day-to-date and volume/amount cumulative.
                # A 5m bar must use sampled-price OHLC and per-bar volume/amount delta.
                fix_5m_cmd = [
                    python, "tools/fix_5m_ohlcv_from_snapshots.py",
                    "--date", trade_date,
                    "--cache-dir", str(realtime_cache_dir),
                    "--cutoff-time", args.cutoff_time,
                    "--symbols-file", str(effective_watchlist),
                ]
                run_cmd(fix_5m_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)
                mark("fix_5m_ohlcv_from_snapshots", "ok")
'''
    if old not in s2:
        raise SystemExit('[ERROR] cannot find build_bars success marker. Patch manually around mark("build_bars", "ok").')
    s2 = s2.replace(old, insert, 1)

p.write_text(s2, encoding="utf-8")
print("[OK] pipeline patched")

cmp_path = Path("tools/compare_collected_vs_baostock.py")
if cmp_path.exists():
    t = cmp_path.read_text(encoding="utf-8")
    old = 'syms = [p.name for p in day_dir.iterdir() if p.is_dir()]'
    new = 'syms = [p.name for p in day_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]'
    if old in t:
        cmp_path.write_text(t.replace(old, new, 1), encoding="utf-8")
        print("[OK] compare tool patched to skip _report dirs")
PY

echo "[CHECK] py_compile"
"$PYTHON" -m py_compile "$TARGET" tools/fix_5m_ohlcv_from_snapshots.py

echo "[DONE]"
echo
echo "To fix today's cached 5m bars now:"
echo "  DATE=20260513 bash scripts/fix_today_5m_ohlcv_from_snapshots.sh"
