#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

PIPELINE="pipelines/run_trading_day_signal_pipeline.py"
HELPER="scripts/fill_sector_range_z20_from_history.py"

if [[ ! -f "$PIPELINE" ]]; then
  echo "[ERROR] run from project root; missing $PIPELINE" >&2
  exit 2
fi
if [[ ! -f "$HELPER" ]]; then
  echo "[ERROR] missing helper script after unzip: $HELPER" >&2
  exit 2
fi

mkdir -p "backups/sector_range_z20_history_patch_${TS}"
cp "$PIPELINE" "backups/sector_range_z20_history_patch_${TS}/run_trading_day_signal_pipeline.py"

"$PYTHON" - <<'PY'
from pathlib import Path

path = Path("pipelines/run_trading_day_signal_pipeline.py")
s = path.read_text(encoding="utf-8")

if "fill_sector_range_z20_from_history.py" in s:
    print("[OK] pipeline already calls fill_sector_range_z20_from_history.py")
    raise SystemExit(0)

old = '''                run_cmd(context_build_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)\n                mark("context_build_features", "ok")\n'''
new = '''                run_cmd(context_build_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)\n                mark("context_build_features", "ok")\n\n                # Post-process sector_range_z20 without changing the realtime source:\n                # estimate today's sector_range_pct from sampled THS sector prices,\n                # then compute z20 from the model training samples referenced by\n                # realtime_context_plan.csv / saved model metadata.\n                sector_range_z20_cmd = [\n                    python, "scripts/fill_sector_range_z20_from_history.py",\n                    "--date", trade_date,\n                    "--context-dir", str(context_dir),\n                    "--cutoff-time", args.cutoff_time,\n                ]\n                try:\n                    run_cmd(sector_range_z20_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)\n                    mark("sector_range_z20_history_fill", "ok")\n                except Exception as exc:\n                    mark("sector_range_z20_history_fill", "failed", {"error": str(exc)})\n                    if not args.keep_going:\n                        raise\n'''
if old not in s:
    raise SystemExit("[ERROR] could not find context_build_features success block to patch")
path.write_text(s.replace(old, new, 1), encoding="utf-8")
print("[OK] patched", path)
PY

chmod +x "$HELPER"

echo "[OK] installed helper: $HELPER"
echo "[OK] patched pipeline: $PIPELINE"
echo "[OK] backup: backups/sector_range_z20_history_patch_${TS}"
echo
echo "Validate with:"
echo "  $PYTHON scripts/fill_sector_range_z20_from_history.py --date \$(date +%Y%m%d) --context-dir saved_data/realtime_context --cutoff-time 14:55"
