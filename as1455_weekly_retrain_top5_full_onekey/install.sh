#!/usr/bin/env bash
set -euo pipefail
REPO="."
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$SRC_DIR/payload"
cd "$REPO"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP="_backup_as1455_weekly_retrain_top5_${TS}"
mkdir -p "$BACKUP"
copy_one() {
  local rel="$1"
  local src="$PAYLOAD/$rel"
  local dst="$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    mkdir -p "$BACKUP/$(dirname "$dst")"
    cp -a "$dst" "$BACKUP/$dst"
  fi
  cp -a "$src" "$dst"
}
copy_one code/training/run_as1455_weekly_retrain_predict_v1.py
copy_one scripts/run_as1455_top5_weekly_retrain_full_v7.sh
copy_one docs/README_AS1455_WEEKLY_RETRAIN_TOP5_FULL.md
chmod +x code/training/run_as1455_weekly_retrain_predict_v1.py scripts/run_as1455_top5_weekly_retrain_full_v7.sh
python3 -m py_compile code/training/run_as1455_weekly_retrain_predict_v1.py
if [[ -z "$(find "$BACKUP" -mindepth 1 -print -quit)" ]]; then
  rmdir "$BACKUP"
else
  echo "[INFO] Existing files backed up to: $BACKUP"
fi
echo "[OK] Installed weekly retrain top5 full backtest patch."
echo "Run: bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh"
