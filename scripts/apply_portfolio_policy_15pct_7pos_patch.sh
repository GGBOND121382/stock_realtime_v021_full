#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_policy_15pct_7pos_${TS}"

FILES=(
  "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
  "configs/portfolio_confirm_config.json"
)

mkdir -p "$BACKUP_DIR"
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    exit 2
  fi
  mkdir -p "$BACKUP_DIR/$(dirname "$f")"
  cp -p "$f" "$BACKUP_DIR/$f"
  echo "[BACKUP] $f -> $BACKUP_DIR/$f"
done

"$PYTHON" scripts/patch_portfolio_policy_15pct_7pos.py

"$PYTHON" -m py_compile portfolio_decision/daily_portfolio_confirm_pyscipopt.py

"$PYTHON" - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("configs/portfolio_confirm_config.json").read_text(encoding="utf-8"))
print("max_policy_weight=", cfg.get("max_policy_weight"))
print("max_positions=", cfg.get("max_positions"))
assert abs(float(cfg.get("max_policy_weight")) - 0.15) < 1e-12
assert int(cfg.get("max_positions")) == 7
PY

echo "[DONE] portfolio policy patch applied: max_policy_weight=0.15, max_positions=7"
echo "[BACKUP_DIR] $BACKUP_DIR"
