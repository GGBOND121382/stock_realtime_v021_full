#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/portfolio_allinone_replace_${TS}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FILES=(
  "portfolio_decision/portfolio_confirm_from_buy_signals.py"
  "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
  "scripts/run_portfolio_confirm_from_signals.sh"
  "configs/portfolio_confirm_config.json"
)

mkdir -p "$BACKUP_DIR"
for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp -p "$f" "$BACKUP_DIR/$f"
    echo "[BACKUP] $f -> $BACKUP_DIR/$f"
  else
    echo "[WARN] file not found before patch: $f"
  fi
done

# Replace only the three portfolio source/wrapper files. This is deliberate:
# it recovers from earlier partial string-patch failures.
install -m 0644 "$PATCH_DIR/replacements/portfolio_decision/portfolio_confirm_from_buy_signals.py" "portfolio_decision/portfolio_confirm_from_buy_signals.py"
install -m 0644 "$PATCH_DIR/replacements/portfolio_decision/daily_portfolio_confirm_pyscipopt.py" "portfolio_decision/daily_portfolio_confirm_pyscipopt.py"
install -m 0755 "$PATCH_DIR/replacements/scripts/run_portfolio_confirm_from_signals.sh" "scripts/run_portfolio_confirm_from_signals.sh"

echo "[REPLACED] portfolio source files"

mkdir -p configs
if [[ ! -f configs/portfolio_model_overrides.csv ]]; then
  cp "$PATCH_DIR/configs/portfolio_model_overrides.template.csv" configs/portfolio_model_overrides.csv
  echo "[INIT] configs/portfolio_model_overrides.csv created from template"
else
  echo "[KEEP] configs/portfolio_model_overrides.csv already exists"
fi

# Preserve existing config fields, only set account-level policy values.
"$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("configs/portfolio_confirm_config.json")
cfg = {}
if p.exists():
    cfg = json.loads(p.read_text(encoding="utf-8"))
cfg["max_policy_weight"] = 0.15
cfg["max_positions"] = 7
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("[CONFIG] max_policy_weight=", cfg.get("max_policy_weight"))
print("[CONFIG] max_positions=", cfg.get("max_positions"))
PY

"$PYTHON" -m py_compile \
  portfolio_decision/portfolio_confirm_from_buy_signals.py \
  portfolio_decision/daily_portfolio_confirm_pyscipopt.py

bash -n scripts/run_portfolio_confirm_from_signals.sh

"$PYTHON" - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path("configs/portfolio_confirm_config.json").read_text(encoding="utf-8"))
assert abs(float(cfg.get("max_policy_weight")) - 0.15) < 1e-12
assert int(cfg.get("max_positions")) == 7
print("[CHECK] config ok")
PY

echo "[DONE] portfolio all-in-one replacement patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
