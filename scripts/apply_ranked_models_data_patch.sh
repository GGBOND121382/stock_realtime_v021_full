#!/usr/bin/env bash
set -euo pipefail

# Safely apply data-collection config changes required by auto-selected models.
# This script is intentionally conservative:
#   - Always backs up files before editing.
#   - Refuses to append duplicate TOML tables unless FORCE_APPEND=1.
#   - Validates TOML parseability before replacing the real config.
#   - The stock quote source-priority change only removes the disabled EM realtime source;
#     it does not remove Sina or THS capability.
#
# Run from project root:
#   bash scripts/apply_ranked_models_data_patch.sh
#
# Options:
#   FORCE_APPEND=1       append even if target TOML tables already exist
#   SKIP_SOURCE_PATCH=1  do not change SPOT_SOURCE_PRIORITY default
#   VALIDATE_ONLY=1      only validate the already-installed TOML config

ROOT="${1:-$(pwd)}"
cd "$ROOT"

CONFIG="configs/realtime_context_sources.toml"
APPEND="patches/realtime_context_ranked_models_append.toml"
TRADING_SCRIPT="scripts/run_trading_day_signal_and_portfolio_all_models.sh"
MARKER="BEGIN ranked-model realtime context patch v2"
TS="$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${PYTHON:-python3}"

FORCE_APPEND="${FORCE_APPEND:-0}"
SKIP_SOURCE_PATCH="${SKIP_SOURCE_PATCH:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

validate_toml() {
  local f="$1"
  "$PYTHON_BIN" - "$f" <<'PY'
import sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
p = Path(sys.argv[1])
with p.open('rb') as fh:
    tomllib.load(fh)
print(f"[OK] TOML parses: {p}")
PY
}

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] missing $CONFIG"
  exit 1
fi

validate_toml "$CONFIG"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "[DONE] VALIDATE_ONLY=1"
  exit 0
fi

if [[ ! -f "$APPEND" ]]; then
  echo "[ERROR] missing $APPEND"
  exit 1
fi

mkdir -p "saved_data/patch_backups"
CONFIG_BAK="saved_data/patch_backups/realtime_context_sources.toml.$TS.bak"
cp "$CONFIG" "$CONFIG_BAK"
echo "[BACKUP] $CONFIG_BAK"

if grep -q "$MARKER" "$CONFIG"; then
  echo "[SKIP] TOML patch marker already exists in $CONFIG"
else
  DUP_TABLES=()
  for table in \
    'stocks\."600522\.SH"' 'stocks\."600487\.SH"' 'stocks\."002518\.SZ"' \
    'contexts\.ocg_stocks' 'contexts\.ocg_etfs' 'contexts\.ocg_futures' 'contexts\.ocg_boards' \
    'contexts\.sp_stocks' 'contexts\.sp_etfs' 'contexts\.sp_futures' 'contexts\.sp_boards'
  do
    if grep -Eq "^\[$table\]" "$CONFIG"; then
      DUP_TABLES+=("$table")
    fi
  done

  if [[ ${#DUP_TABLES[@]} -gt 0 && "$FORCE_APPEND" != "1" ]]; then
    echo "[ERROR] Target TOML tables already exist; refusing to append duplicates."
    printf '  - %s\n' "${DUP_TABLES[@]}"
    echo "[HINT] Merge $APPEND manually, or rerun with FORCE_APPEND=1 only if you know it is safe."
    exit 2
  fi

  TMP_CONFIG="$(mktemp)"
  cat "$CONFIG" "$APPEND" > "$TMP_CONFIG"
  if validate_toml "$TMP_CONFIG"; then
    mv "$TMP_CONFIG" "$CONFIG"
    echo "[APPLY] appended ranked-model realtime context TOML to $CONFIG"
  else
    rm -f "$TMP_CONFIG"
    echo "[ERROR] TOML validation failed; restored original config unchanged. Backup: $CONFIG_BAK"
    exit 3
  fi
fi

if [[ "$SKIP_SOURCE_PATCH" == "1" ]]; then
  echo "[SKIP] source-priority patch because SKIP_SOURCE_PATCH=1"
elif [[ -f "$TRADING_SCRIPT" ]]; then
  SCRIPT_BAK="saved_data/patch_backups/run_trading_day_signal_and_portfolio_all_models.sh.$TS.bak"
  cp "$TRADING_SCRIPT" "$SCRIPT_BAK"
  if grep -q 'SPOT_SOURCE_PRIORITY:-sina,ths,em,xq' "$TRADING_SCRIPT"; then
    perl -0pi -e 's/SPOT_SOURCE_PRIORITY:-sina,ths,em,xq/SPOT_SOURCE_PRIORITY:-sina_batch,ths_etf,xq/g' "$TRADING_SCRIPT"
    echo "[APPLY] changed default SPOT_SOURCE_PRIORITY to sina_batch,ths_etf,xq"
    echo "[BACKUP] $SCRIPT_BAK"
  elif grep -q 'SPOT_SOURCE_PRIORITY:-sina_batch,ths_etf,xq' "$TRADING_SCRIPT"; then
    echo "[SKIP] SPOT_SOURCE_PRIORITY already uses sina_batch,ths_etf,xq"
  else
    echo "[WARN] Did not find the known SPOT_SOURCE_PRIORITY default in $TRADING_SCRIPT"
    echo "[WARN] You can still run with: SPOT_SOURCE_PRIORITY=sina_batch,ths_etf,xq"
  fi
else
  echo "[WARN] missing $TRADING_SCRIPT; skip default source-priority patch"
fi

echo
cat <<'MSG'
[CHECK] Recommended checks after applying:
  python3 data_collection/collect_realtime_context.py plan \
    --watchlist selected_watchlist.txt \
    --models-dir saved_models \
    --model-policy all \
    --config configs/realtime_context_sources.toml \
    --out-dir saved_data/realtime_context \
    --date $(date +%Y%m%d) \
    --cutoff-time 14:55 \
    --refresh-plan

Open saved_data/realtime_context/YYYYMMDD/realtime_context_plan.csv and check:
  - 600522.SH / 600487.SH use ocg_stocks,ocg_etfs,ocg_futures,ocg_boards
  - 002518.SZ uses sp_stocks,sp_etfs,sp_futures,sp_boards if its model is saved
  - missing_context_config_features is empty for selected artifacts
MSG
