#!/usr/bin/env bash
set -euo pipefail

# One-key installer for AS1455 max-position grid backtest v7.
# Usage from anywhere:
#   bash install.sh --repo ~/stock_realtime_v021_full
# Or after extracting under repo root:
#   bash as1455_grid_backtest_v7_onekey/install.sh --repo .

REPO="."
DRY_RUN=0
SKIP_COMPILE=0
KEEP_BACKUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="${2:?missing value for --repo}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-compile)
      SKIP_COMPILE=1
      shift
      ;;
    --no-backup)
      KEEP_BACKUP=0
      shift
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="$SCRIPT_DIR/payload"
REPO="$(cd "$REPO" && pwd)"

if [[ ! -d "$REPO" ]]; then
  echo "[ERROR] Repo directory not found: $REPO" >&2
  exit 1
fi
if [[ ! -f "$REPO/README.md" && ! -d "$REPO/.git" ]]; then
  echo "[ERROR] $REPO does not look like the stock_realtime_v021_full repo root." >&2
  echo "        Run from the repo root or pass --repo ~/stock_realtime_v021_full" >&2
  exit 1
fi
if [[ ! -d "$PAYLOAD_DIR" ]]; then
  echo "[ERROR] Payload directory missing: $PAYLOAD_DIR" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$REPO/_backup_as1455_grid_backtest_v7_$TS"

TARGETS=(
  "code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py"
  "code/backtest/run_as1455_close_auction_grid_v1.py"
  "scripts/run_as1455_grid_smoke_v7.sh"
  "scripts/run_as1455_grid_full_v7.sh"
)

copy_one() {
  local rel="$1"
  local src="$PAYLOAD_DIR/$rel"
  local dst="$REPO/$rel"
  if [[ ! -f "$src" ]]; then
    echo "[ERROR] Payload file missing: $src" >&2
    exit 1
  fi
  echo "[INSTALL] $rel"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$KEEP_BACKUP" == "1" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$rel")"
    cp -p "$dst" "$BACKUP_DIR/$rel"
  fi
  cp -f "$src" "$dst"
}

for rel in "${TARGETS[@]}"; do
  copy_one "$rel"
done

if [[ "$DRY_RUN" == "0" ]]; then
  chmod +x "$REPO/scripts/run_as1455_grid_smoke_v7.sh" "$REPO/scripts/run_as1455_grid_full_v7.sh"
fi

if [[ "$SKIP_COMPILE" == "0" ]]; then
  echo "[CHECK] python3 -m py_compile"
  if [[ "$DRY_RUN" == "0" ]]; then
    python3 -m py_compile \
      "$REPO/code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py" \
      "$REPO/code/backtest/run_as1455_close_auction_grid_v1.py"
  fi
fi

cat <<EOF
[DONE] AS1455 grid backtest v7 installed.

Target repo:
  $REPO

Installed files:
  code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
  code/backtest/run_as1455_close_auction_grid_v1.py
  scripts/run_as1455_grid_smoke_v7.sh
  scripts/run_as1455_grid_full_v7.sh
EOF

if [[ "$KEEP_BACKUP" == "1" && -d "$BACKUP_DIR" ]]; then
  echo
  echo "Backup of overwritten files:"
  echo "  $BACKUP_DIR"
fi

cat <<'EOF'

Next commands:
  bash scripts/run_as1455_grid_smoke_v7.sh
  # If smoke is ok:
  bash scripts/run_as1455_grid_full_v7.sh

Before committing:
  git status --short
  git diff --stat
EOF
