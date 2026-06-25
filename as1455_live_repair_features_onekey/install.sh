#!/usr/bin/env bash
set -Eeuo pipefail

REPO="."
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash install.sh --repo /path/to/stock_realtime_v021_full"; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

REPO="$(cd "$REPO" && pwd)"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$PKG_DIR/payload"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP="$REPO/_backup_as1455_live_repair_features_$TS"

mkdir -p "$BACKUP"
cd "$REPO"

echo "[INFO] repo=$REPO"
echo "[INFO] backup=$BACKUP"

install_file() {
  local rel="$1"
  local src="$PAYLOAD/$rel"
  local dst="$REPO/$rel"
  if [[ ! -f "$src" ]]; then
    echo "[ERROR] missing payload file: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    mkdir -p "$BACKUP/$(dirname "$rel")"
    cp -p "$dst" "$BACKUP/$rel"
  fi
  cp -f "$src" "$dst"
  chmod +x "$dst" || true
  echo "[OK] installed $rel"
}

install_file tools/repair_as1455_live_collect_and_features_v1.py
install_file scripts/run_as1455_live_repair_and_features_v1.sh

# Patch the root-level live common helper if present. This fixes future collect/finalize runs.
COMMON="$REPO/features/as1455_live_common.py"
if [[ -f "$COMMON" ]]; then
  mkdir -p "$BACKUP/features"
  cp -p "$COMMON" "$BACKUP/features/as1455_live_common.py"
  python3 - "$COMMON" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
old = '''        quality = "ok"
        miss = str(r.get("missing_core_fields", "") or "")
        try:
            lo = float(r["low"]); hi = float(r["high"]); op = float(r["open"]); cl = float(r["last_price"])
            if not (lo <= op <= hi and lo <= cl <= hi):
                quality = "price_order_invalid"
        except Exception:
            quality = "invalid_numeric"
        if miss:
            quality = "missing_core_fields"
'''
new = '''        quality = "ok"
        raw_miss = r.get("missing_core_fields", "")
        if raw_miss is None or pd.isna(raw_miss):
            miss = ""
        else:
            miss = str(raw_miss).strip()
            if miss.lower() in {"", "nan", "none", "null", "na", "n/a", "[]", "{}"}:
                miss = ""
        try:
            lo = float(r["low"]); hi = float(r["high"]); op = float(r["open"]); cl = float(r["last_price"])
            if not (lo <= op <= hi and lo <= cl <= hi):
                quality = "price_order_invalid"
        except Exception:
            quality = "invalid_numeric"
        if miss:
            quality = "missing_core_fields"
'''
if old in text:
    text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")
    print("[OK] patched features/as1455_live_common.py missing_core_fields NaN handling")
elif "raw_miss = r.get(\"missing_core_fields\"" in text:
    print("[OK] features/as1455_live_common.py already appears patched")
else:
    raise SystemExit("cannot find expected snapshots_to_raw_panel quality_status block; common helper not patched")
PY
else
  echo "[WARN] features/as1455_live_common.py not found; only repair script installed."
fi

python3 -m py_compile tools/repair_as1455_live_collect_and_features_v1.py
if [[ -f features/as1455_live_common.py ]]; then
  python3 -m py_compile features/as1455_live_common.py
fi
if [[ -f features/build_as1455_live_features.py ]]; then
  python3 -m py_compile features/build_as1455_live_features.py
fi

echo "[DONE] AS1455 live repair/features patch installed."
echo "Run existing collected data through features with:"
echo "  TRADE_DATE=20260625 bash scripts/run_as1455_live_repair_and_features_v1.sh"
