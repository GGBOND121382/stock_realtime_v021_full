#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-all}"
REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"
STATE_DIR="${STATE_DIR:-$REBUILD_ROOT/state}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"

mkdir -p "$STATE_DIR"
if [[ -s "$STATE_DIR/run_stamp.txt" ]]; then
  RUN_STAMP="$(tr -d '\r\n' < "$STATE_DIR/run_stamp.txt")"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  printf '%s\n' "$RUN_STAMP" > "$STATE_DIR/run_stamp.txt"
fi

rebalance_for_target() {
  case "$1" in
    r01_fwd) printf '1\n' ;;
    r05_fwd) printf '5\n' ;;
    r21_fwd) printf '21\n' ;;
    *) printf '[ERROR] unsupported target: %s\n' "$1" >&2; exit 2 ;;
  esac
}

# The existing strict-OOS implementation writes its manifest under
# 01_close_auction_grid/.  The controller also checks a root-level manifest.
# Create a stable relative symlink before the run so both locations refer to the
# same generated file.  No existing file is replaced or deleted.
for target in $TARGETS; do
  rebalance="$(rebalance_for_target "$target")"
  for preset in $FEATURE_PRESETS; do
    out_root="saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/${preset}_${target}_reb${rebalance}_${RUN_STAMP}"
    mkdir -p "$out_root"
    link="$out_root/strict_oos_manifest.json"
    if [[ ! -e "$link" && ! -L "$link" ]]; then
      ln -s "01_close_auction_grid/strict_oos_manifest.json" "$link"
    fi
  done
done

exec bash scripts/rebuild_ch17_as1455_from_scratch.sh "$MODE"
