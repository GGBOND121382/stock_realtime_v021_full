#!/usr/bin/env bash
set -euo pipefail

# Cleanup the accidental quick/small-batch 603308 ane_live_board_v2 training outputs.
# It does NOT remove rebuilt external data:
#   saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment_live_board_v2
# By default, it moves files/dirs to cleanup_trash/ so they are out of the live pool.
# Set HARD_DELETE=1 to permanently delete instead.

TS="$(date +%Y%m%d_%H%M%S)"
TRASH_ROOT="${TRASH_ROOT:-cleanup_trash/603308_ane_quick_garbage_${TS}}"
HARD_DELETE="${HARD_DELETE:-0}"

PATHS=(
  "saved_models/603308.SH/nextday_all_days_close_profit_xgb_d3_all_no_ak_ane_live_board_v2"
  "saved_models/603308.SH/nextday_vwap_low_close_profit_xgb_d3_all_no_ak_ane_live_board_v2"
  "saved_data/603308_pipeline_out/05_search_ane_live_board_v2_all_days"
  "saved_data/603308_pipeline_out/05_search_ane_live_board_v2_vwap_low"
  "saved_data/603308_pipeline_out/99_summary_ane_live_board_v2"
  "scripts/retrain_603308_ane_live_board_v2_and_summary.sh"
)

KEEP_PATH="saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment_live_board_v2"

printf '\n[KEEP] rebuilt external data will NOT be removed:\n  %s\n' "$KEEP_PATH"
printf '\n[CLEANUP TARGETS]\n'
for p in "${PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    echo "  FOUND  $p"
  else
    echo "  MISS   $p"
  fi
done

if [[ "$HARD_DELETE" == "1" ]]; then
  printf '\n[MODE] HARD_DELETE=1, permanently deleting found targets ...\n'
  for p in "${PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      rm -rf "$p"
      echo "  deleted: $p"
    fi
  done
else
  printf '\n[MODE] move to trash: %s\n' "$TRASH_ROOT"
  mkdir -p "$TRASH_ROOT"
  for p in "${PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      dest="$TRASH_ROOT/$p"
      mkdir -p "$(dirname "$dest")"
      mv "$p" "$dest"
      echo "  moved: $p -> $dest"
    fi
  done
fi

printf '\n[VERIFY ACTIVE PATHS]\n'
for p in "${PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    echo "  STILL EXISTS: $p"
  else
    echo "  clean: $p"
  fi
done

printf '\n[DONE] quick/small-batch garbage is out of active paths.\n'
if [[ "$HARD_DELETE" != "1" ]]; then
  echo "Trash saved at: $TRASH_ROOT"
fi
