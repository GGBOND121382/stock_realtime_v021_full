Cleanup patch for accidental quick/small-batch 603308 ane_live_board_v2 outputs.

Usage from project root:
  unzip -o 603308_ane_quick_cleanup_patch.zip
  bash scripts/cleanup_603308_ane_quick_garbage.sh

Default behavior: move targets to cleanup_trash/, not permanent delete.
Permanent delete:
  HARD_DELETE=1 bash scripts/cleanup_603308_ane_quick_garbage.sh

The script does NOT remove:
  saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment_live_board_v2
