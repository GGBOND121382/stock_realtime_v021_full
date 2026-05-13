# 603308 ane_live_board_v2 external-only search patch

Purpose: test whether rebuilt `ane_live_board_v2` external features improve the model.

This fixed version avoids Bash's special `$GROUPS` variable. The feature group list is stored in `FEATURE_GROUPS`.

Usage from repo root:

```bash
unzip -o 603308_ane_live_board_v2_external_search_patch_v2_fixed.zip
bash scripts/run_603308_ane_external_full_search.sh
```

Outputs:

```text
saved_data/603308_pipeline_out/99_summary_ane_live_board_v2_external_full/final_leaderboard.csv
saved_data/603308_pipeline_out/99_summary_ane_live_board_v2_external_full/best_by_target_top5.csv
saved_data/603308_pipeline_out/99_summary_ane_live_board_v2_external_full/save_top_artifacts_commands.sh
```

Only after reviewing the leaderboard:

```bash
bash saved_data/603308_pipeline_out/99_summary_ane_live_board_v2_external_full/save_top_artifacts_commands.sh
```
