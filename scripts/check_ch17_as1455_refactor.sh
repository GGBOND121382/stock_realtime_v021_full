#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo '===== Python syntax ====='
"$PYTHON_BIN" -m compileall -q \
  features/as1455_live_common.py \
  pipelines/as1455_update_history_to_prevday.py \
  pipelines/as1455_update_history_to_prevday_fast_v4.py \
  utils/as1455_paths.py \
  utils/as1455_ch17_common.py \
  utils/as1455_forward_features.py \
  utils/as1455_rebalance_phase.py \
  utils/as1455_strict_oos.py \
  utils/as1455_artifact_retention.py \
  utils/as1455_cli.py \
  utils/as1455_signal_specs.py \
  utils/as1455_model_selection.py \
  utils/as1455_rank_cache.py \
  utils/as1455_backtest_io.py \
  utils/as1455_grid_runner.py \
  utils/as1455_plotting.py \
  scripts/build_ashare_ch12_as1455_model_data.py \
  scripts/as1455_target_label_common.py \
  scripts/run_as1455_sector_rotation_fold0_param_search.py \
  scripts/run_as1455_first_batch_features_fold0_param_search.py \
  scripts/run_as1455_target_fold_param_search.py \
  scripts/run_as1455_target_one_lag_backtest.py \
  scripts/run_as1455_fold0_forward_backtest.py \
  scripts/run_as1455_rotation_one_lag_daily_backtest.py \
  scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py \
  scripts/plot_as1455_backtest_return_curves.py \
  scripts/plot_as1455_fold_sequence_curves.py \
  scripts/resolve_as1455_common_forward_start.py \
  scripts/resolve_as1455_existing_result_pairs.py \
  scripts/run_as1455_independent_fold_backtests.py \
  scripts/run_as1455_r05_addon_fold_comparison.py \
  scripts/check_ch17_as1455_refactor.py \
  scripts/check_as1455_historical_model_selection.py \
  scripts/check_as1455_storage_oos_fixes.py \
  scripts/check_as1455_artifact_retention.py \
  scripts/check_as1455_exact_offset_filter.py \
  scripts/check_as1455_disk_space.py \
  scripts/cleanup_as1455_storage.py \
  scripts/run_as1455_cleanup_safe.py \
  scripts/export_as1455_storage_diagnostics.py \
  scripts/compact_as1455_prediction_artifacts.py \
  scripts/materialize_as1455_best_run.py \
  scripts/compare_as1455_backtest_runs.py \
  code/backtest/run_as1455_close_auction_grid_v1.py \
  code/backtest/run_as1455_close_auction_grid_inprocess.py \
  code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py

echo '===== Shell syntax ====='
for script in \
  scripts/run_as1455_live_data_feature_pipeline.sh \
  scripts/build_ashare_ch12_as1455_lowmem.sh \
  scripts/run_ch17_as1455_full_rebuild.sh \
  scripts/run_ch17_as1455_backtest_only.sh \
  scripts/run_ch17_as1455_existing_results.sh \
  scripts/run_as1455_r05_addon_fold_comparison.sh \
  scripts/as1455_python_memory_guard.sh \
  scripts/run_as1455_target_search_all.sh \
  scripts/run_as1455_r05_target_search_all.sh \
  scripts/run_as1455_r21_target_search_all.sh \
  scripts/run_as1455_target_natural_backtest.sh \
  scripts/run_as1455_r05_natural_backtest.sh \
  scripts/run_as1455_r21_natural_backtest.sh \
  scripts/run_as1455_fold0_forward_backtests.sh \
  scripts/refresh_as1455_forward_model_data.sh \
  scripts/plot_as1455_default_ab_nav_curves.sh \
  scripts/run_as1455_storage_maintenance.sh; do
  bash -n "$script"
done

PYTHON="$PYTHON_BIN" bash scripts/run_as1455_live_data_feature_pipeline.sh check

echo '===== Default protocol policy ====='
grep -F 'MODEL_SELECTION_MODE="${MODEL_SELECTION_MODE:-strict_oos}"' scripts/run_as1455_fold0_forward_backtests.sh >/dev/null
grep -F 'SELECTION_RANK_METRIC="${SELECTION_RANK_METRIC:-sharpe}"' scripts/run_as1455_fold0_forward_backtests.sh >/dev/null
grep -F 'OUTPUT_MODE="${OUTPUT_MODE:-summary}"' scripts/run_as1455_target_natural_backtest.sh >/dev/null
grep -F 'scripts/run_as1455_target_fold_param_search.py' scripts/run_as1455_target_search_all.sh >/dev/null
grep -F 'scripts/run_as1455_target_one_lag_backtest.py' scripts/run_as1455_target_natural_backtest.sh >/dev/null
grep -F 'DEFAULT_FOLDS="0 1 2 3 4 5"' scripts/run_as1455_target_search_all.sh >/dev/null
grep -F 'DEFAULT_TARGET_FOLDS="0,1,2,3,4"' scripts/run_as1455_target_natural_backtest.sh >/dev/null

grep -F '[MODE] independent folds with frozen configurations' scripts/run_ch17_as1455_backtest_only.sh >/dev/null
grep -F 'prediction_generation=false grid=false training=false data_refresh=false' scripts/run_ch17_as1455_backtest_only.sh >/dev/null
grep -F 'expected_backtests=40' scripts/run_ch17_as1455_backtest_only.sh >/dev/null
grep -F 'scripts/run_as1455_independent_fold_backtests.py' scripts/run_ch17_as1455_backtest_only.sh >/dev/null
grep -F 'initial_state=empty_positions_and_initial_cash' scripts/run_ch17_as1455_backtest_only.sh >/dev/null
grep -F 'independent-folds|backtest|backtest-only' scripts/run_ch17_as1455_full_rebuild.sh >/dev/null
grep -F 'r05-addon-comparison|r05-addon-folds' scripts/run_ch17_as1455_full_rebuild.sh >/dev/null
grep -F 'existing-results|plots' scripts/run_ch17_as1455_full_rebuild.sh >/dev/null

if grep -E 'run_as1455_target_natural_backtest|run_as1455_r05_natural_backtest|run_as1455_r21_natural_backtest|run_as1455_fold0_forward_backtests|run_as1455_close_auction_grid|materialize_as1455_best_run|MODEL_DATA=|REFRESH_DATA=' scripts/run_ch17_as1455_backtest_only.sh >/dev/null; then
  echo '[ERROR] independent-fold entry references training/prediction/grid/materialization workflow' >&2
  exit 1
fi
if grep -E 'build_grid_command|grid_runner|force-grid|--force' scripts/run_as1455_independent_fold_backtests.py >/dev/null; then
  echo '[ERROR] independent-fold runner references grid orchestration' >&2
  exit 1
fi
grep -F 'bt.backtest(' scripts/run_as1455_independent_fold_backtests.py >/dev/null
grep -F 'for fold in range(6, -1, -1)' scripts/run_as1455_independent_fold_backtests.py >/dev/null
grep -F 'expected_backtests = 40' scripts/run_as1455_independent_fold_backtests.py >/dev/null
grep -F 'empty_positions_and_initial_cash' scripts/run_as1455_independent_fold_backtests.py >/dev/null
grep -F '(original - skipped) % every' scripts/run_as1455_independent_fold_backtests.py >/dev/null

grep -F '[MODE] r05_fwd rotation_addon_onehot fold comparison' scripts/run_as1455_r05_addon_fold_comparison.sh >/dev/null
grep -F 'independent_backtests=6 continuous_cross_fold=reuse_materialized' scripts/run_as1455_r05_addon_fold_comparison.sh >/dev/null
grep -F -- '--feature-presets rotation_addon_onehot' scripts/run_as1455_r05_addon_fold_comparison.sh >/dev/null
grep -F -- '--targets r05_fwd' scripts/run_as1455_r05_addon_fold_comparison.sh >/dev/null
grep -F 'EXPECTED_TARGET_FOLDS = tuple(range(5, -1, -1))' scripts/run_as1455_r05_addon_fold_comparison.py >/dev/null
grep -F 'continuous_result_source": "reused_authoritative_materialized_run"' scripts/run_as1455_r05_addon_fold_comparison.py >/dev/null
grep -F 'trading_gap_days != 0' scripts/run_as1455_r05_addon_fold_comparison.py >/dev/null
grep -F 'helpers.write_independent_run(' scripts/run_as1455_r05_addon_fold_comparison.py >/dev/null
grep -F 'result = bt.backtest(' scripts/run_as1455_r05_addon_fold_comparison.py >/dev/null
if grep -E 'run_as1455_target_natural_backtest|run_as1455_fold0_forward_backtests|run_as1455_close_auction_grid|build_grid_command|grid_runner|force-grid|--force|materialize_as1455_best_run' scripts/run_as1455_r05_addon_fold_comparison.sh scripts/run_as1455_r05_addon_fold_comparison.py >/dev/null; then
  echo '[ERROR] r05 addon comparison references grid/training/prediction/materialization orchestration' >&2
  exit 1
fi

grep -F '[MODE] existing results only' scripts/run_ch17_as1455_existing_results.sh >/dev/null
grep -F 'prediction=false backtest=false grid=false training=false data_refresh=false' scripts/run_ch17_as1455_existing_results.sh >/dev/null
if grep -E 'run_as1455_independent_fold_backtests|run_as1455_target_natural_backtest|run_as1455_fold0_forward_backtests|run_as1455_close_auction_grid' scripts/run_ch17_as1455_existing_results.sh >/dev/null; then
  echo '[ERROR] existing-results plot-only entry references a backtest path' >&2
  exit 1
fi

if grep -F 'run_ch17_as1455_full_rebuild_aligned.sh' scripts/run_ch17_as1455_full_rebuild.sh >/dev/null; then
  echo '[ERROR] public entry still references aligned rebuild' >&2
  exit 1
fi

grep -F 'RANK_METRIC="${RANK_METRIC:-sharpe}"' scripts/plot_as1455_default_ab_nav_curves.sh >/dev/null
grep -F 'APPLY="${APPLY:-0}"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'INCLUDE_OBSOLETE="${INCLUDE_OBSOLETE:-0}"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'PRUNE_GRID_RUNS="${PRUNE_GRID_RUNS:-0}"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'SHARE_FILE="$OUT_DIR/share_me.txt"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'scripts/run_as1455_cleanup_safe.py' scripts/run_as1455_storage_maintenance.sh >/dev/null
echo '[OK] independent folds use frozen configurations, empty initial state, translated phase and no parameter grid'
echo '[OK] r05 addon comparison runs six independent folds and reuses one authoritative continuous result'

echo '===== Historical model-selection synthetic check ====='
"$PYTHON_BIN" scripts/check_as1455_historical_model_selection.py

echo '===== Forward-date and strict-OOS phase-alignment synthetic check ====='
"$PYTHON_BIN" scripts/check_as1455_storage_oos_fixes.py

echo '===== Exact-offset grid synthetic check ====='
"$PYTHON_BIN" scripts/check_as1455_exact_offset_filter.py

echo '===== Prediction artifact retention synthetic check ====='
"$PYTHON_BIN" scripts/check_as1455_artifact_retention.py

echo '===== Structural and synthetic checks ====='
"$PYTHON_BIN" scripts/check_ch17_as1455_refactor.py

echo '===== CLI imports ====='
"$PYTHON_BIN" scripts/build_ashare_ch12_as1455_model_data.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_target_fold_param_search.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_target_one_lag_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_fold0_forward_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_rotation_one_lag_daily_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/plot_as1455_backtest_return_curves.py --help >/dev/null
"$PYTHON_BIN" scripts/plot_as1455_fold_sequence_curves.py --help >/dev/null
"$PYTHON_BIN" scripts/resolve_as1455_common_forward_start.py --help >/dev/null
"$PYTHON_BIN" scripts/resolve_as1455_existing_result_pairs.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_independent_fold_backtests.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_r05_addon_fold_comparison.py --help >/dev/null
"$PYTHON_BIN" scripts/compare_as1455_backtest_runs.py --help >/dev/null
"$PYTHON_BIN" scripts/check_as1455_disk_space.py --help >/dev/null
"$PYTHON_BIN" scripts/cleanup_as1455_storage.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_cleanup_safe.py --help >/dev/null
"$PYTHON_BIN" scripts/export_as1455_storage_diagnostics.py --help >/dev/null
"$PYTHON_BIN" scripts/compact_as1455_prediction_artifacts.py --help >/dev/null
"$PYTHON_BIN" scripts/materialize_as1455_best_run.py --help >/dev/null
"$PYTHON_BIN" code/backtest/run_as1455_close_auction_grid_inprocess.py --help >/dev/null

echo '[PASS] Ch17 AS1455 clean runtime validation passed'
