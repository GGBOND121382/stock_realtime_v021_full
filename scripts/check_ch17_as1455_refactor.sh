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

bash scripts/run_as1455_live_data_feature_pipeline.sh check

echo '===== Default protocol policy ====='
grep -F 'MODEL_SELECTION_MODE="${MODEL_SELECTION_MODE:-strict_oos}"' scripts/run_as1455_fold0_forward_backtests.sh >/dev/null
grep -F 'SELECTION_RANK_METRIC="${SELECTION_RANK_METRIC:-sharpe}"' scripts/run_as1455_fold0_forward_backtests.sh >/dev/null
grep -F 'OUTPUT_MODE="${OUTPUT_MODE:-summary}"' scripts/run_as1455_target_natural_backtest.sh >/dev/null
grep -F 'FORWARD_ARTIFACT_MODE="${FORWARD_ARTIFACT_MODE:-model_only}"' scripts/refresh_as1455_forward_model_data.sh >/dev/null
grep -F 'RANK_METRIC="${RANK_METRIC:-sharpe}"' scripts/plot_as1455_default_ab_nav_curves.sh >/dev/null
grep -F 'APPLY="${APPLY:-0}"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'INCLUDE_OBSOLETE="${INCLUDE_OBSOLETE:-0}"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'PRUNE_GRID_RUNS="${PRUNE_GRID_RUNS:-0}"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'SHARE_FILE="$OUT_DIR/share_me.txt"' scripts/run_as1455_storage_maintenance.sh >/dev/null
grep -F 'scripts/run_as1455_cleanup_safe.py' scripts/run_as1455_storage_maintenance.sh >/dev/null
echo '[OK] strict OOS, summary-first grid, model-only artifacts, conservative storage maintenance and Sharpe selection are defaults'

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
"$PYTHON_BIN" scripts/compare_as1455_backtest_runs.py --help >/dev/null
"$PYTHON_BIN" scripts/check_as1455_disk_space.py --help >/dev/null
"$PYTHON_BIN" scripts/cleanup_as1455_storage.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_cleanup_safe.py --help >/dev/null
"$PYTHON_BIN" scripts/export_as1455_storage_diagnostics.py --help >/dev/null
"$PYTHON_BIN" scripts/compact_as1455_prediction_artifacts.py --help >/dev/null
"$PYTHON_BIN" scripts/materialize_as1455_best_run.py --help >/dev/null
"$PYTHON_BIN" code/backtest/run_as1455_close_auction_grid_inprocess.py --help >/dev/null

echo '[PASS] Ch17 AS1455 clean runtime validation passed'
