# scripts

This directory contains reusable project entrypoints. Do not add backup copies or one-time maintenance files here.

## Main AS1455 / ML4T entries

- `build_ashare_ch12_as1455_model_data.py`: build AS1455 ML4T-style model data.
- `build_ashare_ch12_as1455_lowmem.sh`: low-memory wrapper for model-data construction.
- `run_as1455_sector_rotation_fold0_param_search.py`: current rotation plus sector-one-hot NN search entry. Use `--fold-index 0..6` and `--sector-encoding onehot`.
- `run_as1455_first_batch_features_fold0_param_search.py`: full rotation plus compact add-on feature experiment.
- `run_as1455_sector_onehot_single_fold_test.py`: single-fold sector-one-hot test entry.
- `run_ashare_ch17_nn_reproduce.py`: original 31-feature ML4T Chapter 17 baseline.
- `run_ashare_ch17_backtest_profiles.py`: Ch17 prediction and backtest profile helper.
- `analyze_ashare_ch17_model_predictions.py`: prediction diagnostics.

## One-fold-lag daily close-auction backtests

These scripts generate target-fold predictions from previous-fold search-time checkpoints and then call the existing close-auction v7 grid with daily rebalance only (`rebalance_every=1`, `rebalance_offset=0`).

- `run_as1455_rotation_one_lag_daily_backtest.py` / `.sh`: original 31 features + full sector rotation + sector one-hot.
- `run_as1455_rotation_addon_one_lag_daily_backtest.py` / `.sh`: original 31 features + full sector rotation + compact add-on features + sector one-hot.

Main metric files are written under `<OUT_ROOT>/01_close_auction_daily_grid/02_summary/`:

- `grid_summary_compact.csv`: compact table with return, drawdown, win-rate, turnover, fee, order and rejection metrics.
- `leaderboard_by_total_return.csv`
- `leaderboard_by_annual_return.csv`
- `leaderboard_by_sharpe.csv`
- `leaderboard_by_calmar.csv`
- `leaderboard_by_max_drawdown.csv`
- `leaderboard_by_trade_win_rate.csv`
- `leaderboard_by_fee_efficiency.csv`

Important columns include `total_return`, `annual_return`, `sharpe`, `calmar`, `max_drawdown`, `daily_win_rate`, `monthly_win_rate`, `trade_win_rate`, `round_trip_win_rate`, `avg_turnover`, `annualized_turnover`, `gross_trade_amount`, `total_fee`, `fee_to_initial_cash`, `avg_positions`, `n_orders`, and `n_rejections`.

## Live and weekly AS1455 pipeline

- `run_as1455_live_data_feature_pipeline.sh`: live data and feature pipeline.
- `run_as1455_top5_weekly_retrain_full_v7.sh`: weekly retrain entry.
- `run_as1455_grid_smoke_v7.sh` / `run_as1455_grid_full_v7.sh`: v7 grid smoke/full search wrappers.
- `run_as1455_live_checkpoint_signal_v2.sh`: checkpoint-based live signal wrapper.
- `run_as1455_live_fast_auto_v1.sh`: fast live pipeline wrapper.
- `run_as1455_live_fast_auto_checkpoint_signal_v1.sh`: fast live pipeline plus checkpoint signals.
- `run_as1455_live_prefast_v2.sh` and `run_as1455_live_postfast_v1.sh`: prefast/postfast wrappers.
- `run_as1455_live_rebuild_features_strict_v2.sh`: strict live feature rebuild.
- `run_as1455_live_repair_and_features_v1.sh`: live repair plus feature generation.

## Diagnostics and historical scripts

Reusable diagnostics and repair scripts such as `diagnose_ashare_ch12_as1455_*.py`, `clean_as1455_5m_cache_oldschema.py`, `compare_today_collected_vs_baostock.sh`, `check_*`, `fill_*`, and `fix_*` are kept for verification.

Older asof1455, portfolio, special-stock, and multi-pipeline scripts are retained only for result reproduction. Revalidate them before using them in a new workflow.

## Cleaned categories

The active directory should not contain `apply_*.sh`, `patch_*.py`, `*.bak*`, or temporary README/test files.
