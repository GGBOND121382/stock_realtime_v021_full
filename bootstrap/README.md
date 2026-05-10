# bootstrap

This directory holds the older BaoStock data/feature bootstrap chain used by
`../pipelines/run_nextday_pipeline.py` for its `update_data` stage.

- `ashare_xgb_dual_opportunity_regression_baostock_full_v19_compressed_trading_axis.py`
  is still called by the main pipeline, primarily in `--mode update_data`.
- `ashare_fetch_and_train_xgb_sell_signal_baostock_state_cache_helper_fix2.py`
  provides BaoStock caching and feature construction helpers.
- `t_strategy_backtest_cv5_split_eval.py` is kept here because the helper imports
  its feature/backtest config objects.

Prefer the current `pipelines/`, `model_training/`, `model_saving/`, `prediction/`, and `visualization/` scripts for training, scoring and plotting.
