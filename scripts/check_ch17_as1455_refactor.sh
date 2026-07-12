#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo '===== Python syntax ====='
"$PYTHON_BIN" -m compileall -q \
  utils/as1455_paths.py \
  utils/as1455_ch17_common.py \
  utils/as1455_cli.py \
  utils/as1455_signal_specs.py \
  utils/as1455_rank_cache.py \
  utils/as1455_backtest_io.py \
  utils/as1455_grid_runner.py \
  utils/as1455_plotting.py \
  scripts/as1455_target_label_common.py \
  scripts/run_as1455_target_fold_param_search.py \
  scripts/run_as1455_target_one_lag_backtest.py \
  scripts/run_as1455_fold0_forward_backtest.py \
  scripts/run_as1455_rotation_one_lag_daily_backtest.py \
  scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py \
  scripts/plot_as1455_backtest_return_curves.py \
  scripts/check_ch17_as1455_refactor.py \
  scripts/compare_as1455_backtest_runs.py \
  code/backtest/run_as1455_close_auction_grid_inprocess.py

echo '===== Shell syntax ====='
for script in \
  scripts/run_as1455_target_search_all.sh \
  scripts/run_as1455_r05_target_search_all.sh \
  scripts/run_as1455_r21_target_search_all.sh \
  scripts/run_as1455_target_natural_backtest.sh \
  scripts/run_as1455_r05_natural_backtest.sh \
  scripts/run_as1455_r21_natural_backtest.sh \
  scripts/run_as1455_fold0_forward_backtests.sh \
  scripts/plot_as1455_default_ab_nav_curves.sh; do
  bash -n "$script"
done

echo '===== Default selection policy ====='
grep -F 'TOP_N="${TOP_N:-5}"' scripts/run_as1455_fold0_forward_backtests.sh >/dev/null
grep -F 'RANK_METRIC="${RANK_METRIC:-sharpe}"' scripts/plot_as1455_default_ab_nav_curves.sh >/dev/null
echo '[OK] fold0 candidates=top5+ensembles; plot selection metric=sharpe'

echo '===== Structural and synthetic checks ====='
"$PYTHON_BIN" scripts/check_ch17_as1455_refactor.py

echo '===== CLI imports ====='
"$PYTHON_BIN" scripts/run_as1455_target_fold_param_search.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_target_one_lag_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_fold0_forward_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_rotation_one_lag_daily_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py --help >/dev/null
"$PYTHON_BIN" scripts/plot_as1455_backtest_return_curves.py --help >/dev/null
"$PYTHON_BIN" scripts/compare_as1455_backtest_runs.py --help >/dev/null
"$PYTHON_BIN" code/backtest/run_as1455_close_auction_grid_inprocess.py --help >/dev/null

echo '[PASS] Ch17 AS1455 refactor validation passed'
