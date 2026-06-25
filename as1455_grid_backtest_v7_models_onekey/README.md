# AS1455 Grid Backtest V7 Models One-Key Patch

目标：一键覆盖安装 AS1455 close-auction 回测矩阵脚本，支持完整模型/模型组合维度。

默认完整矩阵：

```text
7 signals × 5 max_positions × 6 sell_rank × 5 rebalance_every = 1050 runs
```

默认 signal：

```text
model_0: column 0
model_1: column 1
model_2: column 2
model_3: column 3
model_4: column 4
ensemble_first3_mean: mean(0,1,2)
ensemble_all5_mean: mean(0,1,2,3,4)
```

默认交易参数：

```text
initial_cash = 200000
commission_rate = 0.000085
min_commission = 5
stamp_tax_rate = 0.0005
transfer_fee_rate = 0.00001
slippage_bps = 0
```

## 安装

把 zip 解压到仓库根目录后：

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_grid_backtest_v7_models_onekey.zip
bash as1455_grid_backtest_v7_models_onekey/install.sh --repo .
```

安装器会覆盖/写入：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
code/backtest/run_as1455_close_auction_grid_v1.py
scripts/run_as1455_grid_smoke_v7.sh
scripts/run_as1455_grid_full_v7.sh
```

如果目标文件已存在，会自动备份到：

```text
_backup_as1455_grid_backtest_v7_YYYYMMDD_HHMMSS/
```

## Smoke

```bash
bash scripts/run_as1455_grid_smoke_v7.sh
```

默认 smoke 会跑：

```text
7 signals × 4 parameter configs = 28 runs
```

## 完整 1050 组

```bash
bash scripts/run_as1455_grid_full_v7.sh
```

等价于：

```bash
python3 code/backtest/run_as1455_close_auction_grid_v1.py \
  --force \
  --out-root saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_models_$(date +%Y%m%d) \
  --predictions saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5 \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --profile close_auction_skip_limit \
  --capacity-mode none
```

## 全 offset 稳健性 3150 组

```bash
OFFSET_MODE=full bash scripts/run_as1455_grid_full_v7.sh
```

## 自定义 signal

```bash
python3 code/backtest/run_as1455_close_auction_grid_v1.py \
  --out-root saved_data/ashare_ml4t/custom_signal_grid \
  --predictions saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5 \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --signal-spec model_0:0:single \
  --signal-spec ensemble_all5_mean:0,1,2,3,4:mean
```

如果提供 `--signal-spec`，会覆盖默认 7 个 signal；不提供则默认跑完整 7 个。

## 日志/输出

每个 run 目录包含：

```text
config.json
summary.json
close_auction_summary.json
close_auction_nav.csv
close_auction_orders.csv
close_auction_trades.csv
round_trips.csv
daily_drawdown.csv
monthly_summary.csv
yearly_summary.csv
fee_summary.csv
turnover_summary.csv
```

总表和榜单：

```text
00_grid_config.csv
02_summary/grid_summary.csv
02_summary/grid_summary_compact.csv
02_summary/leaderboard_by_sharpe.csv
02_summary/leaderboard_by_total_return.csv
02_summary/leaderboard_by_calmar.csv
02_summary/best_by_signal_sharpe.csv
02_summary/best_by_signal_total_return.csv
02_summary/best_by_signal_calmar.csv
02_summary/best_by_signal_max_drawdown.csv
```

每组 `summary.json` / `config.json` 会记录：

```text
model_family
model_run
prediction_file
prediction_file_sha256
prediction_hdf_key
signal_name
signal_cols
signal_mode
model_params_file
n_prediction_rows
n_prediction_symbols
prediction_date_min
prediction_date_max
```

注意：回测脚本不重新训练模型，只消费 `test_preds.h5` 中已经生成的预测列。
