# AS1455 Grid Backtest v7 一键覆盖包

用途：把 AS1455 回测改为“最大持仓数量 max_positions × 卖出排名 sell_rank × 调仓间隔 rebalance_every”的参数矩阵搜索，并输出详细日志。

## 一键安装

把压缩包上传到服务器仓库根目录，例如 `~/stock_realtime_v021_full/`，然后执行：

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_grid_backtest_v7_onekey.zip
bash as1455_grid_backtest_v7_onekey/install.sh --repo .
```

安装器会覆盖/写入：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
code/backtest/run_as1455_close_auction_grid_v1.py
scripts/run_as1455_grid_smoke_v7.sh
scripts/run_as1455_grid_full_v7.sh
```

如果目标文件已存在，会先备份到：

```text
_backup_as1455_grid_backtest_v7_YYYYMMDD_HHMMSS/
```

## Smoke test

```bash
bash scripts/run_as1455_grid_smoke_v7.sh
```

默认读取：

```text
saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache
```

默认输出：

```text
saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_smoke/
```

## 完整 150 组矩阵

```bash
bash scripts/run_as1455_grid_full_v7.sh
```

矩阵为：

```text
max_positions = 5,10,15,20,25
sell_rank = 75,100,150,200,250,300
rebalance_every = 1,2,3,4,5
rebalance_offset = 0
```

默认资金与费用：

```text
initial_cash = 200000
commission_rate = 0.000085
min_commission = 5
stamp_tax_rate = 0.0005
transfer_fee_rate = 0.00001
slippage_bps = 0
```

## 如果路径不同

可以用环境变量覆盖：

```bash
PREDICTIONS=你的/test_preds.h5 \
RAW_DAILY_CACHE_DIR=你的/baostock_raw_daily_cache \
OUT_ROOT=saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_custom \
bash scripts/run_as1455_grid_smoke_v7.sh
```

## 容量约束

默认先用 `CAPACITY_MODE=none` 做参数初筛。候选参数确定后，再用 last5 容量约束复核：

```bash
CAPACITY_MODE=last5_both bash scripts/run_as1455_grid_smoke_v7.sh
```

如果使用 last5 容量约束，还应直接调用 Python 脚本并传入 `--raw-5m-cache-dir`，例如：

```bash
python3 code/backtest/run_as1455_close_auction_grid_v1.py \
  --smoke --force \
  --out-root saved_data/ashare_ml4t/ch17_as1455_backtest_grid_v7_smoke_last5 \
  --predictions saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7/results/test_preds.h5 \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --raw-5m-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache \
  --profile close_auction_skip_limit \
  --capacity-mode last5_both \
  --capacity-missing-policy reject
```

## 输出重点

每组 run 输出：

```text
summary.json
daily_nav.csv
daily_drawdown.csv
daily_positions.csv
orders.csv
trades.csv
round_trips.csv
rejections.csv
monthly_summary.csv
yearly_summary.csv
fee_summary.csv
turnover_summary.csv
```

总汇总目录：

```text
02_summary/grid_summary.csv
02_summary/grid_summary_compact.csv
02_summary/leaderboard_by_sharpe.csv
02_summary/leaderboard_by_calmar.csv
02_summary/leaderboard_by_return.csv
02_summary/leaderboard_cost_adjusted.csv
```
