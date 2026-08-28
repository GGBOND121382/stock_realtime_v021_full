# Ch17 AS1455 独立 Fold 回测与绘图

本分支默认复用服务器上已经完成的模型预测、历史最优配置和 strict-OOS 结果，对 fold6 至 fold0 分别执行独立投资组合回测。

## 默认入口

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
```

兼容命令：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only
bash scripts/run_ch17_as1455_backtest_only.sh
```

## 回测口径

每个 fold、每个策略均：

```text
统一初始资金 200000
+ 空仓启动
+ 冻结既有 signal_name / signal_cols / signal_mode
+ 冻结既有 max_positions / sell_rank / rebalance_every
+ 复用既有完整执行配置
+ 按独立 fold 起点换算 rebalance_offset
```

调仓相位换算为：

```text
effective_local_offset
= (original_offset - skipped_overlap_dates) mod rebalance_every
```

因此每个 fold 虽然重新从空仓开始，但不会把既有调仓相位擅自重置为 0。

## 运行数量

```text
r01_fwd: 6 个历史 fold × 2 个 preset = 12
r05_fwd: 6 个历史 fold × 2 个 preset = 12
r21_fwd: 5 个历史 fold × 2 个 preset = 10
fold0 strict-OOS: 3 个 target × 2 个 preset = 6
总计：40 次单配置回测
```

这些不是 grid。每次只执行已经冻结的一组配置。

## 明确不会执行

```text
重新训练模型
重新生成预测
交易参数 grid
更新行情缓存
重建 model_data
覆盖 checkpoint / scaler / 训练目录
```

启动日志必须包含：

```text
prediction_generation=false
grid=false
training=false
data_refresh=false
backtest=true
initial_state=empty_positions_and_initial_cash
expected_backtests=40
```

## 输入

主要只读输入：

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/
```

解析器从完整 strict-OOS forward 结果反查其实际使用的历史结果，并校验：

- 历史 `materialized_best_run.json`；
- 历史和 forward 的 `config.json`；
- 历史和 forward 预测 HDF；
- 历史最佳信号和交易配置一致性；
- one-lag fold mapping 完整性；
- strict-OOS 仅保留一个配置。

更新但残缺的目录会被跳过；任一策略缺少完整配对时直接失败，不会补跑 grid。

## Fold 范围

```text
r01_fwd: fold0..fold6
r05_fwd: fold0..fold6
r21_fwd: fold0..fold5
```

fold6 没有 r21，因此包含 r01/r05 两个 preset 的四条曲线；fold5 至 fold0 包含六条可比曲线。

每个 fold 使用所有可用策略的公共可执行交易日集合，并在同一天以相同初始资金和空仓状态启动。

## 输出

```text
saved_data/ashare_ml4t/ch17_as1455_independent_folds/<timestamp>/
```

包括：

- 40 组 compact 单配置回测结果；
- fold6..fold0 的日、周、月图，共 21 张；
- 每张图对应 CSV；
- 每次回测的来源、配置、相位换算和容量检查清单；
- `independent_fold_manifest.json`；
- `independent_fold_report.json`。

默认磁盘余量检查为 1 GiB。

## 旧的连续 NAV 切片模式

只读取已有连续 NAV、按 fold 边界裁剪而不重新回测：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

该模式与默认独立 fold 回测是不同实验口径，不再作为 `backtest-only` 的默认行为。
