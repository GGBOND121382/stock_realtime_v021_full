# AS1455 r1 / r5 / r21 运行指南

## 默认任务：独立 Fold 单配置回测

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
```

兼容入口：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only
```

默认流程不会重新训练、生成预测或搜索交易参数。它读取已经完成的六组历史/strict-OOS 结果配对，取得冻结信号和完整交易配置，然后对每个 fold 从空仓和统一初始资金独立执行一次回测。

启动日志：

```text
[MODE] independent folds with frozen configurations
[MODE] prediction_generation=false grid=false training=false data_refresh=false
[MODE] backtest=true initial_state=empty_positions_and_initial_cash
[MODE] expected_backtests=40 initial_cash=200000
```

## 40 次回测

```text
r01_fwd: fold1..fold6 × 2 preset = 12
r05_fwd: fold1..fold6 × 2 preset = 12
r21_fwd: fold1..fold5 × 2 preset = 10
fold0: 3 target × 2 preset = 6
总计 = 40
```

每次只运行一组冻结配置，不运行 grid。

## 冻结内容

从严格配对的既有结果读取：

- `signal_name`、`signal_cols`、`signal_mode`；
- `max_positions`、`sell_rank`、`rebalance_every`；
- 手续费、印花税、过户费、滑点、100 股整数手；
- 主板/ST/涨跌停/T+1/容量/公司行为配置；
- 历史或 strict-OOS 已保留的 `rebalance_offset`。

独立 fold 起点改变后，offset 按原连续 OOS 日历换算：

```text
effective_local_offset
= (original_offset - skipped_overlap_dates) mod rebalance_every
```

## 数据配对

解析器从完整 strict-OOS forward 结果反查其实际使用的历史回测，并要求：

- 历史 `materialized_best_run.json` 完整；
- 历史/forward NAV、预测 HDF 和 `config.json` 存在；
- 历史最佳信号与 forward strict-OOS 信号一致；
- forward 仅生成并保留一个冻结配置；
- one-lag fold mapping 完整。

新建但残缺的错误目录会被跳过；找不到六组完整配对时立即失败，不会补跑 grid。

## Fold 公共日历

```text
r01_fwd: fold0..fold6
r05_fwd: fold0..fold6
r21_fwd: fold0..fold5
```

每个 fold 使用所有可用策略公共的预测/执行交易日。所有曲线在公共首日以 200000 元现金和空仓状态开始。

fold6 只有 r01/r05 的两个 preset，共四条曲线；fold5..fold0 有六条曲线。

## 输出

```text
saved_data/ashare_ml4t/ch17_as1455_independent_folds/<时间戳>/
```

输出包含：

- `runs/fold*/<strategy>/`：40 组 compact 回测结果；
- `plots/fold*/return_curve_{daily,weekly,monthly}.png`：21 张图；
- 对应 CSV；
- 每次运行的 `independent_fold_run.json`；
- `independent_fold_manifest.json`；
- `independent_fold_report.json`。

## 只切旧 NAV 的兼容模式

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

该模式不做回测，只对既有连续 NAV 切片和绘图，与默认的独立 fold 实验口径不同。
