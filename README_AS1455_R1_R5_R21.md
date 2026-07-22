# AS1455 r1 / r5 / r21 运行指南

## 本分支当前任务

本分支的公开入口只读取已经完整生成的历史 NAV 与 strict-OOS forward NAV，并按原始 fold 边界重新绘图。

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

兼容旧命令：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only
```

两条命令执行相同的 existing-results 工作流。

## 明确不会执行

该入口不会：

- 生成模型预测；
- 运行投资组合回测；
- 运行交易参数 grid；
- 训练或搜索模型；
- 更新行情；
- 重建 model_data；
- materialize 新的最佳回测结果；
- 修改 checkpoint、scaler 或训练目录。

启动日志必须包含：

```text
[MODE] prediction=false backtest=false grid=false training=false data_refresh=false
```

## 输入结果配对

每个 strict-OOS forward 结果必须通过 `strict_oos_manifest.json` 反向指向它实际使用的历史回测目录。解析器同时校验：

- `evaluation_mode=strict_oos`；
- 历史交易参数和调仓相位均已冻结复用；
- forward 只有一个保留配置；
- 历史 `materialized_best_run.json` 与 forward 记录的选择完全一致；
- 历史最佳 NAV 和 forward NAV 均存在；
- one-lag fold mapping 完整且包含真实起止日期。

更新但残缺的目录会被跳过。六组策略中任一组找不到严格配对时，程序会直接失败，不会自动补跑回测或 grid。

## Fold 范围

```text
r01_fwd: fold0..fold6
r05_fwd: fold0..fold6
r21_fwd: fold0..fold5
```

历史 fold1..fold6 读取对应历史 NAV，并按照 `one_lag_prediction_manifest.json` 中的真实目标窗口切片和重新归一化。fold0 读取现有 strict-OOS forward NAV。每个 fold 使用该 fold 可用策略的公共真实日期区间。

## 输出

```text
saved_data/ashare_ml4t/ch17_as1455_backtest_plots/existing_results_<时间戳>/
saved_data/ashare_ml4t/ch17_as1455_existing_results/<时间戳>/
```

生成：

- 3 张六策略 strict-OOS forward 对比图；
- fold6..fold0 的日、周、月图，共 21 张；
- 对应 CSV；
- `existing_result_pairs.json`，记录每条曲线使用的历史和 forward 根目录、NAV、选择配置与 manifest；
- `existing_results_report.json`，明确记录 prediction/backtest/grid/training/data_refresh 均为 false，并记录实际运行秒数。

该流程只读取既有 CSV/JSON/NAV 并绘图，正常运行时间取决于结果文件规模，通常应为几分钟量级。
