# r05_fwd rotation_addon_onehot 分 Fold / 跨 Fold 回测

## 运行

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
```

也可以直接运行：

```bash
bash scripts/run_as1455_r05_addon_fold_comparison.sh
```

## 实验口径

目标策略固定为：

```text
target_col = r05_fwd
feature_preset = rotation_addon_onehot
```

脚本从完整 strict-OOS 结果反查其实际使用的历史 materialized run，并冻结：

- signal_name / signal_cols / signal_mode；
- max_positions / sell_rank / rebalance_every / rebalance_offset；
- 费用、主板、ST、涨跌停、T+1、容量和公司行为配置。

不会训练模型、生成预测、刷新数据或运行交易参数 grid。

## 分 Fold 结果

历史 one-fold-lag 映射按目标测试窗口命名：

```text
source model fold6 -> target fold5
source model fold5 -> target fold4
...
source model fold1 -> target fold0
```

对 target_fold5 至 target_fold0 分别执行 6 次单配置回测。每次：

```text
initial_cash = 200000
initial_positions = empty
```

rebalance offset 按该 fold 在原连续预测日历中的位置换算，避免把五日调仓相位重置为 0。

## 跨 Fold 结果

历史 grid 的 materialized best run 本身就是六个 target fold 连续拼接、账户状态不重置的跨 fold 回测。

专用脚本不会再次运行同一条连续回测，而是：

1. 校验 materialized NAV 日期与六个 fold 的预测日期完全一致；
2. 校验相邻 fold 之间 `trading_gap_days = 0`；
3. 复制 authoritative materialized run 的 compact 结果；
4. 生成跨 fold 曲线和各 fold 分段指标。

这样默认只新增 6 次回测，运行时间主要消耗在一次执行面板构建和六次单配置模拟。

## 输出

```text
saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/<timestamp>/
```

主要文件：

```text
fold_boundary_audit.csv
execution_data_report.csv
r05_addon_backtest_comparison.csv
r05_addon_fold_comparison_manifest.json
r05_addon_fold_comparison_report.json

per_fold/
  target_fold5/
  ...
  target_fold0/
  plots/return_curve_{daily,weekly,monthly}.{png,csv}

cross_fold/
  continuous_nav.csv
  continuous_fold_segments.csv
  materialized_run/
  plots/return_curve_{daily,weekly,monthly}.{png,csv}
```

`fold_boundary_audit.csv` 中任一相邻 fold 出现非零交易日缺口时，脚本直接失败，不生成可误用的连续结果。
