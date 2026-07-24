# r05_fwd rotation_addon_onehot 分 Fold、Forward 与连续回测

## 运行

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
```

也可以直接运行：

```bash
bash scripts/run_as1455_r05_addon_fold_comparison.sh
```

目标策略固定为：

```text
target_col = r05_fwd
feature_preset = rotation_addon_onehot
```

脚本只使用已有历史预测、fold0 strict-OOS forward 预测、materialized 历史最佳配置、strict-OOS 保留配置和行情缓存。不会训练模型、生成预测、刷新数据或运行交易参数 grid。

## 生成的四类结果

### 1. 六个历史 Fold 独立回测

```text
source model fold6 -> target fold5
...
source model fold1 -> target fold0
```

`target_fold5` 至 `target_fold0` 分别从 200000 元和空仓开始。历史调仓相位会转换为各 fold 的本地 offset，不会简单重置为 0。

输出：

```text
per_fold/target_fold5/
...
per_fold/target_fold0/
per_fold/plots/
```

### 2. 历史跨 Fold 连续结果

历史 materialized best run 本身是 `target_fold5 -> ... -> target_fold0` 的连续账户。脚本校验其日期覆盖后直接复制，不重复运行同一结果。

输出：

```text
cross_fold_historical/
  continuous_nav.csv
  materialized_run/
  plots/
```

### 3. Fold0 之后的独立 strict-OOS Forward

脚本直接纳入与历史结果严格配对的 retained strict-OOS forward run。该结果仍是原 forward 回测自己的独立账户口径，用于单独检查 fold0 模型在历史 fold0 结束后的样本外表现。

输出：

```text
forward_strict_oos/
  forward_nav.csv
  retained_run/
  plots/
```

### 4. 历史到 Forward 的真正连续账户

脚本额外执行：

```text
target_fold5 -> ... -> target_fold0
-> bridge execution dates
-> fold0 strict-OOS forward
```

实现规则：

- 历史段从 200000 元和空仓开始；
- 历史段重跑后必须与 authoritative materialized NAV 一致，否则失败；
- 历史末端现金、持仓、买入日期和成本基础传给 forward；
- 历史末日与 forward 首日之间没有预测的交易日只估值并处理公司行为，不虚构排名或交易；
- forward 使用 strict-OOS manifest 已换算好的本地 rebalance offset；
- 全程账户状态不重置，最终日期必须等于 forward 预测的最后日期。

输出：

```text
cross_fold_historical_plus_forward/
  continuous_nav.csv
  continuous_segments.csv
  summary.json
  config.json
  close_auction_summary.json
  plots/
```

## 性能设计

```text
execution panel 构建 1 次
六个历史独立 fold 回测
历史连续账户重跑 1 次
forward 连续账户接续回测 1 次
已有历史/forward 独立结果直接复制
```

当历史与 forward 的冻结配置均为 `capacity_mode=none` 时，脚本自动禁用 5 分钟缓存扫描。默认 `OUTPUT_MODE=compact`，不写大体量逐日持仓和订单明细。

## 根目录与审计文件

```text
saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/<timestamp>/
```

关键文件：

```text
fold_boundary_audit.csv
forward_bridge_execution_dates.csv
execution_data_report.csv
r05_addon_backtest_comparison.csv
r05_addon_fold_comparison_manifest.json
r05_addon_fold_comparison_report.json
```

`fold_boundary_audit.csv` 对历史相邻 folds 要求 `trading_gap_days=0`。历史 fold0 到 forward 的交易日间隔单独记录为 bridge，并明确采用“只估值、不交易”策略。

## 自定义输出位置

```bash
RUN_STAMP=forward_complete \
OUT_ROOT=saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/forward_complete \
bash scripts/run_as1455_r05_addon_fold_comparison.sh
```

默认不要使用 `OUTPUT_MODE=full`；只有需要逐笔订单、持仓和拒单审计时才启用：

```bash
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_addon_fold_comparison.sh
```
