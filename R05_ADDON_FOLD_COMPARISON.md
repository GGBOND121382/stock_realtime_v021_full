# r05_fwd + rotation_addon_onehot：逐 fold 嵌套选择协议

## 修正后的实验协议

旧流程先生成 `source fold6 -> target fold5` 到 `source fold1 -> target fold0` 的六段预测，再把六段拼接后统一运行 1050 组交易参数 grid。该流程会让 target fold 的交易结果参与最终 signal、持仓数、卖出阈值和调仓相位选择，不符合原定的逐 fold 协议。

现在每个 source fold 独立执行：

```text
source fold k 的训练产物
  -> 在 source fold k 自己的 63 日留出段生成候选 checkpoint 预测
  -> 只在这 63 日上运行完整 signal + trading grid
  -> 冻结完整配置 C_k
  -> C_k 仅用于 target fold k-1
```

fold0 同理：

```text
fold0 留出段 grid -> 冻结 C_0 -> 仅用于 fold0 test_end 之后的 forward
```

target fold 和 forward 都不会重新 grid，也不会反向参与配置选择。

## 运行

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
```

或：

```bash
bash scripts/run_as1455_r05_addon_fold_comparison.sh
```

默认复用：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
saved_data/ashare_ml4t/ch17_as1455_target_search/rotation_addon_onehot/r05_fwd/fold{0..6}_search
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache
```

不会训练模型、刷新行情或重建 `model_data`。但是会执行 7 次独立 validation grid，因此运行时间显著高于旧的结果复制脚本。

## 输出

```text
saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/
  rotation_addon_onehot_r05_fwd_<timestamp>/
    source_fold6/
      validation_selection/
      selected_for_next_window.json
      target_fold5/
    ...
    source_fold1/
      validation_selection/
      selected_for_next_window.json
      target_fold0/
    source_fold0/
      validation_selection/
      selected_for_next_window.json
      forward/
    nested_fold_target_results.csv
    nested_fold_protocol_manifest.json
    continuous_target_folds_plus_forward/
```

每个 `selected_for_next_window.json` 明确记录：

- source fold 的验证日期；
- 验证段选中的 checkpoint/ensemble signal；
- `max_positions`、`sell_rank`、`rebalance_every`、验证段 offset；
- 目标段相位换算后的 effective offset；
- `target_results_used_for_selection=false`。

连续账户按 `target_fold5 -> ... -> target_fold0 -> forward` 顺序运行，跨段继承现金、持仓、买入日期和成本基础。配置在边界处切换为对应 source fold 预先冻结的配置。

## 续跑与磁盘

验证 grid 默认使用：

```text
VALIDATION_OUTPUT_MODE=summary
```

只保留每个候选的 JSON 和汇总，不为全部 1050 个候选保存 NAV。目标段默认使用 `compact`。

同一 `OUT_ROOT` 下再次执行会复用已有预测和已完成 grid；强制覆盖：

```bash
FORCE=1 OUT_ROOT=<existing-dir> \
  bash scripts/run_as1455_r05_addon_fold_comparison.sh
```

只生成独立 target 结果、不生成连续账户：

```bash
SKIP_CONTINUOUS=1 \
  bash scripts/run_as1455_r05_addon_fold_comparison.sh
```
