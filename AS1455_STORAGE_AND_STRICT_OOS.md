# AS1455 存储治理与严格样本外协议

本文档定义 AS1455 r1/r5/r21 流程的存储边界、保留策略、forward 日期口径和严格样本外参数选择规则。若其他旧文档与本文冲突，以本文和 `README_AS1455_R1_R5_R21.md` 为准。

## 1. 三个问题与根因

### 1.1 forward 数据重复

`ch12_as1455_forward_latest` 过去在重建 `model_data_as1455.h5` 时，同时永久写入：

```text
as1455_ohlcv_raw.h5
as1455_ohlcv_adj.h5
as1455_execution_metadata.h5
```

这些文件来自 `ch12_as1455` 下的共享缓存，可重建，不应在 forward 目录形成第二份长期副本。

历史和 forward 大网格过去还会为每个参数组合保存 NAV、回撤、月度、年度、费用、换手甚至订单明细，导致数千份重复时间序列。

### 1.2 forward 日期被目标标签截断

`r05_fwd` 需要未来 5 个交易日，`r21_fwd` 需要未来 21 个交易日。过去 fold0-forward 复用了训练数据的 `target_only` 过滤，导致最新 5/21 个交易日虽然特征完整，却因为未来目标未实现而被删除。

训练和历史回测需要目标标签；真实 forward 推理只需要模型输入特征。两者必须分开。

### 1.3 forward 重新选择交易参数

过去 forward 只继承历史最佳模型信号，却重新搜索：

```text
max_positions
sell_rank
rebalance_offset
```

绘图再按 forward Sharpe 选择最佳行，相当于在样本外窗口事后调参。该曲线不能作为严格样本外主结果。

## 2. 修复后的正式协议

### 2.1 历史训练与历史回测

- 仍要求当前 `target_col` 非空；
- 标签定义、特征定义、fold 定义和 one-fold-lag 映射不变；
- 历史完整网格用于选择一个完整配置行；
- 只允许 `status=ok` 行参与选择；
- 所有失败行都不能成为 best run。

### 2.2 fold0-forward 日期

fold0-forward 使用：

```text
row_mode = inference_features_only
require_target = false
```

只删除模型特征不完整的行，不因目标标签为空而删除最新日期。

预测结束日期必须满足：

```text
prediction_end = 当前 start/end 条件下的 feature_valid_max_date
```

manifest 同时记录：

```text
model_data_max_date
feature_valid_max_date
target_valid_max_date
unlabeled_prediction_rows
unlabeled_prediction_dates
expected_prediction_end
prediction_end
```

### 2.3 strict OOS 参数

正式 forward 默认：

```text
MODEL_SELECTION_MODE=strict_oos
```

从历史最佳完整行冻结：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
rebalance_offset
```

forward 只运行并保留该配置，不按 forward 指标重新选择参数。

敏感性分析显式使用：

```text
MODEL_SELECTION_MODE=forward_parameter_sweep
```

该模式不得标记为严格 OOS。

## 3. 存储权威目录

### 3.1 必须长期保留

```text
ch12_as1455/baostock_5m_cache/
ch12_as1455/baostock_raw_daily_cache/
ch12_as1455/as1455_daily_cache/
ch12_as1455/model_data_as1455.h5
ch12_as1455/model_data_contract.json
ch12_as1455/as1455_ohlcv_adj.h5   # 当前 contract 校验仍依赖
```

训练 checkpoint：

```text
ch17_as1455_target_search/<feature>/<target>/fold*_search/
ch17_as1455_sector_rotation_onehot_fold*_search/
ch17_as1455_full_rotation_plus_first_batch_compact_fold*_search/
```

### 3.2 forward 目录

默认仅长期保留：

```text
ch12_as1455_forward_latest/model_data_as1455.h5
ch12_as1455_forward_latest/reports/  # compact / gzip 后
```

默认删除：

```text
as1455_ohlcv_raw.h5
as1455_ohlcv_adj.h5
as1455_execution_metadata.h5
```

### 3.3 历史回测目录

完整参数网格长期保留：

```text
00_predictions/test_preds.h5
01_close_auction_grid/02_summary/*.csv
materialized_best_run.json
```

只为一个选中 run 保留 compact/full 时间序列：

```text
01_close_auction_grid/01_runs/<selected_run>/
```

### 3.4 strict forward 目录

长期保留：

```text
00_predictions/fold0_forward_preds.h5
00_predictions/selected_fold0_checkpoints.csv
00_predictions/fold0_forward_prediction_manifest.json
01_close_auction_grid/01_runs/<frozen_run>/
01_close_auction_grid/02_summary/
01_close_auction_grid/strict_oos_manifest.json
01_close_auction_grid/grid_engine_manifest.json
```

### 3.5 live 目录

默认只保留最近 3 个日期目录。最近日期保留完整审计；更早但仍保留的日期删除可重建历史尾部：

```text
04_history_tail_raw.*
05_history_tail_qfq_livebase.*
10_live_feature_panel_tail.*
```

## 4. 自动化入口

### 4.1 空间门禁

```bash
python3 scripts/check_as1455_disk_space.py \
  --path saved_data/ashare_ml4t \
  --min-free-gb 5 \
  --label manual-check
```

forward 刷新、历史大网格和 fold0-forward wrapper 已默认调用该门禁。

### 4.2 dry-run 清理

```bash
python3 scripts/cleanup_as1455_storage.py \
  --keep-live-dates 3 \
  --include-obsolete \
  --prune-grid-runs \
  --compress-reports
```

### 4.3 正式清理

审核生成的 `cleanup_audit_*.json` 后：

```bash
python3 scripts/cleanup_as1455_storage.py \
  --apply \
  --keep-live-dates 3 \
  --include-obsolete \
  --prune-grid-runs \
  --compress-reports
```

清理器默认会阻止在 AS1455 任务运行时执行删除。

## 5. 迁移现有结果的顺序

1. 停止历史更新、训练、回测和 live 任务；
2. `git pull` 并运行 `bash scripts/check_ch17_as1455_refactor.sh`；
3. 运行清理器 dry-run；
4. 检查 manifest 中的删除目录、预计释放空间和 active process 列表；
5. 使用 `--apply`；
6. 检查 `df -h`、`du -h --max-depth=1 saved_data/ashare_ml4t`；
7. 重新运行 r5 strict forward；
8. 检查 prediction end 和 strict config；
9. 重新绘图。

## 6. 修复后禁止的做法

- 不得为正式 forward 使用 `forward_parameter_sweep` 结果作为 OOS 主结果；
- 不得让 r5/r21 forward 因目标标签为空而停在最新数据前 5/21 个交易日；
- 不得默认对数千个参数组合使用 `OUTPUT_MODE=full`；
- 不得在 forward 目录永久复制完整 raw/adjusted/execution HDF；
- 不得在剩余空间低于门禁时启动高占用任务；
- 不得直接删除训练 checkpoint、共享原始缓存或当前 contract 依赖的主 adjusted HDF；
- 不得仅按目录日期认定结果有效，必须检查 summary、manifest、prediction SHA 和状态。
