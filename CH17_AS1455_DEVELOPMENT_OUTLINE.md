# Ch17 AS1455 开发大纲

运行命令与详细存储政策见：

```text
README_AS1455_R1_R5_R21.md
AS1455_STORAGE_AND_STRICT_OOS.md
```

## 1. 固定口径

- 特征只能使用当日 14:55 及以前的数据；
- 历史执行价使用当日 15:00 收盘价近似收盘集合竞价成交价；
- long-only，默认只交易沪深主板；
- 默认处理 T+1、停牌、涨跌停、100 股整手、费用和公司行为。

目标映射只允许定义在：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

| 目标 | lookahead | 调仓周期 | offset 搜索范围 |
|---|---:|---:|---|
| `r01_fwd` | 1 | 1 | `0` |
| `r05_fwd` | 5 | 5 | `0..4` |
| `r21_fwd` | 21 | 21 | `0..20` |

这里的 offset 是 v7 当前窗口内的本地序号，不是固定日历锚点。跨历史与 forward 窗口时必须继承相位并换算 forward 本地 offset，禁止直接复制整数。

特征方案：

```text
rotation_onehot
rotation_addon_onehot
```

## 2. 端到端协议

```text
共享历史缓存
→ model_data_as1455.h5
→ 训练/历史：特征完整且当前目标已实现
→ 参数搜索
→ one-fold-lag 历史回测
→ 从 status=ok 中选择历史最佳完整配置和真实历史窗口
→ 冻结 signal/max/sell/rebalance 与历史调仓相位
→ forward：只要求模型特征完整，不要求目标已实现
→ 用完整执行日历将历史相位换算为 forward effective offset
→ fold0.test_end 后 strict OOS 单配置回测
→ 统一绘图与审计
```

## 3. 公共模块

### 3.1 训练与历史预测

```text
utils/as1455_ch17_common.py
```

负责目标映射、A/B 特征、fold、checkpoint、scaler、模型输入、prediction artifact 和 grid 命令。其他入口不得复制这些逻辑。

### 3.2 forward 行保留

```text
utils/as1455_forward_features.py
```

只改变行保留条件，不改变特征定义：

```text
训练/历史：模型特征非空 + target_col 非空
forward：模型特征非空；target_col 可为空
```

forward manifest 必须记录：

```text
model_data_max_date
feature_valid_max_date
target_valid_max_date
unlabeled_prediction_rows
expected_prediction_end
prediction_end
```

并满足：

```text
prediction_end == expected_prediction_end
```

### 3.3 历史配置与窗口选择

```text
utils/as1455_model_selection.py
```

- 严格只允许 `status=ok`；
- 没有成功行时直接失败；
- 选择完整 run，而不只选择 signal；
- 提取 signal、max、sell、rebalance 和历史 offset；
- 提取选中 run 的 `date_min/date_max/n_days`；
- 旧 summary 缺少窗口字段时，从 materialized `close_auction_nav.csv` 回填；
- 跳过仅有失败结果的较新目录；
- 绘图和 forward 共用同一选择函数。

### 3.4 调仓相位对齐

```text
utils/as1455_rebalance_phase.py
```

v7 本地调仓条件：

```text
(day_index - local_offset) mod rebalance_every = 0
```

历史窗口与 forward 窗口都会从 `day_index=0` 重新编号，因此 strict OOS 使用：

```text
forward_global_index
  = historical_n_days + bridge_execution_days

effective_forward_offset
  = (historical_offset - forward_global_index) mod rebalance_every
```

`bridge_execution_days` 由共享 grid 的完整 raw daily execution calendar 计算，不使用自然日，也不简单假定 fold 长度。

### 3.5 strict OOS

```text
utils/as1455_strict_oos.py
```

正式模式冻结：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
historical rebalance phase
```

共享 grid 支持：

```text
--rebalance-phase-history-offset
--rebalance-phase-history-first-date
--rebalance-phase-history-last-date
--rebalance-phase-history-n-days
--rebalance-offset-list
```

相位换算在配置生成前完成，最终只生成 forward effective offset 对应的一个配置，不运行其他 offset，也不按 forward 指标重新选择参数。

manifest 必须满足：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
historical_rebalance_phase_reused = true
rebalance_phase_alignment.effective_forward_offset 已记录
generated_config_count = 1
retained_config_count = 1
```

`historical_offset_numeric_reused` 仅表示两个窗口的本地 offset 整数是否恰好相同，不代表相位是否继承。

### 3.6 prediction 保留

```text
utils/as1455_artifact_retention.py
scripts/compact_as1455_prediction_artifacts.py
```

HDF 是预测权威文件。只删除与同名 HDF 重复的 prediction CSV，并同步更新 manifest。`actual_<target>.csv` 保存真实标签，必须保留。

### 3.7 grid 与交易

```text
utils/as1455_grid_runner.py
utils/as1455_rank_cache.py
utils/as1455_backtest_io.py
```

- prediction 每个 signal 加载一次；
- 每个 signal 每日排序一次；
- execution panel 构造一次；
- 相位对齐使用未裁剪的完整 execution calendar；
- effective offset 在配置生成阶段过滤；
- 每个配置只调用唯一 v7 `backtest()`；
- 不允许第二套交易循环。

唯一交易语义来源：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

## 4. 存储边界

共享权威缓存：

```text
ch12_as1455/baostock_5m_cache/
ch12_as1455/baostock_raw_daily_cache/
ch12_as1455/as1455_daily_cache/
```

forward 不得复制这些缓存。

forward 刷新默认：

```text
FORWARD_ARTIFACT_MODE=model_only
FORWARD_REPORT_MODE=compact
MIN_FREE_GB=5
```

验证后删除 forward 目录中的可重建副本：

```text
as1455_ohlcv_raw.h5
as1455_ohlcv_adj.h5
as1455_execution_metadata.h5
```

live 默认只保留最近 3 个日期；旧保留日期删除可重建 history tail。

## 5. 历史回测

协议：

```text
source fold6 -> target fold5
...
source fold1 -> target fold0
```

默认：

```text
OUTPUT_MODE=summary
MATERIALIZE_BEST=1
MATERIALIZED_OUTPUT_MODE=compact
```

完整网格只保留 summary。`scripts/materialize_as1455_best_run.py` 使用历史窗口本地精确 offset 只重跑一个最佳配置，并删除其他 summary-only run 目录和日志。历史 materialization 与 forward 相位换算是两个不同流程，不得调用 forward strict finalizer。

## 6. fold0-forward

入口：

```text
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

默认：

```text
MODEL_SELECTION_MODE=strict_oos
SELECTION_RANK_METRIC=sharpe
OUTPUT_MODE=compact
MIN_FREE_GB=5
```

其他模式：

```text
forward_parameter_sweep  # 仅敏感性分析
all_top_n                 # 兼容旧实验
```

输出目录使用秒级时间戳，避免同日误用旧结果。

## 7. 清理自动化

```text
scripts/check_as1455_disk_space.py
scripts/cleanup_as1455_storage.py
```

清理器默认 dry-run，并具备：

- 活动进程门禁；
- 路径边界保护；
- forward HDF 验证；
- live 日期保留；
- prediction manifest 同步；
- 可选旧目录清理；
- 可选 grid run 裁剪；
- 大型报告 gzip；
- JSON 审计 manifest。

共享行情缓存、训练 checkpoint、训练 model data 和当前 contract 依赖文件不在自动删除清单中。

## 8. 最低验证

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

必须覆盖：

- Python/Shell 语法和 CLI；
- failed-only summary 拒绝；
- 历史窗口 metadata 提取；
- forward 最新无标签日期保留；
- r5 历史 `off3`、378 日窗口换算为 forward `off0` 的相位测试；
- strict OOS 完整配置与历史相位冻结；
- exact-offset 配置生成；
- prediction CSV 清理且 actual 标签保留；
- 唯一 v7 引擎；
- summary-first、model-only 和 strict OOS 默认值。

真实目录 smoke：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

日志必须出现：

```text
[PHASE ALIGN] historical_offset=... history_days=... bridge_days=... forward_global_index=... effective_forward_offset=...
```

`PARITY_CHECK_ONLY=1` 只验证相位对齐后的单个 v7 引擎路径，不代表完整正式结果。
