# Ch17 AS1455 开发大纲

本文档规定 `ch17_as1455` 代码的职责边界、唯一事实来源、严格样本外协议和存储约束。运行命令与清理流程见：

```text
README_AS1455_R1_R5_R21.md
AS1455_STORAGE_AND_STRICT_OOS.md
```

---

## 1. 固定业务口径

### 1.1 数据与交易

- 模型特征只能使用当日 14:55 及以前的数据；
- 历史执行价使用当日 15:00 收盘价近似收盘集合竞价成交价；
- raw 执行字段只进入执行面板、metadata 和质量报告；
- long-only；
- 默认只交易沪深主板；
- 默认处理 T+1、停牌、涨跌停、100 股整手、手续费、印花税、过户费和公司行为。

### 1.2 目标与自然周期

唯一事实来源：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

| 简称 | 标签 | lookahead | 调仓周期 | offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

禁止在 wrapper 或新脚本中维护第二份目标—周期映射。

### 1.3 特征方案

| 名称 | `feature_preset` | 内容 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

---

## 2. 端到端结构

```text
历史/实时行情
    ↓
共享 5 分钟缓存 + 原始日线缓存 + AS1455 日缓存
    ↓
model_data_as1455.h5
    ↓
训练数据加载：特征完整 + 目标标签已实现
    ↓
r1 / r5 / r21 参数搜索
    ↓
one-fold-lag 历史回测
    ↓
从历史 summary 中选择 status=ok 的最佳完整配置
    ↓
冻结 signal + max/sell/rebalance/offset
    ↓
forward 数据加载：只要求模型特征完整，不要求目标已实现
    ↓
fold0.test_end 后 strict OOS 回测
    ↓
统一绘图与审计
```

---

## 3. 公共工具层

### 3.1 路径

```text
utils/as1455_paths.py
```

统一默认 model data、共享执行缓存、target search、target backtest、fold0-forward 和绘图根目录。新入口不得重新硬编码这些路径。

### 3.2 训练与历史预测公共实现

```text
utils/as1455_ch17_common.py
```

职责：

- `TARGET_SPECS`；
- 训练/历史回测的目标标签过滤；
- A/B 特征构造；
- fold 构造；
- checkpoint 目录规则；
- search-time checkpoint、scaler 和 feature manifest 读取；
- 模型输入变换与 checkpoint 推理；
- prediction HDF、审计清单和 manifest；
- grid 命令构造。

标签、特征和模型输入语义不得在其他入口复制。

### 3.3 forward 特征行保留

```text
utils/as1455_forward_features.py
```

该模块只改变**行保留条件**，不改变任何特征定义：

```text
训练/历史：模型特征非空 + 当前 target_col 非空
forward：   模型特征非空；target_col 可为空
```

forward manifest 必须记录：

```text
row_mode = inference_features_only
model_data_max_date
feature_valid_max_date
target_valid_max_date
unlabeled_prediction_rows
unlabeled_prediction_dates
```

### 3.4 公共 CLI

```text
utils/as1455_cli.py
```

统一 prediction、grid、执行缓存、输出模式、smoke、force、dry-run 和 signal spec 参数。wrapper 只设置默认值和组合入口。

### 3.5 signal 定义

```text
utils/as1455_signal_specs.py
```

仅生成真实存在预测列对应的单模型和 ensemble；禁止手工维护另一份 top-N signal 清单。

### 3.6 历史完整配置选择

```text
utils/as1455_model_selection.py
```

职责：

1. 查找 `grid_summary_compact.csv` 或 `grid_summary.csv`；
2. 存在 `status` 列时严格只保留 `status=ok`；
3. 没有成功行时直接失败，禁止退回失败行；
4. 按指标选择最佳完整 run；
5. 提取：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
rebalance_offset
```

6. 推导 fold0 推理所需 checkpoint 数量；
7. 跳过仅有失败结果的较新目录。

默认指标是 `sharpe`。绘图和 forward 都必须调用该模块，禁止各自定义“最佳”。

### 3.7 strict OOS

```text
utils/as1455_strict_oos.py
```

正式模式完整冻结历史配置。若历史 offset 非零，当前共享 grid 可短暂生成同一周期的全部 offset，但不得按 forward 指标选择；执行结束后必须只保留历史 offset 对应 run，并将活动 summary 改为单行严格配置。

manifest 必须记录：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
retained_config_count = 1
retained_run_name
retained_config
```

### 3.8 prediction 保留策略

```text
utils/as1455_artifact_retention.py
scripts/compact_as1455_prediction_artifacts.py
```

HDF 是权威预测文件。删除重复 CSV 时必须同步更新 prediction manifest，并写出 `prediction_artifact_retention.json`，禁止留下指向不存在文件的 manifest。

### 3.9 排名、交易输出和 grid

```text
utils/as1455_rank_cache.py
utils/as1455_backtest_io.py
utils/as1455_grid_runner.py
```

约束：

- prediction 每个 signal 加载一次；
- 每个 signal 每日排序一次；
- execution panel 构造一次；
- 每个组合只调用唯一 v7 `backtest()`；
- 不 monkey-patch；
- 不实现第二套交易循环。

### 3.10 绘图

```text
utils/as1455_plotting.py
scripts/plot_as1455_backtest_return_curves.py
scripts/plot_as1455_default_ab_nav_curves.sh
```

绘图只读取已存在 NAV，不改变日期或交易参数。strict OOS 根目录活动 summary 只有一个冻结 run，因此绘图不会再次在 forward 窗口调参。

---

## 4. 数据层与存储边界

### 4.1 共享历史缓存

入口：

```text
scripts/run_as1455_live_data_feature_pipeline.sh history
pipelines/as1455_update_history_to_prevday_fast_v4.py
```

共享权威缓存位于：

```text
ch12_as1455/baostock_5m_cache/
ch12_as1455/baostock_raw_daily_cache/
ch12_as1455/as1455_daily_cache/
```

不得为 forward 单独复制这些缓存。

### 4.2 model data 构建

```text
scripts/build_ashare_ch12_as1455_model_data.py
```

构造复权 OHLCV、31 个模型特征和三个 forward 标签。训练母版继续保留 contract 校验所需 adjusted artifact。

### 4.3 forward 刷新

```text
scripts/refresh_as1455_forward_model_data.sh
```

默认：

```text
FORWARD_ARTIFACT_MODE=model_only
FORWARD_REPORT_MODE=compact
MIN_FREE_GB=5
```

构建和 schema 验证完成后，删除 forward 目录中可由共享缓存重建的：

```text
as1455_ohlcv_raw.h5
as1455_ohlcv_adj.h5
as1455_execution_metadata.h5
```

长期保留 `model_data_as1455.h5` 和 compact 报告。

### 4.4 live 保留

默认只保留最近 3 个日期。最近日期可保留完整审计；更早保留日期删除可重建的 raw/qfq history tail 和 feature panel tail。

---

## 5. 训练层

底层训练核心：

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

统一入口：

```text
scripts/run_as1455_target_fold_param_search.py
scripts/run_as1455_target_search_all.sh
```

r5/r21 wrapper 只设置目标，不复制训练循环。正式 checkpoint 仅来自：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

---

## 6. 历史回测协议

### 6.1 one-fold-lag

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

统一入口：

```text
scripts/run_as1455_target_one_lag_backtest.py
```

### 6.2 summary-first

```text
scripts/run_as1455_target_natural_backtest.sh
```

默认：

```text
OUTPUT_MODE=summary
MATERIALIZE_BEST=1
MATERIALIZED_OUTPUT_MODE=compact
```

完整网格只保留所有参数的指标 summary。网格结束后调用：

```text
scripts/materialize_as1455_best_run.py
```

仅重跑一个历史最佳配置为 compact/full，并默认删除其余 summary-only run 目录和日志。完整参数指标仍保留在 `02_summary`。

---

## 7. fold0-forward 协议

入口：

```text
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

### 7.1 正式默认

```text
MODEL_SELECTION_MODE=strict_oos
SELECTION_RANK_METRIC=sharpe
OUTPUT_MODE=compact
MIN_FREE_GB=5
```

步骤：

1. 找到对应 preset、target、自然调仓周期的最新成功历史目录；
2. 选择历史最佳完整 run；
3. 冻结模型信号与完整交易参数；
4. 使用 fold0 search-time checkpoint；
5. 用 feature-only 行保留规则选取 `date > fold0.test_end`；
6. 硬校验 `prediction_end == expected_prediction_end`；
7. 从初始现金和空仓开始回测；
8. 只保留冻结配置。

### 7.2 敏感性与兼容模式

```text
MODEL_SELECTION_MODE=forward_parameter_sweep
```

只冻结历史信号，forward 重新遍历交易参数，仅用于敏感性分析，不得作为 OOS 主结果。

```text
MODEL_SELECTION_MODE=all_top_n
```

保留旧全信号网格，仅用于兼容实验。

### 7.3 输出目录

使用秒级时间戳：

```text
<feature>_<target>_reb<period>_YYYYMMDD_HHMMSS
```

避免同日重复执行误用旧 run。跳过 prediction 或复用 run 时，后续还应继续加强 prediction SHA 和配置指纹校验。

---

## 8. 唯一交易语义来源

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

只有该函数决定调仓、买卖、T+1、主板/ST/停牌/涨跌停、整手、容量、公司行为、费用、持仓、NAV、换手、回撤和 round trip。

薄 grid 入口：

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
utils/as1455_grid_runner.py
```

禁止在 grid、wrapper、绘图或清理脚本中实现另一套交易语义。

---

## 9. 存储治理自动化

```text
scripts/check_as1455_disk_space.py
scripts/cleanup_as1455_storage.py
```

清理器规则：

- 默认 dry-run；
- `--apply` 前检测 AS1455 活动进程；
- 所有删除必须位于指定 base 内；
- 验证 forward model HDF 后才能删除重复 HDF；
- prediction HDF 存在时才删除对应 CSV；
- 可选清理明确列入政策的旧目录；
- 可选删除未被保留指标/信号选中的 run；
- 可选 gzip 大型审计 CSV；
- 每次生成 `cleanup_audit_*.json`。

共享行情缓存、训练 checkpoint、主训练 model data 和当前 contract 依赖文件不在自动清理清单中。

---

## 10. 开发规则

新增功能前必须回答：

1. 功能属于数据、特征、训练、预测协议、模型选择、grid、交易、绘图还是保留策略？
2. 唯一事实来源是什么？
3. 是否能增加公共纯函数，而不是复制脚本？
4. 是否改变 14:55 边界、标签、fold、checkpoint、完整配置选择或交易语义？
5. 历史训练与 forward 推理是否错误共用了目标标签过滤？
6. 是否会重复写入大文件？
7. 如何验证日期、参数、SHA、结果和磁盘占用？

wrapper 只允许设置默认参数、组合入口、确定输出目录、检查输入和输出上下文。

wrapper 禁止复制特征、checkpoint 推理、最佳配置排序、交易循环、summary/leaderboard 和目标周期映射。

---

## 11. 最低验证

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

覆盖：

- Python/Shell 语法和 CLI 导入；
- `TARGET_SPECS`；
- signal spec；
- failed-only summary 拒绝；
- forward 最新无标签日期保留；
- strict OOS 参数冻结；
- 排名缓存；
- 唯一交易引擎；
- 薄 wrapper 和统一绘图器；
- summary-first、model-only 和 strict OOS 默认值。

真实目录 engine smoke：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

`PARITY_CHECK_ONLY=1` 只验证单个 v7 引擎路径，不代表完整网格或正式 strict OOS 结果。正式结果仍需检查 prediction manifest、strict OOS manifest、NAV 日期范围和冻结配置一致性。
