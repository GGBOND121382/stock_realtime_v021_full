# Ch17 AS1455 开发大纲

本文档说明 `ch17_as1455` 相关代码的职责边界、唯一事实来源、兼容入口和开发约束。使用命令见：

```text
README_AS1455_R1_R5_R21.md
```

---

## 1. 固定业务口径

### 1.1 数据时点与交易约束

- 模型特征只能使用当日 14:55 及以前的数据；
- 历史执行价使用当日 15:00 收盘价近似收盘集合竞价成交价；
- raw 执行字段只进入执行面板、metadata 和质量报告；
- 回测为 long-only；
- 默认只交易沪深主板；
- 默认处理 T+1、停牌、涨跌停、100 股整手、手续费、印花税、过户费和公司行为。

### 1.2 目标与自然调仓周期

唯一事实来源：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

| 简称 | 标签 | lookahead | 调仓周期 | offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

禁止在新脚本中复制另一份目标—周期映射。

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
5 分钟缓存 + 原始日线缓存 + AS1455 日缓存
    ↓
model_data_as1455.h5
    ↓
公共特征、fold、checkpoint 和预测工具
    ↓
r1 / r5 / r21 参数搜索
    ↓
one-fold-lag 历史回测
    ↓
ch17_as1455_target_backtest 中按指标选历史最佳模型信号
    ↓
用相同 checkpoint 排名或 ensemble 定义组合 fold0 checkpoint
    ↓
fold0.test_end 后从空仓开始 forward 回测
    ↓
统一收益曲线绘制
```

---

## 3. 公共工具层

### 3.1 `utils/as1455_paths.py`

统一工程路径：

- 默认 model data；
- 默认执行日线缓存；
- target search 根目录；
- target backtest 根目录；
- fold0-forward 根目录；
- 绘图根目录。

新入口不得重新硬编码这些路径。

### 3.2 `utils/as1455_ch17_common.py`

训练和预测公共实现：

- `TARGET_SPECS`；
- 目标标签过滤；
- A/B 特征构造；
- fold 构造；
- checkpoint 目录规则；
- `search_best_checkpoints.csv` 读取；
- scaler 和 feature manifest 读取；
- 模型输入变换；
- checkpoint 推理；
- prediction HDF/CSV、actual、checkpoint 清单和 manifest 写出；
- grid 命令构造。

训练、one-fold-lag 和 fold0-forward 不得重新实现这些逻辑。

### 3.3 `utils/as1455_cli.py`

预测到回测阶段的公共 CLI：

- 模型数据、日期、`TOP_N` 和输出目录；
- prediction 文件；
- grid、执行缓存和输出模式；
- smoke、force、dry-run；
- 支持 top-N 自动 signal；
- 支持显式指定一个或多个 signal spec。

fold0-forward 使用显式 signal spec，只运行历史选中的模型信号。

### 3.4 `utils/as1455_signal_specs.py`

根据 checkpoint 数量生成合法信号：

- `TOP_N=1`：`model_0`；
- `TOP_N=5`：`model_0..model_4`、`ensemble_first3_mean`、`ensemble_all5_mean`；
- 其他数量只引用真实存在的预测列。

### 3.5 `utils/as1455_model_selection.py`

历史模型选择和绘图的共同事实来源。

职责：

1. 在给定 backtest root 下查找 `grid_summary_compact.csv` 或 `grid_summary.csv`；
2. 只使用有效的 `status=ok` 结果；
3. 按给定指标选择最佳完整 run；
4. 提取：

```text
signal_name
signal_cols
signal_mode
```

5. 生成明确的 signal spec；
6. 根据最大 signal column 推导 fold0 推理所需的 checkpoint 数量；
7. 记录历史 run 的交易参数，但不决定是否迁移这些参数。

默认指标：

```text
sharpe
```

fold0-forward 和绘图必须共同调用此模块，禁止各自复制一套“最佳”定义。

### 3.6 `utils/as1455_rank_cache.py`

每个 signal 的每日排名缓存：

- 按日期分组；
- 每日执行一次与 v7 一致的 score 降序排序；
- 返回显式预排序 frame；
- 不修改 pandas 全局状态；
- 不 monkey-patch v7；
- 不复制交易循环。

### 3.7 `utils/as1455_backtest_io.py`

回测编排公共函数：

- 构造统一 `TradeConfig`；
- 选择 `summary/compact/full` 输出；
- 写 config、summary、run metadata 和 CSV。

该文件不实现买卖、持仓或 NAV。

### 3.8 `utils/as1455_grid_runner.py`

in-process grid 编排：

- prediction 每个 signal 加载一次；
- 每个 signal 每日排序一次；
- execution panel 构造一次；
- 同一进程遍历所有参数；
- 每个组合调用唯一 v7 `backtest()`；
- 调用公共 output helper 写结果。

### 3.9 `utils/as1455_plotting.py`

统一绘图样式：

- 颜色、线型和 marker 同时区分；
- marker 稀疏显示；
- curve CSV 记录 `line_style` 和 `marker`。

---

## 4. 数据层

### 4.1 历史缓存更新

入口：

```text
scripts/run_as1455_live_data_feature_pipeline.sh history
```

核心：

```text
pipelines/as1455_update_history_to_prevday_fast_v4.py
```

职责：增量更新 5 分钟行情、原始日线和 AS1455 日缓存，并记录状态与错误。

### 4.2 历史 model_data 构建

```text
scripts/build_ashare_ch12_as1455_model_data.py
```

职责：

- 构造 AS1455 OHLCV 和复权因子；
- 构造 Ch12 31 个模型特征；
- 构造 `r01_fwd/r05_fwd/r21_fwd`；
- 输出 34 列 `model_data_as1455.h5`；
- 输出覆盖率、标签对齐、复权和泄漏检查报告。

### 4.3 fold0-forward 数据刷新

```text
scripts/refresh_as1455_forward_model_data.sh
```

只组合已有历史更新和 model data 构建器，不重新实现抓取、复权或特征。

---

## 5. 训练层

### 5.1 底层训练核心

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

提供：

- 基础输入契约；
- `MultipleTimeSeriesCV`；
- sector rotation、one-hot 和 compact add-on；
- TensorFlow 参数搜索；
- checkpoint、scaler 和 manifest 保存。

### 5.2 统一单 fold 入口

```text
scripts/run_as1455_target_fold_param_search.py
```

支持 A/B、r1/r5/r21、指定 fold 和 search-time top-N checkpoint。

### 5.3 批量训练入口

```text
scripts/run_as1455_target_search_all.sh
```

兼容入口：

```text
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
```

兼容入口只设置目标，不复制训练循环。

---

## 6. 预测协议层

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

r1 兼容入口只设置默认值，不维护独立预测循环。

### 6.2 fold0-forward

入口：

```text
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

默认协议：

```text
MODEL_SELECTION_MODE=historical_best
SELECTION_RANK_METRIC=sharpe
```

步骤：

1. 找到：

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
<preset>_<target>_reb<period>_*
```

中最新且包含有效 summary 的目录；
2. 按与绘图相同的规则选历史最佳完整 run；
3. 只继承其模型信号：

```text
signal_name + signal_cols + signal_mode
```

4. 用同样的 checkpoint 排名或 ensemble 定义组合 fold0 checkpoint；
5. 仅预测 `date > fold0.test_end`；
6. 每个 forward 配置从初始现金和空仓开始。

默认不继承历史：

```text
max_positions
sell_rank
rebalance_offset
```

这些字段只写入 manifest。forward 仍重新遍历交易参数。

保留旧实验模式：

```text
MODEL_SELECTION_MODE=all_top_n
```

该模式不是默认协议。

### 6.3 预测文件契约

```text
HDF key: /predictions
index:    (symbol, date)
columns:  0..N-1
```

fold0-forward manifest 必须记录：

```text
historical_model_selection.backtest_root
historical_model_selection.summary_file
historical_model_selection.rank_metric
historical_model_selection.run_name
historical_model_selection.signal_spec
historical_model_selection.required_top_n
historical_trading_parameters_reused
```

---

## 7. 回测层

### 7.1 唯一交易语义来源

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

只有该函数决定：调仓、买卖、T+1、主板/ST/停牌/涨跌停、整手、容量、公司行为、费用、持仓、NAV、换手、回撤和 round trip。

### 7.2 grid 入口

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

该文件是薄入口，编排位于：

```text
utils/as1455_grid_runner.py
```

禁止在 grid 中实现第二套交易循环。

### 7.3 自然周期 wrapper

```text
scripts/run_as1455_target_natural_backtest.sh
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
```

r5/r21 文件只设置目标。

---

## 8. 绘图层

唯一 Python 入口：

```text
scripts/plot_as1455_backtest_return_curves.py
```

唯一 shell 入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

绘图调用：

```text
utils/as1455_model_selection.py::select_best_run
```

因此绘图和 fold0-forward 使用同一个：

- summary 查找顺序；
- `status=ok` 过滤；
- 指标方向；
- 排序稳定性。

---

## 9. 已完成的重复开发治理

| 原问题 | 当前处理 |
|---|---|
| target/lookahead 多处定义 | 统一到 `TARGET_SPECS` |
| one-fold-lag 与 fold0-forward 各写 checkpoint 推理 | 统一到 `as1455_ch17_common.py` |
| wrapper 重复 CLI 和 grid 命令 | 统一到 `as1455_cli.py` |
| `TOP_N` 与 signal 列不匹配 | 统一到 `as1455_signal_specs.py` |
| grid 复制完整交易循环 | 删除；统一调用 v7 |
| 绘图另建 accessible monkey-patch | 删除；样式并入公共绘图层 |
| 绘图与 forward 各自定义“最佳模型” | 统一到 `as1455_model_selection.py` |
| r5/r21 训练和回测 shell 重复 | 改为通用入口 + 薄 wrapper |

---

## 10. 开发规则

新增功能前必须回答：

1. 功能属于数据、特征、训练、预测协议、模型选择、grid、交易还是绘图？
2. 该层唯一事实来源是什么？
3. 是否可以增加参数或公共纯函数，而不是复制脚本？
4. 是否改变 14:55 数据边界、标签、fold、checkpoint、模型选择或交易语义？
5. 如何验证旧结果不变？

wrapper 只允许：

- 设置默认参数；
- 组合已有入口；
- 确定输出目录；
- 检查输入存在性；
- 输出运行上下文。

wrapper 禁止：

- 重新实现特征；
- monkey-patch 业务函数；
- 复制 checkpoint 推理；
- 复制“最佳模型”排序；
- 复制交易循环；
- 复制 summary/leaderboard；
- 维护另一份目标周期映射。

---

## 11. 最低验证集

结构检查：

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

其中包含：

- Python 和 shell 语法；
- CLI 导入；
- 目标映射；
- signal spec；
- 历史模型选择合成测试；
- 排名缓存；
- 唯一交易引擎；
- 薄 wrapper；
- 唯一绘图器。

真实目录 smoke：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

必须先打印 `[MODEL SELECT]`，再完成 v7 单配置 smoke。
