# Ch17 AS1455 开发大纲与重复开发审计

本文档用于统一 `ch17_as1455` 相关开发口径，明确各层代码的职责、唯一事实来源、输入输出契约和兼容边界，并记录当前已经确认的重复开发问题。

使用说明文档见：

```text
README_AS1455_R1_R5_R21.md
```

本文档重点回答两个问题：

1. 当前 AS1455 数据、训练、预测、回测、绘图和实盘代码分别做什么；
2. 新功能应该修改哪一层，避免重新复制一套已有逻辑。

---

## 1. 业务与实验口径

### 1.1 信号与成交时点

- AS1455 特征使用当日不晚于 14:55 的 5 分钟行情；
- 历史回测以当日 15:00 收盘价近似收盘集合竞价成交价；
- 回测是 long-only；
- 默认只交易沪深主板；
- 默认包含 T+1、涨跌停、停牌/不可交易、100 股整手、手续费、印花税和过户费约束。

### 1.2 监督目标与自然调仓周期

| 简称 | 标签 | lookahead | 自然调仓周期 | 完整 offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

### 1.3 特征方案

| 名称 | `feature_preset` | 内容 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + 完整 sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

---

## 2. 端到端架构

```text
历史/实时行情
    |
    v
原始 5 分钟缓存 + 原始日线缓存 + AS1455 日缓存
    |
    v
model_data_as1455.h5
    |
    +--> A/B 特征构造
    |       |
    |       v
    |    r1/r5/r21 分 fold 参数搜索
    |       |
    |       v
    |    search-time checkpoints + scaler + feature manifest
    |       |
    |       +--> one-fold-lag 历史评估
    |       |
    |       +--> fold0 后续日期 forward 评估
    |       |
    |       +--> live/deploy 推理
    |
    v
预测 HDF（key=/predictions）
    |
    v
信号排序 + 参数网格
    |
    v
唯一交易语义引擎
    |
    v
NAV / 订单 / 拒单 / 持仓 / 费用 / 换手 / leaderboard
    |
    v
收益曲线与结果汇总
```

---

## 3. 数据层

### 3.1 历史缓存增量更新

#### 入口

```text
scripts/run_as1455_live_data_feature_pipeline.sh
```

`history` 模式负责将以下缓存更新到最近一个已完成交易日：

```text
saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/
saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache/
```

#### 核心实现

```text
pipelines/as1455_update_history_to_prevday_fast_v4.py
```

职责：

- 扫描实际缓存文件的最后日期；
- 增量抓取原始 5 分钟行情；
- 增量抓取原始日线；
- 从 5 分钟缓存聚合 AS1455 日数据；
- 记录逐股票更新状态和错误；
- 支持只修复 AS1455 聚合、不重新下载原始行情。

#### 兼容实现

```text
pipelines/as1455_update_history_to_prevday.py
```

`fast_v4` 仍会复用其中的数据源查询、字段标准化和基础工具。新开发不应再复制这些函数。

### 3.2 离线 model_data 构建

#### 唯一完整构建器

```text
scripts/build_ashare_ch12_as1455_model_data.py
```

职责：

- 读取 5 分钟缓存；
- 聚合 14:55 截止的 AS1455 OHLCV；
- 使用原始日线 `close/preclose` 构造复权因子；
- 构造 Ch12 风格特征；
- 构造 `r01_fwd/r05_fwd/r21_fwd`；
- 输出 34 列 `model_data_as1455.h5`；
- 输出覆盖率、泄漏、复权、质量、标签对齐等报告。

### 3.3 live 当日特征路径

```text
pipelines/as1455_live_prepare.py
data_collection/collect_as1455_live_quotes_as1455.py
features/as1455_live_common.py
features/build_as1455_live_features.py
```

这条路径服务于盘中/实盘，不应替代历史 `model_data` 构建器。

### 3.4 forward 数据刷新包装

```text
scripts/refresh_as1455_forward_model_data.sh
```

当前职责：

1. 调用已有 `history` 模式更新缓存；
2. 调用已有完整构建器生成：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

该文件是流程包装，不应自行实现行情抓取、复权或特征计算。

---

## 4. 特征与训练层

### 4.1 基础训练与公共函数来源

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
```

当前同时提供：

- 31 列基础输入加载；
- `MultipleTimeSeriesCV`；
- sector rotation 特征；
- sector one-hot；
- 参数网格；
- TensorFlow 训练；
- checkpoint、scaler 和 manifest 保存。

该文件目前承担职责过多。短期内它仍是训练公共函数来源；长期应拆出纯函数模块。

### 4.2 compact add-on 特征

```text
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

核心可复用函数：

```text
add_compact_addon_features()
```

其余命令行和训练流程属于 r1 早期入口。

### 4.3 target-aware 公共逻辑

```text
scripts/as1455_target_label_common.py
```

职责：

- 定义 `r01_fwd/r05_fwd/r21_fwd` 与 lookahead 映射；
- 按目标列执行 `target_only` 缺失值过滤；
- 按目标 lookahead 生成 fold。

### 4.4 当前推荐的单 fold 搜索入口

```text
scripts/run_as1455_target_fold_param_search.py
```

支持：

- A/B 两套特征；
- r1/r5/r21 三个目标；
- fold0..fold6；
- 原参数网格；
- 保存 search-time top-N checkpoints；
- 可选诊断性 retrain。

### 4.5 shell 批量入口

```text
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
```

二者只应该设置默认目标和 fold 范围，不应包含新的训练算法。

### 4.6 训练产物契约

正式回测使用：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

禁止用诊断性 retrain 的 `models/best_*.keras` 替代搜索阶段 checkpoint。

---

## 5. 预测协议层

预测协议必须明确，不能仅以“回测”统称。

### 5.1 one-fold-lag 历史评估

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

核心旧实现：

```text
scripts/run_as1455_rotation_one_lag_daily_backtest.py
```

目标感知入口：

```text
scripts/run_as1455_target_one_lag_backtest.py
```

r1 B 兼容入口：

```text
scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py
```

### 5.2 fold0 forward 评估

协议：

- 使用 fold0 搜索阶段保存的 checkpoint、scaler 和 feature manifest；
- 预测日期严格满足：

```text
date > fold0.test_end
```

- 不重新训练；
- 从初始资金和空仓开始；
- 与 fold0 测试窗口不拼接持仓。

当前入口：

```text
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.3 live/deploy 推理

旧 Ch17 基线模型部署路径：

```text
tools/create_as1455_sharpe1_checkpoint_bundle_v1.py
prediction/run_as1455_live_checkpoint_ensemble_inference_v1.py
scripts/run_as1455_live_checkpoint_signal_v1.sh
```

该路径针对旧 `run_ashare_ch17_nn_reproduce.py` 的 checkpoint 格式，并不等同于 A/B target-aware 搜索产物。不能直接混用。

### 5.4 预测文件契约

```text
HDF key: /predictions
index:    (symbol, date)
columns:  0..N-1
```

默认 top5 时，网格可构造：

```text
model_0..model_4
ensemble_first3_mean
ensemble_all5_mean
```

`top_n=1` 才是严格的“单个最优模型”。

---

## 6. 回测层

### 6.1 唯一交易语义来源

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
```

应由该文件唯一决定：

- 调仓日；
- 卖出条件；
- 买入候选；
- T+1；
- 主板/ST/停牌/涨跌停约束；
- 整手；
- 容量与部分成交；
- 公司行为；
- 费用；
- 持仓成本；
- NAV、换手、胜率、回撤和 round trip。

任何优化版网格都不应再复制这一套交易循环。

### 6.2 旧 subprocess 网格

```text
code/backtest/run_as1455_close_auction_grid_v1.py
```

职责：

- 定义 signal specs；
- 构造参数组合；
- 生成 run name；
- 汇总 summary；
- 生成 leaderboard；
- 每个组合启动一次 v7 子进程。

### 6.3 当前 in-process 网格

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

当前优化目标：

- execution panel 只构造一次；
- 每个 signal 每日排序一次；
- 所有参数组合复用排序；
- 同一进程执行所有组合。

当前存在的重要架构债务见第 9 节。

### 6.4 自然周期 wrapper

```text
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
```

仅设置：

- target；
- rebalance_every；
- offset_mode；
- target folds；
- 输出目录。

---

## 7. 绘图层

### 7.1 基础绘图器

```text
scripts/plot_as1455_backtest_return_curves.py
```

职责：

- 查找 grid summary；
- 按 Sharpe、收益或 Calmar 选择最优 run；
- 读取 NAV；
- 绘制 daily/weekly/monthly 累计收益；
- 保存选中参数清单。

### 7.2 shell 入口

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

职责：

- 接收多个回测根目录；
- 接收对应标签；
- 传递 rank metric、输出目录和频率。

### 7.3 当前无障碍绘图包装

```text
scripts/plot_as1455_backtest_return_curves_accessible.py
```

增加线型和 marker，但当前实现存在重复开发，见第 9 节。

---

## 8. 当前代码的唯一事实来源规则

### 8.1 必须只有一个实现的位置

| 逻辑 | 唯一事实来源 |
|---|---|
| 目标与 lookahead | `as1455_target_label_common.py` |
| sector rotation | `run_as1455_sector_rotation_fold0_param_search.py`，后续应抽离 |
| compact add-on | `run_as1455_first_batch_features_fold0_param_search.py`，后续应抽离 |
| checkpoint/scaler/manifest 读取 | 应抽为单独公共模块 |
| 交易语义 | `run_as1455_close_auction_backtest_v7_maxpos_grid.py` |
| 参数组合和 leaderboard | `run_as1455_close_auction_grid_v1.py` 或其抽离公共模块 |
| 收益曲线选择和加载 | `plot_as1455_backtest_return_curves.py` |
| 历史缓存更新 | `as1455_update_history_to_prevday_fast_v4.py` |
| 完整 model_data 构建 | `build_ashare_ch12_as1455_model_data.py` |

### 8.2 wrapper 允许做的事

wrapper 只能：

- 设置默认参数；
- 组合已有入口；
- 确定输出目录；
- 做输入存在性检查；
- 输出运行上下文。

wrapper 不应：

- 重写特征函数；
- monkey-patch 另一个模块；
- 复制模型预测循环；
- 复制交易循环；
- 复制 summary/leaderboard 逻辑。

### 8.3 新功能开发前检查

新增代码前必须回答：

1. 该功能属于数据、特征、训练、预测协议、交易、网格还是绘图？
2. 这一层现有唯一事实来源是哪一个文件？
3. 能否通过新增参数或纯函数扩展，而不是新建并复制脚本？
4. 修改后如何证明旧结果不变？
5. 新旧路径是否生成相同 schema 和 manifest？

---

## 9. 重复开发审计

### 9.1 高风险：in-process 网格复制了 v7 交易循环

文件：

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

问题：

- `backtest_prepared()` 重新实现了现金、持仓、卖出、买入、T+1、费用、部分成交、公司行为、NAV、持仓明细和 round trip；
- 原 v7 `backtest()` 中已经存在同一套业务逻辑；
- 两套实现以后很容易只修一边；
- 当前 parity check 只抽查第一个配置，不能证明所有 profile、capacity、公司行为和异常分支永远一致。

结论：

```text
存在严重重复开发；这是当前最高优先级架构债务。
```

正确方向：

- 将 v7 的交易循环改造成可接收“预排序信号提供器”和“预构造 execution panel”的核心函数；
- 原 v7 单次回测和 in-process grid 都调用同一个函数；
- in-process grid 只保留缓存、参数循环和结果写出。

### 9.2 高风险：fold0 forward 脚本复制了 one-fold-lag 预测编排

文件：

```text
scripts/run_as1455_fold0_forward_backtest.py
```

已复用部分：

- target-aware 特征构造；
- preprocess 读取；
- feature manifest 对齐；
- checkpoint 路径解析；
- top checkpoint 读取。

重复部分：

- TensorFlow checkpoint 遍历和预测循环；
- prediction HDF/CSV 写出；
- actual 文件写出；
- checkpoint 清单写出；
- prediction manifest 组织；
- grid 命令拼装；
- 大量回测参数 argparse 定义。

这些逻辑在：

```text
run_as1455_rotation_one_lag_daily_backtest.py
run_as1455_target_one_lag_backtest.py
```

中已经存在相同结构。

结论：

```text
存在明确重复开发；协议不同，但公共预测执行层不应该复制。
```

正确方向：

抽出公共函数：

```text
build_target_feature_matrix(...)
load_search_artifacts(...)
predict_checkpoint_set(...)
write_prediction_artifacts(...)
build_grid_command(...)
```

one-fold-lag 和 fold0-forward 只负责生成不同的日期选择/源模型计划。

### 9.3 中风险：fold0 forward shell 重复 target 调度

文件：

```text
scripts/run_as1455_fold0_forward_backtests.sh
```

重复内容：

- r1/r5/r21 到 rebalance_every 的映射；
- r1/r5/r21 到 offset_mode 的映射；
- A/B 循环；
- 参数列表；
- output mode；
- grid 开关。

类似内容已经存在于 r5/r21 natural wrapper。

结论：

```text
存在 wrapper 级重复，风险低于交易和预测复制，但会导致默认值漂移。
```

正确方向：

建立一个通用入口：

```text
run_as1455_target_backtest.sh
```

参数包含：

```text
PROTOCOL=one_lag|fold0_forward
TARGETS=...
FEATURE_PRESETS=...
```

原 wrapper 只做兼容转发。

### 9.4 中风险：无障碍绘图复制并 monkey-patch 基础绘图函数

文件：

```text
scripts/plot_as1455_backtest_return_curves_accessible.py
```

问题：

- 重新实现 `plot_frequency()`；
- 运行时执行 `base.plot_frequency = plot_frequency`；
- 基础绘图器和 accessible 包装会逐渐漂移；
- 这是此前已经出现过的 monkey-patch 型技术债务。

结论：

```text
存在不必要的重复开发。
```

正确方向：

直接在基础绘图器中支持：

```text
--distinguish-mode color_line_marker
--markers ...
--line-styles ...
```

默认使用颜色、线型和 marker 三重区分，然后删除 accessible 包装。

### 9.5 低风险且基本可接受：forward 数据刷新包装

文件：

```text
scripts/refresh_as1455_forward_model_data.sh
```

优点：

- 没有重新实现 BaoStock 抓取；
- 没有重新实现 AS1455 聚合；
- 没有重新实现复权、特征或标签；
- 只组合已有 history 和完整 model_data 构建器。

问题：

- 同一缓存目录同时作为 `--bar-root` 和 `--baostock-5m-cache-dir` 传入，构建器会扫描同一批文件两次，虽然字典 `setdefault` 可避免重复使用，但报告来源会重复；
- 参数名和路径默认值仍与其他 shell 重复；
- 默认每次 forward 回测都刷新和重建，成本较高。

结论：

```text
没有复制核心业务算法；属于可保留的组合 wrapper，但应消除重复扫描并明确缓存策略。
```

### 9.6 已有历史技术债务

不仅是刚新增代码，旧代码中也已存在以下重复：

1. `run_as1455_rotation_addon_one_lag_daily_backtest.py` monkey-patch 基础 one-lag 模块；
2. `run_as1455_target_one_lag_backtest.py` monkey-patch基础 feature builder 和 fold getter；
3. r05 与 r21 搜索 shell 基本相同；
4. r05 与 r21 回测 shell 基本相同；
5. CV splitter、schema 常量、symbol normalize 在训练、live 推理和回测中多处定义；
6. grid 命令构造在 r1、target-aware 和 fold0-forward 中重复；
7. r1 使用旧目录结构，r5/r21 使用统一 target-search 目录，增加特殊分支。

---

## 10. 对刚新增 fold0 forward 功能的准确结论

### 10.1 功能口径

当前代码能表达以下协议：

- 从更新后的 model_data 读取 fold0 之后的有标签日期；
- 加载 fold0 搜索 checkpoint/scaler/manifest；
- 仅预测 `date > fold0.test_end`；
- 从空仓和初始资金开始；
- 运行 r1/r5/r21 自然调仓周期；
- 输出完整回测和绘图所需文件。

### 10.2 不应保留的实现方式

当前不应作为长期正式架构保留：

```text
run_as1455_fold0_forward_backtest.py 中复制的预测循环和 run_grid
run_as1455_fold0_forward_backtests.sh 中复制的 target 分发
plot_as1455_backtest_return_curves_accessible.py 的 monkey-patch
run_as1455_close_auction_grid_inprocess.py 中复制的交易循环
```

### 10.3 语义问题

用户表述“fold0 的最优模型”是单数，而 wrapper 默认：

```text
TOP_N=5
```

这会生成 5 个单模型信号和 2 个 ensemble 信号。正式命令必须明确选择：

```text
TOP_N=1  # 单个最优模型
TOP_N=5  # 延续现有 model_0..4 + ensemble 研究口径
```

不能把两者混称为同一实验。

---

## 11. 整理计划

### 阶段 1：冻结业务语义

先保存以下代表配置的新旧输出：

- r1/r5/r21；
- A/B；
- rebalance offset 0 和非 0；
- max_positions 5/25；
- sell_rank 75/300；
- `close_auction_simple` 与 `close_auction_skip_limit`；
- capacity none 与一个 capacity 模式；
- full 输出。

比较：

```text
NAV
orders
rejections
positions
round_trips
summary
```

### 阶段 2：抽公共预测层

建议新增纯模块：

```text
ch17_as1455/targets.py
ch17_as1455/features.py
ch17_as1455/search_artifacts.py
ch17_as1455/prediction.py
ch17_as1455/grid_command.py
```

要求：

- 不包含 CLI；
- 不写全局 monkey-patch；
- 输入输出可单元测试；
- one-lag 与 forward 只提供 prediction plan。

### 阶段 3：消除双交易引擎

将 v7 拆成：

```text
prepare_execution_data(...)
run_portfolio(dates, daily_rank_provider, exec_by_date, config, ...)
write_backtest_outputs(...)
```

原单次 v7 和 in-process grid 共同调用 `run_portfolio()`。

完成后删除 `backtest_prepared()` 中的复制交易逻辑。

### 阶段 4：合并 wrapper

统一：

```text
run_as1455_target_search_all.sh
run_as1455_target_backtest.sh
```

旧 r05/r21/r1 wrapper 暂时保留，但只转发参数，不再维护各自逻辑。

### 阶段 5：合并绘图

将 line style/marker 直接纳入：

```text
plot_as1455_backtest_return_curves.py
```

删除：

```text
plot_as1455_backtest_return_curves_accessible.py
```

### 阶段 6：统一目录

新实验统一：

```text
saved_data/ashare_ml4t/ch17_as1455/
  search/{preset}/{target}/fold{n}/
  predictions/{protocol}/{preset}/{target}/{run_id}/
  backtests/{protocol}/{preset}/{target}/{run_id}/
  plots/{comparison}/{run_id}/
```

旧目录只读兼容，不再继续扩展新的特殊命名。

---

## 12. 验证门槛

任何重构提交必须同时满足：

1. Python `py_compile` 通过；
2. shell `bash -n` 通过；
3. A/B 特征列顺序与旧 manifest 完全一致；
4. 固定 checkpoint 的预测逐元素一致；
5. 固定回测配置的 NAV/订单/拒单/持仓/round-trip 一致；
6. summary 关键指标一致；
7. r1/r5/r21 日期边界和 lookahead 报告正确；
8. 输出 schema 不变；
9. manifest 记录 protocol、target、preset、checkpoint、日期范围和初始状态；
10. 不允许新增第二套交易循环、特征计算或 checkpoint 读取实现。

---

## 13. 当前审计结论

```text
当前工程确实存在重复开发，而且刚新增代码中也存在。
```

按风险排序：

1. **最高风险**：in-process grid 复制 v7 完整交易循环；
2. **高风险**：fold0-forward 复制 checkpoint 预测和 grid 编排；
3. **中风险**：accessible 绘图复制函数并 monkey-patch；
4. **中低风险**：多个 shell wrapper 重复 target 和参数默认值；
5. **低风险**：forward 数据刷新 wrapper 主要是合理组合，但存在重复扫描和默认参数重复。

在完成上述整理前，新增 r1/r5/r21 协议、特征方案、交易规则或绘图功能时，应先修改现有唯一事实来源，不再新建平行实现。
