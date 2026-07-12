# Ch17 AS1455 开发大纲

本文档说明 `ch17_as1455` 相关代码的职责边界、唯一事实来源、兼容入口和开发约束。具体运行命令见：

```text
README_AS1455_R1_R5_R21.md
```

---

## 1. 固定业务口径

### 1.1 AS1455 数据时点

- 模型特征只能使用当日 14:55 及以前的数据；
- 历史执行价使用当日 15:00 收盘价近似收盘集合竞价成交价；
- raw 执行字段只进入执行面板、metadata 和质量报告，不进入 34 列 `model_data`；
- 回测为 long-only；
- 默认只交易沪深主板；
- 默认处理 T+1、停牌、涨跌停、整手、手续费、印花税、过户费和公司行为。

### 1.2 目标和自然调仓周期

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
| A | `rotation_onehot` | 原始 31 特征 + 完整 sector rotation + sector one-hot |
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
one-fold-lag 或 fold0-forward 预测
    ↓
预测 HDF（key=/predictions）
    ↓
参数网格编排
    ↓
唯一 v7 交易引擎
    ↓
NAV / 订单 / 拒单 / 持仓 / 费用 / 换手 / leaderboard
    ↓
统一收益曲线绘制
```

---

## 3. 公共工具层

### 3.1 `utils/as1455_paths.py`

统一工程路径：

- 工程根目录；
- 默认 `model_data_as1455.h5`；
- 默认原始日线缓存；
- 默认 in-process grid；
- target search、target backtest、fold0-forward 和绘图根目录。

新脚本不得再次硬编码这些默认路径。

### 3.2 `utils/as1455_ch17_common.py`

训练和预测公共实现：

- `TARGET_SPECS`；
- 按目标过滤标签缺失；
- A/B 特征构造；
- 按 lookahead 构造 fold；
- r1/r5/r21 默认训练目录和 checkpoint 目录；
- 读取 `search_best_checkpoints.csv`；
- 读取 `scaler.pkl` 和 `feature_manifest.json`；
- 按训练期列顺序执行 scaler 变换；
- 加载 `.keras` checkpoint 并预测；
- 写 predictions HDF/CSV、actual、checkpoint 清单和 manifest；
- 构造 grid 命令。

训练、one-fold-lag 和 fold0-forward 不得重新实现这些逻辑。

### 3.3 `utils/as1455_cli.py`

预测到回测阶段的公共 CLI：

- 模型数据、日期、`TOP_N`、输出目录；
- prediction 文件和 `--skip-predictions`；
- grid、执行缓存、费用输出模式；
- smoke、force、dry-run；
- 统一调用 grid。

one-fold-lag 和 fold0-forward 只保留各自协议参数，不再复制整套 grid 参数和命令拼装。

### 3.4 `utils/as1455_signal_specs.py`

根据实际 `TOP_N` 生成合法信号：

- `TOP_N=1`：只生成 `model_0`；
- `TOP_N=5`：生成 `model_0..model_4`、`ensemble_first3_mean` 和 `ensemble_all5_mean`；
- 其他数量只引用真实存在的预测列。

### 3.5 `utils/as1455_backtest_io.py`

回测编排公共函数：

- 从 grid tuple 构造统一 `TradeConfig`；
- 按 `summary/compact/full` 选择输出；
- 写 config、summary、run metadata 和 CSV。

该文件不实现买卖、持仓或 NAV 逻辑。

### 3.6 `utils/as1455_plotting.py`

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

职责：

- 检查缓存最后日期；
- 增量更新 5 分钟行情和原始日线；
- 聚合 AS1455 日数据；
- 写状态和错误报告。

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

只负责组合：

```text
history 更新
+ build_ashare_ch12_as1455_model_data.py
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

该脚本不得重新实现行情抓取、复权或特征。

### 4.4 live 当日特征

```text
pipelines/as1455_live_prepare.py
features/as1455_live_common.py
features/build_as1455_live_features.py
```

live 路径与历史 model_data 构建路径用途不同，禁止相互替代。

---

## 5. 特征和训练层

### 5.1 底层训练核心

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
```

当前仍提供：

- 31 列基础输入契约；
- `MultipleTimeSeriesCV`；
- sector rotation；
- sector one-hot；
- TensorFlow 参数搜索；
- checkpoint、scaler 和 manifest 保存。

### 5.2 compact add-on

```text
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

公共特征函数：

```text
add_compact_addon_features()
```

### 5.3 统一单 fold 入口

```text
scripts/run_as1455_target_fold_param_search.py
```

支持：

- A/B；
- r1/r5/r21；
- 指定 fold；
- 原参数网格；
- search-time top-N checkpoint；
- 可选诊断性 retrain。

默认目录通过同一个 fold 目录模板确定：

- r1 继续使用原有历史目录，兼容已训练模型；
- r5/r21 使用 `ch17_as1455_target_search/<preset>/<target>/foldN_search`。

### 5.4 批量训练入口

通用入口：

```text
scripts/run_as1455_target_search_all.sh
```

兼容入口：

```text
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
```

两个兼容入口只设置 `TARGET_COL`。

### 5.5 正式训练产物契约

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

正式回测不得使用诊断性 retrain 的 `models/best_*.keras` 替代 search-time checkpoint。

---

## 6. 预测协议层

### 6.1 one-fold-lag

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

统一实现：

```text
scripts/run_as1455_target_one_lag_backtest.py
```

只负责：

- source/target fold 计划；
- 合并各 target fold 预测；
- 调用公共预测和 grid 工具。

r1 兼容入口：

```text
scripts/run_as1455_rotation_one_lag_daily_backtest.py
scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py
```

这两个文件是薄 wrapper，不再维护 checkpoint 推理或 monkey-patch。

### 6.2 fold0-forward

协议：

- 使用 fold0 search-time checkpoint、scaler 和 feature manifest；
- 日期严格满足 `date > fold0.test_end`；
- 不重新训练；
- 从初始资金和空仓开始；
- 不继承 fold0 测试期持仓。

实现：

```text
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

默认：

```text
TOP_N=1
```

即默认只测试 fold0 最优单模型 `model_0`。需要 top5/ensemble 时显式设置 `TOP_N=5`。

### 6.3 预测文件契约

```text
HDF key: /predictions
index:    (symbol, date)
columns:  0..N-1
```

所有协议统一通过 `write_prediction_artifacts()` 写出。

---

## 7. 回测层

### 7.1 唯一交易语义来源

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

只有该函数决定：

- 调仓日；
- 买卖条件；
- T+1；
- 主板/ST/停牌/涨跌停；
- 整手；
- 容量和部分成交；
- 公司行为；
- 费用；
- 持仓成本；
- NAV、换手、回撤和 round trip。

### 7.2 subprocess 兼容网格

```text
code/backtest/run_as1455_close_auction_grid_v1.py
```

保留：

- signal spec；
- 参数组合；
- run name；
- summary 展平；
- leaderboard。

### 7.3 in-process 网格

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

当前引擎：

```text
inprocess_shared_data_v3
```

已实现：

- prediction 每个 signal 加载一次；
- execution panel 构造一次；
- universe、ST、公司行为和容量输入加载一次；
- 所有参数在一个进程中执行；
- 每个组合直接调用唯一 v7 `backtest()`；
- 统一使用 `utils/as1455_backtest_io.py` 写结果。

已删除：

- `backtest_prepared()` 第二套交易循环；
- `inspect.getsource + exec` 动态改写；
- 两套费用、持仓和 NAV 语义。

### 7.4 当前性能边界

当前仍由 v7 在每个参数组合内部逐日排序：

```text
daily_rankings_built_once_per_signal = false
```

因此：

- 数据加载和 execution panel 重复已消除；
- 4410 个独立 Python 子进程已消除；
- 不同配置之间仍重复排序。

后续要共享每日排名，必须改造 v7 公共接口；禁止在 grid 中复制交易循环。

### 7.5 自然周期 wrapper

通用入口：

```text
scripts/run_as1455_target_natural_backtest.sh
```

兼容入口：

```text
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
```

兼容入口只设置 `TARGET_COL`。

---

## 8. 绘图层

唯一 Python 入口：

```text
scripts/plot_as1455_backtest_return_curves.py
```

shell 入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

职责：

- 查找 grid summary；
- 按指标选择最优 run；
- 读取 NAV；
- 生成 daily/weekly/monthly 曲线；
- 保存实际选中的参数；
- 使用统一线型和 marker。

已删除重复实现：

```text
scripts/plot_as1455_backtest_return_curves_accessible.py
```

---

## 9. 已完成的重复开发治理

| 原问题 | 当前处理 |
|---|---|
| target、lookahead、调仓周期多处定义 | 统一到 `TARGET_SPECS` |
| one-fold-lag 和 fold0-forward 各自复制 checkpoint 推理 | 抽到 `as1455_ch17_common.py` |
| 两个协议各自复制 grid CLI 和命令 | 抽到 `as1455_cli.py` |
| `TOP_N=1` 仍请求 model1..4 | 由 `as1455_signal_specs.py` 动态生成 |
| r1 A/B 各维护一套预测逻辑 | 改成薄 wrapper |
| r05/r21 训练 shell 重复 | 统一到 `run_as1455_target_search_all.sh` |
| r05/r21 回测 shell 重复 | 统一到 `run_as1455_target_natural_backtest.sh` |
| in-process grid 复制完整交易循环 | 删除，统一调用 v7 `backtest()` |
| 回测输出逻辑散落 | 抽到 `as1455_backtest_io.py` |
| 无障碍绘图另建 monkey-patch | 合并到唯一绘图器并删除包装 |
| 默认路径散落 | 抽到 `as1455_paths.py` |

---

## 10. 结构回归检查

统一检查入口：

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

检查内容：

- Python 语法；
- shell 语法；
- 关键 CLI 可导入；
- `TARGET_SPECS` 固定映射；
- `TOP_N` 与 signal 数量匹配；
- in-process grid 不含第二套 `backtest`；
- 兼容 wrapper 不复制 checkpoint 或交易函数；
- 旧 accessible 绘图包装已经删除；
- 唯一绘图器调用公共样式。

数据级 smoke：

```bash
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r05_natural_backtest.sh
```

输出必须包含：

```text
[PARITY] single v7 trade engine smoke run ...
[PARITY] PASS
[PARITY] check-only completed; grid was not executed
```

这里的 `PARITY` 名称为兼容旧参数名，当前实际含义是“唯一 v7 引擎单配置 smoke”，不是比较两套交易实现。

---

## 11. 修改后的结果一致性要求

对固定预测文件和固定参数，重构前后至少比较：

```text
close_auction_nav.csv
close_auction_orders.csv
close_auction_rejections.csv
round_trips.csv
summary.json
```

数值容差：

```text
rtol=1e-12
atol=1e-12
```

允许不同：

- 输出目录名；
- 创建时间；
- grid engine metadata。

不允许不同：

- 交易日期；
- 买卖方向；
- 成交数量；
- 费用；
- 持仓；
- NAV；
- 收益和风险指标。

---

## 12. 剩余技术债务

### 12.1 v7 原生排名缓存

优先级：高。

目标：在不复制交易循环的前提下，使 v7 原生接收预排序信号或排名提供器，从而实现每个 signal 每日只排序一次。

### 12.2 训练底层进一步拆分

当前：

```text
run_as1455_sector_rotation_fold0_param_search.py
```

仍同时承担 CV、特征、训练和 checkpoint 保存。后续可拆为：

```text
utils/as1455_features.py
utils/as1455_cv.py
utils/as1455_training.py
```

拆分前必须以现有 fold report、checkpoint 排名和 IC 结果做回归基准。

### 12.3 live 部署契约

旧 live bundle 使用旧 Ch17 `.weights.h5` 和重新拟合 scaler；target-aware A/B 使用 search-time `.keras + scaler.pkl + feature_manifest.json`。两者不能靠路径猜测混用，需要独立设计 deploy manifest。

---

## 13. 开发规则

wrapper 只允许：

- 设置默认参数；
- 组合已有入口；
- 确定输出目录；
- 检查输入；
- 输出运行上下文。

wrapper 禁止：

- 重新实现特征；
- monkey-patch 业务函数；
- 复制 checkpoint 推理；
- 复制交易循环；
- 复制 summary/leaderboard；
- 维护另一份目标或周期映射。

新增功能前必须确认：

1. 属于数据、特征、训练、预测协议、grid、交易还是绘图；
2. 该层唯一事实来源；
3. 是否可通过参数或公共函数完成；
4. 是否改变 14:55 数据边界、标签、fold、checkpoint 或交易语义；
5. 如何验证结果不变。
