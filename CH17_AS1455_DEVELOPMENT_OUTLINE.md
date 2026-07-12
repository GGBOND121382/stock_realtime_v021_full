# Ch17 AS1455 开发大纲

本文档说明 `ch17_as1455` 相关代码的职责边界、唯一事实来源、兼容入口和开发约束。具体命令见：

```text
README_AS1455_R1_R5_R21.md
```

---

## 1. 固定业务口径

### 1.1 数据时点与交易约束

- 模型特征只能使用当日 14:55 及以前的数据；
- 历史执行价使用当日 15:00 收盘价近似收盘集合竞价成交价；
- raw 执行字段只进入执行面板、metadata 和质量报告，不进入 34 列 `model_data`；
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
公共 grid runner：共享数据 + 每个 signal 每日排序一次
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

- 默认 `model_data_as1455.h5`；
- 默认原始日线缓存；
- 默认 in-process grid；
- target search、target backtest、fold0-forward 和绘图根目录。

新 Python 入口不得重新硬编码这些默认路径。

### 3.2 `utils/as1455_ch17_common.py`

训练和预测公共实现：

- `TARGET_SPECS`；
- 按目标过滤标签缺失；
- A/B 特征构造；
- 按 lookahead 构造 fold；
- r1/r5/r21 checkpoint 目录规则；
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
- grid、执行缓存和输出模式；
- smoke、force、dry-run；
- 统一调用 grid。

one-fold-lag 和 fold0-forward 只保留各自协议参数。

### 3.4 `utils/as1455_signal_specs.py`

根据实际 `TOP_N` 生成合法信号：

- `TOP_N=1`：只生成 `model_0`；
- `TOP_N=5`：生成 `model_0..model_4`、`ensemble_first3_mean` 和 `ensemble_all5_mean`；
- `TOP_N=3` 不重复生成等价的 `first3` 和 `all3`；
- 其他数量只引用真实存在的预测列。

### 3.5 `utils/as1455_rank_cache.py`

每个 signal 的每日排名缓存：

1. 按日期分组；
2. 对每个日期执行一次与 v7 完全相同的：

```python
sort_values("score", ascending=False)
```

3. 返回显式 `PreSortedPredictionFrame`；
4. v7 后续对同一日执行相同排序请求时只返回副本，不再次排序。

该方案不修改 pandas 全局状态，不 monkey-patch v7，也不复制交易循环。

### 3.6 `utils/as1455_backtest_io.py`

回测编排公共函数：

- 从 grid tuple 构造统一 `TradeConfig`；
- 按 `summary/compact/full` 选择输出；
- 写 config、summary、run metadata 和 CSV。

该文件不实现买卖、持仓或 NAV。

### 3.7 `utils/as1455_grid_runner.py`

当前 in-process grid 的完整编排层：

- prediction 每个 signal 加载一次；
- 每个 signal 的每日 score 排序一次；
- execution panel 构造一次；
- universe、ST、公司行为和容量输入加载一次；
- 同一进程遍历全部参数；
- 每个组合调用唯一 v7 `backtest()`；
- 调用 `as1455_backtest_io.py` 写结果。

### 3.8 `utils/as1455_plotting.py`

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

只组合：

```text
history 更新
+ build_ashare_ch12_as1455_model_data.py
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

### 4.4 live 当日特征

```text
pipelines/as1455_live_prepare.py
features/as1455_live_common.py
features/build_as1455_live_features.py
```

live 路径与历史 model_data 构建路径用途不同，禁止混用。

---

## 5. 特征与训练层

### 5.1 底层训练核心

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
```

当前提供：

- 31 列基础输入契约；
- `MultipleTimeSeriesCV`；
- sector rotation 和 one-hot；
- TensorFlow 参数搜索；
- checkpoint、scaler 和 manifest 保存。

compact add-on 位于：

```text
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

### 5.2 统一单 fold 入口

```text
scripts/run_as1455_target_fold_param_search.py
```

支持 A/B、r1/r5/r21、指定 fold、原参数网格和 search-time top-N checkpoint。

默认目录使用同一个 fold 模板：

- r1 继续兼容原有训练目录；
- r5/r21 使用 `ch17_as1455_target_search/<preset>/<target>/foldN_search`。

### 5.3 批量训练入口

通用入口：

```text
scripts/run_as1455_target_search_all.sh
```

兼容入口：

```text
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
```

兼容入口只设置 `TARGET_COL`。

### 5.4 正式训练产物契约

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

r1 兼容入口：

```text
scripts/run_as1455_rotation_one_lag_daily_backtest.py
scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py
```

两个 r1 文件仅设置默认参数，不维护 checkpoint 推理或 monkey-patch。

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

默认 `TOP_N=1`，即只测试 fold0 最优单模型。top5/ensemble 需显式设置 `TOP_N=5`。

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

只有该函数决定：调仓日、买卖条件、T+1、主板/ST/停牌/涨跌停、整手、容量、公司行为、费用、持仓成本、NAV、换手、回撤和 round trip。

### 7.2 subprocess 兼容网格

```text
code/backtest/run_as1455_close_auction_grid_v1.py
```

保留 signal spec、参数组合、run name、summary 展平和 leaderboard。

### 7.3 当前 in-process 入口

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

该文件现在是薄入口，调用：

```text
utils/as1455_grid_runner.py
```

当前引擎：

```text
inprocess_shared_rank_v4
```

已实现：

- 一个进程；
- execution panel 只构造一次；
- 每个 signal 每日只实际排序一次；
- 全部 `max_positions × sell_rank × offset` 共享排序结果；
- 每个组合仍调用唯一 v7 交易函数；
- 不存在第二套买卖、费用或 NAV 实现。

已删除：

- `backtest_prepared()` 第二套交易循环；
- `inspect.getsource + exec` 动态改写；
- 独立 accessible 绘图 monkey-patch。

### 7.4 自然周期 wrapper

通用入口：

```text
scripts/run_as1455_target_natural_backtest.sh
```

兼容入口：

```text
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
```

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

职责：查找 grid summary、按指标选择最优 run、读取 NAV、生成 daily/weekly/monthly 曲线、保存选中参数并使用统一线型与 marker。

已删除：

```text
scripts/plot_as1455_backtest_return_curves_accessible.py
```

---

## 9. 重复开发治理结果

| 原问题 | 当前处理 |
|---|---|
| target、lookahead、调仓周期多处定义 | 统一到 `TARGET_SPECS` |
| one-fold-lag 与 fold0-forward 复制 checkpoint 推理 | 抽到 `as1455_ch17_common.py` |
| 两个协议复制 grid CLI 和命令 | 抽到 `as1455_cli.py` |
| `TOP_N` 与 signal 列不匹配 | 由 `as1455_signal_specs.py` 动态生成 |
| r1 A/B 各维护预测逻辑 | 改成薄 wrapper |
| r05/r21 训练 shell 重复 | 统一到 `run_as1455_target_search_all.sh` |
| r05/r21 回测 shell 重复 | 统一到 `run_as1455_target_natural_backtest.sh` |
| in-process grid 复制交易循环 | 删除，统一调用 v7 `backtest()` |
| 每个 config 重复排序 | 由 `as1455_rank_cache.py` 每个 signal 每日排序一次 |
| 回测输出逻辑散落 | 抽到 `as1455_backtest_io.py` |
| 无障碍绘图另建 monkey-patch | 合并进唯一绘图器 |
| 默认路径散落 | 抽到 `as1455_paths.py` |

---

## 10. 验证入口

结构、语法和合成排名缓存验证：

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

它检查：

- Python 和 shell 语法；
- 关键 CLI 可导入；
- `TARGET_SPECS`；
- `TOP_N` 与 signal 数量；
- 排名缓存与原逐日排序结果一致；
- grid runner 不包含第二套 `backtest`；
- 兼容 wrapper 不复制 checkpoint 或交易函数；
- 唯一绘图器和公共样式。

数据级单配置 smoke：

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

`PARITY` 是兼容旧参数名，当前含义是“唯一 v7 引擎单配置 smoke”。

单 run 结果比较：

```bash
python3 scripts/compare_as1455_backtest_runs.py \
  --left-run <重构前单个run目录> \
  --right-run <重构后单个run目录>
```

默认比较 NAV、订单、拒单、round trip 和 `summary.json`，容差为：

```text
rtol=1e-12
atol=1e-12
```

---

## 11. 剩余技术债务

### 11.1 训练底层进一步拆分

`run_as1455_sector_rotation_fold0_param_search.py` 仍同时承担 CV、特征、训练和 checkpoint 保存。后续可拆为：

```text
utils/as1455_features.py
utils/as1455_cv.py
utils/as1455_training.py
```

拆分前必须以现有 fold report、checkpoint 排名和 IC 结果做回归基准。

### 11.2 live 部署契约

旧 live bundle 使用旧 Ch17 `.weights.h5` 和重新拟合 scaler；target-aware A/B 使用 search-time `.keras + scaler.pkl + feature_manifest.json`。两者不能靠路径猜测混用，需要独立 deploy manifest。

---

## 12. 开发规则

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
