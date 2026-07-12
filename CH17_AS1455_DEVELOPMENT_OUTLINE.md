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
2. 每个日期执行一次与 v7 完全相同的 `sort_values("score", ascending=False)`；
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

当前 in-process grid 编排层：

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

## 6. 预测协议与模型选择

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

### 6.3 默认候选模型与绘图一致性

fold0-forward 默认：

```text
TOP_N=5
```

候选 signal：

```text
model_0
model_1
model_2
model_3
model_4
ensemble_first3_mean
ensemble_all5_mean
```

其中 checkpoint 先按训练阶段的 `daily_ic_median` 排名确定 top5。随后回测对 7 个 signal 执行完整交易参数网格。

绘图默认：

```text
RANK_METRIC=sharpe
```

因此默认选择政策是：

```text
训练阶段 daily_ic_median top5 checkpoint
→ 构造 5 个单模型 signal + 2 个 ensemble
→ 完整回测参数网格
→ 每个 BACKTEST_ROOT 中选择 Sharpe 最高的完整 run
→ 绘制该 run 的 NAV
```

“完整 run”包含：

```text
signal_name
max_positions
sell_rank
rebalance_every
rebalance_offset
```

只测试训练阶段排名第一的单模型时，显式设置 `TOP_N=1`。这属于受控实验覆盖，不是默认策略。

### 6.4 预测文件契约

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

该文件是薄入口，调用：

```text
utils/as1455_grid_runner.py
```

当前引擎：

```text
inprocess_shared_rank_v4
```

保证：

- 每个 signal 每日只排序一次；
- execution panel 只构造一次；
- 所有配置在同一进程运行；
- 每个配置调用唯一 v7 `backtest()`；
- 没有 `backtest_prepared()`、动态源码替换或第二套交易循环。

### 7.4 自然周期 wrapper

通用：

```text
scripts/run_as1455_target_natural_backtest.sh
```

兼容：

```text
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
```

---

## 8. 绘图层

唯一 Python 绘图器：

```text
scripts/plot_as1455_backtest_return_curves.py
```

shell 入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

对每个 `BACKTEST_ROOT`：

1. 读取 `grid_summary_compact.csv` 或 `grid_summary.csv`；
2. 过滤正常完成的 run；
3. 默认按 Sharpe 降序选择第一行；
4. 读取该 `run_name` 的 `close_auction_nav.csv`；
5. 生成 daily、weekly、monthly 曲线；
6. 写 `selected_best_grids.csv/json`。

禁止新建第二个绘图器或使用 monkey-patch 修改基础绘图函数。

---

## 9. 重复开发治理规则

wrapper 只允许：

- 设置默认参数；
- 组合已有入口；
- 确定输出目录；
- 检查输入存在性；
- 输出运行上下文。

wrapper 禁止：

- 重新实现特征；
- monkey-patch 业务函数；
- 复制 checkpoint 推理循环；
- 复制交易循环；
- 复制 summary/leaderboard；
- 维护另一份目标/周期映射。

新增功能前必须回答：

1. 功能属于数据、特征、训练、预测协议、grid、交易还是绘图？
2. 该层唯一事实来源是什么？
3. 是否可以增加公共纯函数或参数，而不是复制脚本？
4. 是否改变 14:55 数据边界、标签、fold、checkpoint 或交易语义？
5. 如何验证旧结果不变？

---

## 10. 自动验证

入口：

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

检查内容：

- Python 编译；
- shell 语法；
- CLI 导入；
- 目标映射；
- signal 数量；
- fold0 默认 `TOP_N=5`；
- 绘图默认 `RANK_METRIC=sharpe`；
- 排名缓存等价性；
- 单一交易引擎；
- 薄 wrapper；
- 统一绘图器。

固定预测文件的重构前后结果比较：

```bash
python3 scripts/compare_as1455_backtest_runs.py \
  --left-run <重构前run目录> \
  --right-run <重构后run目录>
```

数值容差：

```text
rtol=1e-12
atol=1e-12
```

---

## 11. 当前已知限制

1. r21 当前数据下没有可用 source fold6，所以 one-fold-lag 默认只覆盖 target fold0..4。
2. forward HDF 最后 1、5、21 个交易日分别没有完整 r1、r5、r21 前向标签。
3. live 旧基线 checkpoint bundle 与 target-aware A/B `.keras + scaler.pkl + feature_manifest.json` 不是同一产物契约，不能直接混用。
