# Ch17 AS1455 开发大纲

本文档是 `ch17_as1455` 相关代码的结构说明和开发约束。使用命令见：

```text
README_AS1455_R1_R5_R21.md
```

本文档用于回答：

1. 数据、特征、训练、预测、回测、绘图和实盘代码分别负责什么；
2. 每类公共逻辑的唯一实现在哪里；
3. 新功能应修改哪一层；
4. 当前仍有哪些技术债务。

---

## 1. 固定业务与实验口径

### 1.1 AS1455 时点

- 特征只能使用当日 14:55 及以前的数据；
- 历史执行价使用当日 15:00 收盘价近似收盘集合竞价成交价；
- raw 执行字段只进入执行面板、metadata 和质量报告，不进入 34 列 `model_data`；
- 回测为 long-only；
- 默认只交易沪深主板；
- 默认包含 T+1、停牌/不可交易、涨跌停、100 股整手、手续费、印花税、过户费和公司行为处理。

### 1.2 目标与自然调仓周期

目标定义和调仓周期只能从以下公共表读取：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

| 简称 | 标签 | lookahead | 自然调仓周期 | offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

禁止在新的 Python 脚本中再复制一份映射表。

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
    v
utils/as1455_ch17_common.py
    |-- 目标过滤
    |-- A/B 特征构造
    |-- fold 构造
    |-- checkpoint/scaler/manifest 读取
    |-- checkpoint 推理
    |-- predictions/manifest 写出
    |-- grid 命令构造
    |
    +--> r1/r5/r21 参数搜索
    |
    +--> one-fold-lag 历史评估
    |
    +--> fold0 后续日期 forward 评估
    |
    v
预测 HDF（key=/predictions）
    |
    v
参数网格编排
    |
    v
唯一 v7 交易引擎
    |
    v
NAV / 订单 / 拒单 / 持仓 / 费用 / 换手 / leaderboard
    |
    v
统一收益曲线绘制
```

---

## 3. 公共工具层

### 3.1 `utils/as1455_ch17_common.py`

这是 Ch17 AS1455 训练与预测的公共实现层，包含：

- `TARGET_SPECS`：目标、lookahead、自然调仓周期和 offset 模式；
- `FeatureBuildResult`：统一特征构造返回值；
- `load_xy_target()`：按目标列过滤标签缺失；
- `build_target_features()`：构造 A/B 特征；
- `get_fold_target()`：按目标 lookahead 构造 fold；
- `default_search_dir()`、`default_fold_dir_template()`：统一目录规则；
- `read_top_checkpoints()`：读取 search-time top-N checkpoint；
- `load_preprocess()`：读取 scaler 和 feature manifest；
- `transform_for_source_model()`：按训练期列顺序和 scaler 变换输入；
- `predict_checkpoint_set()`：执行一组 checkpoint 推理；
- `write_prediction_artifacts()`：统一写 HDF、CSV、actual、checkpoint 清单和 manifest；
- `build_grid_command()`：统一构造回测命令。

训练、one-fold-lag 和 fold0-forward 不得重新实现上述函数。

### 3.2 `utils/as1455_signal_specs.py`

根据实际 `top_n` 生成合法的 grid signal：

- `top_n=1`：只生成 `model_0`；
- `top_n=5`：保持历史的 `model_0..model_4`、`ensemble_first3_mean`、`ensemble_all5_mean`；
- 其他数量：只引用真实存在的预测列。

这解决了 `top_n=1` 时 grid 仍尝试读取 `model_1..model_4` 的问题。

### 3.3 `utils/as1455_backtest_io.py`

只负责回测编排的公共辅助逻辑：

- 从 grid tuple 构造统一 `TradeConfig`；
- 按 `summary/compact/full` 选择输出文件；
- 写 `config.json`、`summary.json`、CSV 和 run metadata。

该文件不实现买卖、持仓或 NAV 业务逻辑。

### 3.4 `utils/as1455_plotting.py`

提供统一绘图样式：

- 不只依赖颜色；
- 每条曲线同时使用不同线型和 marker；
- marker 稀疏显示；
- 导出的 curve CSV 记录 `line_style` 和 `marker`。

---

## 4. 数据层

### 4.1 历史缓存更新入口

```text
scripts/run_as1455_live_data_feature_pipeline.sh history
```

核心实现：

```text
pipelines/as1455_update_history_to_prevday_fast_v4.py
```

职责：

- 扫描实际缓存最后日期；
- 增量抓取原始 5 分钟行情和原始日线；
- 由 5 分钟缓存聚合 AS1455 日数据；
- 记录逐股票阶段状态和错误；
- 支持只修复 AS1455 聚合。

旧实现：

```text
pipelines/as1455_update_history_to_prevday.py
```

`fast_v4` 仍复用其中的数据源查询和字段标准化函数。不得复制这些函数。

### 4.2 完整 model_data 构建器

```text
scripts/build_ashare_ch12_as1455_model_data.py
```

职责：

- 读取历史 5 分钟、AS1455 日缓存和原始日线；
- 构造 14:55 截止的 AS1455 OHLCV；
- 构造复权因子；
- 构造 Ch12 31 个模型特征；
- 构造 `r01_fwd/r05_fwd/r21_fwd`；
- 输出 34 列 `model_data_as1455.h5`；
- 输出覆盖率、标签对齐、复权、泄漏和质量报告。

### 4.3 fold0-forward 数据刷新包装

```text
scripts/refresh_as1455_forward_model_data.sh
```

该脚本只组合：

```text
history 更新
+ build_ashare_ch12_as1455_model_data.py
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

它不自行实现行情抓取、复权或特征计算。

### 4.4 live 当日特征路径

```text
pipelines/as1455_live_prepare.py
features/as1455_live_common.py
features/build_as1455_live_features.py
```

这条路径服务于盘中/实盘，不替代历史 `model_data` 构建器。

---

## 5. 特征与训练层

### 5.1 当前底层训练实现

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
```

当前仍提供：

- 31 列基础输入契约；
- `MultipleTimeSeriesCV`；
- sector rotation；
- sector one-hot；
- 参数网格；
- TensorFlow 搜索训练；
- checkpoint、scaler 和 manifest 保存。

该文件仍承担较多职责，但上层代码已不再复制其功能。

### 5.2 compact add-on

```text
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

公共函数：

```text
add_compact_addon_features()
```

### 5.3 推荐的统一单 fold 入口

```text
scripts/run_as1455_target_fold_param_search.py
```

该脚本现在只做 CLI 和训练编排，特征和 fold 来自：

```text
utils/as1455_ch17_common.py
```

支持：

- A/B；
- r1/r5/r21；
- fold0..fold6；
- 原始 Ch17 参数网格；
- search-time top-N checkpoint；
- 可选诊断性 retrain。

### 5.4 通用批量训练入口

```text
scripts/run_as1455_target_search_all.sh
```

兼容入口：

```text
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
```

两个兼容入口只设置 `TARGET_COL`，不再复制训练循环。

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

### 6.1 one-fold-lag 历史评估

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

它只负责：

- 生成 source/target fold 计划；
- 调用公共特征和 checkpoint 推理；
- 合并各 target fold 的预测；
- 调用 grid。

r1 兼容入口：

```text
scripts/run_as1455_rotation_one_lag_daily_backtest.py
scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py
```

这两个文件已改成薄 wrapper，只提供 r1 A/B 默认值，不再维护预测循环或 monkey-patch。

### 6.2 fold0-forward 评估

协议：

- 使用 fold0 search-time checkpoint、scaler 和 feature manifest；
- 日期严格满足 `date > fold0.test_end`；
- 不重新训练；
- 从初始资金和空仓开始；
- 不继承 fold0 测试窗口持仓。

实现：

```text
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

Python 入口只负责日期选择；checkpoint 推理和产物写出来自公共 utils。

shell 入口默认：

```text
TOP_N=1
```

因此“fold0 最优模型”默认只回测 `model_0`。需要延续 top5/ensemble 实验时显式设置 `TOP_N=5`。

### 6.3 预测文件契约

```text
HDF key: /predictions
index:    (symbol, date)
columns:  0..N-1
```

所有协议统一通过 `write_prediction_artifacts()` 写出。

### 6.4 live/deploy 推理

旧基线部署路径：

```text
tools/create_as1455_sharpe1_checkpoint_bundle_v1.py
prediction/run_as1455_live_checkpoint_ensemble_inference_v1.py
scripts/run_as1455_live_checkpoint_signal_v1.sh
```

该路径针对旧 `run_ashare_ch17_nn_reproduce.py` checkpoint 格式，不等同于 A/B target-aware search-time `.keras` 产物，禁止直接混用。

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

保留职责：

- signal specs；
- 参数组合；
- run name；
- summary 展平；
- leaderboard。

旧模式为每个组合启动一个 v7 子进程。

### 7.3 当前 in-process 网格

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

当前版本：

```text
inprocess_shared_data_v3
```

职责：

- prediction 每个 signal 加载一次；
- execution panel 构造一次；
- universe/ST/corporate-action/capacity 数据加载一次；
- 在同一进程遍历所有参数；
- 每个参数组合直接调用唯一 v7 `backtest()`；
- 使用 `utils/as1455_backtest_io.py` 写结果。

重要变化：

- 已删除原 `backtest_prepared()` 的第二套交易循环；
- 不再存在两套买卖、费用和 NAV 语义；
- `PARITY_CHECK_ONLY=1` 现在是共享 v7 引擎 smoke check，不再是假设两套交易实现长期等价的 parity test。

### 7.4 当前性能边界

为了消除重复交易逻辑，当前 v3 暂时仍由 v7 在每个参数组合内部执行逐日排序：

```text
daily_rankings_built_once_per_signal = false
```

这意味着：

- 数据加载和 execution panel 重复已经消除；
- 4410 个独立 Python 子进程已经消除；
- 但不同参数组合之间仍重复排序。

后续要恢复“每个 signal 每日只排序一次”，必须重构 v7 公共核心，使它原生接收预排序信号提供器。禁止再在 grid 文件中复制交易循环。

### 7.5 通用自然周期 wrapper

```text
scripts/run_as1455_target_natural_backtest.sh
```

兼容入口：

```text
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
```

r5/r21 入口现在只设置 `TARGET_COL`。

---

## 8. 绘图层

### 8.1 唯一绘图入口

```text
scripts/plot_as1455_backtest_return_curves.py
```

职责：

- 查找 grid summary；
- 按 Sharpe、收益、Calmar 等指标选择最优 run；
- 读取 NAV；
- 生成 daily/weekly/monthly 累计收益曲线；
- 保存选中参数；
- 调用 `utils/as1455_plotting.py` 应用线型和 marker。

### 8.2 shell 入口

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

只负责传入多个根目录、标签、指标、频率和输出目录。

### 8.3 已删除的重复实现

```text
scripts/plot_as1455_backtest_return_curves_accessible.py
```

该 monkey-patch 包装已删除，无障碍样式已合并到唯一绘图器。

---

## 9. 已完成的重复开发治理

| 原问题 | 当前处理 |
|---|---|
| in-process grid 复制完整交易循环 | 已删除；统一调用 v7 `backtest()` |
| fold0-forward 复制 checkpoint 推理和产物写出 | 已抽到 `utils/as1455_ch17_common.py` |
| one-fold-lag r1/r5/r21 各有预测实现 | 已统一到 `run_as1455_target_one_lag_backtest.py` |
| r1 B 通过 monkey-patch 替换特征和 grid | 已改成薄 wrapper |
| target 标签和 lookahead 多处定义 | 已统一到 `TARGET_SPECS` |
| top_n 与 grid signal 数量不匹配 | 已统一由 `as1455_signal_specs.py` 生成 |
| r05/r21 训练 shell 重复 | 已统一到 `run_as1455_target_search_all.sh` |
| r05/r21 回测 shell 重复 | 已统一到 `run_as1455_target_natural_backtest.sh` |
| 无障碍绘图另建 monkey-patch 文件 | 已合并进基础绘图器并删除包装 |

---

## 10. 当前剩余技术债务

### 10.1 v7 内部需要原生支持预排序缓存

优先级：高。

目标接口方向：

```python
backtest(
    predictions,
    execution_panel,
    config,
    corporate_actions,
    ranked_signal_provider=None,
)
```

原单次回测和 in-process grid 必须调用同一核心。

### 10.2 训练基础文件职责过多

```text
run_as1455_sector_rotation_fold0_param_search.py
```

后续可继续拆为：

```text
utils/as1455_features.py
utils/as1455_cv.py
utils/as1455_training.py
```

拆分前必须以现有 checkpoint、fold report 和 IC 结果为基准做一致性测试。

### 10.3 live 基线与 target-aware A/B 模型尚未统一部署契约

旧 live bundle 使用旧 Ch17 `.weights.h5` 和重新拟合 fold scaler；target-aware A/B 使用 search-time `.keras + scaler.pkl + feature_manifest.json`。需要单独设计统一 deploy manifest，不能靠路径猜测混用。

---

## 11. 开发规则

新增功能前必须回答：

1. 功能属于数据、特征、训练、预测协议、grid、交易还是绘图？
2. 该层唯一事实来源是什么？
3. 是否可以增加参数或公共纯函数，而不是复制脚本？
4. 是否改变 14:55 数据边界、标签、fold、checkpoint 选择或交易语义？
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
- 复制 checkpoint 推理循环；
- 复制交易循环；
- 复制 summary/leaderboard；
- 维护另一份目标/周期映射。

---

## 12. 修改后的最低验证集

### 12.1 静态导入和语法

```bash
python3 -m compileall -q \
  utils/as1455_ch17_common.py \
  utils/as1455_signal_specs.py \
  utils/as1455_backtest_io.py \
  utils/as1455_plotting.py \
  scripts/run_as1455_target_fold_param_search.py \
  scripts/run_as1455_target_one_lag_backtest.py \
  scripts/run_as1455_fold0_forward_backtest.py \
  scripts/plot_as1455_backtest_return_curves.py \
  code/backtest/run_as1455_close_auction_grid_inprocess.py
```

### 12.2 CLI smoke

```bash
python3 scripts/run_as1455_target_fold_param_search.py --help >/dev/null
python3 scripts/run_as1455_target_one_lag_backtest.py --help >/dev/null
python3 scripts/run_as1455_fold0_forward_backtest.py --help >/dev/null
python3 scripts/plot_as1455_backtest_return_curves.py --help >/dev/null
python3 code/backtest/run_as1455_close_auction_grid_inprocess.py --help >/dev/null
```

### 12.3 fold0 单模型 signal 契约

```bash
python3 - <<'PY'
from utils.as1455_signal_specs import signal_specs_for_top_n
assert signal_specs_for_top_n(1) == ["model_0:0:single"]
assert len(signal_specs_for_top_n(5)) == 7
print("[OK] signal specs")
PY
```

### 12.4 交易引擎 smoke

```bash
FEATURE_PRESETS="rotation_onehot" \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r05_natural_backtest.sh
```

必须出现：

```text
[PARITY] single v7 trade engine smoke run ...
[PARITY] PASS
[PARITY] check-only completed; grid was not executed
```

### 12.5 结果一致性

对固定预测文件和固定参数，重构前后至少比较：

```text
close_auction_nav.csv
close_auction_orders.csv
close_auction_rejections.csv
round_trips.csv
summary.json
```

数值字段使用 `rtol=1e-12, atol=1e-12`。目录名和 engine metadata 允许不同，交易和绩效结果不允许不同。
