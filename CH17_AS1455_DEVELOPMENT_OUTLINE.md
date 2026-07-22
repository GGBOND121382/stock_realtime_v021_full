# Ch17 AS1455 开发、重训、Grid 检索与回测指南

本文档是 AS1455 Chapter-17 流水线的开发与运行总纲，覆盖：

```text
历史数据与缓存
→ 14:55 model_data 构建
→ 分 target / feature preset / fold 模型重训与参数搜索
→ one-fold-lag 历史预测
→ 交易参数 grid 检索
→ 历史最佳配置 materialize
→ fold0 strict-OOS forward 回测
→ 分 fold、跨 fold 与综合绘图
→ 结果审计、定位和清理
```

相关专题文档：

```text
README_AS1455_R1_R5_R21.md
AS1455_STORAGE_AND_STRICT_OOS.md
AS1455_STORAGE_MAINTENANCE.md
R05_ADDON_FOLD_COMPARISON.md
CH17_AS1455_FROM_SCRATCH.md
CH17_AS1455_CLEAN_TREE.md
```

> **重要边界**：`scripts/run_ch17_as1455_full_rebuild.sh` 当前是受保护的“已有结果回测/绘图”入口，不负责模型重训、model_data 重建或历史大 grid。模型重训、历史 grid 和 strict-OOS forward 必须使用本文列出的专用脚本。

---

## 1. 固定实验口径

### 1.1 时间与泄漏边界

- 模型特征只能使用交易日当日 `14:55` 及以前的数据；
- `r01_fwd`、`r05_fwd`、`r21_fwd` 分别定义为从当日 `14:55` 到未来第 1、5、21 个交易日 `14:55` 的收益；
- 历史交易执行价使用当日 `15:00` 收盘价近似收盘集合竞价成交价；
- 模型输入中禁止出现当日完整日线 `open/high/low/close/volume`、执行状态、主板标记等执行期字段；
- forward 推理只要求模型特征完整，不要求未来标签已经实现。

### 1.2 交易口径

默认交易规则：

```text
long-only
沪深主板 only
ST 默认不可买
停牌不可交易
涨停不可买、跌停不可卖
T+1
100 股整数手
同日 15:00 价格近似成交
券商佣金 + 最低佣金 + 印花税 + 过户费
公司行为默认使用 preclose 合成份额因子保持持仓价值连续
```

默认初始资金：

```text
200000 元
```

默认费用：

| 参数 | 默认值 |
|---|---:|
| `commission_rate` | `0.000085` |
| `min_commission` | `5` |
| `stamp_tax_rate` | `0.0005`，仅卖出 |
| `transfer_fee_rate` | `0.00001`，双边 |
| `slippage_bps` | `0` |
| `lot_size` | `100` |

### 1.3 Target 与调仓周期

唯一 target 定义位于：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

| target | lookahead | `rebalance_every` | 历史 grid offset |
|---|---:|---:|---|
| `r01_fwd` | 1 | 1 | `0` |
| `r05_fwd` | 5 | 5 | `0..4` |
| `r21_fwd` | 21 | 21 | `0..20` |

v7 引擎本地调仓条件：

```text
(day_index - rebalance_offset) mod rebalance_every = 0
```

`rebalance_offset` 是当前回测窗口的本地序号。历史窗口、独立 fold 和 forward 窗口都可能从 `day_index=0` 重新编号，因此跨窗口时必须换算相位，不能机械复制同一个 offset 数字。

### 1.4 Feature preset

正式支持：

```text
rotation_onehot
rotation_addon_onehot
```

- `rotation_onehot`：基础 Ch12 特征 + 同日行业轮动特征 + sector one-hot；
- `rotation_addon_onehot`：在前者基础上加入 compact addon 特征组。

### 1.5 Fold 编号

时间序列 CV 中：

```text
fold0 = 最新测试窗口
fold6 = 最早测试窗口
```

历史 one-fold-lag 协议按**目标测试窗口**解释：

```text
source model fold6 -> target fold5
source model fold5 -> target fold4
...
source model fold1 -> target fold0
```

因此：

- `r01_fwd`、`r05_fwd` 训练 fold 为 `0..6`，历史 one-lag target fold 为 `0..5`；
- `r21_fwd` 当前有效数据只支持训练 fold `0..5`，历史 one-lag target fold 为 `0..4`；
- 报告或图片应同时记录 `source_model_fold` 与 `target_fold`，不能把二者混为一谈。

---

## 2. 代码架构与唯一语义来源

### 2.1 数据构建

```text
scripts/build_ashare_ch12_as1455_model_data.py
scripts/build_ashare_ch12_as1455_lowmem.sh
scripts/refresh_as1455_forward_model_data.sh
```

职责：

- 从 5 分钟缓存聚合 14:55 日级 OHLCV；
- 使用 raw daily `close/preclose` 构造严格 as-of 复权因子；
- 生成 Ch12 风格技术指标、横截面分位数、行业分位数和 forward 标签；
- 写入 34 列 `model_data_as1455.h5`；
- 输出泄漏、覆盖、复权、标签对齐和 Chapter-17 读取检查报告。

### 2.2 特征、fold、checkpoint 与预测

```text
utils/as1455_ch17_common.py
utils/as1455_forward_features.py
```

`as1455_ch17_common.py` 是以下逻辑的公共实现：

- target 定义；
- A/B feature preset；
- CV fold；
- checkpoint 选择与加载；
- scaler 和模型输入列；
- 历史 one-lag 预测；
- prediction artifact 写入；
- grid 命令构造。

`as1455_forward_features.py` 只改变 forward 行保留条件：

```text
训练/历史：模型特征非空 + 当前 target 非空
forward：模型特征非空；当前 target 可以为空
```

### 2.3 模型参数搜索

```text
scripts/run_as1455_target_fold_param_search.py
scripts/run_as1455_target_search_all.sh
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

正式通用入口是：

```text
scripts/run_as1455_target_fold_param_search.py
```

底层神经网络、训练循环、IC 计算与 checkpoint 保存由：

```text
scripts/run_as1455_sector_rotation_fold0_param_search.py
```

提供公共实现。

### 2.4 历史 one-fold-lag 与 grid

```text
scripts/run_as1455_target_one_lag_backtest.py
scripts/run_as1455_target_natural_backtest.sh
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
utils/as1455_cli.py
utils/as1455_signal_specs.py
utils/as1455_grid_runner.py
utils/as1455_rank_cache.py
utils/as1455_backtest_io.py
```

### 2.5 历史选择与 materialize

```text
utils/as1455_model_selection.py
scripts/materialize_as1455_best_run.py
```

选择规则：

- 只允许 `status=ok`；
- 按 `sharpe` 等指定指标选择完整 run；
- 冻结 signal、max positions、sell rank、rebalance period 和历史 offset；
- 新但残缺或 failed-only 的目录必须跳过；
- 最佳 run 以 compact/full 模式重新执行一次并保存 NAV；
- `materialized_best_run.json` 是历史最佳配置的权威索引。

### 2.6 Strict OOS 与调仓相位

```text
utils/as1455_rebalance_phase.py
utils/as1455_strict_oos.py
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

正式 strict-OOS 冻结：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
historical rebalance phase
完整交易执行配置
```

历史到 forward 的相位换算：

```text
forward_global_index
  = historical_n_days + bridge_execution_days

effective_forward_offset
  = (historical_offset - forward_global_index) mod rebalance_every
```

`bridge_execution_days` 使用完整 raw-daily execution calendar 计算，不使用自然日。

### 2.7 唯一交易引擎

唯一交易语义来源：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

共享 grid CLI：

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
```

该文件只转发到：

```text
utils/as1455_grid_runner.py
```

禁止在其他脚本复制第二套买卖循环。

---

## 3. 数据、模型和结果目录

### 3.1 Canonical path

统一路径定义：

```text
utils/as1455_paths.py
```

主要目录：

| 类型 | 位置 |
|---|---|
| 静态股票池 | `saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv` |
| 历史 AS1455 根目录 | `saved_data/ashare_ml4t/ch12_as1455/` |
| 历史 model_data | `saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5` |
| 5 分钟原始缓存 | `saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/` |
| raw daily 缓存 | `saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/` |
| 14:55 日级缓存 | `saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache/` |
| 历史数据构建报告 | `saved_data/ashare_ml4t/ch12_as1455/reports/` |
| forward model_data | `saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5` |
| 模型搜索根目录 | `saved_data/ashare_ml4t/ch17_as1455_target_search/` 及 r01 兼容目录 |
| 历史 grid 根目录 | `saved_data/ashare_ml4t/ch17_as1455_target_backtest/` |
| fold0 strict-OOS 根目录 | `saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/` |
| 全策略独立 fold 结果 | `saved_data/ashare_ml4t/ch17_as1455_independent_folds/` |
| r05 addon 专项结果 | `saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/` |
| 通用绘图根目录 | `saved_data/ashare_ml4t/ch17_as1455_backtest_plots/` |

### 3.2 5 分钟缓存格式

典型文件：

```text
saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/<symbol>_5m_raw.csv
```

核心字段：

```text
symbol
trade_date
datetime
open
high
low
close
volume
amount
source
bar_freq
bar_label
```

只有当日 `14:55` 及以前的 bar 可以进入特征构建。

### 3.3 Raw daily 缓存

典型文件：

```text
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/<symbol>_daily_raw.csv
```

回测和复权至少依赖：

```text
date
open
high
low
close
preclose
volume
amount
tradestatus
isST
```

### 3.4 Model data

权威文件：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
HDF key: model_data
index: symbol, date
columns: 34
```

34 列：

```text
dollar_vol, dollar_vol_rank,
rsi, bb_high, bb_low, NATR, ATR, PPO, MACD, sector,
r01, r05, r10, r21, r42, r63,
r01dec, r05dec, r10dec, r21dec, r42dec, r63dec,
r01q_sector, r05q_sector, r10q_sector,
r21q_sector, r42q_sector, r63q_sector,
r01_fwd, r05_fwd, r21_fwd,
year, month, weekday
```

中间文件：

```text
as1455_ohlcv_raw.h5          key=ohlcv
as1455_ohlcv_adj.h5          key=ohlcv
as1455_execution_metadata.h5 key=metadata
```

历史根目录可以保留这些文件；forward 默认 `model_only`，验证后删除可重建中间 HDF。

### 3.5 Prediction HDF

历史 one-lag：

```text
<历史回测根>/00_predictions/test_preds.h5
HDF key: predictions
```

fold0 forward：

```text
<forward 根>/00_predictions/fold0_forward_preds.h5
HDF key: predictions
```

HDF 是预测权威文件。与 HDF 重复的 prediction CSV 可由：

```bash
python3 scripts/compact_as1455_prediction_artifacts.py \
  --prediction-dir <root>/00_predictions
```

清理；`actual_<target>.csv` 是真实标签，不得误删。

---

## 4. 历史 Model Data 构建

### 4.1 推荐离线低内存入口

```bash
cd /root/stock_realtime_v021_full

bash scripts/build_ashare_ch12_as1455_lowmem.sh
```

该入口：

- 使用已有 `baostock_5m_cache`；
- 使用已有 qfq daily 缓存做兼容性对照；
- 禁止从网络补拉 5 分钟和 qfq daily；
- 使用 raw daily `preclose` 完成 as-of 复权；
- 输出内存检查点。

默认输出：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455/reports/
```

### 4.2 强制重建 14:55 日级缓存

```bash
REBUILD_DAILY_CACHE=1 \
  bash scripts/build_ashare_ch12_as1455_lowmem.sh
```

只在以下情况启用：

- 5 分钟缓存内容发生变化；
- timestamp convention 修复；
- 原有 `as1455_daily_cache` 被确认损坏。

### 4.3 直接调用 Python 构建器

```bash
python3 scripts/build_ashare_ch12_as1455_model_data.py \
  --universe saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv \
  --out-dir saved_data/ashare_ml4t/ch12_as1455 \
  --bar-root saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache \
  --bar-glob '*_5m_raw.csv' \
  --baostock-5m-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache \
  --as1455-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --no-fetch-missing-baostock \
  --no-fetch-missing-qfq-daily \
  --profile-memory
```

### 4.4 小样本/检查模式

```bash
python3 scripts/build_ashare_ch12_as1455_model_data.py \
  --max-symbols 20 \
  --allow-partial-coverage \
  --out-dir saved_data/ashare_ml4t/ch12_as1455_smoke
```

小样本结果不能替代正式 model_data。

### 4.5 必查报告

```text
reports/as1455_build_summary.json
reports/as1455_chapter17_read_smoke_test.json
reports/as1455_feature_column_check.csv
reports/as1455_cutoff_leakage_check.csv
reports/as1455_label_alignment_samples.csv
reports/as1455_5min_coverage_check.csv
reports/as1455_daily_cache_build_report.csv
reports/as1455_raw_daily_fetch_report.csv
reports/as1455_vs_daily_summary.json
```

正式数据至少满足：

```text
chapter17_smoke_passed = true
used_after_cutoff_count = 0
model_columns = 34
model_data HDF 可用 key=model_data 读取
```

### 4.6 快速检查

```bash
python3 - <<'PY'
import pandas as pd
p = 'saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5'
df = pd.read_hdf(p, 'model_data')
dates = pd.DatetimeIndex(df.index.get_level_values('date'))
print('shape=', df.shape)
print('symbols=', df.index.get_level_values('symbol').nunique())
print('date=', dates.min(), dates.max())
print('columns=', list(df.columns))
for col in ['r01_fwd', 'r05_fwd', 'r21_fwd']:
    valid = df[col].notna()
    d = pd.DatetimeIndex(df.index.get_level_values('date')[valid])
    print(col, 'valid_rows=', int(valid.sum()), 'valid_end=', d.max())
PY
```

---

## 5. Forward Model Data 刷新

forward 需要覆盖 fold0 `test_end` 之后的最新特征完整日期。

### 5.1 刷新历史缓存并构建 forward model_data

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

默认行为：

```text
更新共享 5min/raw-daily/as1455-daily 缓存
→ 在 ch12_as1455_forward_latest 重建扩展 model_data
→ 验证 34 列与日期范围
→ 删除 forward 目录内可重建中间 HDF
→ 压缩超大报告 CSV
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455_forward_latest/reports/
```

### 5.2 已更新缓存时跳过网络/历史更新

```bash
SKIP_HISTORY_UPDATE=1 \
  bash scripts/refresh_as1455_forward_model_data.sh
```

### 5.3 仅使用已存在 forward model_data

在 strict-OOS 回测中设置：

```bash
REFRESH_DATA=0 \
  MODEL_DATA=saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

---

## 6. 模型重训与参数搜索

### 6.1 正式参数空间

默认神经网络搜索：

```text
dense_layers: (16,8), (32,16), (32,32), (64,32)
activation: tanh
dropout: 0, 0.1, 0.2
batch_size: 64, 256
epochs: 20
```

即：

```text
4 × 1 × 3 × 2 = 24 组网络参数
```

每组网络连续训练 20 个 epoch，并在每个 epoch 末计算：

```text
pooled_spearman
daily_ic_mean
daily_ic_median
daily_ic_positive_rate
```

默认按 `daily_ic_median` 保存全搜索中前 5 个 model-epoch checkpoint。

> `RETRAIN_BEST=1` 生成的 `models/best_*` 是诊断性重新训练结果。正式 one-fold-lag 与 forward 预测读取的是搜索期 `search_checkpoints/*.keras`，以 `search_best_checkpoints.csv` 为索引。

### 6.2 先做输入检查，不训练

以 `r05_fwd + rotation_addon_onehot + fold0` 为例：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_addon_onehot \
  --target-col r05_fwd \
  --model-data saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 \
  --fold-index 0 \
  --sector-encoding onehot \
  --dropna-mode target_only \
  --input-check-only \
  --out-dir /tmp/as1455_r05_addon_fold0_check
```

检查：

```text
fold_report.json / fold_report.csv
feature_cols_final.json
rotation_feature_cols.json
addon_feature_cols.json
feature_group_cols.json
sector_onehot_cols.json
param_grid.csv
run_summary.json
```

### 6.3 单个 fold 正式重训

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_addon_onehot \
  --target-col r05_fwd \
  --model-data saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 \
  --fold-index 0 \
  --epochs 20 \
  --best-n 5 \
  --seed 42 \
  --force
```

`--force` 会在 canonical fold 目录中重建搜索产物。它不是断点续跑；不要在已有有效 checkpoint 时误用。

### 6.4 一个 target 的全部 preset 与 fold

#### r01

```bash
TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_search_all.sh
```

默认：

```text
feature presets = rotation_onehot rotation_addon_onehot
folds = 0 1 2 3 4 5 6
```

#### r05

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

等价于：

```bash
TARGET_COL=r05_fwd \
  bash scripts/run_as1455_target_search_all.sh
```

#### r21

```bash
bash scripts/run_as1455_r21_target_search_all.sh
```

默认 folds：

```text
0 1 2 3 4 5
```

### 6.5 只训练一个 preset 或部分 folds

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
FOLDS='0 1 2 3 4 5 6' \
EPOCHS=20 \
BEST_N=5 \
SEED=42 \
FORCE=0 \
  bash scripts/run_as1455_target_search_all.sh
```

单 fold smoke：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
FOLDS='0' \
SMOKE=1 \
FORCE=1 \
  bash scripts/run_as1455_target_search_all.sh
```

### 6.6 训练输出目录

#### r01 兼容目录

```text
rotation_onehot:
saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold{fold}_search/

rotation_addon_onehot:
saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold{fold}_search/
```

#### r05/r21 通用目录

```text
saved_data/ashare_ml4t/ch17_as1455_target_search/
  <feature_preset>/
    <target_col>/
      fold{fold}_search/
```

日志：

```text
saved_data/ashare_ml4t/ch17_as1455_target_search/logs/
  <preset>_<target>_fold<fold>_search.log
```

### 6.7 每个 fold 的核心产物

```text
run_summary.json
fold_report.json
fold_report.csv
param_grid.csv
scores_summary.csv
scores_by_day.csv
scores.h5                         key=ic_by_day
search_progress.csv
search_events.jsonl
param_start_log.csv
param_end_log.csv
best_params.csv
search_manifest.json
search_best_checkpoints.csv

preprocess/
  scaler.pkl
  feature_manifest.json

search_checkpoints/
  search_paramXXX_epochYYY.keras
  search_paramXXX_epochYYY.weights.h5
  search_paramXXX_epochYYY.architecture.json
  search_paramXXX_epochYYY.manifest.json
  search_checkpoints_manifest.json
  final_top_checkpoints.json
```

权威关系：

```text
search_best_checkpoints.csv
→ checkpoint_name / keras_model
→ search_checkpoints/*.keras
→ preprocess/scaler.pkl
→ preprocess/feature_manifest.json
```

### 6.8 搜索结果快速检查

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('saved_data/ashare_ml4t/ch17_as1455_target_search/rotation_addon_onehot/r05_fwd/fold0_search')
print(pd.read_csv(root / 'search_best_checkpoints.csv')[
    ['checkpoint_name', 'daily_ic_median', 'daily_ic_mean', 'pooled_spearman', 'checkpoint_saved']
])
print('keras=', len(list((root / 'search_checkpoints').glob('*.keras'))))
PY
```

---

## 7. 历史 One-Fold-Lag 预测与交易 Grid

### 7.1 协议

```text
source model fold(target+1)
→ 在 target fold 测试窗口生成预测
→ 合并所有 target fold prediction
→ 对完整历史 one-lag 序列执行交易参数 grid
```

历史 prediction 文件不是每个 fold 一个连续账户，而是将各目标窗口预测按日期拼接后交给同一个 grid 回测。

### 7.2 通用历史入口

```bash
TARGET_COL=r05_fwd \
  bash scripts/run_as1455_target_natural_backtest.sh
```

快捷入口：

```bash
bash scripts/run_as1455_r05_natural_backtest.sh
bash scripts/run_as1455_r21_natural_backtest.sh
```

r01：

```bash
TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_natural_backtest.sh
```

### 7.3 默认历史设置

```text
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot'
TOP_N=5
OUTPUT_MODE=summary
MATERIALIZE_BEST=1
MATERIALIZED_OUTPUT_MODE=compact
RANK_METRIC=sharpe
MAX_POSITIONS_LIST=5,10,15,20,25
SELL_RANK_LIST=75,100,150,200,250,300
CAPACITY_MODE=none
FORCE_GRID=1
```

fold 范围：

| target | target folds | source model folds |
|---|---|---|
| `r01_fwd` | `0..5` | `1..6` |
| `r05_fwd` | `0..5` | `1..6` |
| `r21_fwd` | `0..4` | `1..5` |

### 7.4 Top-5 checkpoint 对应信号

`TOP_N=5` 生成 7 个 signal：

```text
model_0
model_1
model_2
model_3
model_4
ensemble_first3_mean
ensemble_all5_mean
```

定义位于：

```text
utils/as1455_signal_specs.py
```

### 7.5 Grid 规模

每个配置元组：

```text
signal
× max_positions
× sell_rank
× rebalance_every
× rebalance_offset
```

默认每个 preset：

| target | signals | max | sell | offsets | configs/preset |
|---|---:|---:|---:|---:|---:|
| `r01_fwd` | 7 | 5 | 6 | 1 | 210 |
| `r05_fwd` | 7 | 5 | 6 | 5 | 1050 |
| `r21_fwd` | 7 | 5 | 6 | 21 | 4410 |

两个 preset、三个 target 全部运行：

```text
2 × (210 + 1050 + 4410) = 11340 个交易配置
```

这是真正的大 grid，不应与“固定配置的 6/40 次回测”混淆。

### 7.6 安全运行建议

正式全新搜索：

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
FORCE_GRID=1 \
OUTPUT_MODE=summary \
MATERIALIZE_BEST=1 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

复用同一时间戳目录、跳过已成功配置：

```bash
RUN_STAMP=<已有时间戳> \
FORCE_GRID=0 \
OUTPUT_MODE=summary \
MATERIALIZE_BEST=1 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

仅检查命令和配置数量：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
DRY_RUN=1 \
MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_target_natural_backtest.sh
```

只验证唯一 v7 引擎路径：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
PARITY_CHECK_ONLY=1 \
MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_target_natural_backtest.sh
```

### 7.7 历史输出目录

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
  <feature_preset>_<target_col>_reb<period>_<timestamp>/
```

目录结构：

```text
00_predictions/
  test_preds.h5
  one_lag_prediction_manifest.json
  selected_checkpoints.csv
  actual_<target>.csv

01_close_auction_grid/
  00_grid_config.csv
  grid_engine_manifest.json
  grid_summary.csv
  01_runs/<run_name>/
  02_summary/
    grid_summary.csv
    grid_summary_compact.csv          # 视版本/后处理而定
    leaderboard_by_sharpe.csv
    leaderboard_by_calmar.csv
  04_logs/

materialized_best_run.json
```

历史根目录中最关键的三个权威文件：

```text
00_predictions/test_preds.h5
01_close_auction_grid/02_summary/grid_summary*.csv
materialized_best_run.json
```

### 7.8 Materialize 最佳历史 run

历史 wrapper 默认自动执行：

```text
scripts/materialize_as1455_best_run.py
```

手工执行示例：

```bash
python3 scripts/materialize_as1455_best_run.py \
  --backtest-root saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_addon_onehot_r05_fwd_reb5_<timestamp> \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --rank-metric sharpe \
  --capacity-mode none \
  --output-mode compact \
  --force
```

它会：

1. 从完整 summary 中选取最佳 `status=ok` 行；
2. 使用原 signal、max、sell、period、offset 只运行一个配置；
3. 将 compact/full 文件复制到正式 `01_runs/<run_name>/`；
4. 写 `materialized_best_run.json`；
5. 默认删除非选中 summary-only run 目录和日志。

要保留所有 summary-only run 目录：

```text
--keep-summary-run-dirs
```

---

## 8. Grid 与回测输出语义

### 8.1 In-process grid 的复用策略

`utils/as1455_grid_runner.py` 保证：

```text
每个 signal 的 prediction 只加载一次
每个 signal 每日排名只构建一次
execution panel 只构建一次
每个配置只调用一次 v7 backtest()
```

### 8.2 输出模式

| `OUTPUT_MODE` / `run-output-mode` | 内容 |
|---|---|
| `summary` | 仅 JSON summary/config，适合大 grid |
| `compact` | summary + NAV、回撤、月/年、费用、换手 |
| `full` | compact + 订单、持仓、拒单、公司行为、round trips 等完整审计 |

每个 run 必有：

```text
config.json
summary.json
close_auction_summary.json
```

compact 增加：

```text
close_auction_nav.csv
daily_drawdown.csv
monthly_summary.csv
yearly_summary.csv
fee_summary.csv
turnover_summary.csv
```

full 再增加：

```text
close_auction_orders.csv
close_auction_trades.csv
close_auction_rejections.csv
close_auction_positions.csv
close_auction_corporate_actions.csv
round_trips.csv
```

### 8.3 Run name

run name 由以下字段确定：

```text
signal_name
max_positions
sell_rank
rebalance_every
rebalance_offset
```

不要手工猜测 run 目录；优先读取：

```text
materialized_best_run.json::selection.run_name
strict_oos_manifest.json::retained_run_name
```

---

## 9. Fold0 Strict-OOS Forward

### 9.1 正式入口

全部 target 和 preset：

```bash
bash scripts/run_as1455_fold0_forward_backtests.sh
```

默认：

```text
REFRESH_DATA=1
MODEL_SELECTION_MODE=strict_oos
SELECTION_RANK_METRIC=sharpe
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot'
TARGETS='r01_fwd r05_fwd r21_fwd'
OUTPUT_MODE=compact
CAPACITY_MODE=none
FORCE_GRID=1
```

### 9.2 已有最新 forward model_data 时

```bash
REFRESH_DATA=0 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 9.3 只跑 r05 addon

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 9.4 显式绑定历史选择目录

为防止“最新目录”歧义，可指定：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
SELECTION_BACKTEST_ROOT='saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_addon_onehot_r05_fwd_reb5_<timestamp>' \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

`SELECTION_BACKTEST_ROOT` 只能在单 target、单 preset 时使用。

### 9.5 Strict-OOS 执行过程

```text
读取 fold0 search-time checkpoint
→ 使用 forward model_data 构造特征完整行
→ 仅保留 fold0.test_end 之后日期
→ 从历史 materialized/grid summary 冻结完整配置
→ 使用 execution calendar 换算 forward offset
→ 只生成一个 strict-OOS 配置
→ 执行 v7 回测
→ 写 strict_oos_manifest.json
```

正式 manifest 必须满足：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
historical_rebalance_phase_reused = true
generated_config_count = 1
retained_config_count = 1
```

### 9.6 Forward 输出目录

```text
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
  <feature_preset>_<target_col>_reb<period>_<timestamp>/
```

核心文件：

```text
00_predictions/
  fold0_forward_preds.h5
  fold0_forward_prediction_manifest.json
  selected_fold0_checkpoints.csv

01_close_auction_grid/
  strict_oos_manifest.json
  grid_engine_manifest.json
  00_grid_config.csv
  01_runs/<retained_run_name>/
  02_summary/grid_summary*.csv
```

### 9.7 其他 forward 模式

```text
forward_parameter_sweep  # 敏感性分析，不能作为正式 strict OOS
all_top_n                 # 兼容旧实验，运行全部信号
```

---

## 10. 回测与绘图工作流

### 10.1 r05 addon 分 fold / 跨 fold 专项结果

策略固定：

```text
target_col = r05_fwd
feature_preset = rotation_addon_onehot
```

运行：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
```

或：

```bash
bash scripts/run_as1455_r05_addon_fold_comparison.sh
```

执行内容：

```text
6 次 target_fold5..target_fold0 独立空仓回测
+ 复用并审计 authoritative materialized 跨 fold 连续 NAV
+ 生成分 fold 与跨 fold 日/周/月图
```

输出：

```text
saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/<timestamp>/
```

主要文件：

```text
fold_boundary_audit.csv
execution_data_report.csv
r05_addon_backtest_comparison.csv
r05_addon_fold_comparison_manifest.json
r05_addon_fold_comparison_report.json

per_fold/
  target_fold5/
  target_fold4/
  target_fold3/
  target_fold2/
  target_fold1/
  target_fold0/
  plots/return_curve_{daily,weekly,monthly}.{png,csv}

cross_fold/
  continuous_nav.csv
  continuous_fold_segments.csv
  materialized_run/
  plots/return_curve_{daily,weekly,monthly}.{png,csv}
```

`fold_boundary_audit.csv` 中相邻 target fold 必须：

```text
trading_gap_days = 0
```

### 10.2 全部策略独立 Fold 回测

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
```

含义：

```text
r01: 6 历史 folds × 2 presets = 12
r05: 6 历史 folds × 2 presets = 12
r21: 5 历史 folds × 2 presets = 10
fold0 strict-OOS: 3 targets × 2 presets = 6
合计 40 次固定配置回测
```

每个 fold：

```text
200000 元
空仓启动
冻结历史最佳 signal 和交易配置
换算本地 rebalance offset
```

输出：

```text
saved_data/ashare_ml4t/ch17_as1455_independent_folds/<timestamp>/
```

### 10.3 只切已有连续 NAV，不重新回测

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

该模式：

```text
prediction=false
backtest=false
grid=false
training=false
data_refresh=false
```

它只适合复用已有连续账户 NAV 绘图，不能替代“每个 fold 空仓独立启动”的实验。

### 10.4 受保护入口支持的模式

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

以下模式在该入口中故意禁用：

```text
all, data, training, historical, forward, audit, status ...
```

模型重训和历史 grid 应调用第 6、7 节脚本。

---

## 11. 已完成结果如何定位

GitHub 不保存服务器上的大模型、预测 HDF 和回测 CSV。文档使用 `<timestamp>` 表示实际运行目录。

### 11.1 最新 r05 addon 专项结果

```bash
LATEST=$(ls -dt saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/* | head -1)
echo "$LATEST"
cat "$LATEST/r05_addon_fold_comparison_report.json"
```

查看图：

```bash
find "$LATEST" -type f -name '*.png' | sort
```

查看对照表：

```bash
python3 - <<PY
import pandas as pd
root = '$LATEST'
print(pd.read_csv(f'{root}/r05_addon_backtest_comparison.csv').to_string(index=False))
print(pd.read_csv(f'{root}/fold_boundary_audit.csv').to_string(index=False))
PY
```

### 11.2 最新历史 grid

```bash
ls -dt saved_data/ashare_ml4t/ch17_as1455_target_backtest/* | head
```

检查一个目录：

```bash
ROOT='<历史结果目录>'
cat "$ROOT/materialized_best_run.json"
find "$ROOT/01_close_auction_grid/02_summary" -maxdepth 1 -type f -printf '%f\n' | sort
```

### 11.3 最新 strict-OOS forward

```bash
ls -dt saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/* | head
```

检查：

```bash
ROOT='<forward 结果目录>'
cat "$ROOT/01_close_auction_grid/strict_oos_manifest.json"
find "$ROOT/01_close_auction_grid/01_runs" -maxdepth 2 -type f | sort
```

### 11.4 训练 fold 完整性清单

```bash
find saved_data/ashare_ml4t \
  -type f \
  \( -name 'search_best_checkpoints.csv' \
     -o -name 'scaler.pkl' \
     -o -name 'feature_manifest.json' \) \
  | sort
```

---

## 12. 推荐端到端 Runbook

以下是“数据已经下载完成，从 model_data 开始重新训练并获得正式结果”的顺序。

### 12.1 环境与代码

```bash
cd /root/stock_realtime_v021_full
git fetch origin
git switch agent/ch17-as1455-clean
git pull --ff-only origin agent/ch17-as1455-clean
```

推荐 Python：

```bash
export PYTHON_BIN="$PWD/.venv_as1455/bin/python"
export PYTHON="$PYTHON_BIN"
```

关键依赖：

```text
numpy
pandas
PyTables/tables
scikit-learn
scipy
TensorFlow/Keras
TA-Lib
matplotlib
baostock（仅需要刷新/审计时）
```

### 12.2 验证现有数据

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

若 model_data 需要重建：

```bash
PYTHON="$PYTHON_BIN" \
  bash scripts/build_ashare_ch12_as1455_lowmem.sh
```

### 12.3 输入检查

```bash
PYTHON_BIN="$PYTHON_BIN" \
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
FOLDS='0' \
INPUT_CHECK_ONLY=1 \
FORCE=1 \
  bash scripts/run_as1455_target_search_all.sh
```

注意：不要让输入检查覆盖正式 canonical fold 目录。更稳妥的是直接调用 Python 并指定 `/tmp/... --out-dir`。

### 12.4 模型重训

```bash
PYTHON_BIN="$PYTHON_BIN" TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_search_all.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r05_target_search_all.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r21_target_search_all.sh
```

只重训 r05 addon：

```bash
PYTHON_BIN="$PYTHON_BIN" \
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_target_search_all.sh
```

### 12.5 历史预测与 grid

```bash
PYTHON_BIN="$PYTHON_BIN" TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_natural_backtest.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r05_natural_backtest.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r21_natural_backtest.sh
```

只跑 r05 addon：

```bash
PYTHON_BIN="$PYTHON_BIN" \
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_target_natural_backtest.sh
```

### 12.6 刷新 forward 数据

```bash
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/refresh_as1455_forward_model_data.sh
```

### 12.7 Strict-OOS forward

```bash
PYTHON_BIN="$PYTHON_BIN" \
REFRESH_DATA=0 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

只跑 r05 addon：

```bash
PYTHON_BIN="$PYTHON_BIN" \
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 12.8 r05 addon 分 fold / 跨 fold结果

```bash
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
```

### 12.9 全策略独立 fold 图

```bash
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
```

---

## 13. 开发与修改规则

### 13.1 修改 target

只修改：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

并同步检查：

- lookahead；
- CV 可用 folds；
- one-lag target folds；
- rebalance period 与 offset mode；
- forward phase alignment 测试。

### 13.2 修改 feature preset

公共特征构建必须集中在：

```text
utils/as1455_ch17_common.py
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

训练、历史预测和 forward 必须共享同一组：

```text
feature_cols_final
scale_cols
no_scale_cols
model_input_cols
sector categories
```

### 13.3 修改交易规则

唯一修改位置：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
```

配置构造统一在：

```text
utils/as1455_backtest_io.py::build_trade_config
```

修改后必须验证：

- 单一 v7 引擎；
- grid 与单配置回测一致；
- summary/compact/full 不改变交易逻辑；
- 旧 config 兼容性；
- strict-OOS 配置冻结字段。

### 13.4 修改 grid

集中在：

```text
utils/as1455_grid_runner.py
utils/as1455_signal_specs.py
```

不得在 Shell wrapper 中复制第二套组合逻辑。

### 13.5 修改存储策略

集中在：

```text
utils/as1455_artifact_retention.py
scripts/compact_as1455_prediction_artifacts.py
scripts/materialize_as1455_best_run.py
scripts/cleanup_as1455_storage.py
scripts/run_as1455_storage_maintenance.sh
```

共享行情缓存、model_data、checkpoint、scaler 和当前 contract 依赖文件不得进入默认删除清单。

---

## 14. 验证与 Smoke Test

### 14.1 总体验证

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

必须覆盖：

- Python/Shell 语法；
- CLI `--help`；
- failed-only summary 拒绝；
- 历史窗口 metadata；
- 最新 feature-complete forward 行保留；
- strict-OOS 相位换算；
- exact-offset 配置生成；
- prediction CSV 清理且 actual 保留；
- 单一 v7 引擎；
- r05 addon 固定单策略、6 次独立回测与连续 NAV 审计；
- 清理入口默认 dry-run。

### 14.2 Strict-OOS parity check

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
PARITY_CHECK_ONLY=1 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

日志应出现：

```text
[PHASE ALIGN] historical_offset=... history_days=... bridge_days=... forward_global_index=... effective_forward_offset=...
[PARITY] PASS
```

`PARITY_CHECK_ONLY=1` 只验证相位对齐后的单引擎路径，不生成正式完整结果。

### 14.3 检查 strict-OOS manifest

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('<forward root>/01_close_auction_grid/strict_oos_manifest.json')
x = json.loads(p.read_text(encoding='utf-8'))
for key in [
    'evaluation_mode',
    'historical_trading_parameters_reused',
    'historical_rebalance_phase_reused',
    'generated_config_count',
    'retained_config_count',
    'retained_run_name',
]:
    print(key, '=', x.get(key))
PY
```

---

## 15. 常见问题

### 15.1 为什么完整历史 grid 很慢、很大？

因为默认不是 6 或 40 次固定配置，而是最多：

```text
11340 个交易配置
```

解决：

- 大 grid 使用 `OUTPUT_MODE=summary`；
- grid 完成后只 materialize 最佳 run；
- 只跑目标 preset；
- 固定配置回测不要调用 natural-backtest grid wrapper。

### 15.2 为什么后一个 fold 与前一个 fold 日期有大缺口？

多 target 同图时，如果分别使用 `r01/r05/r21` 的 target-specific 有效日历再求交集，21 日标签会截掉更多日期。专项连续分析应固定单 target，并检查：

```text
fold_boundary_audit.csv::trading_gap_days = 0
```

### 15.3 “已选模型信号”在哪里？

不是独立目录。它保存在历史最佳完整配置中：

```text
materialized_best_run.json::selection
```

包含：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
rebalance_offset
run_name
```

模型 checkpoint 本身位于训练 fold 目录的：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
```

### 15.4 新的残缺目录会不会改变自动选择？

解析器会验证完整 strict-OOS/历史配对并跳过残缺目录，但仍建议：

- 错误运行结果立即移出 canonical 根目录并隔离；
- 单策略正式回测通过 `SELECTION_BACKTEST_ROOT` 显式绑定历史目录；
- 不依赖目录名“最新”推断权威结果。

### 15.5 分 fold 与跨 fold 有什么区别？

分 fold：

```text
每个 fold 200000 元 + 空仓重新启动
```

跨 fold：

```text
账户现金和持仓连续继承
fold 边界只切换 source model prediction
不强制清仓
```

两者是不同实验口径，不能通过简单切 NAV 或首尾拼接互相替代。

### 15.6 如何避免覆盖已有训练？

- 默认 `FORCE=0`；
- 新实验使用显式 `--out-dir`；
- 正式 canonical 目录只有确认重训时才 `--force`；
- 重训前先备份 `search_best_checkpoints.csv`、`search_checkpoints/` 和 `preprocess/`；
- 不要使用模糊 `rm -rf *fold*`。

---

## 16. 存储维护与清理

工具：

```text
scripts/check_as1455_disk_space.py
scripts/cleanup_as1455_storage.py
scripts/run_as1455_cleanup_safe.py
scripts/run_as1455_storage_maintenance.sh
```

默认维护：

```text
APPLY=0
INCLUDE_OBSOLETE=0
PRUNE_GRID_RUNS=0
```

先 dry-run：

```bash
bash scripts/run_as1455_storage_maintenance.sh
```

审核：

```text
share_me.txt
JSON audit manifest
```

再显式执行。共享行情缓存、训练 checkpoint、scaler、model_data 和当前结果 contract 依赖文件不在默认删除范围。

---

## 17. 核心文件索引

### 数据

```text
scripts/build_ashare_ch12_as1455_model_data.py
scripts/build_ashare_ch12_as1455_lowmem.sh
scripts/refresh_as1455_forward_model_data.sh
features/as1455_live_common.py
pipelines/as1455_update_history_to_prevday.py
pipelines/as1455_update_history_to_prevday_fast_v4.py
```

### 训练

```text
scripts/run_as1455_target_fold_param_search.py
scripts/run_as1455_target_search_all.sh
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

### 预测与历史 Grid

```text
scripts/run_as1455_target_one_lag_backtest.py
scripts/run_as1455_target_natural_backtest.sh
scripts/run_as1455_r05_natural_backtest.sh
scripts/run_as1455_r21_natural_backtest.sh
utils/as1455_cli.py
utils/as1455_signal_specs.py
utils/as1455_grid_runner.py
utils/as1455_rank_cache.py
utils/as1455_backtest_io.py
```

### 历史选择与 Strict OOS

```text
utils/as1455_model_selection.py
scripts/materialize_as1455_best_run.py
utils/as1455_rebalance_phase.py
utils/as1455_strict_oos.py
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

### 回测与绘图

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
code/backtest/run_as1455_close_auction_grid_inprocess.py
scripts/run_as1455_independent_fold_backtests.py
scripts/run_ch17_as1455_backtest_only.sh
scripts/run_as1455_r05_addon_fold_comparison.py
scripts/run_as1455_r05_addon_fold_comparison.sh
scripts/run_ch17_as1455_existing_results.sh
scripts/plot_as1455_backtest_return_curves.py
scripts/plot_as1455_fold_sequence_curves.py
scripts/plot_as1455_default_ab_nav_curves.sh
```

### 验证与存储

```text
scripts/check_ch17_as1455_refactor.sh
scripts/check_as1455_historical_model_selection.py
scripts/check_as1455_storage_oos_fixes.py
scripts/check_as1455_exact_offset_filter.py
scripts/check_as1455_artifact_retention.py
scripts/check_as1455_disk_space.py
scripts/cleanup_as1455_storage.py
scripts/run_as1455_cleanup_safe.py
scripts/run_as1455_storage_maintenance.sh
```
