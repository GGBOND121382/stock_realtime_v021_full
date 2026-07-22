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
→ 分 fold、跨 fold与综合绘图
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

> **入口边界**：`scripts/run_ch17_as1455_full_rebuild.sh` 当前只负责受保护的已有结果回测/绘图，不负责模型重训、model_data 重建、历史大 grid 或 forward 数据刷新。重训、历史 grid 和 strict-OOS forward 必须调用本文列出的专用脚本。

---

## 1. 流水线总览

| 阶段 | 正式入口 | 主要输入 | 主要输出 |
|---|---|---|---|
| 历史 model_data | `scripts/build_ashare_ch12_as1455_lowmem.sh` | 5 分钟、raw daily、静态股票池 | `ch12_as1455/model_data_as1455.h5` |
| Forward model_data | `scripts/refresh_as1455_forward_model_data.sh` | 共享历史缓存 | `ch12_as1455_forward_latest/model_data_as1455.h5` |
| 单 fold 模型搜索 | `scripts/run_as1455_target_fold_param_search.py` | 历史 model_data | checkpoint、scaler、IC 报告 |
| 一个 target 全 fold 搜索 | `scripts/run_as1455_target_search_all.sh` | 历史 model_data | 各 fold 搜索目录 |
| 历史 one-lag + grid | `scripts/run_as1455_target_natural_backtest.sh` | 搜索期 checkpoint、raw daily | 历史 grid、最佳 materialized run |
| Fold0 strict-OOS | `scripts/run_as1455_fold0_forward_backtests.sh` | fold0 checkpoint、forward model_data、历史最佳配置 | strict-OOS 单配置结果 |
| r05 addon 专项 | `run_ch17_as1455_full_rebuild.sh r05-addon-comparison` | 已完成历史/forward 结果 | 6 个独立 fold + 跨 fold结果 |
| 全策略独立 fold | `run_ch17_as1455_full_rebuild.sh independent-folds` | 已完成历史/forward 结果 | 40 次固定配置回测 |
| 只重画已有 NAV | `run_ch17_as1455_full_rebuild.sh existing-results` | 已有 NAV | 图和 CSV，不重新回测 |

---

## 2. 固定实验口径

### 2.1 时间与泄漏边界

- 特征只能使用交易日当日 `14:55` 及以前的数据；
- `r01_fwd`、`r05_fwd`、`r21_fwd` 分别是从当日 `14:55` 到未来第 1、5、21 个交易日 `14:55` 的收益；
- 历史交易执行价使用当日 `15:00` 收盘价近似收盘集合竞价成交价；
- 模型输入中禁止出现当日完整日线 `open/high/low/close/volume`、主板标记和交易状态等执行期字段；
- forward 推理只要求模型特征完整，不要求未来 target 已经实现。

### 2.2 默认交易口径

```text
long-only
沪深主板 only
ST 默认不可买
停牌不可交易
涨停不可买、跌停不可卖
T+1
100 股整数手
同日 15:00 价格近似成交
默认无滑点
公司行为默认使用 preclose 合成份额因子
```

默认资金与费用：

| 参数 | 默认值 |
|---|---:|
| `initial_cash` | `200000` |
| `commission_rate` | `0.000085` |
| `min_commission` | `5` |
| `stamp_tax_rate` | `0.0005`，仅卖出 |
| `transfer_fee_rate` | `0.00001`，双边 |
| `slippage_bps` | `0` |
| `lot_size` | `100` |

### 2.3 Target 与调仓周期

唯一 target 定义：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

| target | lookahead | `rebalance_every` | 历史 offset |
|---|---:|---:|---|
| `r01_fwd` | 1 | 1 | `0` |
| `r05_fwd` | 5 | 5 | `0..4` |
| `r21_fwd` | 21 | 21 | `0..20` |

v7 本地调仓条件：

```text
(day_index - rebalance_offset) mod rebalance_every = 0
```

`rebalance_offset` 是当前回测窗口的本地序号。历史、独立 fold 和 forward 窗口都可能从 `day_index=0` 重新编号，因此跨窗口必须换算相位，禁止机械复制整数。

### 2.4 Feature preset

```text
rotation_onehot
rotation_addon_onehot
```

- `rotation_onehot`：基础 Ch12 特征 + 同日行业轮动特征 + sector one-hot；
- `rotation_addon_onehot`：在前者基础上加入 compact addon 特征组。

### 2.5 Fold 编号与 one-fold-lag

时间序列 CV 中：

```text
fold0 = 最新测试窗口
fold6 = 最早测试窗口
```

历史 one-fold-lag：

```text
source model fold6 -> target fold5
source model fold5 -> target fold4
...
source model fold1 -> target fold0
```

范围：

| target | 训练 fold | 历史 target fold |
|---|---|---|
| `r01_fwd` | `0..6` | `0..5` |
| `r05_fwd` | `0..6` | `0..5` |
| `r21_fwd` | `0..5` | `0..4` |

报告与图片应同时记录 `source_model_fold` 和 `target_fold`。

---

## 3. 核心代码与唯一语义来源

### 3.1 数据构建

```text
scripts/build_ashare_ch12_as1455_model_data.py
scripts/build_ashare_ch12_as1455_lowmem.sh
scripts/refresh_as1455_forward_model_data.sh
```

职责：聚合 14:55 日级 OHLCV、严格 as-of 复权、技术指标、横截面与行业特征、forward 标签、34 列 model_data 和数据审计报告。

### 3.2 特征、fold、checkpoint、预测

```text
utils/as1455_ch17_common.py
utils/as1455_forward_features.py
```

`as1455_ch17_common.py` 统一实现 target、feature preset、CV、checkpoint、scaler、模型输入列、历史预测和 grid 命令。其他入口不得复制这些逻辑。

`as1455_forward_features.py` 仅改变行保留：

```text
训练/历史：模型特征非空 + 当前 target 非空
forward：模型特征非空；当前 target 可以为空
```

### 3.3 模型搜索

```text
scripts/run_as1455_target_fold_param_search.py
scripts/run_as1455_target_search_all.sh
scripts/run_as1455_r05_target_search_all.sh
scripts/run_as1455_r21_target_search_all.sh
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

正式通用入口是 `run_as1455_target_fold_param_search.py`；底层神经网络、训练循环、IC 与 checkpoint 保存由 `run_as1455_sector_rotation_fold0_param_search.py` 提供公共实现。

### 3.4 历史预测、grid 与输出

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

### 3.5 历史选择与 materialize

```text
utils/as1455_model_selection.py
scripts/materialize_as1455_best_run.py
```

- 只接受 `status=ok`；
- 选择完整 run，而不是只选 signal；
- 冻结 signal、max positions、sell rank、rebalance period、历史 offset 和历史窗口；
- 跳过更新但残缺或 failed-only 的目录；
- `materialized_best_run.json` 是历史最佳 run 的权威索引。

### 3.6 Strict-OOS 与相位

```text
utils/as1455_rebalance_phase.py
utils/as1455_strict_oos.py
scripts/run_as1455_fold0_forward_backtest.py
scripts/run_as1455_fold0_forward_backtests.sh
```

Formal strict-OOS 从历史选择中冻结：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
historical rebalance phase
historical date_min/date_max/n_days
```

需要精确区分：

- `max_positions/sell_rank/rebalance_every/offset` 是历史 grid 搜索和选择出来的；
- 佣金、主板、ST、涨跌停、容量、公司行为等执行策略**不是**从历史 grid summary 中二次选择出来的；
- formal forward 通过同一共享 wrapper 的参数保持这些执行策略不变。若历史运行曾使用非默认执行参数，forward 必须显式传入同样参数并核对 `config.json`；
- r05 addon 专项解析器会读取历史 materialized `config.json`，因此该专项会冻结完整存储配置。

相位换算：

```text
forward_global_index
  = historical_n_days + bridge_execution_days

effective_forward_offset
  = (historical_offset - forward_global_index) mod rebalance_every
```

`bridge_execution_days` 来自完整 raw-daily execution calendar，不使用自然日。

### 3.7 唯一交易引擎

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

共享 grid CLI：

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
→ utils/as1455_grid_runner.py
```

禁止维护第二套买卖循环。

---

## 4. 数据、模型与结果位置

Canonical path：

```text
utils/as1455_paths.py
```

| 类型 | 位置 |
|---|---|
| 静态股票池 | `saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv` |
| 历史 AS1455 根 | `saved_data/ashare_ml4t/ch12_as1455/` |
| 历史 model_data | `saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5` |
| 5 分钟缓存 | `saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/` |
| raw daily 缓存 | `saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/` |
| 14:55 日级缓存 | `saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache/` |
| 数据构建报告 | `saved_data/ashare_ml4t/ch12_as1455/reports/` |
| forward model_data | `saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5` |
| 通用模型搜索 | `saved_data/ashare_ml4t/ch17_as1455_target_search/` |
| 历史 grid | `saved_data/ashare_ml4t/ch17_as1455_target_backtest/` |
| strict-OOS forward | `saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/` |
| 40 次独立 fold | `saved_data/ashare_ml4t/ch17_as1455_independent_folds/` |
| r05 addon 专项 | `saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/` |
| 通用绘图 | `saved_data/ashare_ml4t/ch17_as1455_backtest_plots/` |

### 4.1 5 分钟缓存

```text
saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/<symbol>_5m_raw.csv
```

典型字段：

```text
symbol, trade_date, datetime, open, high, low, close,
volume, amount, source, bar_freq, bar_label
```

### 4.2 Raw daily

```text
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/<symbol>_daily_raw.csv
```

至少依赖：

```text
date, open, high, low, close, preclose, volume, amount,
tradestatus, isST
```

### 4.3 Model data

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

中间 HDF：

```text
as1455_ohlcv_raw.h5          key=ohlcv
as1455_ohlcv_adj.h5          key=ohlcv
as1455_execution_metadata.h5 key=metadata
```

### 4.4 Prediction HDF

历史 one-lag：

```text
<历史根>/00_predictions/test_preds.h5
HDF key: predictions
```

fold0 forward：

```text
<forward 根>/00_predictions/fold0_forward_preds.h5
HDF key: predictions
```

HDF 是权威预测文件；`actual_<target>.csv` 是真实标签，不得误删。

---

## 5. 历史 Model Data 构建

### 5.1 推荐离线低内存入口

```bash
cd /root/stock_realtime_v021_full
bash scripts/build_ashare_ch12_as1455_lowmem.sh
```

默认：

- 使用已有 5 分钟和 qfq daily 缓存；
- 禁止补拉 5 分钟和 qfq daily；
- 使用 raw daily `close/preclose` 构造复权；
- 输出内存检查点。

输出：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455/reports/
```

### 5.2 强制重建 14:55 日级缓存

```bash
REBUILD_DAILY_CACHE=1 \
  bash scripts/build_ashare_ch12_as1455_lowmem.sh
```

只在 5 分钟源数据、timestamp convention 或日级缓存发生变化时启用。

### 5.3 直接调用构建器

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

### 5.4 安全 smoke

```bash
python3 scripts/build_ashare_ch12_as1455_model_data.py \
  --max-symbols 20 \
  --allow-partial-coverage \
  --out-dir /tmp/ch12_as1455_smoke
```

不要把 smoke 输出写进正式 `ch12_as1455`。

### 5.5 必查报告

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
```

快速检查：

```bash
python3 - <<'PY'
import pandas as pd
p = 'saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5'
df = pd.read_hdf(p, 'model_data')
dates = pd.DatetimeIndex(df.index.get_level_values('date'))
print('shape=', df.shape)
print('symbols=', df.index.get_level_values('symbol').nunique())
print('date=', dates.min(), dates.max())
for col in ['r01_fwd', 'r05_fwd', 'r21_fwd']:
    valid = df[col].notna()
    d = pd.DatetimeIndex(df.index.get_level_values('date')[valid])
    print(col, int(valid.sum()), d.max())
PY
```

---

## 6. Forward Model Data

刷新历史缓存并构建：

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

默认流程：

```text
更新共享历史缓存
→ 在 ch12_as1455_forward_latest 重建 model_data
→ 验证 34 列与日期
→ 删除 forward 可重建中间 HDF
→ 压缩超大报告 CSV
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455_forward_latest/reports/
```

共享缓存已经更新时：

```bash
SKIP_HISTORY_UPDATE=1 \
  bash scripts/refresh_as1455_forward_model_data.sh
```

strict-OOS 直接复用现有 forward model_data：

```bash
REFRESH_DATA=0 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

---

## 7. 模型重训与参数搜索

### 7.1 参数空间

```text
dense_layers: (16,8), (32,16), (32,32), (64,32)
activation: tanh
dropout: 0, 0.1, 0.2
batch_size: 64, 256
epochs: 20
```

共 `24` 组网络参数。每组连续训练 20 个 epoch，每个 epoch 计算：

```text
pooled_spearman
daily_ic_mean
daily_ic_median
daily_ic_positive_rate
```

默认按 `daily_ic_median` 保留前 5 个搜索期 model-epoch checkpoint。

> `RETRAIN_BEST=1` 生成的 `models/best_*` 只用于诊断。正式历史与 forward 预测读取 `search_best_checkpoints.csv` 指向的 `search_checkpoints/*.keras`。

### 7.2 安全输入检查，不训练、不碰 canonical 目录

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
run_summary.json
fold_report.json / fold_report.csv
feature_cols_final.json
rotation_feature_cols.json
addon_feature_cols.json
feature_group_cols.json
sector_onehot_cols.json
param_grid.csv
```

### 7.3 安全训练 smoke

```bash
rm -rf /tmp/as1455_r05_addon_fold0_smoke
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_addon_onehot \
  --target-col r05_fwd \
  --model-data saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 \
  --fold-index 0 \
  --smoke \
  --out-dir /tmp/as1455_r05_addon_fold0_smoke
```

不要用 `SMOKE=1 FORCE=1` 指向已有正式 fold 目录。

### 7.4 单 fold 正式重训

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

`--force` 是从头重建 canonical 搜索产物，不是断点续跑。

### 7.5 一个 target 全部 preset/fold

r01：

```bash
TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_search_all.sh
```

r05：

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

r21：

```bash
bash scripts/run_as1455_r21_target_search_all.sh
```

只训练 r05 addon：

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

### 7.6 训练目录

r01 兼容目录：

```text
rotation_onehot:
saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold{fold}_search/

rotation_addon_onehot:
saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold{fold}_search/
```

r05/r21：

```text
saved_data/ashare_ml4t/ch17_as1455_target_search/
  <feature_preset>/<target_col>/fold{fold}_search/
```

日志：

```text
saved_data/ashare_ml4t/ch17_as1455_target_search/logs/
  <preset>_<target>_fold<fold>_search.log
```

### 7.7 每个 fold 的核心产物

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

权威链：

```text
search_best_checkpoints.csv
→ search_checkpoints/*.keras
→ preprocess/scaler.pkl
→ preprocess/feature_manifest.json
```

快速检查：

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('saved_data/ashare_ml4t/ch17_as1455_target_search/rotation_addon_onehot/r05_fwd/fold0_search')
print(pd.read_csv(root / 'search_best_checkpoints.csv')[
    ['checkpoint_name', 'daily_ic_median', 'daily_ic_mean',
     'pooled_spearman', 'checkpoint_saved']
])
print('keras=', len(list((root / 'search_checkpoints').glob('*.keras'))))
PY
```

---

## 8. 历史 One-Fold-Lag 与交易 Grid

### 8.1 协议

```text
source model fold(target+1)
→ 在 target fold 窗口生成预测
→ 合并各 target fold prediction
→ 对完整历史 one-lag 序列执行交易参数 grid
```

### 8.2 入口

r01：

```bash
TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_natural_backtest.sh
```

r05：

```bash
bash scripts/run_as1455_r05_natural_backtest.sh
```

r21：

```bash
bash scripts/run_as1455_r21_natural_backtest.sh
```

只跑 r05 addon：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_target_natural_backtest.sh
```

### 8.3 默认设置

```text
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

### 8.4 Top-5 对应 7 个 signal

```text
model_0
model_1
model_2
model_3
model_4
ensemble_first3_mean
ensemble_all5_mean
```

定义：

```text
utils/as1455_signal_specs.py
```

### 8.5 Grid 规模

配置元组：

```text
signal × max_positions × sell_rank × rebalance_every × offset
```

| target | configs/preset |
|---|---:|
| `r01_fwd` | `7 × 5 × 6 × 1 = 210` |
| `r05_fwd` | `7 × 5 × 6 × 5 = 1050` |
| `r21_fwd` | `7 × 5 × 6 × 21 = 4410` |

两个 preset、全部 target：

```text
2 × (210 + 1050 + 4410) = 11340 个交易配置
```

这是历史大 grid，不应与 6/40 次冻结配置回测混淆。

### 8.6 安全运行

全新 r05 grid：

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
FORCE_GRID=1 \
OUTPUT_MODE=summary \
MATERIALIZE_BEST=1 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

复用同一时间戳并跳过成功配置：

```bash
RUN_STAMP=<已有时间戳> \
FORCE_GRID=0 \
OUTPUT_MODE=summary \
MATERIALIZE_BEST=1 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

仅检查配置：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
DRY_RUN=1 \
MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_target_natural_backtest.sh
```

单引擎 parity：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
PARITY_CHECK_ONLY=1 \
MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_target_natural_backtest.sh
```

### 8.7 历史结果目录

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
  <feature_preset>_<target_col>_reb<period>_<timestamp>/
```

核心结构：

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
    grid_summary_compact.csv
    leaderboard_by_sharpe.csv
    leaderboard_by_calmar.csv
  04_logs/

materialized_best_run.json
```

### 8.8 Materialize 最佳历史 run

```bash
python3 scripts/materialize_as1455_best_run.py \
  --backtest-root saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_addon_onehot_r05_fwd_reb5_<timestamp> \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --rank-metric sharpe \
  --capacity-mode none \
  --output-mode compact \
  --force
```

流程：

1. 从完整 summary 选择最佳 `status=ok` 行；
2. 原 signal/max/sell/period/offset 只执行一次；
3. 保存 compact/full NAV 与审计文件；
4. 写 `materialized_best_run.json`；
5. 默认删除非选中 summary-only run 目录和日志。

保留 summary-only run：

```text
--keep-summary-run-dirs
```

---

## 9. Grid 与单次回测输出

`utils/as1455_grid_runner.py` 保证：

```text
prediction 每个 signal 加载一次
每个 signal 每日排序一次
execution panel 构造一次
每个配置只调用唯一 v7 backtest()
```

输出模式：

| mode | 文件 |
|---|---|
| `summary` | JSON summary/config，适合大 grid |
| `compact` | summary + NAV、回撤、月/年、费用、换手 |
| `full` | compact + 订单、持仓、拒单、公司行为、round trips |

每个 run 必有：

```text
config.json
summary.json
close_auction_summary.json
```

compact：

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

不要猜 run name。优先读取：

```text
materialized_best_run.json::selection.run_name
strict_oos_manifest.json::retained_run_name
```

---

## 10. Fold0 Strict-OOS Forward

### 10.1 正式入口

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

已有 forward model_data：

```bash
REFRESH_DATA=0 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

只跑 r05 addon：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 10.2 显式绑定历史根目录

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
SELECTION_BACKTEST_ROOT='saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_addon_onehot_r05_fwd_reb5_<timestamp>' \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

`SELECTION_BACKTEST_ROOT` 仅允许单 target、单 preset。

### 10.3 Formal strict-OOS 流程

```text
读取 fold0 search-time checkpoint
→ 在 forward model_data 上构造特征完整行
→ 只保留 fold0.test_end 之后日期
→ 从历史结果冻结 signal/max/sell/period/历史相位
→ 用 execution calendar 换算 effective forward offset
→ 保持同一套执行策略参数
→ 只生成一个 phase-aligned 配置
→ 执行 v7 回测
→ 写 strict_oos_manifest.json
```

Manifest：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
historical_rebalance_phase_reused = true
generated_config_count = 1
retained_config_count = 1
```

### 10.4 结果位置

```text
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
  <feature_preset>_<target_col>_reb<period>_<timestamp>/
```

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

其他模式：

```text
forward_parameter_sweep  # 敏感性分析，不是正式 strict OOS
all_top_n                 # 兼容旧实验
```

---

## 11. 分 Fold、跨 Fold 与绘图

### 11.1 r05 addon 专项

固定：

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

执行：

```text
6 次 target_fold5..target_fold0 独立空仓回测
+ 复用并审计 authoritative materialized 跨 fold连续 NAV
+ 分 fold和跨 fold日/周/月图
```

该专项读取历史 materialized run 的 `config.json`，冻结 signal、交易参数、费用、主板/ST、容量和公司行为配置。

输出：

```text
saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/<timestamp>/
```

```text
fold_boundary_audit.csv
execution_data_report.csv
r05_addon_backtest_comparison.csv
r05_addon_fold_comparison_manifest.json
r05_addon_fold_comparison_report.json

per_fold/
  target_fold5/ ... target_fold0/
  plots/return_curve_{daily,weekly,monthly}.{png,csv}

cross_fold/
  continuous_nav.csv
  continuous_fold_segments.csv
  materialized_run/
  plots/return_curve_{daily,weekly,monthly}.{png,csv}
```

必须：

```text
fold_boundary_audit.csv::trading_gap_days = 0
```

### 11.2 全策略独立 fold

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
```

```text
r01: 6 × 2 = 12
r05: 6 × 2 = 12
r21: 5 × 2 = 10
fold0 strict-OOS: 3 × 2 = 6
总计 40 次冻结配置回测
```

每个 fold 以 `200000` 元、空仓独立启动并换算本地 offset。

输出：

```text
saved_data/ashare_ml4t/ch17_as1455_independent_folds/<timestamp>/
```

### 11.3 只重画已有连续 NAV

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

```text
prediction=false
backtest=false
grid=false
training=false
data_refresh=false
```

该模式不能替代独立 fold实验。

### 11.4 受保护入口模式

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

模型重训、历史 grid 和 forward 使用专用脚本，不通过该入口。

---

## 12. 已完成结果定位

服务器结果不存入 GitHub；以下用 `<timestamp>` 表示实际目录。

最新 r05 addon：

```bash
LATEST=$(ls -dt saved_data/ashare_ml4t/ch17_as1455_r05_addon_fold_comparison/* | head -1)
echo "$LATEST"
cat "$LATEST/r05_addon_fold_comparison_report.json"
find "$LATEST" -type f -name '*.png' | sort
```

查看对照：

```bash
python3 - <<PY
import pandas as pd
root = '$LATEST'
print(pd.read_csv(f'{root}/r05_addon_backtest_comparison.csv').to_string(index=False))
print(pd.read_csv(f'{root}/fold_boundary_audit.csv').to_string(index=False))
PY
```

最新历史 grid：

```bash
ls -dt saved_data/ashare_ml4t/ch17_as1455_target_backtest/* | head
```

```bash
ROOT='<历史结果目录>'
cat "$ROOT/materialized_best_run.json"
find "$ROOT/01_close_auction_grid/02_summary" -maxdepth 1 -type f -printf '%f\n' | sort
```

最新 strict-OOS：

```bash
ls -dt saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/* | head
```

```bash
ROOT='<forward 结果目录>'
cat "$ROOT/01_close_auction_grid/strict_oos_manifest.json"
find "$ROOT/01_close_auction_grid/01_runs" -maxdepth 2 -type f | sort
```

训练产物清单：

```bash
find saved_data/ashare_ml4t \
  -type f \
  \( -name 'search_best_checkpoints.csv' \
     -o -name 'scaler.pkl' \
     -o -name 'feature_manifest.json' \) \
  | sort
```

---

## 13. 推荐端到端 Runbook

### 13.1 环境

```bash
cd /root/stock_realtime_v021_full
git fetch origin
git switch agent/ch17-as1455-clean
git pull --ff-only origin agent/ch17-as1455-clean

export PYTHON_BIN="$PWD/.venv_as1455/bin/python"
export PYTHON="$PYTHON_BIN"
```

关键依赖：

```text
numpy, pandas, PyTables, scikit-learn, scipy,
TensorFlow/Keras, TA-Lib, matplotlib, baostock
```

### 13.2 验证与数据

```bash
bash scripts/check_ch17_as1455_refactor.sh
PYTHON="$PYTHON_BIN" bash scripts/build_ashare_ch12_as1455_lowmem.sh
```

数据无变化时无需重复构建 model_data。

### 13.3 模型重训

```bash
PYTHON_BIN="$PYTHON_BIN" TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_search_all.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r05_target_search_all.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r21_target_search_all.sh
```

仅 r05 addon：

```bash
PYTHON_BIN="$PYTHON_BIN" \
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_target_search_all.sh
```

### 13.4 历史 grid

```bash
PYTHON_BIN="$PYTHON_BIN" TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_natural_backtest.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r05_natural_backtest.sh

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_as1455_r21_natural_backtest.sh
```

仅 r05 addon：

```bash
PYTHON_BIN="$PYTHON_BIN" \
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_target_natural_backtest.sh
```

### 13.5 Forward 数据与 strict-OOS

```bash
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/refresh_as1455_forward_model_data.sh

PYTHON_BIN="$PYTHON_BIN" \
REFRESH_DATA=0 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

仅 r05 addon：

```bash
PYTHON_BIN="$PYTHON_BIN" \
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 13.6 最终图和专项结果

```bash
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison

PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
```

---

## 14. 开发规则

### 14.1 Target

只在：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

定义，并同步验证 lookahead、fold 范围、one-lag mapping、period、offset 和 forward phase。

### 14.2 Feature

集中在：

```text
utils/as1455_ch17_common.py
scripts/run_as1455_sector_rotation_fold0_param_search.py
scripts/run_as1455_first_batch_features_fold0_param_search.py
```

训练、历史和 forward 必须共享：

```text
feature_cols_final
scale_cols
no_scale_cols
model_input_cols
sector categories
```

### 14.3 交易规则

唯一修改：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
```

配置统一在：

```text
utils/as1455_backtest_io.py::build_trade_config
```

### 14.4 Grid

集中在：

```text
utils/as1455_grid_runner.py
utils/as1455_signal_specs.py
```

### 14.5 存储

集中在：

```text
utils/as1455_artifact_retention.py
scripts/compact_as1455_prediction_artifacts.py
scripts/materialize_as1455_best_run.py
scripts/cleanup_as1455_storage.py
scripts/run_as1455_storage_maintenance.sh
```

共享行情缓存、model_data、checkpoint、scaler 和当前 contract 文件不得进入默认删除清单。

---

## 15. 验证与 Smoke

总体验证：

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

必须覆盖：

- Python/Shell 语法和 CLI；
- failed-only summary 拒绝；
- 历史窗口 metadata；
- forward feature-complete 行保留；
- strict-OOS 相位换算；
- exact-offset 生成；
- prediction CSV 清理且 actual 保留；
- 唯一 v7 引擎；
- r05 addon 固定单策略、6 次独立回测与连续 NAV 审计；
- 清理默认 dry-run。

Strict-OOS parity：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
PARITY_CHECK_ONLY=1 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

日志：

```text
[PHASE ALIGN] historical_offset=... history_days=... bridge_days=... forward_global_index=... effective_forward_offset=...
[PARITY] PASS
```

检查 strict manifest：

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

## 16. 常见问题

### 16.1 为什么历史 grid 很慢、很大？

因为完整默认值是 `11340` 个配置，不是 6/40 次冻结配置回测。大 grid 应使用：

```text
OUTPUT_MODE=summary
MATERIALIZE_BEST=1
```

### 16.2 为什么相邻 fold 有大段日期缺口？

同时比较 r01/r05/r21 时，target-specific 有效日历不同，逐 fold 取交集会被 r21 的 21 日标签截短。单 target 连续分析应检查：

```text
fold_boundary_audit.csv::trading_gap_days = 0
```

### 16.3 已选模型信号在哪里？

不是独立目录。历史最佳完整配置位于：

```text
materialized_best_run.json::selection
```

模型 checkpoint 位于：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
```

### 16.4 残缺新目录会不会污染选择？

解析器会验证历史/strict-OOS 完整配对并跳过残缺目录，但仍应：

- 将错误结果移出 canonical 根目录并隔离；
- 单策略正式运行使用 `SELECTION_BACKTEST_ROOT` 显式绑定；
- 不仅凭“最新目录”判断权威结果。

### 16.5 分 fold 与跨 fold 的区别

```text
分 fold：每段 200000 元 + 空仓重新启动
跨 fold：现金和持仓连续继承，边界只切换模型预测
```

不能通过简单切 NAV 或首尾拼接互相替代。

### 16.6 如何避免覆盖已有训练？

- 默认 `FORCE=0`；
- input-check/smoke 使用 `/tmp/... --out-dir`；
- 只有明确重训 canonical fold 才用 `--force`；
- 重训前备份 `search_best_checkpoints.csv`、`search_checkpoints/` 和 `preprocess/`；
- 禁止模糊删除 `*fold*`。

---

## 17. 存储维护

```text
scripts/check_as1455_disk_space.py
scripts/cleanup_as1455_storage.py
scripts/run_as1455_cleanup_safe.py
scripts/run_as1455_storage_maintenance.sh
```

默认：

```text
APPLY=0
INCLUDE_OBSOLETE=0
PRUNE_GRID_RUNS=0
```

先运行 dry-run：

```bash
bash scripts/run_as1455_storage_maintenance.sh
```

审核 `share_me.txt` 和 JSON audit manifest 后再显式执行。

---

## 18. 核心文件索引

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

### 预测与历史 grid

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

### 历史选择与 strict-OOS

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
