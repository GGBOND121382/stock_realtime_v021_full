# AS1455 r1 / r5 / r21 训练、回测、严格样本外与存储指南

代码结构和开发约束见：

```text
CH17_AS1455_DEVELOPMENT_OUTLINE.md
AS1455_STORAGE_AND_STRICT_OOS.md
```

以下命令均在工程根目录执行：

```bash
cd ~/stock_realtime_v021_full
```

---

## 1. 固定实验口径

| 简称 | 标签 | lookahead | 自然调仓周期 | offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

特征方案：

| 名称 | `feature_preset` | 内容 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

正式训练产物：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

正式回测只使用 search-time checkpoint，不使用诊断性 retrain 模型。

---

## 2. 拉取代码与最低验证

```bash
git pull origin master
bash scripts/check_ch17_as1455_refactor.sh
```

正确结束标志：

```text
[PASS] Ch17 AS1455 refactor validation passed
```

检查内容包括：

- Python 和 Shell 语法；
- 历史模型选择只允许 `status=ok`；
- forward 无标签尾部日期不会被删除；
- strict OOS 完整继承历史交易参数；
- summary-first 与 model-only 存储默认值；
- 主要 CLI 可导入。

---

## 3. r1 / r5 / r21 训练

### 3.1 单 fold

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_onehot \
  --target-col r05_fwd \
  --fold-index 0 \
  --epochs 20 \
  --best-n 5
```

### 3.2 批量训练

r1：

```bash
TARGET_COL=r01_fwd \
  bash scripts/run_as1455_target_search_all.sh
```

r5：

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

r21：

```bash
bash scripts/run_as1455_r21_target_search_all.sh
```

当前 r1/r5 默认训练 fold0..fold6；r21 默认训练 fold0..fold5。

---

## 4. one-fold-lag 历史回测

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

### 4.1 存储安全的默认流程

历史大网格默认：

```text
OUTPUT_MODE=summary
```

即所有参数组合只保留指标汇总；网格完成后，wrapper 自动调用：

```text
scripts/materialize_as1455_best_run.py
```

仅把按 `RANK_METRIC` 选出的一个最佳 run 重新执行为 `compact`，保留其 NAV、回撤、月度、年度、费用和换手文件。这样绘图仍可用，但不会为数千个参数组合重复保存时间序列。

### 4.2 r5

```bash
bash scripts/run_as1455_r05_natural_backtest.sh
```

默认参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 5 offsets
= 1050 runs / feature preset
```

### 4.3 r21

```bash
bash scripts/run_as1455_r21_natural_backtest.sh
```

默认参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 21 offsets
= 4410 runs / feature preset
```

### 4.4 显式保留全部 compact/full 文件

仅在确有审计需要时使用：

```bash
OUTPUT_MODE=compact MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

或：

```bash
OUTPUT_MODE=full MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

这两种模式会显著增加存储占用，不是服务器默认协议。

### 4.5 历史输出

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
  <feature>_<target>_reb<period>_YYYYMMDD_HHMMSS/
    00_predictions/test_preds.h5
    01_close_auction_grid/02_summary/
    01_close_auction_grid/01_runs/<selected_run>/close_auction_nav.csv
    materialized_best_run.json
```

预测 HDF 是权威文件；默认删除重复的 `test_preds.csv` 和 `actual_*.csv`。

---

## 5. fold0-forward 严格样本外协议

### 5.1 默认行为：`strict_oos`

正式 forward 流程：

```text
从对应历史 target backtest 中选择 status=ok 且 Sharpe 最佳的完整行
→ 冻结 signal_name / signal_cols / signal_mode
→ 冻结 max_positions
→ 冻结 sell_rank
→ 冻结 rebalance_every
→ 冻结 rebalance_offset
→ 使用 fold0 checkpoint 在 date > fold0.test_end 的区间推理
→ 从空仓和初始现金开始，只评估该冻结配置
```

因此，历史图和 strict forward 图的以下字段必须一致：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
rebalance_offset
```

forward 数据只用于评价，不再用于选择交易参数。

### 5.2 最新日期不再受未来标签截断

训练和历史回测仍要求目标标签非空。

fold0-forward 改用：

```text
feature_row_mode = inference_features_only
```

只要求模型输入特征非空，不要求 `r01_fwd`、`r05_fwd` 或 `r21_fwd` 已实现。因此，数据更新到 2026-07-10 时，r5/r21 的预测结束日期也应到 2026-07-10，而不是分别停在 5/21 个交易日前。

manifest 记录：

```text
feature_meta.model_data_max_date
feature_meta.feature_valid_max_date
feature_meta.target_valid_max_date
feature_meta.unlabeled_prediction_rows
expected_prediction_end
prediction_end
```

并硬校验：

```text
prediction_end == 当前筛选条件下的 feature_valid_max_date
```

### 5.3 r5 strict OOS

```bash
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
MODEL_SELECTION_MODE='strict_oos' \
OUTPUT_MODE='compact' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

`strict_oos`、`compact` 已是默认值，可简写为：

```bash
TARGETS='r05_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.4 r21 strict OOS

```bash
TARGETS='r21_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.5 不重复刷新数据

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.6 固定历史来源目录

只运行一个 target 和一个 feature preset 时：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
SELECTION_BACKTEST_ROOT='saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_onehot_r05_fwd_reb5_YYYYMMDD_HHMMSS' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.7 forward 参数敏感性分析

需要观察 forward 区间不同交易参数时，显式运行：

```bash
MODEL_SELECTION_MODE='forward_parameter_sweep' \
TARGETS='r05_fwd' \
OUTPUT_MODE='summary' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

该结果只能标记为参数敏感性分析，不能作为严格样本外主结果。

### 5.8 旧全信号网格

```bash
MODEL_SELECTION_MODE='all_top_n' \
TOP_N=5 \
TARGETS='r05_fwd' \
OUTPUT_MODE='summary' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

这不是当前默认协议。

---

## 6. forward 输出与审计

输出根目录使用秒级时间戳，避免同一天重复执行误用旧结果：

```text
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
  <feature>_<target>_reb<period>_YYYYMMDD_HHMMSS/
```

预测目录：

```text
00_predictions/fold0_forward_preds.h5
00_predictions/selected_fold0_checkpoints.csv
00_predictions/fold0_forward_prediction_manifest.json
```

默认不长期保留重复的 `fold0_forward_preds.csv` 和 `actual_*.csv`。

strict OOS 回测目录额外包含：

```text
01_close_auction_grid/strict_oos_manifest.json
01_close_auction_grid/grid_engine_manifest.json
```

关键审计字段：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
historical_model_selection.run_name
historical_model_selection.historical_max_positions
historical_model_selection.historical_sell_rank
historical_model_selection.historical_rebalance_every
historical_model_selection.historical_rebalance_offset
strict_oos_config
retained_run_name
```

---

## 7. forward model_data 存储策略

刷新入口：

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

forward 目录默认使用：

```text
FORWARD_ARTIFACT_MODE=model_only
FORWARD_REPORT_MODE=compact
```

长期保留：

```text
ch12_as1455_forward_latest/model_data_as1455.h5
少量构建报告
```

构建验证完成后自动删除重复中间文件：

```text
as1455_ohlcv_raw.h5
as1455_ohlcv_adj.h5
as1455_execution_metadata.h5
```

这些文件可由 `ch12_as1455` 的共享 5 分钟、原始日线和 AS1455 日缓存重建，不应在 forward 目录永久复制。

所有高占用入口默认设置：

```text
MIN_FREE_GB=5
```

磁盘剩余空间低于阈值时任务会在写入前失败。

---

## 8. 绘图

统一入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

### 8.1 r5 历史 A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_target_backtest'
A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r05_fwd_reb5_*' | sort | tail -n 1)

BACKTEST_ROOTS="$A,$B" \
LABELS='r5-A,r5-B' \
RANK_METRIC='sharpe' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r5_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 8.2 r5 strict fold0-forward A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest'
A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r05_fwd_reb5_*' | sort | tail -n 1)

BACKTEST_ROOTS="$A,$B" \
LABELS='r5-A-fold0-forward,r5-B-fold0-forward' \
RANK_METRIC='sharpe' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r5_fold0_forward_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

strict 根目录的活动 summary 只有一行冻结配置，因此绘图不会在 forward 区间再次挑选 `sell/max/off`。

---

## 9. 存储审计与清理

### 9.1 先 dry-run

```bash
python3 scripts/cleanup_as1455_storage.py \
  --keep-live-dates 3 \
  --include-obsolete \
  --prune-grid-runs \
  --compress-reports
```

命令只输出候选动作并生成：

```text
saved_data/ashare_ml4t/cleanup_audit_YYYYMMDD_HHMMSS.json
```

### 9.2 审核 manifest 后执行

```bash
python3 scripts/cleanup_as1455_storage.py \
  --apply \
  --keep-live-dates 3 \
  --include-obsolete \
  --prune-grid-runs \
  --compress-reports
```

清理器会：

- 验证 forward `model_data_as1455.h5` 后删除重复中间 HDF；
- 只保留最近若干 live 日期；
- 删除旧保留日期中的可重建历史尾部；
- 删除已有 HDF 对应的重复 prediction CSV；
- 可选删除明确列入清单的旧 smoke/legacy 目录；
- 可选保留各指标/各信号优胜 run，删除其余重复 run 目录；
- 可选 gzip 大型审计 CSV；
- 检测运行中的 AS1455 进程并阻止误删。

---

## 10. 结果核对

### 10.1 检查 r5 forward 日期

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

base = Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')
for path in sorted(base.glob('rotation_*_r05_fwd_reb5_*/00_predictions/fold0_forward_preds.h5')):
    df = pd.read_hdf(path, 'predictions')
    dates = pd.DatetimeIndex(df.index.get_level_values('date'))
    print(path, dates.min().date(), dates.max().date(), dates.nunique())
PY
```

预测结束日期应等于 forward model_data 的最新特征有效日期，而不是 `r05_fwd` 的最后非空日期。

### 10.2 检查 strict 参数一致性

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')
for path in sorted(base.glob('*/01_close_auction_grid/strict_oos_manifest.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    print(path)
    print('historical:', {
        k: obj['historical_selection'].get(k)
        for k in [
            'historical_max_positions',
            'historical_sell_rank',
            'historical_rebalance_every',
            'historical_rebalance_offset',
        ]
    })
    print('retained:', obj['retained_config'])
PY
```

两者必须一致，且：

```text
historical_trading_parameters_reused = true
retained_config_count = 1
```
