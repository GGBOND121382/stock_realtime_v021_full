# AS1455 r1 / r5 / r21 运行指南

详细开发约束和存储政策见：

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

正式训练只使用：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

---

## 2. 拉取代码与验证

```bash
git pull origin master
bash scripts/check_ch17_as1455_refactor.sh
```

正确结束标志：

```text
[PASS] Ch17 AS1455 refactor validation passed
```

验证集覆盖：

- Python/Shell 语法；
- 目标—周期映射；
- failed-only summary 拒绝；
- forward 最新无标签日期保留；
- strict OOS 参数冻结；
- prediction CSV 压缩时保留真实标签文件；
- 唯一 v7 交易引擎；
- summary-first、model-only 和磁盘门禁默认值。

---

## 3. 训练

单 fold 示例：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_onehot \
  --target-col r05_fwd \
  --fold-index 0 \
  --epochs 20 \
  --best-n 5
```

批量训练：

```bash
TARGET_COL=r01_fwd bash scripts/run_as1455_target_search_all.sh
bash scripts/run_as1455_r05_target_search_all.sh
bash scripts/run_as1455_r21_target_search_all.sh
```

r1/r5 默认 fold0..fold6；r21 默认 fold0..fold5。

---

## 4. one-fold-lag 历史回测

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

### 4.1 存储安全默认值

完整历史网格默认：

```text
OUTPUT_MODE=summary
MATERIALIZE_BEST=1
MATERIALIZED_OUTPUT_MODE=compact
RANK_METRIC=sharpe
```

全部参数组合只保存指标 summary。网格完成后，系统只把一个历史最佳 run 重新执行为 compact，并删除其他 summary-only run 目录和日志；完整参数指标仍保留在 `02_summary`。

r5：

```bash
bash scripts/run_as1455_r05_natural_backtest.sh
```

r21：

```bash
bash scripts/run_as1455_r21_natural_backtest.sh
```

r5 默认每个特征方案 1050 个组合；r21 默认每个特征方案 4410 个组合。不要对整个网格默认使用 `full`。

### 4.2 确需保留全部明细

```bash
OUTPUT_MODE=compact MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

或：

```bash
OUTPUT_MODE=full MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

这两种模式会显著增加磁盘占用。

### 4.3 历史结果结构

```text
ch17_as1455_target_backtest/
  <feature>_<target>_reb<period>_YYYYMMDD_HHMMSS/
    00_predictions/test_preds.h5
    00_predictions/actual_<target>.csv
    00_predictions/selected_checkpoints.csv
    00_predictions/*manifest.json
    01_close_auction_grid/02_summary/
    01_close_auction_grid/01_runs/<selected_run>/
    materialized_best_run.json
```

`test_preds.h5` 是预测权威文件；同内容的 `test_preds.csv` 默认删除并同步更新 manifest。`actual_<target>.csv` 是真实标签，不属于重复数据，必须保留。

---

## 5. fold0-forward 严格样本外

### 5.1 日期口径

训练和历史回测要求当前目标标签非空。

fold0-forward 只要求模型特征非空，不要求未来目标已实现：

```text
feature_row_mode = inference_features_only
```

因此，forward model data 更新到 2026-07-10 时，r5/r21 的 prediction end 也应到最新特征有效日期，而不是分别停在 5/21 个交易日前。

manifest 记录：

```text
feature_meta.model_data_max_date
feature_meta.feature_valid_max_date
feature_meta.target_valid_max_date
feature_meta.unlabeled_prediction_rows
feature_meta.unlabeled_prediction_dates
expected_prediction_end
prediction_end
```

并硬校验：

```text
prediction_end == expected_prediction_end
```

### 5.2 正式默认：`strict_oos`

系统从历史 summary 中选择 `status=ok` 且指标最佳的完整行，冻结：

```text
signal_name
signal_cols
signal_mode
max_positions
sell_rank
rebalance_every
rebalance_offset
```

然后用 fold0 checkpoint 在 `date > fold0.test_end` 区间从空仓和初始现金开始回测。forward 数据只用于评价，不再用于调参。

r5 A/B：

```bash
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
REFRESH_DATA=0 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

默认值已经是：

```text
MODEL_SELECTION_MODE=strict_oos
OUTPUT_MODE=compact
SELECTION_RANK_METRIC=sharpe
```

r21：

```bash
TARGETS='r21_fwd' REFRESH_DATA=0 \
  bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.3 自动更新数据后运行

```bash
TARGETS='r05_fwd' \
REFRESH_DATA=1 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

服务器空间紧张时，应先清理，再执行 `REFRESH_DATA=1`。

### 5.4 固定历史来源

只运行一个 target 和一个 feature preset 时：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
SELECTION_BACKTEST_ROOT='saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_onehot_r05_fwd_reb5_YYYYMMDD_HHMMSS' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.5 forward 参数敏感性分析

```bash
MODEL_SELECTION_MODE='forward_parameter_sweep' \
TARGETS='r05_fwd' \
OUTPUT_MODE='summary' \
REFRESH_DATA=0 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

该结果只能作为敏感性分析，不能作为严格样本外主结果。

旧全信号网格：

```bash
MODEL_SELECTION_MODE='all_top_n' \
TOP_N=5 \
TARGETS='r05_fwd' \
OUTPUT_MODE='summary' \
REFRESH_DATA=0 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.6 strict 输出结构

```text
ch17_as1455_fold0_forward_backtest/
  <feature>_<target>_reb<period>_YYYYMMDD_HHMMSS/
    00_predictions/fold0_forward_preds.h5
    00_predictions/actual_<target>.csv
    00_predictions/selected_fold0_checkpoints.csv
    00_predictions/fold0_forward_prediction_manifest.json
    00_predictions/prediction_artifact_retention.json
    01_close_auction_grid/01_runs/<frozen_run>/
    01_close_auction_grid/02_summary/
    01_close_auction_grid/strict_oos_manifest.json
    01_close_auction_grid/grid_engine_manifest.json
```

与 HDF 重复的 `fold0_forward_preds.csv` 默认删除；`actual_<target>.csv` 保留。

strict manifest 必须满足：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
retained_config_count = 1
```

---

## 6. forward model data 存储

刷新：

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

默认：

```text
FORWARD_ARTIFACT_MODE=model_only
FORWARD_REPORT_MODE=compact
MIN_FREE_GB=5
```

长期保留：

```text
ch12_as1455_forward_latest/model_data_as1455.h5
ch12_as1455_forward_latest/reports/
```

构建验证完成后自动删除 forward 目录中的可重建副本：

```text
as1455_ohlcv_raw.h5
as1455_ohlcv_adj.h5
as1455_execution_metadata.h5
```

主 `ch12_as1455` 的共享缓存、训练 model data 和当前 contract 依赖文件不在此清理范围内。

---

## 7. 绘图

### 7.1 历史 r5 A/B

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

### 7.2 strict r5 forward A/B

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

strict 根目录活动 summary 只有一个冻结配置，因此绘图不会在 forward 区间重新选择 `sell/max/off`。

---

## 8. 存储清理

### 8.1 先检查进程

```bash
pgrep -af 'as1455|build_ashare_ch12|run_as1455'
```

### 8.2 dry-run

```bash
python3 scripts/cleanup_as1455_storage.py \
  --keep-live-dates 3 \
  --include-obsolete \
  --prune-grid-runs \
  --compress-reports
```

默认不会删除，输出候选动作并生成：

```text
saved_data/ashare_ml4t/cleanup_audit_YYYYMMDD_HHMMSS.json
```

### 8.3 审核后执行

```bash
python3 scripts/cleanup_as1455_storage.py \
  --apply \
  --keep-live-dates 3 \
  --include-obsolete \
  --prune-grid-runs \
  --compress-reports
```

清理器会：

- 验证 forward model HDF 后删除重复中间 HDF；
- 只保留最近若干 live 日期；
- 删除旧保留日期中的可重建 history tail；
- 只删除与 HDF 内容重复的 prediction CSV；
- 保留 `actual_<target>.csv`；
- 可选删除明确列入政策的旧 smoke/legacy 目录；
- 可选保留各指标和各信号代表 run，删除其他重复 run；
- 可选 gzip 大型审计 CSV；
- 检测运行中的 AS1455 进程并阻止误删。

---

## 9. 结果核对

### 9.1 日期

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

### 9.2 冻结参数

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')
for path in sorted(base.glob('*/01_close_auction_grid/strict_oos_manifest.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    hist = obj['historical_selection']
    print(path)
    print('historical:', {
        'max_positions': hist['historical_max_positions'],
        'sell_rank': hist['historical_sell_rank'],
        'rebalance_every': hist['historical_rebalance_every'],
        'rebalance_offset': hist['historical_rebalance_offset'],
    })
    print('retained:', obj['retained_config'])
PY
```

历史参数和 `retained_config` 必须完全一致。

### 9.3 磁盘

```bash
df -h .
du -h --max-depth=1 saved_data/ashare_ml4t | sort -h
```
