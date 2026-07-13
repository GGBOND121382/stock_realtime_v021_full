# AS1455 r1 / r5 / r21 运行指南

详细开发约束和存储政策见：

```text
CH17_AS1455_DEVELOPMENT_OUTLINE.md
AS1455_STORAGE_AND_STRICT_OOS.md
AS1455_STORAGE_MAINTENANCE.md
```

以下命令均在工程根目录执行：

```bash
cd ~/stock_realtime_v021_full
```

## 1. 固定实验口径

| 简称 | 标签 | lookahead | 自然调仓周期 | 历史 offset 搜索范围 |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

注意：v7 中 `rebalance_offset` 是相对于当前回测窗口第一个预测—执行重叠交易日的本地序号。历史窗口和 fold0-forward 窗口都会重新从 `day_index=0` 开始，因此 strict OOS 冻结的是历史调仓相位，而不是直接复制 offset 数字。

特征方案：

| 名称 | `feature_preset` | 内容 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

正式训练使用：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

## 2. 拉取代码与验证

```bash
git switch master
git pull --ff-only origin master
bash scripts/check_ch17_as1455_refactor.sh
```

正确结束标志：

```text
[PASS] Ch17 AS1455 refactor validation passed
```

验证集覆盖：

- Python/Shell 语法和 CLI；
- 目标—周期映射；
- failed-only summary 拒绝；
- 历史窗口 `date_min/date_max/n_days` 提取；
- forward 最新无标签日期保留；
- strict OOS 参数和调仓相位冻结；
- exact-offset 单配置生成；
- prediction CSV 清理时保留真实标签文件；
- 唯一 v7 交易引擎；
- summary-first、model-only 和磁盘门禁默认值。

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

## 4. one-fold-lag 历史回测

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

### 4.1 存储安全默认值

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

不要对完整网格默认使用 `full`。

确需保留全部明细：

```bash
OUTPUT_MODE=compact MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

或：

```bash
OUTPUT_MODE=full MATERIALIZE_BEST=0 \
  bash scripts/run_as1455_r05_natural_backtest.sh
```

### 4.2 历史结果结构

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

`test_preds.h5` 是预测权威文件；同内容的 `test_preds.csv` 默认删除。`actual_<target>.csv` 是真实标签，必须保留。

历史最佳 summary 或 materialized NAV 必须能够提供：

```text
date_min
date_max
n_days
```

这些字段用于 forward 调仓相位对齐。

## 5. fold0-forward 严格样本外

### 5.1 日期口径

训练和历史回测要求当前目标标签非空。fold0-forward 只要求模型特征非空，不要求未来目标已实现：

```text
feature_row_mode = inference_features_only
```

因此 forward model data 更新到最新交易日时，r1/r5/r21 prediction end 都应到最新特征有效日期，而不是因目标尚未实现而提前截止。

manifest 记录并硬校验：

```text
feature_meta.model_data_max_date
feature_meta.feature_valid_max_date
feature_meta.target_valid_max_date
feature_meta.unlabeled_prediction_rows
feature_meta.unlabeled_prediction_dates
expected_prediction_end
prediction_end

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
historical rebalance phase
```

调仓相位换算：

```text
forward_global_index
  = historical_n_days + bridge_execution_days

effective_forward_offset
  = (historical_offset - forward_global_index) mod rebalance_every
```

其中 `bridge_execution_days` 来自共享 grid 构造的完整 raw daily execution calendar，而不是自然日或固定 fold 长度。

forward 使用 fold0 checkpoint，在 `date > fold0.test_end` 区间从空仓和初始现金开始回测。forward 数据只用于评价，不再用于调参。

#### r1/r5/r21 A/B 一次性正式运行

```bash
REFRESH_DATA=0 \
TARGETS='r01_fwd r05_fwd r21_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

该命令会生成 6 组 strict OOS 结果：

```text
r1-A   rotation_onehot + r01_fwd
r1-B   rotation_addon_onehot + r01_fwd
r5-A   rotation_onehot + r05_fwd
r5-B   rotation_addon_onehot + r05_fwd
r21-A  rotation_onehot + r21_fwd
r21-B  rotation_addon_onehot + r21_fwd
```

建议在内存和运行时间受限时按单组顺序运行，不要并行执行。

#### 仅运行 r1 A/B

```bash
REFRESH_DATA=0 \
TARGETS='r01_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

单独补跑 r1-A：

```bash
REFRESH_DATA=0 \
TARGETS='r01_fwd' \
FEATURE_PRESETS='rotation_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

单独补跑 r1-B：

```bash
REFRESH_DATA=0 \
TARGETS='r01_fwd' \
FEATURE_PRESETS='rotation_addon_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

#### 仅运行 r5 A/B

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

#### 仅运行 r21 A/B

```bash
REFRESH_DATA=0 \
TARGETS='r21_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

正式默认值：

```text
MODEL_SELECTION_MODE=strict_oos
OUTPUT_MODE=compact
SELECTION_RANK_METRIC=sharpe
```

运行日志必须出现：

```text
[PHASE ALIGN] historical_offset=... history_days=... bridge_days=... forward_global_index=... effective_forward_offset=...
```

### 5.3 自动更新数据后运行

先刷新最新 forward model data，再运行 r1/r5/r21 A/B：

```bash
REFRESH_DATA=1 \
TARGETS='r01_fwd r05_fwd r21_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

服务器空间紧张时，应先清理，再执行 `REFRESH_DATA=1`。

### 5.4 固定历史来源

`SELECTION_BACKTEST_ROOT` 只能与单个目标和单个特征方案一起使用：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
SELECTION_BACKTEST_ROOT='saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_onehot_r05_fwd_reb5_YYYYMMDD_HHMMSS' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

如果所选旧历史结果既没有 `date_min/date_max/n_days`，也没有 materialized `close_auction_nav.csv`，strict OOS 会拒绝运行。先执行：

```bash
python3 scripts/materialize_as1455_best_run.py \
  --backtest-root '<历史回测根目录>' \
  --raw-daily-cache-dir saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache \
  --rank-metric sharpe \
  --output-mode compact \
  --force
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
    01_close_auction_grid/01_runs/<phase-aligned-run>/
    01_close_auction_grid/02_summary/
    01_close_auction_grid/strict_oos_manifest.json
    01_close_auction_grid/grid_engine_manifest.json
```

strict manifest 必须满足：

```text
evaluation_mode = strict_oos
historical_trading_parameters_reused = true
historical_rebalance_phase_reused = true
generated_config_count = 1
retained_config_count = 1
```

以下两个值可能不同：

```text
historical_config.rebalance_offset
retained_config.rebalance_offset
```

是否对齐应检查 `rebalance_phase_alignment`，不能要求两个本地 offset 整数相同。

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

## 7. 绘图

绘图脚本：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

它支持用逗号分隔的 `BACKTEST_ROOTS` 和 `LABELS`，并输出 PNG、CSV 和 JSON。

选择 forward 根目录时，必须要求存在：

```text
01_close_auction_grid/strict_oos_manifest.json
```

这样可以避免误选 parity-only、dry-run 或未完成目录。

### 7.1 strict r1 forward A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest'

A=$(
  find "$BASE" -maxdepth 1 -type d \
    -name 'rotation_onehot_r01_fwd_reb1_*' \
    -exec test -f '{}/01_close_auction_grid/strict_oos_manifest.json' \; \
    -print | sort | tail -n 1
)

B=$(
  find "$BASE" -maxdepth 1 -type d \
    -name 'rotation_addon_onehot_r01_fwd_reb1_*' \
    -exec test -f '{}/01_close_auction_grid/strict_oos_manifest.json' \; \
    -print | sort | tail -n 1
)

printf 'r1-A=%s\nr1-B=%s\n' "$A" "$B"

test -n "$A" && test -n "$B" || {
  echo '[ERROR] 未找到完整的 r1 strict OOS A/B 结果' >&2
  exit 1
}

OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r1_fold0_forward_$(date +%Y%m%d_%H%M%S)"

BACKTEST_ROOTS="$A,$B" \
LABELS='r1-A-fold0-forward,r1-B-fold0-forward' \
RANK_METRIC='sharpe' \
OUT_DIR="$OUT_DIR" \
bash scripts/plot_as1455_default_ab_nav_curves.sh

echo "绘图输出目录：$OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
```

### 7.2 strict r5 forward A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest'

A=$(
  find "$BASE" -maxdepth 1 -type d \
    -name 'rotation_onehot_r05_fwd_reb5_*' \
    -exec test -f '{}/01_close_auction_grid/strict_oos_manifest.json' \; \
    -print | sort | tail -n 1
)

B=$(
  find "$BASE" -maxdepth 1 -type d \
    -name 'rotation_addon_onehot_r05_fwd_reb5_*' \
    -exec test -f '{}/01_close_auction_grid/strict_oos_manifest.json' \; \
    -print | sort | tail -n 1
)

printf 'r5-A=%s\nr5-B=%s\n' "$A" "$B"

test -n "$A" && test -n "$B" || {
  echo '[ERROR] 未找到完整的 r5 strict OOS A/B 结果' >&2
  exit 1
}

BACKTEST_ROOTS="$A,$B" \
LABELS='r5-A-fold0-forward,r5-B-fold0-forward' \
RANK_METRIC='sharpe' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r5_fold0_forward_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 7.3 strict r21 forward A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest'

A=$(
  find "$BASE" -maxdepth 1 -type d \
    -name 'rotation_onehot_r21_fwd_reb21_*' \
    -exec test -f '{}/01_close_auction_grid/strict_oos_manifest.json' \; \
    -print | sort | tail -n 1
)

B=$(
  find "$BASE" -maxdepth 1 -type d \
    -name 'rotation_addon_onehot_r21_fwd_reb21_*' \
    -exec test -f '{}/01_close_auction_grid/strict_oos_manifest.json' \; \
    -print | sort | tail -n 1
)

printf 'r21-A=%s\nr21-B=%s\n' "$A" "$B"

test -n "$A" && test -n "$B" || {
  echo '[ERROR] 未找到完整的 r21 strict OOS A/B 结果' >&2
  exit 1
}

BACKTEST_ROOTS="$A,$B" \
LABELS='r21-A-fold0-forward,r21-B-fold0-forward' \
RANK_METRIC='sharpe' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r21_fold0_forward_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

strict 根目录活动 summary 只有一个相位对齐配置，因此绘图不会在 forward 区间重新选择 `sell/max/off`。

## 8. 存储维护

默认保守 dry-run：

```bash
bash scripts/run_as1455_storage_maintenance.sh
```

默认保守正式清理：

```bash
APPLY=1 bash scripts/run_as1455_storage_maintenance.sh
```

默认参数：

```text
INCLUDE_OBSOLETE=0
PRUNE_GRID_RUNS=0
COMPRESS_REPORTS=1
```

因此默认不会删除显式 obsolete 目录，也不会裁剪历史大网格明细。

显式清理 obsolete 目录时，必须先独立 dry-run：

```bash
INCLUDE_OBSOLETE=1 \
PRUNE_GRID_RUNS=0 \
COMPRESS_REPORTS=0 \
SKIP_FORWARD_ARTIFACTS=1 \
SKIP_LIVE=1 \
SKIP_PREDICTION_CSV=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

审核 `share_me.txt` 和 `cleanup_dry_run.json` 后，再执行：

```bash
APPLY=1 \
INCLUDE_OBSOLETE=1 \
PRUNE_GRID_RUNS=0 \
COMPRESS_REPORTS=0 \
SKIP_FORWARD_ARTIFACTS=1 \
SKIP_LIVE=1 \
SKIP_PREDICTION_CSV=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

`baostock_5m_cache` 和历史大网格明细不会在保守清理中删除。是否裁剪必须单独评估。

## 9. 结果核对

### 9.1 r1/r5/r21 forward 日期

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

base = Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')
patterns = (
    'rotation_*_r01_fwd_reb1_*/00_predictions/fold0_forward_preds.h5',
    'rotation_*_r05_fwd_reb5_*/00_predictions/fold0_forward_preds.h5',
    'rotation_*_r21_fwd_reb21_*/00_predictions/fold0_forward_preds.h5',
)

for pattern in patterns:
    for path in sorted(base.glob(pattern)):
        manifest = path.parents[1] / '01_close_auction_grid' / 'strict_oos_manifest.json'
        if not manifest.is_file():
            continue
        df = pd.read_hdf(path, 'predictions')
        dates = pd.DatetimeIndex(df.index.get_level_values('date'))
        print(path, dates.min().date(), dates.max().date(), dates.nunique())
PY
```

### 9.2 参数与相位

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path('saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest')
for path in sorted(base.glob('*/01_close_auction_grid/strict_oos_manifest.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    phase = obj['rebalance_phase_alignment']
    print('\n', path)
    print('historical_config:', obj['historical_config'])
    print('phase:', {
        'historical_n_days': phase['historical_n_days'],
        'bridge_execution_days': phase['bridge_execution_days'],
        'forward_global_index': phase['forward_global_index'],
        'historical_offset': phase['historical_offset'],
        'effective_forward_offset': phase['effective_forward_offset'],
    })
    print('retained_config:', obj['retained_config'])

    assert obj['evaluation_mode'] == 'strict_oos'
    assert obj['historical_trading_parameters_reused'] is True
    assert obj['historical_rebalance_phase_reused'] is True
    assert obj['generated_config_count'] == 1
    assert obj['retained_config_count'] == 1
    assert obj['retained_config']['rebalance_offset'] == \
        phase['effective_forward_offset']
PY
```

### 9.3 磁盘

```bash
df -h .
du -h --max-depth=1 saved_data/ashare_ml4t | sort -h
```
