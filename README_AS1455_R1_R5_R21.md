# AS1455 r1 / r5 / r21 训练、回测与绘图指南

代码结构和开发约束见：

```text
CH17_AS1455_DEVELOPMENT_OUTLINE.md
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

正式回测使用 search-time checkpoint，不使用诊断性 retrain 的 `models/best_*.keras`。

---

## 2. 拉取代码和结构检查

```bash
cd ~/stock_realtime_v021_full
git pull origin master
bash scripts/check_ch17_as1455_refactor.sh
```

正确结束标志：

```text
[PASS] Ch17 AS1455 refactor validation passed
```

---

## 3. r1 / r5 / r21 训练

### 3.1 单 fold

A、r1、fold0：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_onehot \
  --target-col r01_fwd \
  --fold-index 0 \
  --epochs 20 \
  --best-n 5
```

B、r5、fold3：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_addon_onehot \
  --target-col r05_fwd \
  --fold-index 3 \
  --epochs 20 \
  --best-n 5
```

A、r21、fold0：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_onehot \
  --target-col r21_fwd \
  --fold-index 0 \
  --epochs 20 \
  --best-n 5
```

### 3.2 批量训练 r1

```bash
TARGET_COL=r01_fwd \
bash scripts/run_as1455_target_search_all.sh
```

### 3.3 批量训练 r5

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

等价：

```bash
TARGET_COL=r05_fwd \
bash scripts/run_as1455_target_search_all.sh
```

默认训练 A/B 的 fold0..fold6。

### 3.4 批量训练 r21

```bash
bash scripts/run_as1455_r21_target_search_all.sh
```

等价：

```bash
TARGET_COL=r21_fwd \
bash scripts/run_as1455_target_search_all.sh
```

当前数据默认训练 A/B 的 fold0..fold5；r21 有效日期不足以生成 fold6。

### 3.5 指定特征和 fold

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
FOLDS='0 3 6' \
bash scripts/run_as1455_target_search_all.sh
```

输入检查：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_onehot' \
FOLDS='0' \
INPUT_CHECK_ONLY=1 \
bash scripts/run_as1455_target_search_all.sh
```

smoke：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_onehot' \
FOLDS='0' \
SMOKE=1 \
bash scripts/run_as1455_target_search_all.sh
```

---

## 4. one-fold-lag 历史回测

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

### 4.1 r1

A：

```bash
python3 scripts/run_as1455_rotation_one_lag_daily_backtest.py \
  --output-mode full \
  --force-grid
```

B：

```bash
python3 scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py \
  --output-mode full \
  --force-grid
```

### 4.2 r5

单配置 smoke：

```bash
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r05_natural_backtest.sh
```

正式运行：

```bash
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_natural_backtest.sh
```

默认参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 5 offsets
= 1050 runs / feature preset
```

### 4.3 r21

单配置 smoke：

```bash
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r21_natural_backtest.sh
```

正式运行：

```bash
OUTPUT_MODE=full \
bash scripts/run_as1455_r21_natural_backtest.sh
```

当前默认 target folds 为 `0,1,2,3,4`。

默认参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 21 offsets
= 4410 runs / feature preset
```

历史结果默认写入：

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
  rotation_onehot_r05_fwd_reb5_YYYYMMDD/
  rotation_addon_onehot_r05_fwd_reb5_YYYYMMDD/
  rotation_onehot_r21_fwd_reb21_YYYYMMDD/
  rotation_addon_onehot_r21_fwd_reb21_YYYYMMDD/
```

---

## 5. fold0 模型用于 fold0 后续日期

### 5.1 默认模型选择规则

fold0-forward 不再在后续日期重新比较全部模型信号。默认流程是：

```text
从 saved_data/ashare_ml4t/ch17_as1455_target_backtest
找到与 feature_preset、target_col、rebalance_every 对应的最新完整目录
→ 读取 grid_summary_compact.csv（找不到时读取 grid_summary.csv）
→ 只保留 status=ok
→ 默认按 Sharpe 选出历史最佳完整 run
→ 提取该 run 的 signal_name、signal_cols、signal_mode
→ 用相同 checkpoint 排名或 ensemble 定义组合 fold0 checkpoint
→ 在 date > fold0.test_end 的区间从空仓开始回测
```

例如历史最佳 signal 是：

```text
model_2:2:single
```

forward 只加载 fold0 排名前 3 个 checkpoint，并只回测 `model_2`。

若历史最佳 signal 是：

```text
ensemble_first3_mean:0,1,2:mean
```

forward 只加载 fold0 前 3 个 checkpoint，并只回测三模型均值。

若历史最佳 signal 是：

```text
ensemble_all5_mean:0,1,2,3,4:mean
```

forward 加载 fold0 前 5 个 checkpoint，并只回测五模型均值。

历史回测选择的是**模型信号**。历史窗口的：

```text
max_positions
sell_rank
rebalance_offset
```

会写入 forward manifest 作为来源记录，但默认不直接套用；forward 窗口仍重新遍历交易参数。这避免把历史窗口的交易参数未经检验直接迁移到后续日期。

模型选择和绘图共用：

```text
utils/as1455_model_selection.py
```

两者默认指标都是：

```text
sharpe
```

### 5.2 更新最新数据

默认 wrapper 会自动执行：

```text
history 缓存更新
→ 重建 forward model_data
→ 历史最佳模型信号选择
→ fold0 checkpoint 推理
→ forward 回测
```

单独更新：

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

### 5.3 r5 fold0-forward

```bash
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

运行时会打印类似：

```text
[MODEL SELECT] root=... metric=sharpe value=... historical_run=... signal=ensemble_first3_mean:0,1,2:mean required_top_n=3
```

### 5.4 r21 fold0-forward

```bash
TARGETS='r21_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.5 r1 fold0-forward

必须先在 `ch17_as1455_target_backtest` 下存在对应的 r1 历史回测目录：

```text
rotation_onehot_r01_fwd_reb1_*
rotation_addon_onehot_r01_fwd_reb1_*
```

然后运行：

```bash
TARGETS='r01_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

找不到对应历史目录时会明确失败，不会静默改用其他模型。

### 5.6 不重复刷新数据

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.7 指定历史回测目录

只运行一个 target 和一个 feature preset 时，可以固定来源目录：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
SELECTION_BACKTEST_ROOT='saved_data/ashare_ml4t/ch17_as1455_target_backtest/rotation_onehot_r05_fwd_reb5_20260712' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 5.8 改变历史选择指标

```bash
SELECTION_RANK_METRIC='calmar' \
TARGETS='r05_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

绘图时也应使用相同指标：

```bash
RANK_METRIC='calmar' \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 5.9 保留旧的全信号网格模式

需要让 forward 重新回测 top-N 的全部单模型和 ensemble 时，显式设置：

```bash
MODEL_SELECTION_MODE='all_top_n' \
TOP_N=5 \
TARGETS='r05_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

这不是当前默认协议。

---

## 6. fold0-forward 输出与审计

输出根目录：

```text
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
```

预测目录：

```text
00_predictions/fold0_forward_preds.h5
00_predictions/fold0_forward_preds.csv
00_predictions/selected_fold0_checkpoints.csv
00_predictions/fold0_forward_prediction_manifest.json
```

manifest 中记录：

```text
model_selection_mode
historical_model_selection.backtest_root
historical_model_selection.summary_file
historical_model_selection.rank_metric
historical_model_selection.run_name
historical_model_selection.signal_name
historical_model_selection.signal_cols
historical_model_selection.signal_mode
historical_model_selection.historical_max_positions
historical_model_selection.historical_sell_rank
historical_model_selection.historical_rebalance_offset
historical_trading_parameters_reused = false
```

---

## 7. 回测输出模式

`OUTPUT_MODE` 只控制文件保留范围，不改变交易逻辑。

| 模式 | 文件 |
|---|---|
| `summary` | JSON 汇总 |
| `compact` | JSON + NAV、回撤、月度、年度、费用和换手 |
| `full` | compact + 订单、成交、拒单、每日持仓、公司行为和 round trip |

正式审计建议：

```bash
OUTPUT_MODE=full
```

---

## 8. 绘图

统一入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

绘图和 fold0 历史模型选择共用同一个 best-run 选择函数。默认按 Sharpe 选取每个传入根目录的最佳完整 run。

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

### 8.2 r5 fold0-forward A/B

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

输出：

```text
return_curve_daily.png
return_curve_weekly.png
return_curve_monthly.png
return_curve_daily.csv
return_curve_weekly.csv
return_curve_monthly.csv
selected_best_grids.csv
selected_best_grids.json
```

曲线同时使用颜色、线型和 marker。

---

## 9. 最低验证

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

真实历史目录选择检查：

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

必须先打印 `[MODEL SELECT]`，然后完成单配置 v7 smoke。
