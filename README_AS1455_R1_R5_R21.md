# AS1455 r1 / r5 / r21 训练、回测与绘图指南

代码结构和开发规则见：

```text
CH17_AS1455_DEVELOPMENT_OUTLINE.md
```

以下命令均在工程根目录执行：

```bash
cd ~/stock_realtime_v021_full
```

---

## 1. 当前公共实现

```text
utils/as1455_paths.py
utils/as1455_ch17_common.py
utils/as1455_cli.py
utils/as1455_signal_specs.py
utils/as1455_rank_cache.py
utils/as1455_backtest_io.py
utils/as1455_grid_runner.py
utils/as1455_plotting.py
```

当前 in-process 引擎：

```text
inprocess_shared_rank_v4
```

它会：

- prediction 每个 signal 加载一次；
- 每个 signal 的每个交易日只实际排序一次；
- execution panel 只构造一次；
- 所有 `max_positions × sell_rank × offset` 共用排序结果；
- 每个参数组合仍调用唯一 v7 交易函数；
- 不存在第二套买卖、费用或 NAV 实现。

---

## 2. 固定实验口径

### 2.1 目标与自然调仓周期

| 简称 | 监督目标 | lookahead | 调仓周期 | offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

### 2.2 特征方案

| 名称 | `feature_preset` | 内容 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + 完整 sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

### 2.3 正式训练产物

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

正式回测使用 search-time checkpoint，不使用诊断性 retrain 的 `models/best_*.keras`。

---

## 3. 拉取代码并检查结构

```bash
cd ~/stock_realtime_v021_full
git pull origin master
bash scripts/check_ch17_as1455_refactor.sh
```

正确结束标志：

```text
[PASS] Ch17 AS1455 refactor validation passed
```

检查内容包括：

- Python 和 shell 语法；
- 关键 CLI 导入；
- r1/r5/r21 目标映射；
- `TOP_N` 与 signal 列；
- 排名缓存与原逐日排序结果一致；
- grid 不含第二套交易函数；
- wrapper 不复制 checkpoint 或交易逻辑；
- 唯一绘图器及线型/marker。

---

## 4. r1 / r5 / r21 训练

### 4.1 通用单 fold

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

### 4.2 批量训练 r1

```bash
TARGET_COL=r01_fwd \
bash scripts/run_as1455_target_search_all.sh
```

只训练 A：

```bash
TARGET_COL=r01_fwd \
FEATURE_PRESETS='rotation_onehot' \
bash scripts/run_as1455_target_search_all.sh
```

r1 默认继续写入原有兼容目录：

```text
saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_foldN_search
saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_foldN_search
```

### 4.3 批量训练 r5

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

等价：

```bash
TARGET_COL=r05_fwd \
bash scripts/run_as1455_target_search_all.sh
```

默认训练：

```text
A/B × fold0..fold6
```

只补跑 B 的 fold0、fold3、fold6：

```bash
TARGET_COL=r05_fwd \
FEATURE_PRESETS='rotation_addon_onehot' \
FOLDS='0 3 6' \
bash scripts/run_as1455_target_search_all.sh
```

### 4.4 批量训练 r21

```bash
bash scripts/run_as1455_r21_target_search_all.sh
```

等价：

```bash
TARGET_COL=r21_fwd \
bash scripts/run_as1455_target_search_all.sh
```

当前默认训练：

```text
A/B × fold0..fold5
```

当前数据下 r21 有效日期不足以生成 fold6，默认已排除。

### 4.5 输入检查与 smoke

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

确认覆盖旧目录时才使用：

```bash
FORCE=1
```

---

## 5. one-fold-lag 历史回测

协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

### 5.1 r1

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

两个 r1 文件只是兼容 wrapper，实际统一调用：

```text
scripts/run_as1455_target_one_lag_backtest.py
```

### 5.2 r5

先做单配置 smoke：

```bash
FEATURE_PRESETS='rotation_onehot' \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r05_natural_backtest.sh
```

正确输出：

```text
[PARITY] single v7 trade engine smoke run ...
[PARITY] PASS
[PARITY] check-only completed; grid was not executed
```

正式运行 A/B：

```bash
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_natural_backtest.sh
```

只跑 A：

```bash
FEATURE_PRESETS='rotation_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_natural_backtest.sh
```

默认参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 5 offsets
= 1050 runs / feature preset
```

### 5.3 r21

先做单配置 smoke：

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

当前默认 target folds：

```text
0,1,2,3,4
```

参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 21 offsets
= 4410 runs / feature preset
```

---

## 6. fold0 最优模型用于 fold0 后续日期

协议：

- 使用 fold0 search-time checkpoint、scaler 和 feature manifest；
- 仅预测 `date > fold0.test_end`；
- 不重新训练；
- 每个参数组合从初始资金和空仓开始；
- 不继承 fold0 测试期持仓。

### 6.1 更新最新数据

默认 forward wrapper 会自动执行：

```text
history 缓存更新
→ 重建 forward model_data
→ checkpoint 推理
→ 回测
```

单独更新：

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

### 6.2 r1

```bash
TARGETS='r01_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 6.3 r5

```bash
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 6.4 r21

```bash
TARGETS='r21_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 6.5 单模型与 top5

fold0-forward 默认：

```text
TOP_N=1
```

因此只回测 `model_0`，与“fold0 最优模型”定义一致。

需要 top5 和 ensemble：

```bash
TOP_N=5 \
TARGETS='r05_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 6.6 已刷新后只重跑回测

```bash
REFRESH_DATA=0 \
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
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

完整 run 通常包括：

```text
config.json
summary.json
close_auction_summary.json
close_auction_nav.csv
close_auction_orders.csv
close_auction_trades.csv
close_auction_rejections.csv
close_auction_positions.csv
close_auction_corporate_actions.csv
round_trips.csv
daily_drawdown.csv
monthly_summary.csv
yearly_summary.csv
fee_summary.csv
turnover_summary.csv
```

默认资金和费用：

```text
initial_cash = 200000
commission_rate = 0.000085
min_commission = 5
stamp_tax_rate = 0.0005
transfer_fee_rate = 0.00001
```

---

## 8. 绘图

统一入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

曲线同时使用颜色、线型和 marker，并输出实际选中的 signal 和参数。

### 8.1 r1 A/B

```bash
R1_A=$(find saved_data/ashare_ml4t -maxdepth 1 -type d \
  -name 'ch17_as1455_rotation_one_lag_daily_backtest_*' | sort | tail -n 1)
R1_B=$(find saved_data/ashare_ml4t -maxdepth 1 -type d \
  -name 'ch17_as1455_rotation_addon_one_lag_daily_backtest_*' | sort | tail -n 1)

BACKTEST_ROOTS="$R1_A,$R1_B" \
LABELS='r1-A,r1-B' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r1_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 8.2 r5 A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_target_backtest'
R5_A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
R5_B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r05_fwd_reb5_*' | sort | tail -n 1)

BACKTEST_ROOTS="$R5_A,$R5_B" \
LABELS='r5-A,r5-B' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r5_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 8.3 r21 A/B

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_target_backtest'
R21_A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r21_fwd_reb21_*' | sort | tail -n 1)
R21_B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r21_fwd_reb21_*' | sort | tail -n 1)

BACKTEST_ROOTS="$R21_A,$R21_B" \
LABELS='r21-A,r21-B' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r21_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 8.4 fold0-forward A/B

以 r5 为例：

```bash
BASE='saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest'
A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r05_fwd_reb5_*' | sort | tail -n 1)

BACKTEST_ROOTS="$A,$B" \
LABELS='r5-A-fold0-forward,r5-B-fold0-forward' \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r5_fold0_forward_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

切换最优指标：

```bash
RANK_METRIC=total_return bash scripts/plot_as1455_default_ab_nav_curves.sh
RANK_METRIC=calmar bash scripts/plot_as1455_default_ab_nav_curves.sh
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

---

## 9. 重构前后单 run 对比

```bash
python3 scripts/compare_as1455_backtest_runs.py \
  --left-run <重构前单个run目录> \
  --right-run <重构后单个run目录>
```

默认比较：

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

正确结束标志：

```text
[PASS] AS1455 run outputs are equivalent
```

---

## 10. 当前已知限制

1. r21 当前数据下没有可用 source fold6，所以 one-fold-lag 默认只覆盖 target fold0..4。
2. forward HDF 最后 1、5、21 个交易日分别没有完整 r1、r5、r21 前向标签。
3. live 旧基线 checkpoint bundle 与 target-aware A/B `.keras + scaler.pkl + feature_manifest.json` 不是同一产物契约，不能直接混用。
