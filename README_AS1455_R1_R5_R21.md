# AS1455 r1 / r5 / r21 训练、回测与绘图指南

本文档说明 AS1455 三个预测周期的标准操作：

| 简称 | 监督目标 | 自然调仓周期 | 完整 offset |
|---|---|---:|---|
| `r1` | `r01_fwd` | 1 个交易日 | `0` |
| `r5` | `r05_fwd` | 5 个交易日 | `0,1,2,3,4` |
| `r21` | `r21_fwd` | 21 个交易日 | `0..20` |

以下命令均在工程根目录执行：

```bash
cd ~/stock_realtime_v021_full
```

## 1. 数据与特征口径

默认模型数据：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
```

两组特征方案：

| 名称 | `feature_preset` | 含义 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + 完整 sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

训练与正式回测必须使用搜索阶段保存的以下产物：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

正式 one-fold-lag 回测不要把 `models/best_*.keras` 或诊断性 retrain 产物作为搜索最优模型依据。

## 2. r1 训练

### 2.1 A：rotation + one-hot

完整训练 7 个 fold：

```bash
mkdir -p logs/as1455_folds

for FOLD in 0 1 2 3 4 5 6; do
  python3 scripts/run_as1455_sector_rotation_fold0_param_search.py \
    --model-data saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 \
    --fold-index "$FOLD" \
    --sector-encoding onehot \
    --dropna-mode r01_only \
    --epochs 20 \
    --best-n 5 \
    --out-dir "saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold${FOLD}_search" \
    2>&1 | tee "logs/as1455_folds/sector_rotation_onehot_fold${FOLD}.log"
done
```

输出目录：

```text
saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold0_search
...
saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold6_search
```

### 2.2 B：rotation + compact add-on + one-hot

```bash
mkdir -p logs/as1455_folds

for FOLD in 0 1 2 3 4 5 6; do
  python3 scripts/run_as1455_first_batch_features_fold0_param_search.py \
    --model-data saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 \
    --fold-index "$FOLD" \
    --sector-encoding onehot \
    --dropna-mode r01_only \
    --epochs 20 \
    --best-n 5 \
    --out-dir "saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold${FOLD}_search" \
    2>&1 | tee "logs/as1455_folds/full_rotation_plus_addon_fold${FOLD}.log"
done
```

输出目录：

```text
saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold0_search
...
saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold6_search
```

### 2.3 只检查输入或做 smoke test

A 组 fold0 输入检查：

```bash
python3 scripts/run_as1455_sector_rotation_fold0_param_search.py \
  --fold-index 0 \
  --sector-encoding onehot \
  --dropna-mode r01_only \
  --out-dir saved_data/ashare_ml4t/ch17_as1455_sector_rotation_onehot_fold0_input_check \
  --input-check-only
```

B 组 fold0 smoke test：

```bash
python3 scripts/run_as1455_first_batch_features_fold0_param_search.py \
  --fold-index 0 \
  --sector-encoding onehot \
  --dropna-mode r01_only \
  --epochs 2 \
  --best-n 1 \
  --out-dir saved_data/ashare_ml4t/ch17_as1455_full_rotation_plus_first_batch_compact_fold0_smoke \
  --smoke
```

已有非空输出目录默认拒绝覆盖。确认需要重跑时再加 `--force`。

## 3. r5 训练

两个特征方案、7 个 fold 全部运行：

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

默认参数：

```text
TARGET_COL=r05_fwd
FEATURE_PRESETS="rotation_onehot rotation_addon_onehot"
FOLDS="0 1 2 3 4 5 6"
EPOCHS=20
BEST_N=5
SEED=42
```

只训练 A：

```bash
FEATURE_PRESETS="rotation_onehot" \
bash scripts/run_as1455_r05_target_search_all.sh
```

只训练 B：

```bash
FEATURE_PRESETS="rotation_addon_onehot" \
bash scripts/run_as1455_r05_target_search_all.sh
```

只补跑指定 fold：

```bash
FEATURE_PRESETS="rotation_addon_onehot" \
FOLDS="0 3 6" \
bash scripts/run_as1455_r05_target_search_all.sh
```

输入检查：

```bash
FEATURE_PRESETS="rotation_onehot" \
FOLDS="0" \
INPUT_CHECK_ONLY=1 \
bash scripts/run_as1455_r05_target_search_all.sh
```

smoke test：

```bash
FEATURE_PRESETS="rotation_onehot" \
FOLDS="0" \
SMOKE=1 \
bash scripts/run_as1455_r05_target_search_all.sh
```

重跑已有 fold：

```bash
FEATURE_PRESETS="rotation_addon_onehot" \
FOLDS="0" \
FORCE=1 \
bash scripts/run_as1455_r05_target_search_all.sh
```

输出结构：

```text
saved_data/ashare_ml4t/ch17_as1455_target_search/
  rotation_onehot/r05_fwd/fold0_search ... fold6_search
  rotation_addon_onehot/r05_fwd/fold0_search ... fold6_search
```

## 4. r21 训练

当前 AS1455 数据下，`r21_fwd` 的有效日期不足以构造 source fold6。当前推荐训练 fold0..5：

```bash
FOLDS="0 1 2 3 4 5" \
bash scripts/run_as1455_r21_target_search_all.sh
```

只训练 A：

```bash
FEATURE_PRESETS="rotation_onehot" \
FOLDS="0 1 2 3 4 5" \
bash scripts/run_as1455_r21_target_search_all.sh
```

只训练 B：

```bash
FEATURE_PRESETS="rotation_addon_onehot" \
FOLDS="0 1 2 3 4 5" \
bash scripts/run_as1455_r21_target_search_all.sh
```

只补跑一个 fold：

```bash
FEATURE_PRESETS="rotation_addon_onehot" \
FOLDS="0" \
bash scripts/run_as1455_r21_target_search_all.sh
```

输入检查、smoke 和强制重跑与 r5 相同：

```bash
FEATURE_PRESETS="rotation_onehot" FOLDS="0" INPUT_CHECK_ONLY=1 \
  bash scripts/run_as1455_r21_target_search_all.sh

FEATURE_PRESETS="rotation_onehot" FOLDS="0" SMOKE=1 \
  bash scripts/run_as1455_r21_target_search_all.sh

FEATURE_PRESETS="rotation_onehot" FOLDS="0" FORCE=1 \
  bash scripts/run_as1455_r21_target_search_all.sh
```

输出结构：

```text
saved_data/ashare_ml4t/ch17_as1455_target_search/
  rotation_onehot/r21_fwd/fold0_search ... fold5_search
  rotation_addon_onehot/r21_fwd/fold0_search ... fold5_search
```

## 5. 训练产物检查

检查一个 fold 是否具备正式 one-fold-lag 回测需要的产物：

```bash
DIR="saved_data/ashare_ml4t/ch17_as1455_target_search/rotation_onehot/r05_fwd/fold1_search"

for f in \
  "$DIR/search_best_checkpoints.csv" \
  "$DIR/preprocess/scaler.pkl" \
  "$DIR/preprocess/feature_manifest.json" \
  "$DIR/fold_report.json"; do
  [[ -s "$f" ]] || { echo "[MISSING] $f"; exit 1; }
done

find "$DIR/search_checkpoints" -maxdepth 1 -type f -name '*.keras' -print
```

## 6. r1 回测

r1 使用 one-fold-lag：source fold6 预测 target fold5，依次到 source fold1 预测 target fold0。回测不重新训练模型。

### 6.1 A 组

正式回测并保留完整审计文件：

```bash
python3 scripts/run_as1455_rotation_one_lag_daily_backtest.py \
  --grid-script code/backtest/run_as1455_close_auction_grid_inprocess.py \
  --output-mode full \
  --force-grid
```

默认输出：

```text
saved_data/ashare_ml4t/ch17_as1455_rotation_one_lag_daily_backtest_YYYYMMDD/
```

### 6.2 B 组

```bash
python3 scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py \
  --grid-script code/backtest/run_as1455_close_auction_grid_inprocess.py \
  --output-mode full \
  --force-grid
```

默认输出：

```text
saved_data/ashare_ml4t/ch17_as1455_rotation_addon_one_lag_daily_backtest_YYYYMMDD/
```

### 6.3 先生成预测、不跑 grid

A：

```bash
python3 scripts/run_as1455_rotation_one_lag_daily_backtest.py --skip-grid
```

B：

```bash
python3 scripts/run_as1455_rotation_addon_one_lag_daily_backtest.py --skip-grid
```

预测文件位于对应输出根目录的：

```text
00_predictions/test_preds.h5
00_predictions/test_preds.csv
00_predictions/selected_checkpoints.csv
00_predictions/one_lag_prediction_manifest.json
```

## 7. r5 回测

优化后的回测引擎会：

- 每个特征组合只构造一次 execution panel；
- 每个 signal 每个交易日只排序一次；
- 所有 `max_positions × sell_rank × offset` 复用相同排名；
- 正式运行前默认做一组新旧引擎一致性检查。

先只做一致性检查，不执行完整 grid：

```bash
FEATURE_PRESETS="rotation_onehot" \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r05_natural_backtest.sh
```

必须看到：

```text
[PARITY] PASS
[PARITY] check-only completed; grid was not executed
```

正式回测两个特征组合，并保留完整审计文件：

```bash
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_natural_backtest.sh
```

只跑 A：

```bash
FEATURE_PRESETS="rotation_onehot" \
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_natural_backtest.sh
```

只跑 B：

```bash
FEATURE_PRESETS="rotation_addon_onehot" \
OUTPUT_MODE=full \
bash scripts/run_as1455_r05_natural_backtest.sh
```

默认参数空间：

```text
7 signals
× 5 max_positions
× 6 sell_rank
× 5 offsets
= 1050 runs / feature preset
```

输出目录：

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
  rotation_onehot_r05_fwd_reb5_YYYYMMDD/
  rotation_addon_onehot_r05_fwd_reb5_YYYYMMDD/
```

## 8. r21 回测

当前数据没有可用的 source fold6，所以默认只回测 target folds 0..4：

```text
source fold5 -> target fold4
source fold4 -> target fold3
source fold3 -> target fold2
source fold2 -> target fold1
source fold1 -> target fold0
```

先做一致性检查：

```bash
FEATURE_PRESETS="rotation_onehot" \
PARITY_CHECK_ONLY=1 \
bash scripts/run_as1455_r21_natural_backtest.sh
```

正式回测两个特征组合，并保留完整审计文件：

```bash
OUTPUT_MODE=full \
bash scripts/run_as1455_r21_natural_backtest.sh
```

只跑一个特征组合：

```bash
FEATURE_PRESETS="rotation_onehot" OUTPUT_MODE=full \
  bash scripts/run_as1455_r21_natural_backtest.sh

FEATURE_PRESETS="rotation_addon_onehot" OUTPUT_MODE=full \
  bash scripts/run_as1455_r21_natural_backtest.sh
```

默认参数空间：

```text
7 signals
× 5 max_positions
× 6 sell_rank
× 21 offsets
= 4410 runs / feature preset
```

输出目录：

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
  rotation_onehot_r21_fwd_reb21_YYYYMMDD/
  rotation_addon_onehot_r21_fwd_reb21_YYYYMMDD/
```

未来数据能够生成 source fold6 后，可显式加入 target fold5：

```bash
TARGET_FOLDS="0,1,2,3,4,5" OUTPUT_MODE=full \
  bash scripts/run_as1455_r21_natural_backtest.sh
```

## 9. 回测输出模式

`OUTPUT_MODE` 只控制文件保留范围，不改变交易、费用、持仓或收益逻辑。

| 模式 | 输出内容 |
|---|---|
| `summary` | 只保留 JSON 汇总 |
| `compact` | JSON + NAV、回撤、月度、年度、费用、换手等核心 CSV |
| `full` | 在 compact 基础上增加订单、成交、拒单、每日持仓、公司行为和 round trip 明细 |

正式审计建议显式使用：

```bash
OUTPUT_MODE=full
```

单个 run 的完整输出通常包括：

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

默认交易资金和费用口径由回测引擎配置：

```text
initial_cash = 200000
commission_rate = 0.000085
min_commission = 5
stamp_tax_rate = 0.0005
transfer_fee_rate = 0.00001
```

## 10. 绘图

绘图入口：

```text
scripts/plot_as1455_default_ab_nav_curves.sh
```

Python 主程序会从每个回测根目录的 grid summary 中，按 `RANK_METRIC` 选择最优 run，再读取该 run 的 `close_auction_nav.csv`。默认按 Sharpe 选择，并生成 daily、weekly、monthly 三套累计收益曲线。

### 10.1 r1 A/B

```bash
R1_A=$(find saved_data/ashare_ml4t -maxdepth 1 -type d \
  -name 'ch17_as1455_rotation_one_lag_daily_backtest_*' | sort | tail -n 1)
R1_B=$(find saved_data/ashare_ml4t -maxdepth 1 -type d \
  -name 'ch17_as1455_rotation_addon_one_lag_daily_backtest_*' | sort | tail -n 1)

BACKTEST_ROOTS="$R1_A,$R1_B" \
LABELS="r1-A-rotation-onehot,r1-B-rotation-addon-onehot" \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r1_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 10.2 r5 A/B

```bash
BASE="saved_data/ashare_ml4t/ch17_as1455_target_backtest"
R5_A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
R5_B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r05_fwd_reb5_*' | sort | tail -n 1)

BACKTEST_ROOTS="$R5_A,$R5_B" \
LABELS="r5-A-rotation-onehot,r5-B-rotation-addon-onehot" \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r5_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 10.3 r21 A/B

```bash
BASE="saved_data/ashare_ml4t/ch17_as1455_target_backtest"
R21_A=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_onehot_r21_fwd_reb21_*' | sort | tail -n 1)
R21_B=$(find "$BASE" -maxdepth 1 -type d \
  -name 'rotation_addon_onehot_r21_fwd_reb21_*' | sort | tail -n 1)

BACKTEST_ROOTS="$R21_A,$R21_B" \
LABELS="r21-A-rotation-onehot,r21-B-rotation-addon-onehot" \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r21_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

### 10.4 r1、r5、r21 六条曲线总览

```bash
R1_A=$(find saved_data/ashare_ml4t -maxdepth 1 -type d -name 'ch17_as1455_rotation_one_lag_daily_backtest_*' | sort | tail -n 1)
R1_B=$(find saved_data/ashare_ml4t -maxdepth 1 -type d -name 'ch17_as1455_rotation_addon_one_lag_daily_backtest_*' | sort | tail -n 1)
BASE="saved_data/ashare_ml4t/ch17_as1455_target_backtest"
R5_A=$(find "$BASE" -maxdepth 1 -type d -name 'rotation_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
R5_B=$(find "$BASE" -maxdepth 1 -type d -name 'rotation_addon_onehot_r05_fwd_reb5_*' | sort | tail -n 1)
R21_A=$(find "$BASE" -maxdepth 1 -type d -name 'rotation_onehot_r21_fwd_reb21_*' | sort | tail -n 1)
R21_B=$(find "$BASE" -maxdepth 1 -type d -name 'rotation_addon_onehot_r21_fwd_reb21_*' | sort | tail -n 1)

for d in "$R1_A" "$R1_B" "$R5_A" "$R5_B" "$R21_A" "$R21_B"; do
  [[ -d "$d" ]] || { echo "[MISSING] $d"; exit 1; }
done

BACKTEST_ROOTS="$R1_A,$R1_B,$R5_A,$R5_B,$R21_A,$R21_B" \
LABELS="r1-A,r1-B,r5-A,r5-B,r21-A,r21-B" \
OUT_DIR="saved_data/ashare_ml4t/ch17_as1455_backtest_plots/r1_r5_r21_ab_$(date +%Y%m%d_%H%M%S)" \
bash scripts/plot_as1455_default_ab_nav_curves.sh
```

切换最优参数选择指标：

```bash
RANK_METRIC=total_return bash scripts/plot_as1455_default_ab_nav_curves.sh
RANK_METRIC=calmar bash scripts/plot_as1455_default_ab_nav_curves.sh
```

绘图输出：

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

`selected_best_grids.csv` 记录每条曲线实际选中的 signal、持仓数、卖出排名、调仓周期、offset 和绩效指标。

## 11. 常见问题

### 11.1 `Killed`

`Killed` 只表示 Python 进程收到外部 SIGKILL，不能仅凭这一行判定 OOM。检查：

```bash
dmesg -T | egrep -i 'killed process|out of memory|oom|oom-killer' | tail -80
journalctl -k --since '2 hours ago' | egrep -i 'killed process|out of memory|oom|oom-killer' | tail -80
ps -eo pid,ppid,stat,%mem,rss,etime,cmd --sort=-rss | head -40
```

### 11.2 r21 fold6

当前数据下 r21 source fold6 不可用，不要默认训练或回测它。训练显式使用：

```bash
FOLDS="0 1 2 3 4 5"
```

回测 wrapper 已默认使用：

```text
TARGET_FOLDS=0,1,2,3,4
```

### 11.3 已有输出目录

训练脚本发现非空目录时会拒绝覆盖。确认旧结果不再使用后，才设置：

```bash
FORCE=1
```

回测 wrapper 当前默认 `FORCE_GRID=1`，会重跑 grid。需要保留已有 `summary.json` 并跳过已完成 run 时：

```bash
FORCE_GRID=0 bash scripts/run_as1455_r05_natural_backtest.sh
FORCE_GRID=0 bash scripts/run_as1455_r21_natural_backtest.sh
```

### 11.4 不同周期曲线日期不完全一致

r1、r5、r21 的有效标签日期和 target folds 不同，曲线起止日期可能不同。六曲线图适合总览；正式横向比较时应同时检查 `selected_best_grids.csv` 和每条曲线实际覆盖日期。
