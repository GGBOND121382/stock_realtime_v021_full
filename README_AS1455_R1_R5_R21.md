# AS1455 r1 / r5 / r21 训练、回测与绘图指南

本文档给出当前推荐命令。代码分层、唯一事实来源和开发规则见：

```text
CH17_AS1455_DEVELOPMENT_OUTLINE.md
```

以下命令均在工程根目录执行：

```bash
cd ~/stock_realtime_v021_full
```

---

## 1. 当前结构

公共逻辑已统一放入：

```text
utils/as1455_ch17_common.py
utils/as1455_signal_specs.py
utils/as1455_backtest_io.py
utils/as1455_plotting.py
```

职责如下：

| 文件 | 职责 |
|---|---|
| `as1455_ch17_common.py` | 目标定义、A/B 特征、fold、checkpoint/scaler/manifest、推理、预测产物和 grid 命令 |
| `as1455_signal_specs.py` | 根据 `TOP_N` 生成合法的模型与 ensemble 信号 |
| `as1455_backtest_io.py` | 统一 TradeConfig 和回测结果写出 |
| `as1455_plotting.py` | 统一线型、marker 和绘图样式 |

核心入口：

```text
训练：scripts/run_as1455_target_fold_param_search.py
批量训练：scripts/run_as1455_target_search_all.sh
one-fold-lag：scripts/run_as1455_target_one_lag_backtest.py
自然周期回测：scripts/run_as1455_target_natural_backtest.sh
fold0-forward：scripts/run_as1455_fold0_forward_backtest.py
回测引擎：code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
网格编排：code/backtest/run_as1455_close_auction_grid_inprocess.py
绘图：scripts/plot_as1455_backtest_return_curves.py
```

r1、r5、r21 的旧命令入口仍保留，但已经改成薄 wrapper。

---

## 2. 固定实验口径

### 2.1 目标与自然调仓周期

| 简称 | 监督目标 | lookahead | 调仓周期 | offset |
|---|---|---:|---:|---|
| r1 | `r01_fwd` | 1 | 1 | `0` |
| r5 | `r05_fwd` | 5 | 5 | `0..4` |
| r21 | `r21_fwd` | 21 | 21 | `0..20` |

该映射只维护在：

```text
utils/as1455_ch17_common.py::TARGET_SPECS
```

### 2.2 特征方案

| 名称 | `feature_preset` | 内容 |
|---|---|---|
| A | `rotation_onehot` | 原始 31 特征 + 完整 sector rotation + sector one-hot |
| B | `rotation_addon_onehot` | A + compact add-on 特征 |

### 2.3 训练数据

默认训练数据：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
```

正式搜索产物契约：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
fold_report.json
```

正式回测使用 search-time checkpoint，不使用诊断性 retrain 的 `models/best_*.keras`。

---

## 3. 代码拉取和静态检查

```bash
cd ~/stock_realtime_v021_full
git pull origin master
```

语法检查：

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

CLI 检查：

```bash
python3 scripts/run_as1455_target_fold_param_search.py --help >/dev/null
python3 scripts/run_as1455_target_one_lag_backtest.py --help >/dev/null
python3 scripts/run_as1455_fold0_forward_backtest.py --help >/dev/null
python3 scripts/plot_as1455_backtest_return_curves.py --help >/dev/null
python3 code/backtest/run_as1455_close_auction_grid_inprocess.py --help >/dev/null

echo '[OK] AS1455 CLI imports passed'
```

---

## 4. r1 / r5 / r21 训练

### 4.1 通用单 fold 命令

A 组、r1、fold0：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_onehot \
  --target-col r01_fwd \
  --fold-index 0 \
  --epochs 20 \
  --best-n 5
```

B 组、r5、fold3：

```bash
python3 scripts/run_as1455_target_fold_param_search.py \
  --feature-preset rotation_addon_onehot \
  --target-col r05_fwd \
  --fold-index 3 \
  --epochs 20 \
  --best-n 5
```

A 组、r21、fold0：

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

### 4.3 批量训练 r5

兼容命令：

```bash
bash scripts/run_as1455_r05_target_search_all.sh
```

等价通用命令：

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

兼容命令：

```bash
bash scripts/run_as1455_r21_target_search_all.sh
```

等价通用命令：

```bash
TARGET_COL=r21_fwd \
bash scripts/run_as1455_target_search_all.sh
```

当前默认训练：

```text
A/B × fold0..fold5
```

当前数据下 r21 的有效日期不足以生成 fold6，通用 wrapper 已默认排除 fold6。

### 4.5 输入检查和 smoke

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

one-fold-lag 协议：

```text
source fold6 -> target fold5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

每个 target fold 只使用更早一个 fold 的 search-time checkpoint，不在 target fold 上重新训练或选模型。

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

这两个入口目前只是 r1 A/B 的兼容 wrapper，实际实现统一调用：

```text
scripts/run_as1455_target_one_lag_backtest.py
```

### 5.2 r5

先做单配置引擎 smoke，不执行完整网格：

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

默认 `TOP_N=5`，因此每个特征方案包含：

```text
model_0..model_4
ensemble_first3_mean
ensemble_all5_mean
```

参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 5 offsets
= 1050 runs / feature preset
```

### 5.3 r21

先做引擎 smoke：

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

即：

```text
source fold5 -> target fold4
...
source fold1 -> target fold0
```

参数空间：

```text
7 signals × 5 max_positions × 6 sell_rank × 21 offsets
= 4410 runs / feature preset
```

### 5.4 当前网格优化边界

当前 `inprocess_shared_data_v3` 已做到：

- prediction 每个 signal 只加载一次；
- execution panel 只构造一次；
- universe、ST、公司行为和容量数据只加载一次；
- 所有配置在同一个 Python 进程中执行；
- 每个配置调用同一个 v7 `backtest()`，不存在第二套交易循环。

当前尚未做到：

```text
每个 signal 的每日排序只计算一次并由所有配置共享
```

原因是排序仍位于唯一 v7 回测核心内部。下一步优化必须改造 v7 公共接口，不能在 grid 文件中复制买卖、费用和 NAV 逻辑。

---

## 6. fold0 最优模型用于 fold0 后续日期

该协议与 one-fold-lag 不同：

- 使用 fold0 search-time checkpoint、scaler 和 feature manifest；
- 仅预测 `date > fold0.test_end`；
- 不重新训练；
- 每个回测配置从初始资金和空仓开始；
- 不继承 fold0 测试窗口的持仓。

### 6.1 更新最新历史数据并重建 forward HDF

默认 forward wrapper 会自动执行：

```text
history 缓存更新
→ 重建 forward model_data
→ checkpoint 推理
→ 回测
```

单独更新数据：

```bash
bash scripts/refresh_as1455_forward_model_data.sh
```

输出：

```text
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

### 6.2 r1 fold0-forward

```bash
TARGETS='r01_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 6.3 r5 fold0-forward

```bash
TARGETS='r05_fwd' \
FEATURE_PRESETS='rotation_onehot rotation_addon_onehot' \
OUTPUT_MODE=full \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

### 6.4 r21 fold0-forward

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

因此只生成和回测：

```text
model_0
```

这与“fold0 的最优模型”这一实验定义一致。

需要 top5 和 ensemble 实验时显式运行：

```bash
TOP_N=5 \
TARGETS='r05_fwd' \
bash scripts/run_as1455_fold0_forward_backtests.sh
```

`utils/as1455_signal_specs.py` 会根据实际 `TOP_N` 生成合法 signal，不会在 `TOP_N=1` 时错误读取不存在的 `model_1..model_4`。

### 6.6 已刷新数据后只重跑回测

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

绘图会：

- 从每个根目录的 grid summary 中按 `RANK_METRIC` 选择最优 run；
- 读取对应 `close_auction_nav.csv`；
- 生成 daily、weekly、monthly 曲线；
- 同时使用颜色、线型和 marker；
- 在 CSV 中记录 `line_style` 和 `marker`；
- 输出实际选中的 signal 和参数。

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

切换最优参数指标：

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

## 9. 最低回归验证

### 9.1 signal 数量

```bash
python3 - <<'PY'
from utils.as1455_signal_specs import signal_specs_for_top_n
assert signal_specs_for_top_n(1) == ['model_0:0:single']
assert len(signal_specs_for_top_n(5)) == 7
print('[OK] signal specs')
PY
```

### 9.2 固定预测文件对比

对重构前后相同预测文件、signal 和参数，至少比较：

```text
close_auction_nav.csv
close_auction_orders.csv
close_auction_rejections.csv
round_trips.csv
summary.json
```

数值字段要求：

```text
rtol = 1e-12
atol = 1e-12
```

允许不同：

```text
输出目录名
grid_engine metadata
创建时间
```

不允许不同：

```text
交易日期
买卖方向
成交数量
费用
持仓
NAV
收益和风险指标
```

---

## 10. 当前已知限制

1. r21 当前数据下没有可用 source fold6，因此 one-fold-lag 默认只覆盖 target fold0..4。
2. forward HDF 的最后 1、5、21 个交易日分别无法得到完整的 r1、r5、r21 前向标签。
3. 当前 in-process v3 已消除第二套交易循环，但尚未在所有 grid 配置之间共享每日排名缓存。
4. live 旧基线 checkpoint bundle 与 target-aware A/B `.keras + scaler.pkl + feature_manifest.json` 不是同一产物契约，不能直接混用。
