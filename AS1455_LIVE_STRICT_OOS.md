# AS1455 clean 分支自动盯盘与严格 OOS 单日组合计划

## 1. 旧版代码的最终理解

旧 `master` 中存在两条先后演进、但没有完全合流的代码线。

### 1.1 自动化盯盘线

旧版最新一键入口是：

```text
scripts/run_as1455_live_fast_auto_checkpoint_signal_v1.sh
```

它已经覆盖：

```text
T-1 历史更新
→ 09:35 preclose 与复权事件准备
→ 盘前 252 日快速状态
→ 14:50–14:55 循环采集
→ 固化 <=14:55 的最后有效快照
→ 快速生成当天 31 个基础特征
→ checkpoint 推理与排名
→ 读取现金/持仓
→ 生成计划订单
→ 更新 paper state
```

数据采集和快速特征部分经过多轮修复，包括：

- 只使用不晚于 14:55 的快照；
- 修复 `dollar_vol_rank` 的 MultiIndex 对齐；
- 使用训练数据中的 sector 映射；
- 14:55 后只计算当天一行，而不是重建完整 252 日面板；
- 用 execution sidecar 隔离模型特征与执行期字段。

### 1.2 rotation/addon 与严格 OOS 线

旧版后期又形成了：

```text
rotation_onehot
rotation_addon_onehot
r01/r05/r21 target-aware
one-fold-lag 历史预测
历史完整 Grid
materialized best run
fold0 strict-OOS
连续调仓相位
```

其中 addon 明确是：

```text
基础 31 特征 + 完整 sector rotation + compact addon + sector one-hot
```

而不是用 addon 替代 rotation。

### 1.3 未完成的合流

旧自动盯盘的 checkpoint 层仍然采用：

```text
固定 31 特征
旧 .weights.h5
现场重建 scaler
多个 fold 求均值
Top-5 再平均
独立的单日买卖循环
计划订单默认视为已成交并更新 paper state
```

没有发现它最终接入 `rotation_onehot/rotation_addon_onehot`、`materialized_best_run.json`、严格调仓相位和 v7 唯一交易引擎的证据。

因此 clean 的恢复原则是：

- 恢复旧版较成熟的盘前准备、实时采集和快速基础特征；
- 不恢复旧 checkpoint 部署协议；
- 不恢复旧 paper state 自动推进；
- 不恢复第二套买卖循环；
- 使用 clean 当前模型、历史选择、相位和 v7 语义完成最后半段。

## 2. clean 中的新端到端流程

```text
T-1 历史缓存增量更新
→ 09:35 获取 preclose、识别复权事件、准备 qfq 历史尾部
→ 盘前生成压缩特征状态与 raw-daily 执行日历
→ 14:50–14:55 循环采集新浪累计行情
→ 固化 <=14:55:00 的最后有效截面
→ 快速生成当天 31 个基础特征
→ 用 clean 公共代码构造 rotation/addon/sector one-hot
→ 加载 target/preset 对应 fold0 .keras checkpoint 与已保存 scaler
→ 复用历史最佳完整 run 的 signal 和交易参数
→ 换算连续调仓相位
→ 读取券商当前现金、持仓与买入日期
→ 直接调用 v7 唯一交易引擎的单日模式
→ 输出计划订单和假设成交后的目标组合
```

权威语义源：

```text
utils/as1455_ch17_common.py
utils/as1455_model_selection.py
utils/as1455_rebalance_phase.py
utils/as1455_strict_oos.py
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
```

## 3. 入口

### 静态检查

```bash
bash scripts/run_as1455_live_strict_oos_pipeline.sh check
```

### 上午准备

```bash
TRADE_DATE=today \
TARGET_COL=r05_fwd \
FEATURE_PRESET=rotation_addon_onehot \
bash scripts/run_as1455_live_strict_oos_pipeline.sh pre
```

### 14:50 后采集、推理与组合计划

持仓文件至少包含：

```csv
symbol,shares,buy_date,avg_entry_price
600000.SH,1000,2026-07-21,12.34
000001.SZ,1200,2026-07-18,10.50
```

执行：

```bash
TRADE_DATE=today \
TARGET_COL=r05_fwd \
FEATURE_PRESET=rotation_addon_onehot \
POSITIONS_FILE=/secure/current_positions.csv \
CASH=35678.90 \
bash scripts/run_as1455_live_strict_oos_pipeline.sh post
```

现金也可以放在文件中：

```bash
CASH_FILE=/secure/current_cash.txt
```

### 一键守候

```bash
TRADE_DATE=today \
TARGET_COL=r05_fwd \
FEATURE_PRESET=rotation_addon_onehot \
POSITIONS_FILE=/secure/current_positions.csv \
CASH_FILE=/secure/current_cash.txt \
bash scripts/run_as1455_live_strict_oos_pipeline.sh auto
```

### 已完成交易日回放

准备该日期的 `08_live_raw_row_as1455.csv` 后：

```bash
TRADE_DATE=20260722 \
SKIP_COLLECT=1 \
POSITIONS_FILE=/secure/positions_before_20260722.csv \
CASH_FILE=/secure/cash_before_20260722.txt \
bash scripts/run_as1455_live_strict_oos_pipeline.sh post
```

## 4. 输出

目录：

```text
saved_data/ashare_ml4t/live_as1455/YYYYMMDD/
```

关键文件：

```text
01_universe.csv
02_preclose_snapshot_0935.csv
03_adjustment_events.csv
05_history_tail_qfq_livebase.parquet|csv
05_execution_calendar.csv
06_live_feature_state_fast.npz
08_live_raw_row_as1455.csv
08_live_execution_sidecar.csv
11_live_model_features_for_prediction.csv
14_live_predictions.csv
14_live_checkpoints.csv
15_live_rank.csv
16_live_nav.csv
16_live_orders.csv
16_live_rejections.csv
16_live_positions_after_plan.csv
16_live_target_portfolio.csv
17_live_strict_oos_manifest.json
```

`16_live_orders.csv` 中订单状态被改写为：

```text
planned_not_submitted
```

v7 内部的模拟成交状态保存在 `simulated_fill_status`。这些文件不代表券商真实成交。

## 5. 强制约束

1. `post/plan/auto` 必须显式提供实际现金和持仓，禁止默认从 20 万元空仓开始。
2. 持仓默认必须带 `buy_date`，否则无法严格执行 T+1。
3. signal、max positions、sell rank、rebalance period 和历史相位来自历史最佳完整 run，盘中不调参；相位桥接天数来自盘前按 raw-daily 缓存构造的执行日历。
4. 模型只加载 target/preset 对应的 fold0 搜索期 `.keras` checkpoint、`scaler.pkl` 和 feature manifest。
5. rotation/addon 由 clean 公共代码生成，实时脚本不复制这些公式。
6. 历史与盯盘共同调用 v7；没有第二套买卖循环。
7. 账户持仓已经反映公司行为，因此单日 live 模式将 `corporate_action_mode` 固定为 `none`，防止重复调整。
8. 14:55 时尚不知道完整的 14:55–15:00 最终成交量，当前强制 `CAPACITY_MODE=none`。
9. 14:55 价格只是收盘集合竞价计划价格近似；实际成交必须从券商回报核对。
10. 不自动下单，不自动把计划订单写回账户真值。

## 6. 特征质量与 TA-Lib

生产模式要求安装 `requirements.txt` 中的 `TA-Lib`，以保持 RSI、BBANDS、NATR、ATR、PPO 和 MACD 与历史构建一致。

只有轻量测试环境可以显式开启：

```bash
ALLOW_INDICATOR_FALLBACK=1
```

fallback 会写入报告，不能默默用于生产。

## 7. 服务器前置产物

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/
saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache/
saved_data/ashare_ml4t/ch17_as1455_target_search/.../fold0_search/
saved_data/ashare_ml4t/ch17_as1455_target_backtest/.../
```

fold0 目录必须包含：

```text
search_best_checkpoints.csv
search_checkpoints/*.keras
preprocess/scaler.pkl
preprocess/feature_manifest.json
```

历史最佳 run 必须保留：

```text
status=ok summary row
01_runs/<run_name>/config.json
date_min/date_max/n_days 或 materialized NAV
```

## 8. 验证

本地：

```bash
pytest -q tests/test_as1455_live_strict_oos_helpers.py
bash scripts/run_as1455_live_strict_oos_pipeline.sh check
bash scripts/check_ch17_as1455_refactor.sh
```

服务器上线前至少回放一个已完成交易日，并检查：

- 快速基础特征与历史重建同日特征；
- rotation/addon 最终列与训练 manifest；
- live fold0 预测与离线同日预测；
- 调仓日与连续相位；
- v7 订单、拒单原因、T+1、涨跌停、ST、费用和手数；
- 券商实际成交与计划订单差异。
