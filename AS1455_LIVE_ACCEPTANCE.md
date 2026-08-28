# AS1455 strict-OOS 盯盘验收

## 1. 静态验收

在仓库根目录运行：

```bash
bash scripts/accept_as1455_live_strict_oos.sh static
```

通过标准：

```text
8 passed
[PASS] Ch17 AS1455 clean runtime validation passed
[PASS] live strict-OOS pipeline static check
[PASS] AS1455 live strict-OOS static acceptance
```

## 2. 已完成交易日隔离回放

选择一个此前真实运行过盘前准备和 14:55 采集的日期。该日期目录至少应包含：

```text
03_adjustment_events.csv
05_execution_calendar.csv
06_live_feature_state_fast.npz
08_live_raw_row_as1455.csv
```

准备该日开盘前或 14:55 决策前的账户状态：

```csv
symbol,shares,buy_date,avg_entry_price
600000.SH,1000,2026-07-21,12.34
000001.SZ,1200,2026-07-18,10.50
```

现金文件只放一个非负数字，例如：

```text
35678.90
```

执行：

```bash
TRADE_DATE=20260722 \
POSITIONS_FILE=/secure/positions_before_20260722.csv \
CASH_FILE=/secure/cash_before_20260722.txt \
TARGET_COL=r05_fwd \
FEATURE_PRESET=rotation_addon_onehot \
bash scripts/accept_as1455_live_strict_oos.sh replay
```

脚本不会覆盖原始实时目录。默认隔离输出到：

```text
saved_data/ashare_ml4t/live_as1455_acceptance/20260722/
```

最终必须出现：

```text
[PASS] acceptance report: .../18_live_acceptance_report.json
[PASS] AS1455 completed-date replay acceptance
```

验收报告检查：

- 快速特征、严格特征契约和 execution sidecar 均通过；
- 使用 clean 的 rotation/addon 与 fold0 checkpoint；
- 历史最佳完整 run 的信号和交易参数被复用；
- 连续调仓相位覆盖完整；
- 实际现金、持仓和 T+1 买入日期被读取；
- 只调用 canonical v7 交易引擎；
- 订单股数为正且满足 100 股整数手；
- 所有订单状态均为 `planned_not_submitted`；
- 没有调用券商接口，也没有把计划订单写回账户真值。

## 3. 常见覆盖参数

历史结果或 fold0 目录不能自动定位时：

```bash
SELECTION_BACKTEST_ROOT=/path/to/ch17_as1455_target_backtest/... \
FOLD0_DIR=/path/to/ch17_as1455_target_search/.../fold0_search \
... \
bash scripts/accept_as1455_live_strict_oos.sh replay
```

原始实时目录不在默认位置时：

```bash
SOURCE_LIVE_DIR=/path/to/live_as1455/20260722 \
... \
bash scripts/accept_as1455_live_strict_oos.sh replay
```

轻量环境没有 TA-Lib 时只能用于测试：

```bash
ALLOW_INDICATOR_FALLBACK=1 \
... \
bash scripts/accept_as1455_live_strict_oos.sh replay
```

正式生产验收不应开启 indicator fallback。
