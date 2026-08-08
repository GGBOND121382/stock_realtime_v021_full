# AS1455 九策略收益与 14:55 盯盘前端

## 1. 功能

统一前端读取：

```text
saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1
```

展示 9 个固定信号策略：

```text
r01_fwd × all5 / first3 / best
r05_fwd × all5 / first3 / best
r21_fwd × all5 / first3 / best
```

新增能力：

1. 用户设置“开始持仓 / 收益起算日”，9 条曲线从所选日期或之后第一个可用交易日重新归零；
2. 工作日 20:00 自动更新行情、forward model data、9 组 strict-forward 结果、收益曲线和最新账户状态；
3. 工作日 09:35 自动准备 live 特征状态；
4. 工作日 14:50 开始实时采集，14:55 冻结快照后为 9 个策略生成调仓计划；
5. 每个策略显示计划买入、计划卖出、拒单、目标组合、当日排名和是否为调仓日；
6. 所有 live 订单均为 `planned_not_submitted`，不会调用券商 API。

“开始持仓 / 收益起算日”是同步加入既有策略的收益观察起点：按起算日策略 NAV 重新归零，不改变历史 Grid、调仓相位或后台模拟账户状态，也不补计首次同步建仓成本。

## 2. 部署前端

```bash
cd /root/stock_realtime_v021_full
git pull --ff-only
.venv_as1455/bin/pip install -r requirements-dashboard.txt
```

推荐只绑定本机：

```bash
AS1455_DASHBOARD_REFRESH_TOKEN='set-a-strong-token' \
HOST=127.0.0.1 \
PORT=8501 \
bash scripts/run_as1455_backtest_dashboard.sh
```

本地通过 SSH 端口转发访问：

```bash
ssh -L 8501:127.0.0.1:8501 root@SERVER
```

浏览器打开 `http://127.0.0.1:8501`。

## 3. 第一次生成 T-1 策略账户状态

14:55 九策略盯盘需要每个策略上一完整交易日的模拟现金和持仓。升级后先执行一次 9 组刷新：

```bash
SKIP_DATA_REFRESH=1 \
time bash scripts/run_ch17_as1455_full_rebuild.sh refresh-all-fixed-signals
```

刷新后每个实验根目录新增：

```text
strict_forward_latest_state.json
strict_forward_latest_positions.csv
```

矩阵根目录新增：

```text
strict_forward_latest_states.csv
strict_forward_latest_states_manifest.json
```

这些文件在 full strict-forward audit 被清理前提取，随后仍只永久保留紧凑状态，不长期保存全日期 positions/orders 大文件。

## 4. 自动任务

一条命令安装全部工作日任务：

```bash
bash scripts/install_as1455_strategy_dashboard_automation.sh
```

默认北京时间：

```text
09:35  live pre：T-1 历史更新、preclose/调整状态、紧凑特征状态
14:50  live post：开始采集；14:55:05 前冻结快照；生成 9 策略计划
20:00  daily refresh：更新数据和 9 组 strict-forward，刷新收益曲线和 T 日最终账户状态
```

生成两个独立 cron 文件：

```text
/etc/cron.d/as1455-nine-strategy-live
/etc/cron.d/as1455-dashboard-refresh
```

检查：

```bash
cat /etc/cron.d/as1455-nine-strategy-live
cat /etc/cron.d/as1455-dashboard-refresh
```

## 5. 14:55 九策略盯盘流程

```text
09:35 shared pre
→ 14:50~14:55 shared quote collection
→ current-day base feature finalization once
→ execution sidecar once
→ r01 fold0 Top-5 inference once
→ r05 fold0 Top-5 inference once
→ r21 fold0 Top-5 inference once
→ each target derives all5 / first3 / best scores
→ each strategy reads its own validated historical winner
→ each strategy reads its T-1 strict-forward cash/positions
→ historical rebalance phase alignment
→ canonical v7 single-day planning
→ 9 strategy summaries / orders / target portfolios
```

三种固定信号：

```text
all5   = ensemble_all5_mean:0,1,2,3,4:mean
first3 = ensemble_first3_mean:0,1,2:mean
best   = model_0:0:single
```

输出：

```text
saved_data/ashare_ml4t/live_as1455/YYYYMMDD/nine_strategy/
├── shared_predictions/
│   ├── r01/
│   ├── r05/
│   └── r21/
├── strategies/
│   ├── r01_all5_.../
│   ├── ...
│   └── r21_best_.../
├── live_nine_strategy_summary.csv
├── live_rebalance_strategies.csv
└── live_nine_strategy_manifest.json
```

每个策略目录含：

```text
16_live_orders.csv
16_live_rejections.csv
16_live_positions_after_plan.csv
16_live_target_portfolio.csv
current_positions_before_plan.csv
live_rank.csv
strategy_manifest.json
```

## 6. 模拟账户与公司行动口径

现有单账户 strict-OOS monitor 默认假设券商持仓已反映公司行动，因此 live 当天不重复调整。

9 策略页面使用的是上一完整交易日 strict-forward 模拟账户状态，不是券商状态。若 live 当天发生除权除息，9 策略 planner 使用历史回测对应的 synthetic corporate-action 连续性口径，避免把 T-1 模拟股数误当成已经由券商调整过的股数。

20:00 最终 strict-forward 刷新仍是权威模拟账户状态；14:55 的 plan 不会被自动持久化为账户真值。

## 7. 手工运行

检查：

```bash
bash scripts/run_as1455_live_nine_strategy_pipeline.sh check
```

上午预处理：

```bash
bash scripts/run_as1455_live_nine_strategy_job.sh pre
```

14:50 后生成当日策略：

```bash
bash scripts/run_as1455_live_nine_strategy_job.sh post
```

查看结果：

```bash
bash scripts/run_as1455_live_nine_strategy_pipeline.sh status
```

## 8. 失败保护

- 20:00 前端刷新要求 9 组历史 Grid 均可严格复用；缺失任一历史结果即失败关闭，不会静默重新 Grid；
- live post 要求上午 prefast state 存在；没有则失败，不会临时重建一套不一致的输入；
- live planner 要求每个策略的账户状态日期严格早于当前 live 日期，防止同日状态被重复执行；
- 固定信号必须严格匹配 all5/first3/best 预期 spec，否则失败；
- 14:55 capacity 默认 `none`，因为 14:55 时尚不知道完整 14:55~15:00 成交容量；
- 不提交券商订单。

## 9. 测试

静态/轻量测试：

```bash
.venv_as1455/bin/python -m py_compile \
  dashboard/as1455_strategy_dashboard.py \
  dashboard/as1455_live_data.py \
  scripts/export_as1455_global_forward_latest_states.py \
  scripts/run_as1455_live_target_predictions.py \
  scripts/run_as1455_live_nine_strategy_planner.py

bash -n scripts/run_as1455_live_nine_strategy_pipeline.sh
bash -n scripts/run_as1455_live_nine_strategy_job.sh
bash -n scripts/install_as1455_live_nine_strategy_cron.sh
bash -n scripts/install_as1455_dashboard_daily_refresh_cron.sh
bash -n scripts/install_as1455_strategy_dashboard_automation.sh

.venv_as1455/bin/python -m pytest -q tests/test_as1455_strategy_dashboard_v2.py
```

真实 14:55 流程依赖服务器 HDF、历史 Grid、checkpoint、raw daily cache 和当日 live snapshot，应在服务器完成一次 replay/已有快照验收后再开启 cron。
