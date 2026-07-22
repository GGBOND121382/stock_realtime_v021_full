# Ch17 AS1455 空盘重建

本流程用于服务器上的 AS1455 缓存、模型和回测结果丢失后，从静态 1000 股票池重新构建完整 Ch17 流程。

## 1. 原则

唯一总入口：

```text
scripts/run_ch17_as1455_full_rebuild.sh
```

它只负责顺序调用现有成熟入口、保存阶段标记、识别缺失 fold、在重任务间做内存门禁，并审计最终产物；不复制特征、模型选择、调仓相位或交易逻辑。

fold 日期由 `utils/as1455_fold_calendar.py` 统一确定：先保留模型特征完整的交易日，再把日历截断到 r1/r5/r21 均已有真实标签的共同截止日。三个目标和 A/B 特征方案共享同一组 fold0..fold6 起止日期；目标 lookahead 只影响训练区间与 fold 之间的 embargo，不再改变 fold 日期。

唯一交易语义来源：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

## 2. 实际调用链

```text
scripts/run_ch17_as1455_full_rebuild.sh
└─ scripts/run_ch17_as1455_full_rebuild_aligned.sh
   ├─ 三类历史缓存恢复并更新到自动解析的 T-1
   │  └─ scripts/run_as1455_live_data_feature_pipeline.sh history
   ├─ 低内存构建 34 列 model_data
   │  └─ scripts/build_ashare_ch12_as1455_lowmem.sh
   ├─ 协议与语法自检
   │  └─ scripts/check_ch17_as1455_refactor.sh
   ├─ 共享 fold 日历上的批量训练
   │  ├─ scripts/run_as1455_target_search_all.sh
   │  └─ scripts/run_as1455_target_fold_param_search_aligned.py
   ├─ one-fold-lag 历史回测
   │  ├─ scripts/run_as1455_target_natural_backtest.sh
   │  └─ scripts/run_as1455_target_one_lag_backtest_aligned.py
   ├─ forward model data
   │  └─ scripts/refresh_as1455_forward_model_data.sh
   ├─ fold0 strict-OOS forward
   │  └─ scripts/run_as1455_fold0_forward_backtests.sh
   └─ 统一绘图
      ├─ scripts/plot_as1455_default_ab_nav_curves.sh
      └─ scripts/plot_as1455_fold_sequence_curves.py
```

总控不解析、生成或缓存交易日期。history 与 forward 入口使用各自原有 `TRADE_DATE=today`、`HISTORY_END_DATE=auto` 逻辑。

## 3. 固定范围

- 股票池：`saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv`；
- 历史默认起点：`2020-01-01`；
- 特征：`rotation_onehot`、`rotation_addon_onehot`；
- 目标：`r01_fwd`、`r05_fwd`、`r21_fwd`；
- 训练：42 folds，即 3 个目标 × 2 个特征方案 × fold0..fold6；
- 历史回测：6 组，每组覆盖 source fold6→target fold5 至 source fold1→target fold0；
- strict-OOS forward：6 组，均从共同 fold0 结束日之后开始；
- 绘图：3 张完整 forward 图，以及 fold6..fold0 各 3 张日/周/月图，共 21 张 fold 图。

## 4. fold 图语义

```text
fold6：source fold6 在 target fold5 日期段的 one-fold-lag 结果
fold5：source fold5 在 target fold4 日期段的 one-fold-lag 结果
...
fold1：source fold1 在 target fold0 日期段的 one-fold-lag 结果
fold0：fold0 checkpoint 在共同 fold0 结束日后的 strict-OOS forward 结果
```

历史组合回测的仓位在 fold6→fold1 之间连续传递。分 fold 图按各 fold 的首个共同执行日重新归一化，用于观察该时间段的收益变化；fold0 forward 仍按空仓和初始现金启动。

输出目录：

```text
saved_data/ashare_ml4t/ch17_as1455_backtest_plots/full_rebuild_<RUN_STAMP>/
├─ return_curve_daily.png
├─ return_curve_weekly.png
├─ return_curve_monthly.png
└─ fold_sequence/
   ├─ fold6/return_curve_{daily,weekly,monthly}.png
   ├─ ...
   └─ fold0/return_curve_{daily,weekly,monthly}.png
```

## 5. 内存与断点

`scripts/as1455_python_memory_guard.sh` 将重型 Python 任务放在独立进程中。进程退出后记录内存、等待冷却并检查 `MemAvailable`，不执行系统级缓存清理或终止无关进程。

已存在的有效缓存会从各股票实际最后日期续传。只有带有 `fold_calendar_mode=shared_feature_complete_all_targets` 的完整训练 fold 才会跳过；旧日期口径的 fold 会被改名保留并重跑。新流程阶段标记位于：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/state_v3_aligned/
```

## 6. 启动

```bash
cd /root/stock_realtime_v021_full
bash scripts/run_ch17_as1455_full_rebuild.sh preflight

HISTORY_WORKERS=3 SYMBOL_RETRIES=2 \
  bash scripts/run_ch17_as1455_full_rebuild.sh all
```

也可放入 tmux。中断后重复执行 `all` 即可续跑。查看状态：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh status
```

## 7. 完成条件

最终报告：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/final_report.json
```

必须满足：

```text
training_expected = 42
training_ok = 42
historical_ok = 6
forward_ok = 6
forward_plots_ok = true
fold_plot_expected = 21
fold_plot_ok = 21
all_ok = true
```

并打印：

```text
[PASS] full Ch17 AS1455 rebuild completed
```
