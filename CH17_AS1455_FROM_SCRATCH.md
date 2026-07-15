# Ch17 AS1455 空盘重建

本流程用于服务器上的 AS1455 缓存、模型和回测结果丢失后，从静态 1000 股票池重新构建完整 Ch17 流程。

## 1. 原则

唯一总入口：

```text
scripts/run_ch17_as1455_full_rebuild.sh
```

它只负责顺序调用现有成熟入口、保存阶段标记、识别缺失 fold、在重任务间做内存门禁，并审计最终产物；不复制特征、训练、模型选择、调仓相位或交易逻辑。

唯一交易语义来源：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

## 2. 实际调用链

```text
scripts/run_ch17_as1455_full_rebuild.sh
├─ 三类历史缓存恢复并更新到自动解析的 T-1
│  └─ scripts/run_as1455_live_data_feature_pipeline.sh history
│     └─ pipelines/as1455_update_history_to_prevday_fast_v4.py
│        ├─ raw 5m
│        ├─ raw daily
│        └─ AS1455 daily
├─ 低内存构建 34 列 model_data
│  └─ scripts/build_ashare_ch12_as1455_lowmem.sh
├─ 协议与语法自检
│  └─ scripts/check_ch17_as1455_refactor.sh
├─ 批量训练
│  ├─ scripts/run_as1455_target_search_all.sh
│  ├─ scripts/run_as1455_r05_target_search_all.sh
│  └─ scripts/run_as1455_r21_target_search_all.sh
├─ one-fold-lag 历史回测
│  ├─ scripts/run_as1455_target_natural_backtest.sh
│  ├─ scripts/run_as1455_r05_natural_backtest.sh
│  └─ scripts/run_as1455_r21_natural_backtest.sh
├─ forward model data
│  └─ scripts/refresh_as1455_forward_model_data.sh
├─ fold0 strict-OOS forward
│  └─ scripts/run_as1455_fold0_forward_backtests.sh
└─ 统一绘图
   └─ scripts/plot_as1455_default_ab_nav_curves.sh
```

总控不解析、生成或缓存交易日期。history 与 forward 入口使用各自原有 `TRADE_DATE=today`、`HISTORY_END_DATE=auto` 逻辑。

## 3. 固定范围

- 股票池：`saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv`；
- 历史默认起点：`2020-01-01`；
- 特征：`rotation_onehot`、`rotation_addon_onehot`；
- 目标：`r01_fwd`、`r05_fwd`、`r21_fwd`；
- 训练：40 folds；
- 历史回测：6 组；
- strict-OOS forward：6 组。

## 4. 内存与断点

`scripts/as1455_python_memory_guard.sh` 将重型 Python 任务放在独立进程中。进程退出后记录内存、等待冷却并检查 `MemAvailable`，不执行 `drop_caches`、`pkill`、`kill -9` 或仓库级清理。

已存在的有效缓存会从各股票实际最后日期续传。完整训练 fold 会跳过；不完整训练目录会改名保存后重跑。阶段标记位于：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/state_v2/
```

## 5. 启动

```bash
cd /root/stock_realtime_v021_full
bash scripts/run_ch17_as1455_full_rebuild.sh preflight

tmux new-session -d -s ch17_as1455_rebuild \
  "cd /root/stock_realtime_v021_full && bash scripts/run_ch17_as1455_full_rebuild.sh all"

tmux attach -t ch17_as1455_rebuild
```

中断后重复执行 `all` 即可续跑。查看状态：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh status
```

## 6. 完成条件

最终报告：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/final_report.json
```

必须满足：

```text
training_ok = 40
historical_ok = 6
forward_ok = 6
plots_ok = true
all_ok = true
```

并打印：

```text
[PASS] full Ch17 AS1455 rebuild completed
```
