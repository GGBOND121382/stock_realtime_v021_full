# Ch17 AS1455 空盘重建

本流程用于服务器上的 AS1455 缓存、模型和回测结果丢失后，从 GitHub 代码与仓库内静态 1000 股票池重新构建完整 Ch17 流程。

## 1. 设计原则

本次只新增一个薄编排入口：

```text
scripts/run_ch17_as1455_full_rebuild.sh
```

它只负责：

- 顺序调用原工程已有成熟脚本；
- 保存阶段完成标记；
- 识别缺失训练 fold；
- 在重任务之间检查内存并冷却；
- 验证最终训练、历史回测、strict-OOS forward 和绘图产物。

它不复制特征构建、fold、checkpoint、模型选择、相位换算、交易网格或交易循环。唯一交易语义仍来自：

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

没有新增多股票并行下载器。`fast_v4` 只通过现有 history 自动化用于已建立缓存后的增量同步，不再承担空缓存首次全量下载。

## 2. 实际调用链

```text
scripts/run_ch17_as1455_full_rebuild.sh
├─ 空缓存 5 分钟数据分批恢复
│  └─ scripts/build_ashare_ch12_as1455_model_data.py
│     ├─ --daily-cache-only
│     ├─ --allow-partial-coverage
│     └─ --baostock-fetch-limit 250
├─ 同步三类缓存到固定 T-1
│  └─ scripts/run_as1455_live_data_feature_pipeline.sh history
│     └─ pipelines/as1455_update_history_to_prevday_fast_v4.py
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

训练、历史预测和 forward 继续共用 `utils/as1455_ch17_common.py`；strict-OOS 继续使用现有模型选择、forward 行保留、调仓相位和冻结配置模块。

## 3. 固定范围

- 数据起点：`2020-01-02`；
- 股票池：仓库内静态 1000 股票池；
- 特征方案：`rotation_onehot`、`rotation_addon_onehot`；
- 目标：`r01_fwd`、`r05_fwd`、`r21_fwd`；
- 训练：r1/r5 各 2×7 folds，r21 为 2×6 folds，共 40 folds；
- 历史回测：3 个目标 × 2 个特征方案，共 6 组；
- forward：3 个目标 × 2 个特征方案，共 6 组；
- 历史网格默认 `summary`，只物化 Sharpe 最佳 run；
- forward 默认 `strict_oos`、`compact`、Sharpe 选择。

## 4. 内存隔离

`scripts/as1455_python_memory_guard.sh` 包装现有 Python 入口，不修改业务代码。

默认行为：

```text
MIN_AVAILABLE_MEMORY_MB=2048
TRAIN_COOLDOWN_SECONDS=20
BACKTEST_COOLDOWN_SECONDS=20
DATA_COOLDOWN_SECONDS=30
CPU_THREADS=2
```

每个训练、历史回测、forward 或数据构建 Python 进程退出后：

1. 记录 `MemAvailable` 与 Swap；
2. 等待对应冷却时间；
3. 再次检查可用内存；
4. 内存不足时等待，超过最大等待次数则停止。

不执行 `drop_caches`、`pkill`、`kill -9`、仓库级 `rm -rf`、`git clean` 或 `git reset --hard`。

## 5. 首次启动

先停止旧版重建进程，再拉取本分支：

```bash
tmux send-keys -t ch17_as1455_rebuild C-c 2>/dev/null || true
tmux kill-session -t ch17_as1455_rebuild 2>/dev/null || true

cd /root/stock_realtime_v021_full
git fetch origin agent/rebuild-ch17-as1455-from-scratch
git switch agent/rebuild-ch17-as1455-from-scratch
git pull --ff-only origin agent/rebuild-ch17-as1455-from-scratch

bash -n scripts/run_ch17_as1455_full_rebuild.sh
bash -n scripts/as1455_python_memory_guard.sh
bash scripts/run_ch17_as1455_full_rebuild.sh preflight

tmux new-session -d -s ch17_as1455_rebuild \
  "cd /root/stock_realtime_v021_full && bash scripts/run_ch17_as1455_full_rebuild.sh all"

tmux attach -t ch17_as1455_rebuild
```

旧版流程使用的 `state/` 不会被新流程信任。新版使用：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/state_v2/
```

已下载的有效缓存文件会继续复用，不会因为使用新状态目录而重新下载。

## 6. 断点续跑

中断后直接重复执行：

```bash
cd /root/stock_realtime_v021_full
bash scripts/run_ch17_as1455_full_rebuild.sh all
```

断点粒度：

- 数据 bootstrap 每次只处理一批缺失股票，重复执行继续补缺；
- 完整训练 fold 直接跳过；
- 不完整训练目录改名为 `.incomplete.<run_stamp>.<time>` 后重跑，不删除；
- 历史回测按目标保存完成标记；
- forward 按目标保存完成标记；
- 全阶段完成标记只在对应脚本成功退出后写入。

查看状态：

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh status
```

## 7. 输出与完成条件

状态与日志：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/state_v2/
saved_data/ashare_ml4t/rebuild_ch17_as1455/logs_v2/
saved_data/ashare_ml4t/rebuild_ch17_as1455/search_logs_v2/
```

最终报告：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/final_report.json
```

正式完成必须满足：

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

## 8. 验证边界

已完成：

- 单入口 Shell 静态语法检查；
- 内存守卫 Shell 静态语法检查；
- 现有大纲、README 与成熟入口调用链核对；
- 删除重复第二层总控和自建并行 history 脚本。

仍需在服务器完成真实验证：

- 1000 股票 BaoStock 全量恢复；
- 40 folds 正式训练；
- 6 组历史回测；
- 6 组 strict-OOS forward；
- 最终审计报告。
