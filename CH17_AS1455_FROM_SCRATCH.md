# Ch17 AS1455 空盘重建

本流程用于服务器上代码、缓存、模型和回测结果全部丢失后的完整重建。

## 1. 固定约束

- 数据起点固定为 `2020-01-02`。
- 使用仓库中的静态 1000 股票池。
- 5 分钟缓存、原始日线缓存、AS1455 日线缓存均从 BaoStock 重建。
- 模型数据必须为 `[symbol, date]` 双层索引和固定 34 列。
- 训练覆盖：
  - `r01_fwd`：2 个特征方案 × fold0..fold6；
  - `r05_fwd`：2 个特征方案 × fold0..fold6；
  - `r21_fwd`：2 个特征方案 × fold0..fold5；
  - 合计 40 个训练 fold。
- 历史回测覆盖 r1/r5/r21 × A/B，共 6 组；大网格默认只保存 summary，并物化 Sharpe 最佳 run。
- fold0-forward 覆盖 r1/r5/r21 × A/B，共 6 组；正式口径为 `strict_oos`。
- 总控脚本不执行仓库级删除、`git clean`、`git reset --hard` 或 Linux `drop_caches`。

## 2. 实际调用链

总控只负责编排、断点标记、输出验证和内存门禁。训练、特征、历史预测、交易网格与 strict-OOS 业务逻辑继续使用仓库现有实现。

```text
scripts/run_ch17_as1455_full_rebuild.sh
└─ scripts/rebuild_ch17_as1455_from_scratch.sh
   ├─ pipelines/as1455_update_history_to_prevday_fast_v4.py
   ├─ scripts/build_ashare_ch12_as1455_model_data.py
   ├─ scripts/check_ch17_as1455_refactor.sh
   ├─ scripts/run_as1455_target_fold_param_search.py
   │  └─ utils/as1455_ch17_common.py
   ├─ scripts/run_as1455_target_natural_backtest.sh
   │  └─ scripts/run_as1455_target_one_lag_backtest.py
   │     └─ utils/as1455_ch17_common.py
   │        └─ utils/as1455_grid_runner.py
   │           └─ code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
   └─ scripts/run_as1455_fold0_forward_backtests.sh
      └─ scripts/run_as1455_fold0_forward_backtest.py
         ├─ utils/as1455_forward_features.py
         ├─ utils/as1455_model_selection.py
         ├─ utils/as1455_rebalance_phase.py
         └─ utils/as1455_strict_oos.py
```

其中：

- 目标映射、A/B 特征、fold、checkpoint、scaler 和 prediction artifact 继续由 `utils/as1455_ch17_common.py` 统一实现；
- forward 最新无标签行保留继续由 `utils/as1455_forward_features.py` 实现；
- 历史完整配置选择继续由 `utils/as1455_model_selection.py` 实现；
- 调仓相位换算继续由 `utils/as1455_rebalance_phase.py` 实现；
- strict-OOS 单配置冻结继续由 `utils/as1455_strict_oos.py` 实现；
- 交易只调用大纲指定的唯一 v7 `backtest()`，总控没有复制第二套训练或交易循环。

历史缓存阶段使用 `pipelines/as1455_update_history_to_prevday_fast_v4.py`，因为它就是现有 `scripts/run_as1455_live_data_feature_pipeline.sh history` 背后的实际历史更新实现，并负责三类缓存的断点续传。

## 3. 内存隔离与冷却

所有重任务均作为独立 Python 进程运行。单个训练 fold、历史预测/回测、forward 回测或模型数据构建结束后，进程退出，Linux 会回收该进程的私有 RSS、TensorFlow allocator 和工作线程。

`scripts/as1455_python_memory_guard.sh` 在不修改业务脚本的前提下统一执行：

1. 重任务启动前检查 `/proc/meminfo` 的 `MemAvailable`；
2. 低于阈值时按固定间隔等待，达到最大次数仍未恢复则停止；
3. 任务结束后记录 `MemAvailable`、Swap 和 `free -h`；
4. 训练任务默认冷却 20 秒；
5. 历史/forward 回测默认冷却 20 秒；
6. 数据下载与模型数据构建默认冷却 30 秒；
7. 冷却后再次检查 `MemAvailable`，通过后才启动下一项任务。

默认参数：

```text
CPU_THREADS=2
MIN_AVAILABLE_MEMORY_MB=1024
MEMORY_WAIT_ATTEMPTS=30
MEMORY_WAIT_SECONDS=10
TRAIN_COOLDOWN_SECONDS=20
BACKTEST_COOLDOWN_SECONDS=20
DATA_COOLDOWN_SECONDS=30
```

同时限制：

```text
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
TF_NUM_INTRAOP_THREADS=2
TF_NUM_INTEROP_THREADS=1
MALLOC_ARENA_MAX=2
```

不执行 `echo 3 > /proc/sys/vm/drop_caches`，因为页缓存属于可回收内存，强制清空会增加后续读取和重建成本。也不在一个无关的新 Python 进程里调用 `gc.collect()`，因为它无法清理已经退出的训练进程；进程退出本身才是可靠的内存隔离边界。

可以提高冷却和门禁，例如：

```bash
TRAIN_COOLDOWN_SECONDS=40 \
BACKTEST_COOLDOWN_SECONDS=30 \
MIN_AVAILABLE_MEMORY_MB=1536 \
  bash scripts/run_ch17_as1455_full_rebuild.sh all
```

## 4. 从空服务器启动

```bash
cd /root

git clone \
  --branch agent/rebuild-ch17-as1455-from-scratch \
  --single-branch \
  https://github.com/GGBOND121382/stock_realtime_v021_full.git \
  stock_realtime_v021_full

cd /root/stock_realtime_v021_full

tmux new-session -d -s ch17_as1455_rebuild \
  "cd /root/stock_realtime_v021_full && bash scripts/run_ch17_as1455_full_rebuild.sh all"
```

查看实时输出：

```bash
tmux attach -t ch17_as1455_rebuild
```

脱离 tmux 而不中止任务：按 `Ctrl-b`，再按 `d`。

## 5. 查看状态

```bash
cd /root/stock_realtime_v021_full
bash scripts/run_ch17_as1455_full_rebuild.sh status
```

阶段包括：

```text
preflight
history
model_data
selfcheck
training
historical_backtests
fold0_forward
final_audit
```

完成标记位于：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/state/
```

日志位于：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/logs/
```

最终审计报告：

```text
saved_data/ashare_ml4t/rebuild_ch17_as1455/final_report.json
```

完成时必须出现：

```text
[PASS] full Ch17 AS1455 rebuild completed
```

并满足：

```text
training_ok = 40
historical_ok = 6
forward_ok = 6
all_ok = true
```

## 6. 中断后续跑

重复执行同一入口即可：

```bash
cd /root/stock_realtime_v021_full
bash scripts/run_ch17_as1455_full_rebuild.sh all
```

已完成阶段会跳过；已完成训练 fold 会逐个跳过。若某个训练 fold 只有不完整输出，其目录会被改名保存为 `.incomplete.<run_stamp>.<time>`，随后重新训练该 fold，不删除原目录。

## 7. 单独执行阶段

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh preflight
bash scripts/run_ch17_as1455_full_rebuild.sh history
bash scripts/run_ch17_as1455_full_rebuild.sh model_data
bash scripts/run_ch17_as1455_full_rebuild.sh selfcheck
bash scripts/run_ch17_as1455_full_rebuild.sh training
bash scripts/run_ch17_as1455_full_rebuild.sh historical
bash scripts/run_ch17_as1455_full_rebuild.sh forward
bash scripts/run_ch17_as1455_full_rebuild.sh audit
```

正常情况下无需手工逐阶段执行，使用 `all` 即可。

## 8. 其他默认参数

```text
HISTORY_START_DATE=2020-01-02
HISTORY_END_DATE=auto
TRADE_DATE=today
EPOCHS=20
BEST_N=5
SEED=42
MIN_INITIAL_FREE_GB=20
MIN_FREE_GB=5
FEATURE_PRESETS="rotation_onehot rotation_addon_onehot"
TARGETS="r01_fwd r05_fwd r21_fwd"
```

依赖优先通过 `.venv_as1455` 的 `--system-site-packages` 复用服务器现有 Python 包；缺失时才通过 pip 安装。
