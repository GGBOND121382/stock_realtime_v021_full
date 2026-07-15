# Ch17 AS1455 空盘重建

本流程用于服务器上代码、缓存、模型和回测结果全部丢失后的完整重建。

## 约束

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
- 总控脚本不执行仓库级删除、`git clean` 或 `git reset --hard`。

## 从空服务器启动

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

## 查看状态

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

## 中断后续跑

重复执行同一入口即可：

```bash
cd /root/stock_realtime_v021_full
bash scripts/run_ch17_as1455_full_rebuild.sh all
```

已完成阶段会跳过；已完成训练 fold 会逐个跳过。若某个训练 fold 只有不完整输出，其目录会被改名保存为 `.incomplete.<run_stamp>.<time>`，随后重新训练该 fold，不删除原目录。

## 单独执行阶段

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

## 默认参数

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
