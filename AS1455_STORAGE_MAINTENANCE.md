# AS1455 一键存储检查、诊断与清理

本文档对应：

```text
scripts/run_as1455_storage_maintenance.sh
scripts/run_as1455_cleanup_safe.py
scripts/export_as1455_storage_diagnostics.py
scripts/cleanup_as1455_storage.py
```

一键入口默认只检查和模拟清理，不删除任何文件。只有显式设置 `APPLY=1` 才会执行删除和压缩。

默认正式模式采用保守策略：不删除 hard-coded obsolete 目录，不裁剪历史大网格 run。两类高破坏性操作必须分别显式设置 `INCLUDE_OBSOLETE=1` 和 `PRUNE_GRID_RUNS=1`。

## 1. 一键检查，不删除

在工程根目录执行：

```bash
bash scripts/run_as1455_storage_maintenance.sh
```

默认执行：

1. Python 和 Shell 语法检查；
2. prediction artifact 保留策略合成检查；
3. 清理前磁盘、inode、内存和 Git 状态采集；
4. AS1455 活动进程检查；
5. 两层目录占用统计；
6. 最大文件和文件类型统计；
7. 关键 AS1455 路径存在性检查；
8. 保守清理 dry-run；
9. 生成一个可以直接复制用于分析的 `share_me.txt`。

输出目录：

```text
saved_data/ashare_ml4t/storage_maintenance_YYYYMMDD_HHMMSS/
```

主要文件：

```text
run_config.env
console.log
diagnostics_before.txt
cleanup_dry_run.json
share_me.txt
```

命令结束时会打印：

```text
share_file=/.../storage_maintenance_YYYYMMDD_HHMMSS/share_me.txt
```

需要进一步分析时，只需复制该 `share_me.txt`。它已经包含运行配置、诊断结果、dry-run manifest 和控制台日志尾部，不需要再分别复制多条命令输出。

## 2. 一键正式清理

必须先完成一次默认检查并审核 `cleanup_dry_run.json`。确认删除候选正确后执行：

```bash
APPLY=1 bash scripts/run_as1455_storage_maintenance.sh
```

默认正式模式只处理低风险、可验证或可重建内容：

- 验证 forward `model_data_as1455.h5` 后删除三份重复构建中间 HDF；
- live 只保留最近 3 个日期目录，并删除较旧保留日期中的可重建 history tail；
- 仅在同目录存在权威 prediction HDF 时删除重复 prediction CSV；
- 保留 `actual_<target>.csv`；
- gzip 大于 20 MiB 的 audit/report CSV；
- 不删除 obsolete 目录；
- 不裁剪 grid run 目录。

正式模式会在同一流程中再次生成 dry-run，然后执行 apply，并补充清理后诊断：

```text
cleanup_apply.json
diagnostics_after.txt
share_me.txt
```

`share_me.txt` 会同时包含清理前后磁盘状态和实际执行 manifest。

## 3. 默认参数

```text
KEEP_LIVE_DATES=3
INCLUDE_OBSOLETE=0
PRUNE_GRID_RUNS=0
COMPRESS_REPORTS=1
COMPRESS_MIN_MB=20
SKIP_FORWARD_ARTIFACTS=0
SKIP_LIVE=0
SKIP_PREDICTION_CSV=0
ALLOW_ACTIVE_PROCESSES=0
RUN_FULL_CHECKS=0
```

不会自动删除：

```text
ch12_as1455/baostock_5m_cache/
ch12_as1455/baostock_raw_daily_cache/
ch12_as1455/as1455_daily_cache/
ch12_as1455/model_data_as1455.h5
ch12_as1455/model_data_contract.json
ch12_as1455/as1455_ohlcv_adj.h5
训练 checkpoint
actual_<target>.csv
obsolete 清单目录（默认）
非选中 grid run（默认）
```

## 4. 激进清理必须显式开启

只把 obsolete 目录加入候选：

```bash
INCLUDE_OBSOLETE=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

只把历史/forward grid run 裁剪加入候选：

```bash
PRUNE_GRID_RUNS=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

两者同时加入 dry-run：

```bash
INCLUDE_OBSOLETE=1 \
PRUNE_GRID_RUNS=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

必须先审核新的 `share_me.txt`，再显式执行：

```bash
APPLY=1 \
INCLUDE_OBSOLETE=1 \
PRUNE_GRID_RUNS=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

不得把激进模式作为日常默认命令。

## 5. 跳过某类保守清理

正式清理时跳过 forward 中间文件、live 或 prediction CSV：

```bash
APPLY=1 \
SKIP_FORWARD_ARTIFACTS=1 \
SKIP_LIVE=1 \
SKIP_PREDICTION_CSV=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

关闭大报告压缩：

```bash
COMPRESS_REPORTS=0 \
bash scripts/run_as1455_storage_maintenance.sh
```

## 6. 同时运行完整 AS1455 检查

默认一键存储入口只运行与存储直接相关的轻量检查。需要同时运行完整重构检查时：

```bash
RUN_FULL_CHECKS=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

正式清理并运行完整检查：

```bash
APPLY=1 RUN_FULL_CHECKS=1 \
bash scripts/run_as1455_storage_maintenance.sh
```

## 7. 只导出诊断文件

不运行 cleanup dry-run，只输出服务器存储信息：

```bash
python3 scripts/export_as1455_storage_diagnostics.py \
  --base saved_data/ashare_ml4t \
  --out as1455_storage_diagnostics.txt
```

可调整最大文件数量和目录深度：

```bash
python3 scripts/export_as1455_storage_diagnostics.py \
  --base saved_data/ashare_ml4t \
  --out as1455_storage_diagnostics.txt \
  --top-files 120 \
  --du-depth 3 \
  --du-lines 240
```

诊断器不会读取大型模型内容到内存，只扫描文件元数据并调用受限深度的 `du`。

## 8. 自定义输出目录

```bash
OUT_DIR=saved_data/ashare_ml4t/manual_storage_audit \
bash scripts/run_as1455_storage_maintenance.sh
```

重复使用同一个 `OUT_DIR` 会覆盖 manifest 和诊断文件，因此日常应保留默认时间戳目录。

## 9. 活动进程保护

正式模式仍会检测 AS1455 活动任务。`run_as1455_cleanup_safe.py` 只忽略本次维护脚本自身及其 Bash 日志重定向子进程；其他训练、回测、数据刷新、live 或另一份维护脚本仍会阻止 `--apply`。

人工检查：

```bash
pgrep -af 'as1455|build_ashare_ch12|run_as1455'
```

不建议使用：

```text
ALLOW_ACTIVE_PROCESSES=1
```

它会跳过外部活动进程门禁，只用于已经人工确认进程不会写入清理目录的特殊场景，常规清理必须保持 `0`。

## 10. 推荐操作顺序

```bash
# 1. 更新修复分支
# 当前位于 agent/as1455-storage-oos-fixes 时：
git pull --ff-only origin agent/as1455-storage-oos-fixes

# 2. 完整静态/合成验证
bash scripts/check_ch17_as1455_refactor.sh

# 3. 一键保守审计，不删除
bash scripts/run_as1455_storage_maintenance.sh

# 4. 审核终端打印的 cleanup_dry_run.json 和 share_me.txt

# 5. 正式执行保守清理
APPLY=1 bash scripts/run_as1455_storage_maintenance.sh

# 6. 保存最终 share_me.txt 作为本次清理审计记录
```
