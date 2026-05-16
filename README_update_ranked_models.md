# update_ranked_models_latest patch v2

把本补丁包在项目根目录解压，会新增/替换：

- `scripts/update_ranked_models_latest.sh`
- `model_saving/auto_update_ranked_models.py`
- `patches/realtime_context_ranked_models_append.toml`
- `scripts/apply_ranked_models_data_patch.sh`

v2 新增了数据采集所需配置：

- `600522.SH` / `600487.SH` 的 `optical_cable_grid` 实盘上下文配置，前缀 `ocg_*`
- `002518.SZ` 的 `storage_power` 实盘上下文配置，前缀 `sp_*`
- 将 `scripts/run_trading_day_signal_and_portfolio_all_models.sh` 的默认 `SPOT_SOURCE_PRIORITY` 从 `sina,ths,em,xq` 改成 `sina_batch,ths_etf,xq`

## 一键应用数据采集配置补丁

```bash
unzip update_ranked_models_patch_v2.zip -d /root/stock_realtime_v021_full
cd /root/stock_realtime_v021_full

bash scripts/apply_ranked_models_data_patch.sh
```

脚本会先备份：

```text
saved_data/patch_backups/realtime_context_sources.toml.YYYYMMDD_HHMMSS.bak
saved_data/patch_backups/run_trading_day_signal_and_portfolio_all_models.sh.YYYYMMDD_HHMMSS.bak
```

然后幂等追加 TOML；如果已经追加过，会跳过。

## 一键更新核心有价值模型

```bash
chmod +x scripts/update_ranked_models_latest.sh
PYTHON=python3 END_DATE=2026-05-15 ./scripts/update_ranked_models_latest.sh
```

## 只用已有 pipeline 输出筛选/保存

```bash
SKIP_PIPELINE=1 ./scripts/update_ranked_models_latest.sh
```

## 只看筛选结果，不保存

```bash
DRY_RUN=1 SKIP_PIPELINE=1 ./scripts/update_ranked_models_latest.sh
```

## 输出报告

每次运行会输出到：

```text
saved_data/model_search_queue_logs/auto_ranked_models_YYYYMMDD_HHMMSS/
```

重点看：

- `auto_model_candidates.csv`：所有 leaderboard 模型及拒绝原因
- `auto_model_selected.csv`：最终选中要保存的模型
- `auto_model_save_report.csv`：保存结果和新 metadata 指标
- `auto_model_summary.json`：本次汇总

## 可选参数

```bash
INCLUDE_HIT_AUX=1       # 允许 hit80 辅助模型进入保存清单
INCLUDE_HIGH_DD=1       # 放宽最大回撤约束
MAX_TOTAL=12            # 最多保存多少个 artifact
MAX_PER_STOCK=2         # 每只股票最多保存多少个 artifact
ONLY=603308.SH,600522.SH
```

## 实盘 context plan 检查

```bash
python3 data_collection/collect_realtime_context.py plan \
  --watchlist selected_watchlist.txt \
  --models-dir saved_models \
  --model-policy all \
  --config configs/realtime_context_sources.toml \
  --out-dir saved_data/realtime_context \
  --date $(date +%Y%m%d) \
  --cutoff-time 14:55 \
  --refresh-plan
```

检查：

```text
saved_data/realtime_context/YYYYMMDD/realtime_context_plan.csv
```

确认：

- `600522.SH` / `600487.SH` 的 `context_groups` 包含 `ocg_stocks,ocg_etfs,ocg_futures,ocg_boards`
- `002518.SZ` 如果保存了模型，`context_groups` 包含 `sp_stocks,sp_etfs,sp_futures,sp_boards`
- `missing_context_config_features` 为空

## v3 safety note

`scripts/apply_ranked_models_data_patch.sh` now validates TOML before replacing the real config and refuses to append duplicate TOML tables unless `FORCE_APPEND=1` is explicitly set. This prevents a bad or duplicate TOML append from breaking realtime context collection.

To only validate the current config:

```bash
VALIDATE_ONLY=1 bash scripts/apply_ranked_models_data_patch.sh
```

To apply TOML but keep the old source priority string unchanged:

```bash
SKIP_SOURCE_PATCH=1 bash scripts/apply_ranked_models_data_patch.sh
```
