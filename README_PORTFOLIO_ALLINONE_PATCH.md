# Portfolio All-in-One Patch

这是一份合并版 portfolio 补丁，替代之前分散的这些补丁：

```text
portfolio_optimizer_improvements_patch.zip
portfolio_policy_15pct_10pos_patch.zip
portfolio_policy_15pct_7pos_patch.zip
```

它不包含 new27 pipeline 和 THS 板块名补丁；那两个是新增标的搜索用的，不属于 portfolio optimizer。

## 包含改动

```text
1. portfolio_confirm_from_buy_signals.py
   - 从 configs/realtime_context_sources.toml 补 sector，减少 UNKNOWN。
   - 支持 configs/portfolio_model_overrides.csv。
   - 支持 RECENT_PERF 指向的近期表现降权表。
   - 输出 enabled / weight_multiplier / max_weight_override / max_add_weight_override。

2. daily_portfolio_confirm_pyscipopt.py
   - 支持 enabled=0 禁用候选，并写 rejected reason=disabled_by_override。
   - 支持 weight_multiplier 调整 utility_bps。
   - 支持 max_weight_override / max_add_weight_override。
   - 设置账户级 max_policy_weight=0.15。
   - 设置 max_positions=7。
   - 所有 stock_overrides 仍会被 max_policy_weight=15% 统一压顶。

3. scripts/run_portfolio_confirm_from_signals.sh
   - 自动传入 context-config。
   - 自动传入 model-overrides。
   - 支持 RECENT_PERF=/path/to/recent_perf.csv。

4. configs/portfolio_confirm_config.json
   - 写入 max_policy_weight=0.15。
   - 写入 max_positions=7。

5. configs/portfolio_model_overrides.csv
   - 如果不存在，则从模板创建。
   - 如果已存在，保持不覆盖。

6. backtest_historical_score_portfolio.py
   - 如果该文件存在，额外输出：
     model_perf_summary.csv
     stock_perf_summary.csv
     suggested_portfolio_model_recent_perf.csv
```

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_allinone_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_allinone_patch.sh
```

备份目录：

```text
saved_data/patch_backups/portfolio_allinone_YYYYMMDD_HHMMSS/
```

## 验证

```bash
grep -n '"max_policy_weight"' configs/portfolio_confirm_config.json
grep -n '"max_positions"' configs/portfolio_confirm_config.json
grep -n 'max_policy_weight' portfolio_decision/daily_portfolio_confirm_pyscipopt.py
grep -n 'model-overrides' portfolio_decision/portfolio_confirm_from_buy_signals.py
grep -n 'CONTEXT_CONFIG' scripts/run_portfolio_confirm_from_signals.sh
```

预期：

```text
max_policy_weight = 0.15
max_positions = 7
```

## 模型启停/降权配置

文件：

```text
configs/portfolio_model_overrides.csv
```

格式：

```csv
stock_code,artifact_pattern,enabled,weight_multiplier,max_weight_override,max_add_weight_override,notes
```

示例：

```csv
600312.SH,.*,1,0.60,,,reviewed recent underperformance; reduce weight
601899.SH,.*zijin.*,0,0.00,,,temporarily disabled after review
603308.SH,.*external_full.*extra_trees.*,1,0.50,,,observe only; lower weight
```

## 使用 recent perf

回测后如果生成：

```text
portfolio_reports/backtests/historical_score_portfolio/suggested_portfolio_model_recent_perf.csv
```

可这样启用：

```bash
RECENT_PERF=portfolio_reports/backtests/historical_score_portfolio/suggested_portfolio_model_recent_perf.csv \
PYTHON=python3 bash scripts/run_portfolio_confirm_from_signals.sh
```

注意：`suggested_portfolio_model_recent_perf.csv` 需要人工检查后再用，不建议自动直接实盘启用。
