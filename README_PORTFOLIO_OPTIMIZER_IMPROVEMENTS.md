# Portfolio Optimizer Improvements Patch

这版补丁实现 4 个改进：

```text
1. portfolio_confirm_from_buy_signals.py:
   - 从 configs/realtime_context_sources.toml 补 sector，避免 UNKNOWN。
   - 支持 configs/portfolio_model_overrides.csv。
   - 支持可选 recent_perf CSV，用于模型近期表现降权。

2. daily_portfolio_confirm_pyscipopt.py:
   - 支持 enabled / weight_multiplier。
   - 支持 max_weight_override / max_add_weight_override。
   - disabled 模型会进入 rejected，而不是静默消失。

3. scripts/run_portfolio_confirm_from_signals.sh:
   - 自动传 context-config 和 model-overrides。
   - 可通过 RECENT_PERF=/path/to/recent_perf.csv 传近期表现降权表。

4. backtest_historical_score_portfolio.py:
   - 输出 model_perf_summary.csv / stock_perf_summary.csv。
   - 输出 suggested_portfolio_model_recent_perf.csv，供人工审查后作为 RECENT_PERF 使用。
```

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_optimizer_improvements_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_optimizer_improvements_patch.sh
```

脚本会备份文件到：

```text
saved_data/patch_backups/portfolio_optimizer_improvements_YYYYMMDD_HHMMSS/
```

## 模型启停/降权配置

默认创建：

```text
configs/portfolio_model_overrides.csv
```

格式：

```csv
stock_code,artifact_pattern,enabled,weight_multiplier,max_weight_override,max_add_weight_override,notes
```

示例：

```csv
600312.SH,.*,1,0.60,,,recent backtest underperformed; reduce weight only
601899.SH,.*zijin.*,0,0.00,,,temporarily disabled after review
603308.SH,.*external_full.*extra_trees.*,1,0.50,,,observe only; lower weight
```

说明：

```text
enabled=0               该模型/股票不进入优化器候选
weight_multiplier=0.6   该模型 utility_bps 乘 0.6
max_weight_override     覆盖个股最终权重上限，例如 0.10
max_add_weight_override 覆盖单日新增权重上限，例如 0.05
artifact_pattern        支持正则；* 表示所有 artifact
```

## 近期表现降权

历史回测后会生成：

```text
portfolio_reports/backtests/historical_score_portfolio/
  model_perf_summary.csv
  stock_perf_summary.csv
  suggested_portfolio_model_recent_perf.csv
```

你可以人工检查后，把它作为 live portfolio 的近期表现输入：

```bash
RECENT_PERF=portfolio_reports/backtests/historical_score_portfolio/suggested_portfolio_model_recent_perf.csv \
PYTHON=python3 bash scripts/run_portfolio_confirm_from_signals.sh
```

## 日常运行不变

```bash
PYTHON=python3 bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

只要 `configs/portfolio_model_overrides.csv` 存在，portfolio adapter 会自动读取它。
