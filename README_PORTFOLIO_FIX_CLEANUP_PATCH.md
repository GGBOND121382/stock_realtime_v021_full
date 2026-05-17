# Portfolio cleanup + model-usage bug fix patch

这版同时做两件事：

```text
1. 把旧的 portfolio 回测输出目录移动到 cleanup_trash，不删除；
2. 修复 portfolio 回测 / adapter / optimizer 的模型使用一致性问题。
```

## 修复点

### A. 回测样本重新计算 pipeline 评价列

`training_samples*.csv` 里不一定已有：

```text
trade_net_close_return
trade_target_or_close_return
entry_signal
```

这些列在 pipeline 搜索阶段是通过 `add_trade_returns(...)` 生成的。因此 backtest 也必须在加载每个 artifact 的 samples 后，用该 artifact 的 metadata 重新计算：

```text
round_trip_cost_bps
target_hit_bps
entry_policy
entry_vwap_premium_bps
```

### B. adapter / optimizer 保留 artifact 使用信息

现在会继续传播：

```text
entry_policy
entry_vwap_premium_bps
samples
expected_return_col
metadata_path
```

这样 `daily_portfolio_orders_*.csv` 和最终 trades 能追溯到具体 artifact 的样本与收益口径。

### C. 一致性审计脚本重新计算 return columns

新增：

```text
scripts/settle_portfolio_backtest_consistently.py
scripts/run_portfolio_consistency_audit.sh
```

它不会假设 sample CSV 已经有收益列，而是按 pipeline 同一函数 `add_trade_returns(...)` 重新计算后再审计。

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_fix_cleanup_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_fix_cleanup_patch.sh
```

默认会把旧目录：

```text
portfolio_reports/backtests/historical_score_portfolio
```

移动到：

```text
cleanup_trash/portfolio_backtest_old_YYYYMMDD_HHMMSS/
```

如果不想清理旧回测目录：

```bash
CLEAN_OLD_BACKTEST=0 PYTHON=python3 bash scripts/apply_portfolio_fix_cleanup_patch.sh
```

## 重新跑回测

```bash
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

## 跑一致性审计

```bash
PYTHON=python3 bash scripts/run_portfolio_consistency_audit.sh
```

严格模式：

```bash
STRICT=1 PYTHON=python3 bash scripts/run_portfolio_consistency_audit.sh
```

## 检查结果

```bash
python3 - <<'PY'
import pandas as pd
p = "portfolio_reports/backtests/historical_score_portfolio/pipeline_consistency_audit/portfolio_trade_pipeline_consistency_audit.csv"
df = pd.read_csv(p)
print(df["status"].value_counts(dropna=False))
print(df["entry_policy_checked"].value_counts(dropna=False))
print(df["label_mode_checked"].value_counts(dropna=False))
cols = [
    "stock_code","model_name","buy_date",
    "entry_policy_checked","label_mode_checked",
    "entry_signal_recomputed","expected_pipeline_return_col",
    "pipeline_return_col","portfolio_realized_return","pipeline_realized_return",
    "return_diff_portfolio_minus_pipeline","status"
]
print(df[[c for c in cols if c in df.columns]].head(80).to_string())
PY
```

可信交易必须满足：

```text
status = OK
entry_signal_recomputed = True
close_profit -> trade_net_close_return
hit -> trade_target_or_close_return
```
