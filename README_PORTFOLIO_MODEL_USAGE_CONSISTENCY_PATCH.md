# Portfolio model usage consistency patch

这版统一解决 portfolio 回测与 pipeline 模型定义不一致的问题。

## 代码事实

- `vwap_low` 不是 VWAP 买入，也不是 low 买入。
- `vwap_low` 是 entry universe 过滤：
  `close <= daily_vwap * (1 + entry_vwap_premium_bps / 10000)`。
- `all_days` 是 close 有效即可。
- `close_profit` 的真实评价收益列：`trade_net_close_return`。
- `hit` 的真实评价收益列：`trade_target_or_close_return`。
- 每笔交易必须满足对应 sample row 的 `entry_signal == True`。

## 补丁内容

新增：

```text
scripts/settle_portfolio_backtest_consistently.py
scripts/run_portfolio_consistency_audit.sh
```

修改：

```text
portfolio_decision/portfolio_confirm_from_buy_signals.py
portfolio_decision/daily_portfolio_confirm_pyscipopt.py
```

修改后的 adapter / optimizer 会继续传播：

```text
entry_policy
entry_vwap_premium_bps
samples
expected_return_col
metadata_path
```

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_model_usage_consistency_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_model_usage_consistency_patch.sh
```

## 审计现有回测

```bash
PYTHON=python3 bash scripts/run_portfolio_consistency_audit.sh
```

严格模式：

```bash
STRICT=1 PYTHON=python3 bash scripts/run_portfolio_consistency_audit.sh
```

## 输出

```text
portfolio_reports/backtests/historical_score_portfolio/pipeline_consistency_audit/
  portfolio_trade_pipeline_consistency_audit.csv
  pipeline_consistent_model_perf_summary.csv
  pipeline_consistent_stock_perf_summary.csv
  pipeline_consistency_summary.json
```

## 检查

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

正式可用的交易必须满足：

```text
status = OK
entry_signal_recomputed = True
close_profit -> trade_net_close_return
hit -> trade_target_or_close_return
```
