# Backtest Point-in-Time Risk History Patch

这版修复历史 portfolio 回测里 optimizer 没有拿到历史收益矩阵的问题。

## 修复点

每个回测交易日 `t`：

```text
1. backtest 脚本已有的 close history 宽表仍然来自 saved model metadata 指向的 samples；
2. 在当天 portfolio_runs/YYYYMMDD/ 下写入：
   risk_history_until_YYYYMMDD.csv
3. 该 CSV 只包含 <= 当天 t 的历史价格，避免未来函数；
4. 调用 daily_portfolio_confirm_pyscipopt.py 时传入：
   --history portfolio_runs/YYYYMMDD/risk_history_until_YYYYMMDD.csv
```

这样 optimizer 才能计算：

```text
returns
vol_daily
correlation pairs
scenario_count
cov_matrix
```

并且在 `USE_COVARIANCE_PENALTY=1` 时，协方差惩罚才会进入目标函数。

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/backtest_point_in_time_risk_history_patch.zip -d .
PYTHON=python3 bash scripts/apply_backtest_point_in_time_risk_history_patch.sh
```

## 回测命令

不启用协方差惩罚，只让相关性/情景/真实波动生效：

```bash
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

启用协方差惩罚：

```bash
USE_COVARIANCE_PENALTY=1 COV_RISK_AVERSION=3 \
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

## 验证

```bash
grep -R '"scenario_count"' portfolio_reports/backtests/historical_score_portfolio/portfolio_runs/*/daily_portfolio_report_*.json | head
grep -R '"use_covariance_penalty"' portfolio_reports/backtests/historical_score_portfolio/portfolio_runs/*/daily_portfolio_report_*.json | head
find portfolio_reports/backtests/historical_score_portfolio/portfolio_runs -name 'risk_history_until_*.csv' | head
```

预期：

```text
scenario_count 不再全是 0
risk_history_until_YYYYMMDD.csv 存在
USE_COVARIANCE_PENALTY=1 时 use_covariance_penalty=true
```
