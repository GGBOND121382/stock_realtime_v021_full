# Portfolio covariance + live risk history combined patch

这个补丁合并两件事：

```text
1. 修复 USE_COVARIANCE_PENALTY=1 时 PySCIPOpt nonlinear objective 报错；
2. 修复每日实盘 portfolio 阶段没有历史价格矩阵，导致相关性/情景损失/协方差风控不生效的问题。
```

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_cov_live_risk_allinone_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_cov_live_risk_patch.sh
```

## 每日实盘用法

原命令不变：

```bash
PYTHON=python3 bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

或者单独生成 portfolio：

```bash
PYTHON=python3 bash scripts/run_portfolio_confirm_from_signals.sh
```

现在 wrapper 会自动生成：

```text
portfolio_reports/risk_history/risk_history_for_portfolio_YYYYMMDD.csv
```

默认排除当天日期，只使用上一个完整交易日及以前的价格，适合 14:55 实盘决策。

## 关闭自动风险历史

```bash
AUTO_RISK_HISTORY=0 PYTHON=python3 bash scripts/run_portfolio_confirm_from_signals.sh
```

## 启用协方差惩罚

```bash
USE_COVARIANCE_PENALTY=1 COV_RISK_AVERSION=3 \
PYTHON=python3 bash scripts/run_portfolio_confirm_from_signals.sh
```

## 回测

启用协方差惩罚：

```bash
USE_COVARIANCE_PENALTY=1 COV_RISK_AVERSION=3 \
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

不启用协方差惩罚，只验证真实风险输入：

```bash
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

## 验证

```bash
grep -n 'build_portfolio_risk_history.py\|AUTO_RISK_HISTORY\|RISK_HISTORY_DIR' scripts/run_portfolio_confirm_from_signals.sh
grep -n 'cov_linear_penalty_bps\|covariance_penalty_mode\|cov_linear_self_weight' portfolio_decision/daily_portfolio_confirm_pyscipopt.py
grep -n 'amount\[i\].*amount\[j\]\|amount\[j\].*amount\[i\]' portfolio_decision/daily_portfolio_confirm_pyscipopt.py
```

最后一条不应有输出。
