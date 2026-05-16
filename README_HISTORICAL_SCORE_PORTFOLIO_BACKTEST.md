# Historical Score → Portfolio Backtest Patch

这版脚本以**历史数据**为输入，不再依赖已经存在的每日 `buy_signals.csv`。

它会：

```text
saved_models + saved model metadata 指向的 historical samples
  -> 每个历史交易日生成 all_scores / buy_signals / rejected_scores
  -> 调用现有 portfolio_confirm_from_buy_signals.py
  -> 调用现有 daily_portfolio_confirm_pyscipopt.py
  -> 模拟买入、持有、卖出
  -> 输出策略评价
```

## 安装

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/historical_score_portfolio_backtest_patch.zip -d .
```

新增文件：

```text
portfolio_decision/backtest_historical_score_portfolio.py
scripts/backtest_historical_score_portfolio.sh
README_HISTORICAL_SCORE_PORTFOLIO_BACKTEST.md
```

## 默认运行

```bash
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

默认参数：

```text
MODELS_DIR=saved_models
HISTORY=history_close.csv
CONFIG=configs/portfolio_confirm_config.json
OUT_DIR=portfolio_reports/backtests/historical_score_portfolio
MODEL_POLICY=all
INITIAL_CASH=200000
HOLD_DAYS=1
MIN_AMOUNT_YUAN=50000000
```

## 指定区间

```bash
START_DATE=2026-05-01 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

## 只生成每日历史打分，不跑 portfolio

```bash
SCORE_ONLY=1 START_DATE=2026-05-01 END_DATE=2026-05-15 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

生成位置：

```text
portfolio_reports/backtests/historical_score_portfolio/generated_signals/YYYYMMDD/
  all_scores.csv
  buy_signals.csv
  rejected_scores.csv
  run_summary.json
```

## 完整回测输出

```text
portfolio_reports/backtests/historical_score_portfolio/
  historical_score_portfolio_backtest_summary.json
  historical_score_portfolio_backtest_equity.csv
  historical_score_portfolio_backtest_daily.csv
  historical_score_portfolio_backtest_trades.csv
  historical_score_portfolio_backtest_open_lots.csv
  generated_signals/YYYYMMDD/
  portfolio_runs/YYYYMMDD/
```

## 评价指标

`historical_score_portfolio_backtest_summary.json` 包括：

```text
total_return
annualized_return
annualized_volatility
sharpe_rf0
max_drawdown
realized_trades
win_rate
profit_factor
avg_trade_return
median_trade_return
avg_gross_exposure
open_lots_at_end
```

## 注意

这个脚本是“当前 saved_models 模型库在历史样本上的 replay”。如果 saved model 是用完整历史训练出来的，这不是严格 walk-forward 研究回测。它适合验证：

```text
每日历史打分逻辑
portfolio optimizer 约束
资金滚动与交易成本
最终投资组合表现
```

如果要严格无未来信息，需要准备每个历史日期之前训练好的 point-in-time artifact，再用这个脚本对那些 artifact 回放。
