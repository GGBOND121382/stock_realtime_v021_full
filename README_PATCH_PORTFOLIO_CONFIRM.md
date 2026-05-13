# Portfolio Confirm Patch

This patch adds the portfolio confirmation layer after the all-model trading-day signal pipeline.

## Files

```text
portfolio_decision/daily_portfolio_confirm_pyscipopt.py
portfolio_decision/portfolio_confirm_from_buy_signals.py
configs/portfolio_confirm_config.json
configs/account_template.json
scripts/run_portfolio_confirm_from_signals.sh
scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

## Install dependency

```bash
python3 -m pip install -U pandas numpy pyscipopt
```

## Prepare account file

```bash
cp configs/account_template.json account.json
vim account.json
```

`account.json` contains current account state:

```json
{
  "total_asset": 200000,
  "available_cash": 80000,
  "holdings": {
    "600312.SH": {
      "shares": 0,
      "market_value": 0,
      "cost_basis": 0,
      "sector": "电网设备"
    }
  }
}
```

## Run portfolio confirmation after signal pipeline

Assuming the trading-day pipeline already generated:

```text
saved_data/intraday_nextday_signals/YYYYMMDD/buy_signals.csv
```

run:

```bash
chmod +x scripts/run_portfolio_confirm_from_signals.sh
DATE_DASH=2026-05-12 DATE_COMPACT=20260512 \
ACCOUNT=account.json HISTORY=history_close.csv \
PYTHON=python3 ./scripts/run_portfolio_confirm_from_signals.sh
```

Outputs:

```text
portfolio_reports/
  daily_portfolio_orders_YYYY-MM-DD.csv
  daily_portfolio_selected_YYYY-MM-DD.csv
  daily_portfolio_rejected_YYYY-MM-DD.csv
  daily_portfolio_report_YYYY-MM-DD.json
```

## Run signal pipeline + portfolio confirmation together

```bash
chmod +x scripts/run_trading_day_signal_and_portfolio_all_models.sh
ACCOUNT=account.json HISTORY=history_close.csv \
PYTHON=python3 ./scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

Enable covariance penalty:

```bash
USE_COVARIANCE_PENALTY=1 COV_RISK_AVERSION=3.0 \
ACCOUNT=account.json HISTORY=history_close.csv \
PYTHON=python3 ./scripts/run_portfolio_confirm_from_signals.sh
```

## Notes

- The optimizer reads `buy_signals.csv`, not `all_scores.csv`, by default.
- If one stock has multiple model signals, it keeps the highest utility model for that stock before optimization.
- It enforces max 3 final positions by default.
- It uses 100-share lot integer constraints.
- Commission default is 万0.85 不免5.
- It will still run without `history_close.csv`, but volatility/correlation/scenario risk falls back conservatively. Supplying a proper historical close file is recommended.
