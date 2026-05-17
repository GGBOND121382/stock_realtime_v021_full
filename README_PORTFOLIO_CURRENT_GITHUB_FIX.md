# Portfolio current GitHub fix

This patch is based on the current synced GitHub state.

It does not patch `backtest_historical_score_portfolio.py`; it only validates it, because that file already contains:
- `add_trade_returns(...)`
- `OpenLot` / `TradeRecord` propagated fields
- fixed rejected-score mask

It patches:
- `portfolio_confirm_from_buy_signals.py`: replaces `build_inputs()` so `portfolio_signals.csv` and `portfolio_metrics.csv` both propagate `entry_policy`, `entry_vwap_premium_bps`, `samples`, `expected_return_col`, and `metadata_path`.
- `daily_portfolio_confirm_pyscipopt.py`: adds missing `as_text()` used by `build_candidates()`.

Apply:

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_current_github_fix_v2_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_current_github_fix.sh
```

Success marker:

```text
[OK] portfolio current GitHub fix applied and self-tested
```

The script self-tests:
- `py_compile` for adapter/optimizer/backtest
- backtest markers and dataclass order
- synthetic adapter propagation test
- synthetic optimizer candidate propagation test


v2 change: fixed project-root import path before `import portfolio_decision...`.
