# Daily Flow + README Patch

Fixes:

1. `pipelines/run_intraday_nextday_signals.py`
   - `--spot-source-priority` default is changed from `sina,ths,em,xq` to `sina_batch,ths_etf,xq`.

2. `scripts/run_trading_day_signal_and_portfolio_all_models.sh`
   - passes `--date "$DATE_COMPACT"` into `pipelines/run_trading_day_signal_pipeline.py`.

3. `README.md`
   - documents the recommended one-click signal + portfolio flow.
   - documents `DATE_COMPACT` / `DATE_DASH`.
   - replaces stale source examples with `sina_batch,ths_etf,xq`.

Apply:

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/daily_flow_readme_patch.zip -d .
PYTHON=python3 bash scripts/apply_daily_flow_readme_patch.sh
```

Verify:

```bash
grep -n 'spot-source-priority' pipelines/run_intraday_nextday_signals.py
grep -n -- '--date "$DATE_COMPACT"' scripts/run_trading_day_signal_and_portfolio_all_models.sh
grep -n '推荐一键入口：信号 + 组合确认' README.md
```

Recommended daily run:

```bash
PYTHON=python3 bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

Historical/manual rerun:

```bash
DATE_COMPACT=20260515 DATE_DASH=2026-05-15 PYTHON=python3 \
bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```
