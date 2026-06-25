# AS1455 Grid Backtest V7 Models Low-Disk One-Key Patch

This package installs the AS1455 v7 max-position grid backtest with model/signal search and low-disk default outputs.

## Install

Run from the repository root:

```bash
unzip -oq as1455_grid_backtest_v7_models_lowdisk_onekey.zip
bash as1455_grid_backtest_v7_models_lowdisk_onekey/install.sh --repo .
```

The installer backs up any overwritten files to `_backup_as1455_grid_backtest_v7_YYYYMMDD_HHMMSS/` and runs `python3 -m py_compile` on the Python scripts.

## Default grid

Default full run:

```text
7 signals × 5 max_positions × 6 sell_rank × 5 rebalance_every = 1050 runs
```

Signals:

```text
model_0, model_1, model_2, model_3, model_4, ensemble_first3_mean, ensemble_all5_mean
```

## Low-disk output modes

The new default is:

```bash
OUTPUT_MODE=compact
```

Per run, `compact` keeps only:

```text
summary.json
config.json
close_auction_summary.json
close_auction_nav.csv
daily_drawdown.csv
monthly_summary.csv
yearly_summary.csv
fee_summary.csv
turnover_summary.csv
log
```

It suppresses the large middle/audit files:

```text
close_auction_orders.csv
close_auction_trades.csv
close_auction_rejections.csv
close_auction_positions.csv
round_trips.csv
execution_panel_build_report.csv
execution_panel.csv.gz
```

Even in compact mode, the final `02_summary/grid_summary*.csv` and leaderboards still contain the key conclusion metrics: return, Sharpe, Calmar, max drawdown, daily/monthly/trade/round-trip win rates, gross trade amount, total fee, turnover, average positions, orders, rejections, and signal metadata.

Available modes:

```bash
OUTPUT_MODE=summary   # smallest: only JSON per run + final leaderboards
OUTPUT_MODE=compact   # default: JSON + small conclusion CSVs
OUTPUT_MODE=full      # full audit trail, large disk use
```

## Run smoke

```bash
bash scripts/run_as1455_grid_smoke_v7.sh
```

## Run full 1050-grid with low-disk default

```bash
bash scripts/run_as1455_grid_full_v7.sh
```

## Ultra-low disk run

```bash
OUTPUT_MODE=summary bash scripts/run_as1455_grid_full_v7.sh
```

## Full audit rerun for selected candidates

After identifying a top candidate from `02_summary/grid_summary_compact.csv`, rerun only that parameter with full output by calling the single backtest script and setting:

```bash
--output-mode full
```

Do not use `full` for all 1050/3150 runs unless enough disk is available.
