# AS1455 weekly rolling retrain + top5 full backtest

This patch adds a rolling retrain workflow for the Sharpe top-5 strategy configurations found in the previous static-prediction 1050-run grid.

## Leakage-safe update rule

For update date `U`, the model is updated after `U` close. Since `r01_fwd` for `U` is not observable until the next trading day, training rows use labels with dates `<= previous_trading_date(U)`. The updated model is used from the next trading day after `U` through the next weekly update date.

## Default model specs

The script keeps the same fixed hyper-parameters as the 20260622_cv7 five prediction columns:

| col | dense_layers | activation | dropout | batch_size | selected epoch | fit epochs |
|---:|---|---|---:|---:|---:|---:|
| 0 | (16, 8) | tanh | 0.2 | 256 | 3 | 4 |
| 1 | (16, 8) | tanh | 0.1 | 256 | 2 | 3 |
| 2 | (32, 16) | tanh | 0.2 | 64 | 18 | 19 |
| 3 | (64, 32) | tanh | 0.1 | 256 | 2 | 3 |
| 4 | (64, 32) | tanh | 0.1 | 256 | 5 | 6 |

`selected epoch` follows the original training script's 0-based checkpoint convention, so fit epochs are `selected_epoch + 1`.

## Install

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_weekly_retrain_top5_full_onekey.zip
bash as1455_weekly_retrain_top5_full_onekey/install.sh --repo .
```

## Smoke run

Run only the first two weekly model updates first:

```bash
MAX_UPDATES=2 bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh
```

## Full run

```bash
bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh
```

Default outputs:

```text
saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_top5_full_<timestamp>/
├── 00_weekly_retrain/results/weekly_predictions.h5
├── 00_weekly_retrain/results/model_update_schedule.csv
├── 00_weekly_retrain/results/weekly_training_log.csv
├── 01_top5_full_backtests/<five full backtest dirs>/
└── 02_summary/top5_weekly_retrain_full_summary.csv
```

## Capacity option

Default is `CAPACITY_MODE=none`, matching the previous 1050-run grid. To add last-5min capacity checks:

```bash
CAPACITY_MODE=last5_both CAPACITY_MISSING_POLICY=reject bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh
```

## Notes

This is much heavier than the earlier grid backtest because it actually retrains neural networks weekly. Keep it in `tmux`.
