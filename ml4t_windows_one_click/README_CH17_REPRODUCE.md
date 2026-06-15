# ML4T Chapter 17 Reproduction

This document records the end-to-end flow used to reproduce the Chapter 17 neural-network predictions and the local OOS backtest reports in this folder.

## Directory Layout

Run commands from the repository root:

```powershell
cd D:\VSCodeWorkspace\stock_realtime_v021_full
```

Important paths:

```text
ml4t_windows_one_click/
  machine-learning-for-trading/
    data/
      wiki_prices.csv
      wiki_stocks.csv
      us_equities_meta_data.csv
      assets.h5
    12_gradient_boosting_machines/
      data.h5
    17_deep_learning/
      results/
        scores.h5
        test_preds.h5
  run_ml4t_ch17_reproduce.ps1
  run_ml4t_ch17_reproduce_ubuntu.sh
  run_ch17_local_backtest.py
  out/
```

## Data

The scripts reuse local data when it already exists. The minimum raw files for rebuilding `assets.h5` are:

```text
ml4t_windows_one_click/machine-learning-for-trading/data/wiki_prices.csv
ml4t_windows_one_click/machine-learning-for-trading/data/wiki_stocks.csv
ml4t_windows_one_click/machine-learning-for-trading/data/us_equities_meta_data.csv
```

Sources:

```text
wiki_prices.csv              Nasdaq Data Link / Quandl WIKI PRICES export, unzipped and renamed
wiki_stocks.csv              official ML4T repo data file
us_equities_meta_data.csv    official ML4T repo data file
```

Generated market data file:

```text
ml4t_windows_one_click/machine-learning-for-trading/data/assets.h5
```

Required HDF keys:

```text
/quandl/wiki/prices
/quandl/wiki/stocks
/us_equities/stocks
```

To rebuild `assets.h5` from the local CSV files on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ml4t_windows_one_click\run_ml4t_ch17_reproduce.ps1 `
  -ForceAssets `
  -SkipBacktest
```

On Linux/WSL/server:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --force-assets \
  --skip-backtest
```

## Build Chapter 12 Model Dataset

Chapter 17 training uses the Chapter 12 model-data file:

```text
ml4t_windows_one_click/machine-learning-for-trading/12_gradient_boosting_machines/data.h5
```

To force rebuilding it on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ml4t_windows_one_click\run_ml4t_ch17_reproduce.ps1 `
  -ForceChapter12 `
  -SkipBacktest
```

On Linux/WSL/server:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --force-chapter12 \
  --skip-backtest
```

## Train Chapter 17 NN Models

Training executes:

```text
ml4t_windows_one_click/machine-learning-for-trading/17_deep_learning/04_optimizing_a_NN_architecture_for_trading.ipynb
```

Main generated outputs:

```text
ml4t_windows_one_click/machine-learning-for-trading/17_deep_learning/results/scores.h5
ml4t_windows_one_click/machine-learning-for-trading/17_deep_learning/results/test_preds.h5
```

To force model training on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ml4t_windows_one_click\run_ml4t_ch17_reproduce.ps1 `
  -ForceTraining `
  -SkipBacktest
```

On Linux/WSL/server:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --force-training \
  --skip-backtest
```

If `scores.h5` and `test_preds.h5` already exist, the scripts skip training unless the force flag is used.

## Validate Backtest Data

Validate the real data path without running Zipline:

```powershell
py -3 .\ml4t_windows_one_click\validate_ch17_backtest_data.py
```

On Linux/WSL/server:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --backtest-only \
  --preflight-backtest
```

## Local OOS Backtest

The dependency-light local backtest reads:

```text
data/assets.h5::/quandl/wiki/prices
17_deep_learning/results/test_preds.h5::/predictions
```

It does not use Zipline. It computes deterministic one-day open-to-open returns from the prediction signals.

Default safe timing:

```text
signal[t]
-> t+1 open buy
-> t+2 open sell
```

Pure OOS period starts at:

```text
2016-11-30
```

### Reproduce Current Long-Only Top 5 Result

This reproduces:

```text
ml4t_windows_one_click/out/ch17_local_backtest_long_only_L5_S0_leaderboard.csv
ml4t_windows_one_click/out/ch17_local_backtest_long_only_L5_S0_summary.json
```

Run:

```powershell
py -3 .\ml4t_windows_one_click\run_ch17_local_backtest.py `
  --portfolio-mode long_only `
  --n-longs 5 `
  --n-shorts 0 `
  --min-positions 1
```

The same command also writes the explicit execution-model filenames:

```text
ml4t_windows_one_click/out/ch17_local_backtest_long_only_ideal_open_L5_S0_leaderboard.csv
ml4t_windows_one_click/out/ch17_local_backtest_long_only_ideal_open_L5_S0_summary.json
```

The legacy filenames keep the original timing labels, for example `long_only_safe_next_open`.
The explicit `ideal_open` filenames use labels like `long_only_ideal_open_safe_next_open`.

Use `py -3`, not plain `python`, if the machine has Python 2.7 earlier on `PATH`.

### Reproduce Open-Limit Filter Result

This is a stricter A-share-style check:

```text
buy day opens near limit-up   -> skip that long entry
sell day opens near limit-down -> do not force ideal open exit for that symbol
```

Run:

```powershell
py -3 .\ml4t_windows_one_click\run_ch17_local_backtest.py `
  --portfolio-mode long_only `
  --execution-model skip_open_limit `
  --n-longs 5 `
  --n-shorts 0 `
  --min-positions 1 `
  --limit-pct 0.095
```

Output:

```text
ml4t_windows_one_click/out/ch17_local_backtest_long_only_skip_open_limit_L5_S0_leaderboard.csv
ml4t_windows_one_click/out/ch17_local_backtest_long_only_skip_open_limit_L5_S0_summary.json
```

## Zipline Backtest

Zipline is easiest to run on Linux/WSL/server with a compatible Python environment.

Install/use a Zipline environment, then run preflight:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --backtest-only \
  --preflight-backtest \
  --python "/path/to/venv/bin/python"
```

If the local Zipline `quandl` bundle has not been built from `assets.h5`:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --backtest-only \
  --ingest-local-quandl \
  --python "/path/to/venv/bin/python"
```

Then run the Chapter 17 Zipline notebook backtest:

```bash
bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh \
  --backtest-only \
  --python "/path/to/venv/bin/python"
```

## Output Interpretation

The local result files contain all periods and timing variants. For the clean OOS number, filter:

```text
period = oos_from_live
trade_timing = long_only_safe_next_open
```

In the explicit `ideal_open` output file, the equivalent timing label is:

```text
trade_timing = long_only_ideal_open_safe_next_open
```

For the open-limit filter:

```text
period = oos_from_live
trade_timing = long_only_skip_open_limit_safe_next_open
```

Key fields:

```text
annualized_return              annualized return over the selected period
sharpe                         annualized Sharpe using daily returns
max_drawdown                   max drawdown from daily equity curve
average_longs                  average number of long positions actually held
average_blocked_long_entries   average skipped long entries from open limit-up filter
average_blocked_long_exits     average skipped exits from open limit-down filter
```

The ideal-open local backtest assumes every selected stock can trade at the open price. It does not model auction queue priority, partial fills, volume limits, commissions, stamp tax, slippage, or market impact.
