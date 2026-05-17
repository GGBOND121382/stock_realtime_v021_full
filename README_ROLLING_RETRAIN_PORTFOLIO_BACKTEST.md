# Rolling Retrain Portfolio Backtest

This README documents `scripts/rolling_retrain_portfolio_backtest.py` after the
weekly retrain, validation-window, final-refit, and OOF-threshold updates.

## Purpose

The script simulates a point-in-time portfolio backtest where each saved model
artifact is retrained during the backtest using only labels that would have been
known before the trade date.

For trade date `T`, the latest fully known next-day label is treated as `T-2`,
because `T-1`'s next-day close is only known after `T` closes.

## Current Recommendation

Use scheme A:

- weekly retrain
- `valid_rows=126`
- tail validation threshold
- no final refit with validation

In the 6-stock / 9-artifact test from `2026-01-05` to `2026-05-15`, this was
the best variant:

| Variant | Return | Sharpe | Max DD | Trades | Profit Factor |
|---|---:|---:|---:|---:|---:|
| baseline, valid=252 | 2.52% | 1.87 | -1.88% | 48 | 1.45 |
| scheme A, valid=126 | 2.95% | 2.04 | -2.00% | 32 | 1.77 |
| scheme B, final refit | 1.77% | 0.90 | -3.95% | 80 | 1.15 |
| C-light, OOF threshold | 1.66% | 1.01 | -3.59% | 47 | 1.23 |

## Watchlist Used In Tests

The reproducible 6-stock test watchlist is:

```text
portfolio_reports/backtests/weekly_retrain_6_watchlist.txt
```

It contains:

```text
002270.SZ
002311.SZ
002714.SZ
600312.SH
601899.SH
601985.SH
```

## Recommended Command

PowerShell:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  scripts\rolling_retrain_portfolio_backtest.py `
  --start-date 2026-01-05 `
  --end-date 2026-05-15 `
  --watchlist portfolio_reports\backtests\weekly_retrain_6_watchlist.txt `
  --model-policy all `
  --out-dir portfolio_reports\backtests\weekly_retrain_20260105_schemeA_valid126 `
  --retrain-frequency weekly `
  --threshold-mode tail `
  --initial-cash 200000 `
  --hold-days 1 `
  --min-amount-yuan 50000000 `
  --valid-rows 126 `
  --min-train-entries 80 `
  --min-valid-trades 8 `
  --quantiles 0.5,0.6,0.7,0.8
```

## Saving Models With The Recommended Setting

The rolling backtest does not overwrite `saved_models/`. It retrains temporary
models inside the backtest and writes signals/results only under `--out-dir`.
Existing artifacts under `saved_models/` should therefore be treated as the old
production/library models until they are explicitly refreshed.

To refresh saved artifacts with the current recommendation, use `valid_rows=126`.
The core save scripts now default to `126`, but passing it explicitly is still
recommended for reproducibility.

### Retrain Existing Saved Artifacts

Dry run first:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  model_saving\retrain_existing_models_safe.py `
  --models-dir saved_models `
  --only 002270,002311,002714,600312,601899,601985 `
  --artifact-suffix valid126_20260518 `
  --valid-rows 126 `
  --min-train-entries 80 `
  --min-valid-trades 8 `
  --quantiles 0.5,0.6,0.7,0.8 `
  --dry-run
```

Create new artifacts with a suffix:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  model_saving\retrain_existing_models_safe.py `
  --models-dir saved_models `
  --only 002270,002311,002714,600312,601899,601985 `
  --artifact-suffix valid126_20260518 `
  --valid-rows 126 `
  --min-train-entries 80 `
  --min-valid-trades 8 `
  --quantiles 0.5,0.6,0.7,0.8
```

Replace existing artifacts in place only after reviewing the suffixed artifacts:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  model_saving\retrain_existing_models_safe.py `
  --models-dir saved_models `
  --only 002270,002311,002714,600312,601899,601985 `
  --replace-existing `
  --valid-rows 126 `
  --min-train-entries 80 `
  --min-valid-trades 8 `
  --quantiles 0.5,0.6,0.7,0.8
```

`--replace-existing` moves the old artifact under `cleanup_trash` before writing
the replacement.

### Save One Artifact Manually

Example for the current `601899.SH` Zijin artifact:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  model_saving\save_nextday_model.py `
  --stock-code 601899.SH `
  --artifact-name nextday_vwap_low_close_profit_extra_trees_reversal_fundamental_regime_sector_zijin_v1_valid126 `
  --samples saved_data\601899_pipeline_out\04_external\zijin_external\training_samples_with_zijin_external.csv `
  --intraday-bars saved_data\601899_pipeline_out\00_base\601899_5m.csv `
  --out-dir saved_models `
  --feature-group reversal_fundamental_regime_sector `
  --model-name extra_trees_600_d3 `
  --label-mode close_profit `
  --entry-policy vwap_low `
  --entry-vwap-premium-bps 50 `
  --round-trip-cost-bps 1.7 `
  --target-hit-bps 50 `
  --valid-rows 126 `
  --min-train-entries 80 `
  --min-valid-trades 8 `
  --quantiles 0.5,0.6,0.7,0.8
```

The saved `model.joblib` is intentionally the same model that produced
`validation_tail_predictions.csv` and the saved threshold. It is not refit on
the validation rows, because that breaks score/threshold calibration in the
current tests.

## Core Options

`--retrain-frequency`

- `daily`: train every backtest trading day.
- `weekly`: train on the first backtest trading day of each ISO week and reuse
  the model/threshold for the rest of that week.
- `monthly`: train on the first backtest trading day of each month and reuse
  within the month.

`--valid-rows`

Number of trailing rows used for threshold validation. Default is `252`.
Scheme A uses `126`.

`--use-metadata-valid-rows`

Use each artifact's `valid_rows_for_threshold` from `metadata.json` instead of
the CLI `--valid-rows` value.

`--threshold-mode`

- `tail`: train one threshold model on `train`, score the trailing validation
  window, then choose the threshold from that validation window. This is the
  recommended/default mode.
- `oof`: C-light mode. Split the trailing validation window into chronological
  folds, train only on earlier history for each fold, use OOF validation scores
  to choose the threshold, then train the final model on all known entry rows.

`--oof-folds`

Number of chronological folds for `--threshold-mode oof`. The tested C-light
setting used `3`.

`--final-refit-with-valid`

Scheme B. First choose threshold from the tail validation window, then refit the
final trading model on `train + valid`. This tested worse because the threshold
was calibrated on the pre-refit model but applied to the refit model.

## Training And Threshold Logic

For each artifact and retrain period:

1. Build `history = samples[date <= label_cutoff]`.
2. Drop rows without `trade_net_close_return`.
3. Use the last `valid_rows` rows as validation.
4. Use earlier rows as training.
5. Train only on rows where `entry_signal == True`.
6. Fill missing features with the training median.
7. If the model supports `scale_pos_weight`, set it from train class balance.
8. Score the validation window.
9. Candidate thresholds are validation score quantiles from `--quantiles`.
10. Keep thresholds with at least `--min-valid-trades` selected trades.
11. Select the threshold by highest validation `avg_return`, then
    `profit_factor`.

## Output Files

Each run writes:

```text
<out-dir>/historical_score_portfolio_backtest_summary.json
<out-dir>/historical_score_portfolio_backtest_equity.csv
<out-dir>/historical_score_portfolio_backtest_daily.csv
<out-dir>/historical_score_portfolio_backtest_trades.csv
<out-dir>/historical_score_portfolio_backtest_open_lots.csv
<out-dir>/generated_signals/historical_score_generation_summary.csv
<out-dir>/generated_signals/YYYYMMDD/all_scores.csv
<out-dir>/generated_signals/YYYYMMDD/buy_signals.csv
<out-dir>/generated_signals/YYYYMMDD/rejected_scores.csv
```

Useful signal columns:

- `rolling_label_cutoff`: latest label date used for training.
- `rolling_threshold_mode`: `tail` or `oof`.
- `rolling_train_rows`: rows before validation.
- `rolling_valid_rows`: validation rows.
- `rolling_fit_train_entry_rows`: entry rows used by the threshold model.
- `rolling_final_fit_entry_rows`: entry rows used by the final trading model.
- `threshold`: chosen score threshold.
- `hit_score`: model score for the trade date.
- `signal`: whether the row passed entry, score, and amount filters.

## Reproducing Other Tested Variants

Baseline, valid=252:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  scripts\rolling_retrain_portfolio_backtest.py `
  --start-date 2026-01-05 `
  --end-date 2026-05-15 `
  --watchlist portfolio_reports\backtests\weekly_retrain_6_watchlist.txt `
  --model-policy all `
  --out-dir portfolio_reports\backtests\weekly_retrain_20260105_baseline252 `
  --retrain-frequency weekly `
  --threshold-mode tail `
  --valid-rows 252
```

Scheme B:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  scripts\rolling_retrain_portfolio_backtest.py `
  --start-date 2026-01-05 `
  --end-date 2026-05-15 `
  --watchlist portfolio_reports\backtests\weekly_retrain_6_watchlist.txt `
  --model-policy all `
  --out-dir portfolio_reports\backtests\weekly_retrain_20260105_schemeB_all `
  --retrain-frequency weekly `
  --threshold-mode tail `
  --final-refit-with-valid `
  --valid-rows 252
```

C-light:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe `
  scripts\rolling_retrain_portfolio_backtest.py `
  --start-date 2026-01-05 `
  --end-date 2026-05-15 `
  --watchlist portfolio_reports\backtests\weekly_retrain_6_watchlist.txt `
  --model-policy all `
  --out-dir portfolio_reports\backtests\weekly_retrain_20260105_schemeC_light_oof126 `
  --retrain-frequency weekly `
  --threshold-mode oof `
  --oof-folds 3 `
  --valid-rows 126
```

For exact comparison with the experiments above, include the same common
portfolio settings:

```powershell
--initial-cash 200000 `
--hold-days 1 `
--min-amount-yuan 50000000 `
--min-train-entries 80 `
--min-valid-trades 8 `
--quantiles 0.5,0.6,0.7,0.8
```
