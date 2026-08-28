# Ch17 AS1455 Clean Tree

This branch retains the runnable Ch17 AS1455 code and exposes a guarded independent-fold backtest workflow.

## Current public workflow

```text
scripts/run_ch17_as1455_full_rebuild.sh independent-folds
└─ scripts/run_ch17_as1455_backtest_only.sh
   ├─ resolves six complete historical/strict-OOS result pairs
   ├─ loads the already-selected signal and retained run config
   ├─ loads existing historical and forward prediction HDF files
   ├─ builds one shared raw-daily execution panel
   ├─ runs 40 independent single-configuration fold backtests
   ├─ starts every fold from equal cash and empty positions
   ├─ translates the retained rebalance phase to each fold-local calendar
   └─ plots fold6..fold0 daily/weekly/monthly curves
```

The workflow explicitly does not train models, generate predictions, rebuild model_data, refresh data, or run a parameter grid.

## Separate plot-only compatibility workflow

```text
scripts/run_ch17_as1455_full_rebuild.sh existing-results
└─ scripts/run_ch17_as1455_existing_results.sh
   └─ slices and plots already-existing continuous NAV results
```

The plot-only mode is intentionally separate because it does not reset positions and cash at each fold boundary.

## Existing fold protocol

```text
r01_fwd: fold0..fold6
r05_fwd: fold0..fold6
r21_fwd: fold0..fold5
```

Independent backtest count:

```text
r01 historical = 12
r05 historical = 12
r21 historical = 10
fold0 forward = 6
Total = 40
```

fold6 contains the available r1/r5 A/B strategies. fold5..fold0 contain six strategies.

## Phase semantics

Each stored config contains an offset relative to its original continuous OOS calendar. After cropping to an independent fold, the local offset is translated as:

```text
(original_offset - skipped_overlap_dates) mod rebalance_every
```

This preserves the selected rebalance phase while still resetting the portfolio to empty positions and equal initial cash.

## Data boundaries

Read-only inputs remain under:

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/
saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/   # only if a frozen config uses capacity
```

Training checkpoint, scaler, feature and model-data directories are not modified.

## Output policy

```text
saved_data/ashare_ml4t/ch17_as1455_independent_folds/<timestamp>/
```

Only compact results for 40 selected configurations are written, plus 21 plots and manifests. No grid run directories are produced. The default free-space guard is 1 GiB.

## Validation

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

The checks verify that:

- the independent entry expects exactly 40 backtests;
- each fold starts from empty positions and equal initial cash;
- the runner calls the v7 portfolio engine directly;
- the retained phase is translated rather than reset;
- no grid, training, prediction-generation, materialization or refresh workflow is reachable from the independent entry;
- the old continuous-NAV plot-only mode remains separate.

## Restored strict-OOS live workflow

The clean tree now also contains a scoped live/replay workflow:

```text
scripts/run_as1455_live_strict_oos_pipeline.sh
├─ T-1 clean history updater
├─ restored preclose / adjustment / raw-daily execution calendar
├─ <=14:55 collection and current-day 31-base-feature fast path
├─ clean rotation/addon/one-hot feature construction
├─ clean fold0 .keras + saved preprocessing
├─ historical best full-run selection and phase continuation
└─ canonical v7 single-day mode with explicit account cash/positions
```

It produces planned, reviewable orders only. It does not connect to a broker and does not persist planned fills as account truth. See `AS1455_LIVE_STRICT_OOS.md`.
