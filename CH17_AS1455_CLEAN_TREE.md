# Ch17 AS1455 Clean Tree

This branch retains the runnable Ch17 AS1455 code and now exposes a guarded existing-model backtest workflow.

## Current public workflow

```text
scripts/run_ch17_as1455_full_rebuild.sh backtest-only
└─ scripts/run_ch17_as1455_backtest_only.sh
   ├─ validates existing fold checkpoints without modifying them
   ├─ runs original one-fold-lag historical backtests
   ├─ runs strict-OOS fold0 forward with REFRESH_DATA=0
   ├─ aligns the six forward strategies to one common start date
   └─ plots fold6..fold0 daily/weekly/monthly curves
```

The public entry blocks data refresh, model-data rebuilding, and model training on this branch. The obsolete aligned-fold training and rebuild files have been removed.

## Existing model protocol

```text
r01_fwd: fold0..fold6
r05_fwd: fold0..fold6
r21_fwd: fold0..fold5
```

Historical one-fold-lag mapping remains unchanged:

```text
source fold6 -> target fold5   # available for r1/r5
source fold5 -> target fold4
...
source fold1 -> target fold0
```

The fold6 plot therefore contains the available r1/r5 A/B strategies. Later folds include all strategies that have that source fold.

## Data boundaries

Authoritative caches and model inputs remain under:

```text
saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache/
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/
saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache/
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
```

The backtest-only workflow reads these paths. It does not rebuild them. If the existing forward model-data file is absent, the command fails rather than refreshing data.

## Output policy

Historical grids use summary output and materialize only the selected best run in compact mode. Forward uses compact strict-OOS output. Prediction CSV duplicates are removed after the HDF authority file is written.

The default free-space guard for this workflow is 1 GiB, not 20 GiB. The final report records the actual new bytes created in result roots.

## Validation

```bash
bash scripts/check_ch17_as1455_refactor.sh
```

The checks verify that:

- the original target fold training and one-fold-lag code paths are restored;
- the public entry cannot call an aligned rebuild;
- the backtest-only script contains `REFRESH_DATA=0` and `training=false`;
- Python and shell entry points parse successfully;
- the existing strict-OOS, model selection, artifact retention, and single trading-engine checks remain active.
