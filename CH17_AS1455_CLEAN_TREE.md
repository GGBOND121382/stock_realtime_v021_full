# Ch17 AS1455 Clean Tree

This document is the allowlist for `agent/ch17-as1455-clean`.

## Retention rule

A file is retained only when it is one of the following:

1. a canonical Ch17 AS1455 protocol or operations document;
2. a runnable entry point named by the protocol;
3. a Python import or shell-call dependency of a retained entry point;
4. the sole v7 trading engine or a helper required by the in-process grid;
5. a validation, disk guard, artifact-retention, or conservative storage-maintenance component;
6. the static 1000-symbol universe or the minimal dependency/configuration files.

## Data path

```text
scripts/run_as1455_live_data_feature_pipeline.sh history
├─ pipelines/as1455_update_history_to_prevday_fast_v4.py
├─ pipelines/as1455_update_history_to_prevday.py
└─ features/as1455_live_common.py

scripts/build_ashare_ch12_as1455_lowmem.sh
└─ scripts/build_ashare_ch12_as1455_model_data.py
```

Only `fast_v4` remains as the active history updater. `fast_v2` and `fast_v3` are removed. The non-fast updater remains only because `fast_v4` imports its BaoStock query, calendar, merge, and schema helpers.

The history controller supports only `history`, `check`, and `status`; live prepare, quote collection, live feature generation, and live inference are outside this repository.

## Training path

```text
scripts/run_as1455_target_search_all.sh
├─ scripts/run_as1455_r05_target_search_all.sh
├─ scripts/run_as1455_r21_target_search_all.sh
└─ scripts/run_as1455_target_fold_param_search.py
   ├─ utils/as1455_ch17_common.py
   ├─ scripts/run_as1455_sector_rotation_fold0_param_search.py
   └─ scripts/run_as1455_first_batch_features_fold0_param_search.py
```

## Historical and forward path

```text
scripts/run_as1455_target_natural_backtest.sh
└─ scripts/run_as1455_target_one_lag_backtest.py

scripts/run_as1455_fold0_forward_backtests.sh
├─ scripts/refresh_as1455_forward_model_data.sh
└─ scripts/run_as1455_fold0_forward_backtest.py
```

These paths use the shared target, forward-feature, model-selection, strict-OOS, rebalance-phase, artifact, CLI, grid, and plotting modules under `utils/`.

## Backtest path

```text
code/backtest/run_as1455_close_auction_grid_inprocess.py
└─ utils/as1455_grid_runner.py
   ├─ code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py
   └─ code/backtest/run_as1455_close_auction_grid_v1.py
```

`run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest` is the only portfolio simulation. `grid_v1.py` is retained solely for its grid configuration, naming, hashing, summary, and leaderboard helper functions imported by the in-process runner; its subprocess grid `main()` is not used by the formal workflow.

## Removed categories

The clean tree excludes:

- individual-stock and next-day prediction pipelines;
- portfolio decision and optimization systems;
- realtime context, live quote collection, live feature and live inference systems;
- old AS1455 updater versions and duplicate/backup files;
- legacy experiments, notebooks, Windows one-click packages and debug exports;
- model-library maintenance for unrelated models;
- patch payloads, replacement trees and historical patch notes;
- generated models, caches, reports and backtest outputs from Git tracking;
- unrelated configuration and account files.

Runtime outputs remain under `saved_data/` on the server and are ignored by Git. The tracked static universe is the only retained file under that directory.

## Validation

```bash
bash scripts/check_ch17_as1455_refactor.sh
bash scripts/run_as1455_live_data_feature_pipeline.sh check
bash scripts/run_ch17_as1455_full_rebuild.sh preflight
```
