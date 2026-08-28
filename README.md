# Ch17 AS1455 Minimal Runtime

This branch contains only the runnable Ch17 AS1455 workflow and its verified dependency closure.

## Scope

The retained workflow is:

```text
historical raw 5m/raw daily/AS1455 daily caches
→ 34-column model_data_as1455.h5
→ r1/r5/r21 A/B parameter search (40 folds)
→ one-fold-lag historical backtests (6 groups)
→ strict-OOS fold0 forward backtests (6 groups)
→ plots, audit, and conservative storage maintenance
```

Trading semantics come only from:

```text
code/backtest/run_as1455_close_auction_backtest_v7_maxpos_grid.py::backtest
```

The repository excludes unrelated individual-stock pipelines, legacy portfolio systems, experiments, notebooks and debug exports. It now retains one scoped AS1455 live/replay path whose model, selection, phase and trading semantics are shared with the clean Ch17 workflow.

## Validate

```bash
python3 -m venv --system-site-packages .venv_as1455
. .venv_as1455/bin/activate
pip install -r requirements.txt
bash scripts/check_ch17_as1455_refactor.sh
bash scripts/run_as1455_live_data_feature_pipeline.sh check
```

## Guarded result workflows

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh independent-folds
bash scripts/run_ch17_as1455_full_rebuild.sh existing-results
```

Model-data rebuild, training, historical Grid and strict-OOS forward each use the dedicated commands in `CH17_AS1455_DEVELOPMENT_OUTLINE.md`.

## Strict-OOS live/replay workflow

```bash
bash scripts/run_as1455_live_strict_oos_pipeline.sh check
```

See `AS1455_LIVE_STRICT_OOS.md`.

Detailed protocol and operating instructions:

- `CH17_AS1455_DEVELOPMENT_OUTLINE.md`
- `README_AS1455_R1_R5_R21.md`
- `CH17_AS1455_FROM_SCRATCH.md`
- `AS1455_STORAGE_AND_STRICT_OOS.md`
- `AS1455_STORAGE_MAINTENANCE.md`
- `CH17_AS1455_CLEAN_TREE.md`
- `AS1455_LIVE_STRICT_OOS.md`
