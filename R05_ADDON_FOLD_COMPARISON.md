# AS1455 r05_fwd nested fold experiments

## Standard nested signal-and-trading selection

For each source fold `k = 6..0`, reuse the existing `rotation_addon_onehot + r05_fwd` checkpoint search artifacts, generate predictions on that source fold's own held-out window, search the complete signal and trading grid there, freeze the winner, and apply it once to target fold `k-1`. Source fold 0 is applied to strict forward dates after the fold-0 held-out end.

```bash
OUT_ROOT=saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_nested_v1 \
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison
```

The standard validation grid contains seven signal specifications, five maximum-position values, six sell-rank values, and five rebalance offsets: 1050 configurations per source fold.

## Fixed top-three ensemble experiment

This controlled experiment fixes the signal in every source fold to:

```text
ensemble_first3_mean:0,1,2:mean
```

The three inputs are that source fold's top-three checkpoint prediction columns. Their predictions are averaged with equal weights. The signal is not selected by the trading grid.

Only the trading parameters are searched on each source fold's own held-out validation window:

- `max_positions`: `5,10,15,20,25`
- `sell_rank`: `75,100,150,200,250,300`
- `rebalance_every`: fixed at `5`
- `rebalance_offset`: `0,1,2,3,4`

This gives `5 × 6 × 5 = 150` validation configurations per source fold. The highest validation Sharpe is frozen and applied once to the next target fold. Fold 0 is frozen for strict forward evaluation. Target and forward results never participate in grid selection.

```bash
OUT_ROOT=saved_data/ashare_ml4t/ch17_as1455_nested_fold_protocol/r05_addon_first3_ensemble_nested_v1 \
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-first3-ensemble
```

The experiment reuses existing checkpoints, scalers, feature metadata, model data, and raw daily data. It does not retrain models, refresh market data, or rebuild `model_data`.

## Outputs

Both experiments write the same auditable result structure:

- one `source_fold*` directory per selection fold
- source-fold validation predictions and validation grid
- `selected_for_next_window.json`
- one frozen target or forward run
- `nested_fold_target_results.csv`
- `nested_fold_protocol_manifest.json`
- `continuous_target_folds_plus_forward/`
- `plots/`

To regenerate plots only:

```bash
OUT_ROOT=<completed-nested-result-root> \
bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-plots
```

Use a distinct output root for each experiment. Do not point the fixed-ensemble experiment at a result tree created by the seven-signal grid.
