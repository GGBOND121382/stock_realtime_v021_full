# AS1455 Production Model Generations

## Naming contract

Historical `fold0..fold6` keep their existing meaning and paths. They remain the
Chapter-17 time-series CV splits used by historical research, validation, and
frozen strategy selection.

Production live models use a separate namespace:

- `gen000`: compatibility generation referencing the existing fold0 model bundles;
- `gen001`, `gen002`, ...: rolling production model generations;
- `period000`, `period001`, ...: forward service periods for those generations.

Existing nine-strategy experiment IDs are not renamed, even when their names
contain `fold0`. Those names are stable strategy IDs; model lineage is carried in
separate generation metadata.

## Production flow

A live post run freezes one `13_active_model_snapshot.json` before feature
preparation. r01/r05/r21 inference all read that same snapshot, so one trading day
cannot mix model generations. The nine fixed strategies continue to derive
`all5`, `first3`, and `best` signals from each target's Top-5 prediction panel.

A trading day is appended to the current model period only after the full
nine-strategy planner has completed successfully. Failed post runs do not advance
the 63-day counter.

`gen000` is initialized by intersecting the existing r01/r05/r21 fold0-forward
prediction dates. Its model update date is the first common strict-forward
production date, not fold0's historical `test_end`.

## Rolling update rule

Default period length: 63 successful live trading days.

Default rolling training length: 1008 mature-label trading days per target.

When a period is due, `train_as1455_rolling_generation.py`:

1. reads the current generation's ordered Top-5 model recipes;
2. rebuilds each target's Chapter-17 feature matrix using the existing feature
   contract;
3. selects the latest 1008 dates whose corresponding r01/r05/r21 target label is
   already available;
4. refits the scaler and five neural networks from scratch;
5. does **not** rerun the strategy Grid and does **not** change the nine frozen
   strategy parameters;
6. trains r01/r05/r21 serially into a staging generation;
7. validates the bundles with the same production preprocess/checkpoint readers;
8. activates all three targets atomically only after every target bundle succeeds.

The first successful live day after activation becomes that generation's
`model_updated_date` / `effective_from` date.

## Files

Registry root:

```text
saved_data/ashare_ml4t/ch17_as1455_model_registry/
├── registry.json
├── generations/
│   ├── gen001/
│   │   ├── generation_manifest.json
│   │   ├── r01_fwd/
│   │   ├── r05_fwd/
│   │   └── r21_fwd/
│   └── ...
└── .dashboard/
    ├── rollover_status.json
    └── rollover_*.log
```

Daily live lineage:

```text
saved_data/ashare_ml4t/live_as1455/YYYYMMDD/13_active_model_snapshot.json
```

## Dashboard

Streamlit adds the page `模型与滚动更新`, which displays for all nine strategies:

- current `genNNN` model version;
- model update date;
- model source;
- rolling train start/end when applicable;
- current `periodNNN` progress and remaining trading days;
- generation history and rollover task status.

Historical Fold/Grid pages and data remain unchanged.

## Rollover automation

The optional checker/retrain job is deliberately **not** installed just by pulling
new code. After server smoke validation it can be enabled explicitly:

```bash
sudo env INSTALL_MODEL_ROLLOVER=1 \
  bash scripts/install_as1455_strategy_dashboard_automation.sh
```

Default rollover check time is 21:30 on weekdays. The checker shares the existing
AS1455 heavy-compute lock. When the period is not due it exits quickly. When due,
it refreshes extended model data and then performs the serial three-target refit.

A safe one-time status/legacy initialization check that does not train models is:

```bash
.venv_as1455/bin/python scripts/check_as1455_model_rollover.py \
  --registry-root saved_data/ashare_ml4t/ch17_as1455_model_registry \
  --feature-preset rotation_addon_onehot
```

Do not run `run_as1455_model_rollover_job.sh` merely to inspect status: if the
period is already due, that job is allowed to rebuild data, train, and activate a
new generation.
