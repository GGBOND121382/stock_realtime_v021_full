# AS1455 r1/r5/r21 Existing-Model Backtest

Use the existing checkpoints to rerun historical and strict-OOS forward backtests:

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only
```

This command does not update market data, rebuild model data, or train models. It requires the existing historical and forward model-data files and the previously trained fold artifacts.

Default free-space guard:

```text
1 GiB
```

Outputs include six combined forward curves and fold6..fold0 daily/weekly/monthly curves. See `CH17_AS1455_FROM_SCRATCH.md` for the exact inputs, outputs, and safety boundaries.
