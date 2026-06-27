# AS1455 checkpoint signal hardcheck v3

This patch hardens checkpoint live inference compatibility checks.

Changes:

1. `model_data_as1455.h5` is validated by required column names, then training `X` is explicitly reconstructed as the 31 feature columns.
2. Live `date` accepts both `YYYY-MM-DD` and compact `YYYYMMDD`; integer `20260626` is not interpreted as a nanosecond timestamp.
3. Manifest `feature_columns` must exactly match the 31 live/training feature columns.
4. Reconstructed CV splits are compared against `DEPLOY_DIR/cv_split_report.csv`; mismatch fails before inference.
5. Checkpoint count must equal `n_models * n_active_folds`.
6. `run_as1455_live_checkpoint_signal_v1.sh` recreates deploy manifest by default (`RECREATE_DEPLOY=1`) so changes to `MODEL_ROWS/FOLDS/FOLD_MODE` cannot be hidden by a stale manifest.

Install:

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_checkpoint_signal_hardcheck_v3.zip
bash as1455_checkpoint_signal_hardcheck_v3/install.sh --repo .
```

Run preflight:

```bash
TRADE_DATE=20260626 DRY_RUN=1 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

Use `RECREATE_DEPLOY=0` only if you deliberately want to reuse the existing manifest.
