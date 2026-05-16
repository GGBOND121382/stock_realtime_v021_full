# 603308 external_full autodetect patch

This patch replaces:

```text
scripts/train_selected_models_safe.sh
```

The updated script no longer requires manually specifying `EXTERNAL_FULL_SAMPLES` for 603308 by default. It searches under:

```text
saved_data/603308_pipeline_out
```

for external-full sample CSV files.

Apply:

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/model_maintenance_603308_autodetect_patch.zip -d .
```

Use:

```bash
INCLUDE_EXTERNAL_FULL=1 APPLY=1 PYTHON=python3 \
bash scripts/model_library_maintenance_safe.sh train-selected
```

Optional override:

```bash
PIPELINE_603308_ROOT=saved_data/603308_pipeline_out \
INCLUDE_EXTERNAL_FULL=1 APPLY=1 PYTHON=python3 \
bash scripts/model_library_maintenance_safe.sh train-selected
```
