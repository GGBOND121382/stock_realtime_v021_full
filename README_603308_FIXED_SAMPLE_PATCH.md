# 603308 fixed sample path patch

This replaces:

```text
scripts/train_selected_models_safe.sh
```

The script now uses the actual 603308 external sample file directly:

```text
saved_data/603308_pipeline_out/04_external/aero_nuclear_equipment/training_samples_with_aero_nuclear_equipment_external.csv
```

Apply:

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/model_maintenance_603308_fixed_sample_patch.zip -d .
```

Run 603308 external_full candidates:

```bash
INCLUDE_EXTERNAL_FULL=1 APPLY=1 PYTHON=python3 \
bash scripts/model_library_maintenance_safe.sh train-selected
```

Dry run:

```bash
INCLUDE_EXTERNAL_FULL=1 PYTHON=python3 \
bash scripts/model_library_maintenance_safe.sh train-selected
```
