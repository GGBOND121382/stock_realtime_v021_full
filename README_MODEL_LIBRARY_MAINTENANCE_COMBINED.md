# Combined Model Library Maintenance Patch

## Install

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/model_library_maintenance_combined_patch.zip -d .
```

Installed files:

```text
scripts/model_library_maintenance_safe.sh
scripts/train_selected_models_safe.sh
model_saving/retrain_existing_models_safe.py
model_saving/prune_saved_models_keep_good.py
model_saving/restore_keep_good_models_from_trash.py
model_saving/inspect_saved_model_features.py
configs/saved_models_keep_good_families.csv
```

No script deletes files. Cleanup and refresh replacement use `cleanup_trash/`.

---

## 1. Inspect current model dependencies

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh inspect
```

For only 603308:

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh inspect --only 603308.SH
```

Report:

```text
saved_data/model_update_logs/model_feature_inspect/saved_models_feature_dependencies.csv
```

This shows whether a model uses board features.

---

## 2. Restore good models moved earlier

Dry-run:

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh restore-preview
```

Apply:

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh restore-apply
```

---

## 3. Strict cleanup: keep only approved model families

Dry-run first:

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh cleanup-preview
```

Check:

```bash
LATEST=$(ls -td saved_data/model_cleanup_logs/strict_keep_good_* | head -1)
cat "$LATEST/strict_keep_good_summary.json"
cat "$LATEST/strict_keep_good_report.csv"
```

Apply only after checking:

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh cleanup-apply
```

Moved artifacts go to:

```text
cleanup_trash/strict_keep_good_YYYYMMDD_HHMMSS/
```

Keep rules are in:

```text
configs/saved_models_keep_good_families.csv
```

---

## 4. Periodic retrain existing saved model library

After you update data yourself, retrain existing artifacts from their own metadata.

Dry-run:

```bash
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh retrain-existing
```

Recommended actual run: replace existing artifact names safely. Old artifact directories move to `cleanup_trash/model_refresh_backup_*` first.

```bash
APPLY=1 REPLACE_EXISTING=1 PYTHON=python3 bash scripts/model_library_maintenance_safe.sh retrain-existing
```

Only selected stocks:

```bash
ONLY=603308.SH,600312.SH APPLY=1 REPLACE_EXISTING=1 PYTHON=python3 \
  bash scripts/model_library_maintenance_safe.sh retrain-existing
```

If you do not want replacement and prefer side-by-side refreshed models:

```bash
APPLY=1 ARTIFACT_SUFFIX=refresh_20260515 PYTHON=python3 \
  bash scripts/model_library_maintenance_safe.sh retrain-existing
```

---

## 5. Optional selected new models

Default `train-selected` trains nothing unless an include flag is set.

Vetted 600522/600487:

```bash
INCLUDE_VETTED_NEW=1 APPLY=1 PYTHON=python3 \
  bash scripts/model_library_maintenance_safe.sh train-selected
```

External-full 603308 requires a concrete sample CSV path:

```bash
INCLUDE_EXTERNAL_FULL=1 \
EXTERNAL_FULL_SAMPLES=/root/stock_realtime_v021_full/saved_data/603308_pipeline_out/.../your_external_full_samples.csv \
APPLY=1 PYTHON=python3 \
bash scripts/model_library_maintenance_safe.sh train-selected
```

Do not enable external-full unless the sample file is confirmed.

---

## Suggested workflow

```bash
cd /root/stock_realtime_v021_full

# 1. Inspect current model library
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh inspect

# 2. Restore any allowlisted good model that was moved earlier
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh restore-preview
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh restore-apply

# 3. Strict cleanup preview, then apply after reading report
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh cleanup-preview
LATEST=$(ls -td saved_data/model_cleanup_logs/strict_keep_good_* | head -1)
cat "$LATEST/strict_keep_good_summary.json"
cat "$LATEST/strict_keep_good_report.csv"
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh cleanup-apply

# 4. After you update latest data, refresh existing saved model library
PYTHON=python3 bash scripts/model_library_maintenance_safe.sh retrain-existing
APPLY=1 REPLACE_EXISTING=1 PYTHON=python3 bash scripts/model_library_maintenance_safe.sh retrain-existing
```
