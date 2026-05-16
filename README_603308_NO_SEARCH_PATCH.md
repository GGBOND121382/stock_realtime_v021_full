# 603308 no-search patch

This patch replaces only:

```text
scripts/rebuild_603308_pipeline_safe.sh
```

The new script does not run model search. It only rebuilds data required for training/saving models:

```text
update_data,samples,fundamental,sector,external_aero_nuclear_equipment
```

Then it calls `model_saving/save_nextday_model.py` directly for selected 603308 models.

## Apply

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/big_safe_model_retrain_patch_v2_no_search.zip -d .
bash scripts/apply_603308_no_search_patch.sh
```

## Run

```bash
PYTHON=python3 END_DATE=2026-05-15 bash scripts/rebuild_603308_pipeline_safe.sh
```

## Skip data rebuild and only train from existing samples

```bash
SKIP_DATA_REBUILD=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/rebuild_603308_pipeline_safe.sh
```
