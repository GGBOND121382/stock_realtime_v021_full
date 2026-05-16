# Model retrain + cleanup patch

## Files

```text
scripts/train_preselected_models_strict.sh
scripts/cleanup_saved_models_safe.sh
model_saving/cleanup_saved_models_safe.py
configs/model_cleanup_keep_patterns.txt
```

## Periodic retrain

Default trains only 603308 selected models:

```bash
PYTHON=python3 END_DATE=2026-05-15 bash scripts/train_preselected_models_strict.sh
```

Add vetted new models:

```bash
INCLUDE_VETTED_NEW=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/train_preselected_models_strict.sh
```

Add existing strong models:

```bash
INCLUDE_EXISTING_STRONG=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/train_preselected_models_strict.sh
```

## Cleanup model library

Dry-run first:

```bash
PYTHON=python3 bash scripts/cleanup_saved_models_safe.sh
```

Check report:

```bash
LATEST=$(ls -td saved_data/model_cleanup_logs/cleanup_* | head -1)
cat "$LATEST/model_cleanup_report.csv"
cat "$LATEST/model_cleanup_summary.json"
```

Apply move to cleanup_trash:

```bash
APPLY=1 PYTHON=python3 bash scripts/cleanup_saved_models_safe.sh
```

Artifacts are moved to:

```text
cleanup_trash/saved_models_cleanup_YYYYMMDD_HHMMSS/
```

No artifact is deleted.
