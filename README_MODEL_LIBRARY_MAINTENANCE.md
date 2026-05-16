# Model library maintenance patch v2

## 1. Retrain existing saved model library

Dry run first:

```bash
PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
```

Actually train refreshed artifacts:

```bash
APPLY=1 PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
```

Replace existing artifact names safely. Old artifact dirs are moved to cleanup_trash first:

```bash
APPLY=1 REPLACE_EXISTING=1 PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
```

Limit to selected stocks:

```bash
ONLY=603308.SH,600312.SH APPLY=1 PYTHON=python3 bash scripts/retrain_existing_saved_models_safe.sh
```

## 2. Cleanup weak models

Dry run:

```bash
PYTHON=python3 bash scripts/cleanup_saved_models_safe.sh
```

Apply move to cleanup_trash:

```bash
APPLY=1 PYTHON=python3 bash scripts/cleanup_saved_models_safe.sh
```

No files are deleted by these scripts.
