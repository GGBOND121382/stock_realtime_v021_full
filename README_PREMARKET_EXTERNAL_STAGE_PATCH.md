# Premarket external stage patch

This patch updates `pipelines/run_premarket_history_update.py` so external stages
are passed as `external_<profile>` instead of generic `external`.

Run from project root:

```bash
unzip -o premarket_external_stage_patch.zip -d .
python3 scripts/patch_premarket_external_stages.py
python3 -m py_compile pipelines/run_premarket_history_update.py
```

Dry-run first:

```bash
python3 scripts/patch_premarket_external_stages.py --dry-run
```

The patcher creates a backup:

```text
pipelines/run_premarket_history_update.py.bak_premarket_external_stage
```
