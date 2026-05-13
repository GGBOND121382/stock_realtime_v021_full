# REAL full replacement for pipelines/run_premarket_history_update.py

This zip contains the actual full file, not the patcher script.

Install from project root:

```bash
unzip -o premarket_history_update_REAL_full_replacement.zip -d .
python3 -m py_compile pipelines/run_premarket_history_update.py
python3 pipelines/run_premarket_history_update.py --help | grep -E "models-dir|saved-data-dir|context-config|cache-mode|feature-cache-mode|keep-going"
```

Then run:

```bash
python3 pipelines/run_premarket_history_update.py \
  --models-dir saved_models \
  --saved-data-dir saved_data \
  --context-config configs/realtime_context_sources.toml \
  --end-date today \
  --cache-mode incremental \
  --feature-cache-mode incremental \
  --keep-going
```
