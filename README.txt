Full replacement for pipelines/run_premarket_history_update.py.

Change:
- SymbolPlan.only_stages() now emits external_<profile> stages.
- Example: --external aero_nuclear_equipment -> --only-stages ...,external_aero_nuclear_equipment.

Apply from project root:
  unzip -o premarket_history_update_full_replacement.zip -d .
  python3 -m py_compile pipelines/run_premarket_history_update.py
