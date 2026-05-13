# Board prefix fix for v2 all14 pipelines

This patch fixes duplicate columns in `feature_building/build_stock_external_features.py` when several Chinese THS board names are used in one external profile.

Failure symptom:

```text
pandas.errors.MergeError: Passing suffixes which cause duplicate columns ... sp_board_board_* / ocg_board_board_*
```

Run:

```bash
unzip -o stock_external_v2_board_prefix_fix.zip -d .
python3 -m compileall -q feature_building scripts
chmod +x scripts/run_failed3_v2_all14.sh
PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_failed3_v2_all14.sh
```

Only the failed three symbols are rerun: `002518.SZ`, `600522.SH`, `600487.SH`.
