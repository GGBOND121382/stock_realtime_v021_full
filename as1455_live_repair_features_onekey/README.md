# AS1455 live collection repair + feature resume patch

This package fixes the `quality_status=missing_core_fields` false failure caused
by `missing_core_fields` being read back as NaN/`"nan"`, and adds a recovery
entry point that uses already-collected live data to continue to feature
building.

## Install

From repository root:

```bash
unzip -oq as1455_live_repair_features_onekey.zip
bash as1455_live_repair_features_onekey/install.sh --repo .
```

The installer:

- installs `tools/repair_as1455_live_collect_and_features_v1.py`;
- installs `scripts/run_as1455_live_repair_and_features_v1.sh`;
- patches `features/as1455_live_common.py` so future collect/finalize runs do not misclassify NaN `missing_core_fields`;
- backs up overwritten files to `_backup_as1455_live_repair_features_YYYYMMDD_HHMMSS/`;
- runs `python3 -m py_compile` checks.

## Resume today's already-collected data

```bash
TRADE_DATE=20260625 bash scripts/run_as1455_live_repair_and_features_v1.sh
```

This does not re-fetch quotes. It repairs:

- `08_live_raw_row_as1455.csv`
- `08_live_collection_report.csv`
- `08_collection_report.json`

Then it runs feature building to generate:

- `09_live_qfq_row_as1455.csv`
- `10_live_feature_panel_tail.parquet` or `.csv`
- `11_live_model_features.csv`
- `12_feature_build_report.json`

## Options

Only repair collection status, do not build features:

```bash
TRADE_DATE=20260625 NO_FEATURES=1 bash scripts/run_as1455_live_repair_and_features_v1.sh
```

Use explicit live directory:

```bash
LIVE_DIR=saved_data/ashare_ml4t/live_as1455/20260625 \
TRADE_DATE=20260625 \
bash scripts/run_as1455_live_repair_and_features_v1.sh
```

Pass model feature columns if available:

```bash
FEATURE_COLUMNS=saved_data/ashare_ml4t/ch17_as1455_deploy/feature_columns.json \
TRADE_DATE=20260625 \
bash scripts/run_as1455_live_repair_and_features_v1.sh
```

## Expected fixed report

For the uploaded 20260625 live data, `08_collection_report.json` should become approximately:

```json
{
  "valid_panel_rows": 1000,
  "valid_panel_rate": 1.0,
  "collection_passed": true,
  "quality_status_counts": {"ok": 1000}
}
```

Feature pass still depends on TA-Lib being installed and the feature builder
being able to compute at least `MIN_FEATURE_ROWS` usable rows.
