# saved_data asof1455 scale quick validation

Scope:
- Scanned `saved_data/**/pipeline_summary.json`.
- Rebuilt/normalized features in memory with the new `asof1455` valuation policy.
- Compared training feature rows against synthetic 14:55 watch rows for the most recent ~5 available sample dates per completed pipeline.

Coverage:
- Verifiable rows: 25 stock-date rows.
- Verifiable stocks: `600312.SH`, `600487.SH`, `603308.SH`.
- Skipped pipeline directories: 56, because the local workspace has only summary metadata and does not contain the referenced `training_samples*.csv` and/or `*_5m.csv` files.

Outputs:
- `summary.csv`
- `feature_diffs.csv`
- `category_summary.csv`
- `input_report.csv`

Result:

| stock | days | max features | min exactish share | max p90 rel diff | max rel diff | missing live features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 600312.SH | 10 | 96 | 100% | 0 | 4.3e-13 | 0 |
| 600487.SH | 5 | 123 | 100% | 0 | 0 | 0 |
| 603308.SH | 10 | 164 | 100% | 0 | 2.6e-14 | 0 |

Category result:

| category | result |
| --- | --- |
| stock_asof_raw | exact |
| stock_asof_derived | floating point noise only |
| fundamental | exact |
| lagged_daily_external | exact |

Interpretation:

For completed local samples, the new 14:55 training feature scale and the watch reconstruction path are aligned. Remaining nonzero values are floating point noise.

The skipped directories need their sample CSVs and 5m CSVs restored or regenerated before the same validation can cover every historical pipeline under `saved_data`.
