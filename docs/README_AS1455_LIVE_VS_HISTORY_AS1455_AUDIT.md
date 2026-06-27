# AS1455 live vs history AS1455 audit

This audit compares same-cutoff AS1455 data, not 15:00 full-day bars.

For a target date such as 20260625, it compares:

- original live 14:55 row: `live_as1455/20260625/08_live_raw_row_as1455.csv`
- history-reconstructed AS1455 row from `ch12_as1455/as1455_daily_cache`
- original live prediction features: `live_as1455/20260625/11_live_model_features_for_prediction.csv`
- reconstructed prediction features produced by reusing the original `06_live_feature_state_fast.npz` and replacing only the day-T AS1455 row.

Run:

```bash
TRADE_DATE=20260625 bash scripts/run_as1455_audit_live_vs_history_as1455_v1.sh
```

Outputs:

```text
saved_data/ashare_ml4t/live_as1455/20260625_audit_history_as1455/
  08_live_raw_row_as1455.csv
  09_live_qfq_row_as1455.csv
  11_live_model_features_for_prediction.csv
  08_raw_as1455_diff_summary.csv
  09_qfq_as1455_diff_summary.csv
  11_feature_diff_summary.csv
  *_top100_each_field.csv
  audit_live_vs_history_as1455_report.json
```
