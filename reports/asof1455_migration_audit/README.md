# asof1455 Migration Audit

Generated on 2026-05-24.

## Environment

- Created project-local `.venv`.
- Installed `requirements.txt`, `requirements_server.txt`, and `requirements/data88_extract_tool_requirements.txt`.
- Added audit tooling:
  - `tools/audit_model_required_features.py`
  - `tools/run_asof1455_migration_audit.py`

## Data Rebuilt For Spot Checks

Only near-month data was rebuilt, under `saved_data/asof1455_audit_minidata/`, without overwriting original pipeline directories.

- `603308.SH`, `aero_nuclear_equipment`, 2026-04-01 to 2026-05-21
- `600487.SH`, `optical_cable_grid`, 2026-04-01 to 2026-05-21
- `600312.SH`, sector-only, 2026-04-01 to 2026-05-21

All three spot-check pipelines were run only through data/sample/feature stages. No model search or retraining was run.

## Key Findings

1. Saved model training samples are missing locally.
   - 18 saved model metadata files found.
   - 18/18 metadata `samples` paths do not exist in this workspace.
   - 54 `final_leaderboard.csv` files found, 10,320 leaderboard rows.
   - 10,320/10,320 leaderboard `sample_file` paths do not exist locally.

2. High-risk as-of fields are concentrated in external models.
   - 18 saved models, 4,790 trained feature dependencies.
   - 5 models use lagged futures/US external fields.
   - 5 models have high-risk as-of feature ratio above 20%.
   - Highest-risk models:
     - `603308.SH` external extra_trees: 267/410 high-risk, 99 lagged daily.
     - `603308.SH` external xgb: 277/547 high-risk, 99 lagged daily.
     - `603308.SH` sector/external xgb: 283/570 high-risk, 99 lagged daily.
     - `600522.SH` optical_cable_grid: 296/614 high-risk, 70 lagged daily.
     - `600487.SH` optical_cable_grid: 296/616 high-risk, 70 lagged daily.

3. Representative current-day sample rows can provide lagged daily fields.
   - In rebuilt near-month samples, target date `2026-05-21` exists.
   - `603308` external sample: 92 lagged futures/US columns, 92 non-null on T.
   - `600487` external sample: 65 lagged futures/US columns, 65 non-null on T.
   - This supports using `samples[date=T]` for lagged futures/US fields, with hard reject if absent.

4. Realtime dependency classification is correct for lagged daily fields after inspection.
   - `ane_fut_ni0_close`, `ane_future_basket_close_ret1`, and `ane_stock_vs_future_basket_ret20` classify as lagged daily.
   - They are excluded from realtime context dependencies.
   - Domestic prefixed raw fields such as `ane_stk_600893_close` remain realtime/context dependencies.

5. Label/entry policy is still EOD-close based.
   - Current training return logic uses `next_day_close / close - 1 - cost`.
   - Current hit logic uses `next_day_high / close - 1 - cost`.
   - Current `vwap_low` entry uses `close <= daily_vwap * premium`.
   - Rebuilt sample files do not contain `close_asof1455` or `vwap_asof1455`.
   - Therefore as-of training needs explicit new entry/label construction.

6. Reconstruction audit data is absent.
   - No `saved_data/feature_reconstruction_audit/**/model_feature_compare_summary.csv`.
   - No `model_feature_compare_detail.csv`.
   - No `bar_compare_summary.csv`.
   - No snapshot CSVs were found locally, so feature-diff and bar-diff checks cannot be completed from current workspace data.

## Replay Reconstruction Audit

After rebuilding near-month pipeline data, replay inputs were generated from historical 5m bars:

- Snapshot-like cache: `saved_data/fr_audit_tmp/synthetic_snapshot_cache`
- Temporary model metadata: `saved_data/fr_audit_tmp/temp_saved_models`
- Dates: `20260519`, `20260520`, `20260521`
- Stocks: `603308.SH`, `600487.SH`, `600312.SH`
- Models checked: 7 saved model artifacts mapped onto the rebuilt near-month samples

Outputs:

- `saved_data/feature_reconstruction_audit/asof1455_replay/feature_reconstruction_summary.csv`
- `saved_data/feature_reconstruction_audit/asof1455_replay/feature_reconstruction_feature_diffs.csv`
- `saved_data/feature_reconstruction_audit/asof1455_replay/feature_diff_by_category.csv`
- `saved_data/feature_reconstruction_audit/asof1455_replay_bar_compare/bar_compare_summary.csv`
- `saved_data/feature_reconstruction_audit/asof1455_replay_bar_compare/bar_compare_detail.csv`

Important caveat: the replay snapshot cache is synthesized from historical 5m bars, not true archived tick snapshots. It is still useful for checking the 14:55 cutoff effect and reconstruction code path, but true historical snapshot archives would be needed for quote-source error measurement.

### Replay Results

- Feature reconstruction summary: 21 model-date rows and 7,713 feature-level diff rows.
- Bar compare: 9 stock-date rows and 432 bar-level rows.
- The replay has 47 bars through 14:55 versus 48 full-day training bars; the missing bar is 15:00.
- As-of close differs from full close on all checked days:
  - `600312.SH`: -0.02 to -0.06
  - `600487.SH`: -0.03 to -0.32
  - `603308.SH`: -0.17 to +0.06
- As-of amount/volume are lower by about 1.5% to 2.5%, which is expected because 15:00 is excluded.
- VWAP difference is small:
  - roughly -0.08 to +0.06 yuan in the checked rows.

### Feature Diff By Category

| category | comparable | median abs rel diff | p90 abs rel diff | max abs rel diff |
| --- | ---: | ---: | ---: | ---: |
| last_tail | 54 | 1.026750 | 20.274196 | 662.145923 |
| shock_z | 897 | 0.535875 | 4.130774 | 139.150513 |
| range | 1191 | 1.190982 | 3.453978 | 166.796490 |
| lagged_fut_us | 1023 | 0.304194 | 2.964371 | 164.464536 |
| sector_board | 711 | 0.172356 | 2.793734 | 320.648145 |
| price_ret | 1011 | 0.284138 | 2.504870 | 174.914826 |
| volume_amount | 207 | 0.091725 | 0.288875 | 0.560259 |
| other | 1713 | 0.000000 | 0.029018 | 14.444458 |

Interpretation:

- `last_30m/last_60m/afternoon/high_to_open/upper_shadow` type fields are very unstable around the 14:55 cutoff.
- `shock20`, `z20`, `z60`, and `range_pct` families are not safe to reuse from EOD models for 14:55 scoring.
- `volume/amount` as-of values are biased lower but comparatively well-behaved; these should be retrained as as-of features, not median-filled or reused as full-day fields.
- The lagged futures/US group appears large in the diff because the temporary replay compares as-of reconstructed row against full current sample; for live scoring these should be taken directly from `samples[date=T]`, and the hard reject added in `run_intraday_nextday_signals.py` protects against missing values.

## Reports

- `summary.json`
- `saved_model_features.csv`
- `saved_model_high_risk_by_model.csv`
- `saved_model_sample_availability.csv`
- `leaderboard_inventory.csv`
- `lagged_daily_sample_check.csv`
- `minidata_sample_inventory.csv`
- `top_model_risk_compact.csv`
