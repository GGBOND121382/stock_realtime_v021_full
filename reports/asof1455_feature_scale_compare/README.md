# asof1455 feature scale comparison

Compared the new `asof1455` training feature scale against replayed 14:55 live/watch reconstruction.

Inputs:
- `saved_data/asof1455_v1/600312_pipeline_out`
- `saved_data/asof1455_v1/603308_pipeline_out`
- Dates: `2026-05-19`, `2026-05-20`
- `2026-05-21` was skipped because the current training samples were built with `--no-keep-unlabeled-tail`, so the final sample row is absent.

Outputs:
- `summary.csv`
- `feature_diffs.csv`
- `category_summary.csv`
- `input_report.csv`

Main findings:

| group | result |
| --- | --- |
| stock_asof_raw | Exact match after refreshing `*_asof1455` aliases after cutoff bar merge. |
| stock_asof_derived | Numerical match, only floating point noise around `1e-14`. |
| lagged_daily_external | Exact match for 603308, 66 fields filled from `samples[date=T]`, 0 missing. |
| fundamental | Exact match after converting valuation fields to 14:55-known scale and filling daily-known fields from `samples[date=T]`. |

Summary:

| stock | feature_count | exactish_share | p90_abs_rel_diff | max_abs_rel_diff | issue |
| --- | ---: | ---: | ---: | ---: | --- |
| 600312.SH | 96 | 100% | 0 | ~2e-14 | Only floating point noise remains. |
| 603308.SH | 164 | 100% | 0 | ~3e-14 | Only floating point noise remains. |

Interpretation:

The new stock-level as-of scale is aligned with the 14:55 watch reconstruction. Futures/US lagged daily fields also align when filled from `samples[date=T]`.

Valuation fields are now handled with a strict as-of policy:

```text
valuation_asof_T = valuation_eod_T-1 * close_asof1455_T / close_eod_T-1
```

This is applied to `peTTM`, `pbMRQ`, `psTTM`, and `pcfNcfTTM`; their `*_rank252` fields are recomputed from the as-of valuation sequence. Raw same-day BaoStock valuation values are preserved as `*_eod` audit columns but excluded from the asof feature whitelist.

Live scoring now treats these daily-known fields like lagged external fields and fills them from `samples[date=T]`, with missing values triggering the same missing daily-feature path.
