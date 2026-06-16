# A-share Chapter 12 Model Data

Builds an A-share version of:

```text
12_gradient_boosting_machines/data.h5::/model_data
```

Output:

```text
saved_data/ashare_ml4t/ch12_reproduce/model_data.h5::/model_data
```

The table is intentionally compatible with the Chapter 17 read path:

```python
data = pd.read_hdf(model_data_path, "model_data").dropna().sort_index()
outcomes = data.filter(like="fwd").columns.tolist()
X = data.drop(outcomes, axis=1)
y = data["r01_fwd"]
```

## Important Scope

This is a strict Chapter 12/17 data-shape reproduction. It does not add A-share trading masks, open-limit flags, tradestatus, ST flags, next-open labels, board fields, or industry strings to `/model_data`.

Only these 34 columns are written:

```text
dollar_vol, dollar_vol_rank
rsi, bb_high, bb_low, NATR, ATR, PPO, MACD
sector
r01, r05, r10, r21, r42, r63
r01dec, r05dec, r10dec, r21dec, r42dec, r63dec
r01q_sector, r05q_sector, r10q_sector, r21q_sector, r42q_sector, r63q_sector
r01_fwd, r05_fwd, r21_fwd
year, month, weekday
```

Metadata is stored separately:

```text
saved_data/ashare_ml4t/ch12_reproduce/assets_ashare.h5::/ashare/metadata
```

## Input

Default universe:

```text
saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv
```

BaoStock data:

```text
date, code, open, high, low, close, volume
adjustflag=2  # 前复权
```

The default start date is `2010-01-01` because strict Chapter 12 uses:

```text
MONTH = 21
YEAR = 252
min_obs = 7 * YEAR = 1764
```

Pulling only seven natural years of A-share history usually gives fewer than 1764 trading rows.

## Run

Install runtime dependencies:

```bash
python3 -m pip install -r requirements.txt
```

On Ubuntu, TA-Lib may require the system library before `pip install TA-Lib` works.

Full build:

```bash
python3 scripts/build_ashare_ch12_model_data.py --workers 4
```

or

```bash
python3 scripts/build_ashare_ch12_model_data.py   --workers 4   --cache-dir saved_data/ashare_ml4t/ch12_reproduce/baostock_qfq_daily_cache   --source-cache-dir saved_data/ashare_ml4t/ch12_reproduce/baostock_qfq_daily_cache   --source-cache-pattern "{code}_qfq_daily.csv"   --source-cache-adjust qfq   --no-fetch-missing-source-cache   --out-dir saved_data/ashare_ml4t/ch12_reproduce
```

Background run in tmux:

```bash
python3 scripts/build_ashare_ch12_model_data.py --workers 4 \
  2>&1 | tee logs/build_ashare_ch12_model_data_$(date +%Y%m%d_%H%M%S).log
```

Smoke test:

```bash
python3 scripts/build_ashare_ch12_model_data.py \
  --out-dir saved_data/ashare_ml4t/ch12_reproduce_smoke \
  --max-symbols 20 \
  --workers 4
```

## Reports

Reports are written to:

```text
saved_data/ashare_ml4t/ch12_reproduce/reports/
```

Key reports:

```text
column_check.csv
na_report_before_dropna.csv
daily_sample_count_before_dropna.csv
daily_sample_count_after_dropna.csv
label_alignment_samples.csv
outlier_symbols_r01_gt_1.csv
board_distribution.csv
fetch_errors.csv
nobs_by_symbol.csv
chapter17_read_smoke_test.json
pool_validation.json
```
