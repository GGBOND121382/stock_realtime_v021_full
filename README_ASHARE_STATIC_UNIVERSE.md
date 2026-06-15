# A-share Static Universe Builder

Builds the first-pass Chapter 17 style static A-share universes using BaoStock only:

- `07_universe_allA_top1000_static.csv`: training universe, main board + ChiNext + STAR.
- `08_universe_mainboard_top1000_static.csv`: trading universe, Shanghai/Shenzhen main board only.

Implementation:

```text
scripts/build_ashare_static_universe.py
```

## Data Source Policy

BaoStock is the only data source used by this builder:

- `query_stock_basic()` for listing status, security type, IPO date, and delisting date.
- `query_stock_industry()` for industry fields.
- `query_history_k_data_plus()` for 7-year daily completeness, recent tradability, ST status, and approximate circulating market cap.

The script does not call AKShare or Eastmoney market-cap interfaces. Market-cap ranking uses BaoStock's turnover-implied circulating market cap and is explicitly marked:

```text
marketcap_source = baostock_approx_circ_mv
marketcap_confidence = low_confidence_approx
```

## Approximate Market Cap

BaoStock approximate circulating market cap is used for prefiltering and final ranking. The script uses a 5-valid-day median estimate for float shares:

```text
approx_float_shares_1d = volume / (turn / 100)
implied_float_shares_5d_median = median(volume / (turn / 100) over the last 5 valid trading days)
approx_circ_mv = recent_close * implied_float_shares_5d_median
```

Each source day must satisfy:

```text
tradestatus = 1
turn > 0
volume > 0
close > 0
isST != 1
```

The history and cross-check files include float-share stability diagnostics:

```text
implied_float_shares_5d_obs
implied_float_shares_5d_range_pct
implied_float_shares_unstable_3pct
implied_float_shares_unstable_5pct
```

Market-cap rank columns use explicit names:

```text
rank_marketcap_used  # rank of the market cap actually used for final selection
rank_approx          # rank of BaoStock approximate circulating market cap
rank_external        # reserved for external market cap; blank in BaoStock-only mode
```

## Run

On this Windows machine, use the fuller existing virtualenv:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\Activate.ps1
```

Or run directly:

```powershell
& 'D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe' scripts\build_ashare_static_universe.py --help
```

Small real BaoStock smoke test:

```powershell
& 'D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe' scripts\build_ashare_static_universe.py `
  --out-dir saved_data\ashare_static_universe_smoke `
  --max-history-candidates 20 `
  --prefilter-n 20 `
  --top-n 10 `
  --workers 2
```

Full build:

```powershell
& 'D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe' scripts\build_ashare_static_universe.py --workers 4
```

On Ubuntu servers, run from the repository root:

```bash
python3 -m pip install -r requirements.txt
mkdir -p logs

nohup python3 scripts/build_ashare_static_universe.py --workers 4 \
  > logs/build_ashare_static_universe_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Use `--workers 4` as a conservative default. If BaoStock is stable from that server, `--workers 6` or `--workers 8` can reduce wall-clock time. If the logs show many query failures or timeouts, lower it back to `2-4`.

For a server smoke test:

```bash
python3 scripts/build_ashare_static_universe.py \
  --out-dir saved_data/ashare_static_universe_smoke50 \
  --max-history-candidates 50 \
  --prefilter-n 50 \
  --top-n 20 \
  --workers 4
```

Default output directory:

```text
saved_data/ashare_static_universe/
```

Output files:

```text
01_baostock_stock_basic.csv
02_baostock_stock_industry.csv
03_baostock_history_completeness.csv
04_baostock_prefilter_candidates.csv
05_baostock_approx_marketcap.csv
06_marketcap_cross_check.csv
07_universe_allA_top1000_static.csv
08_universe_mainboard_top1000_static.csv
universe_build_summary.json
```

## Rules

- Static as-of date defaults to the latest complete weekday inferred locally.
- Security type must be BaoStock `type = 1`.
- Listing status must be BaoStock `status = 1`.
- Industry must be non-empty.
- IPO date must be at least 7 years before the history start.
- 7-year valid trading days default to at least `7 * 240 * 0.90 = 1512`.
- Recent 252 rows must include at least 240 valid trading days.
- Current and recent ST are excluded.
- Training universe allows main board, ChiNext, and STAR.
- Trading universe allows only Shanghai/Shenzhen main board.

This is a current static snapshot universe. For rigorous historical backtests, replace it later with a rolling point-in-time universe to avoid look-ahead from current listing status and industry.
