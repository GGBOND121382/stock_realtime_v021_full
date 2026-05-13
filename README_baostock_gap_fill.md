# BaoStock 5m intraday gap fill patch

Use from project root:

```bash
unzip -o baostock_intraday_gap_fill_patch.zip
PYTHON=python3 DATE=20260513 bash scripts/fill_today_pre1038_baostock_5m.sh
```

Optional:

```bash
PYTHON=python3 DATE=20260513 SYMBOLS_FILE=selected_watchlist.txt BEFORE_TIME=10:38 \
  bash scripts/fill_today_pre1038_baostock_5m.sh
```

What it does:

1. Query BaoStock 5-minute bars for each symbol under `saved_data/akshare_realtime_cache/pending/<DATE>/`.
2. Merge only bars before `BEFORE_TIME` and before first existing local bar.
3. Save merged bars to `minute_bars_5min.csv`.
4. Rebuild/update `saved_data/akshare_realtime_cache/feature_cache/<SYMBOL>_intraday_reversal_features.csv`.
5. Back up overwritten files under `saved_data/akshare_realtime_cache/backup_baostock_gap_fill/`.

It does not fake 1-minute bars.
