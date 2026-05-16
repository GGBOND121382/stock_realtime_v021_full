# Realtime 5m bar timestamp patch

This patch replaces `tools/fix_5m_ohlcv_from_snapshots.py`.

## Problem fixed

The old realtime 5m bar builder used left-end 5m labels / floor buckets, causing:

- extra collected endpoints: `09:15`, `09:20`, `09:25`, `09:30`, `13:00`
- missing collected endpoint: `14:55`
- false large differences vs BaoStock 5m bars

BaoStock 5m bars are right-end labelled:

- `09:30~09:35 -> 09:35`
- `14:50~14:55 -> 14:55`

## Apply

```bash
unzip realtime_5m_bar_patch.zip -d /root/stock_realtime_v021_full
cd /root/stock_realtime_v021_full
bash scripts/apply_realtime_5m_bar_patch.sh
```

## Validate syntax only

```bash
VALIDATE_ONLY=1 bash scripts/apply_realtime_5m_bar_patch.sh
```

## Validate a trading-day cache

```bash
python3 tools/fix_5m_ohlcv_from_snapshots.py \
  --date 20260515 \
  --cache-dir saved_data/akshare_realtime_cache \
  --symbols-file saved_data/intraday_nextday_signals/20260515/effective_watchlist.txt \
  --cutoff-time 14:55 \
  --dry-run
```

Then run without `--dry-run`.

## Expected result

After the patch, `minute_bars_5min.csv` should use endpoints like:

```text
09:35, 09:40, ..., 11:30, 13:05, ..., 14:55
```

It should not contain:

```text
09:15, 09:20, 09:25, 09:30, 13:00
```
