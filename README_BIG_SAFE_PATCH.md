# Big Safe Model Retrain Patch

## Apply

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/big_safe_model_retrain_patch.zip -d .
bash scripts/apply_big_safe_patch.sh
```

## Safe model update from existing pipeline outputs

This only scans existing leaderboards and saves models. It does not rerun pipelines.

```bash
SKIP_PIPELINE=1 PYTHON=python3 END_DATE=2026-05-15 bash scripts/update_ranked_models_latest.sh
```

## Rebuild only 603308 pipeline/search/model

This writes only:

```text
saved_data/603308_pipeline_out
saved_models/603308.SH
```

```bash
PYTHON=python3 END_DATE=2026-05-15 bash scripts/rebuild_603308_pipeline_safe.sh
```

## Fix current-day 5m realtime bars

```bash
python3 tools/fix_5m_ohlcv_from_snapshots.py \
  --date 20260515 \
  --cache-dir saved_data/akshare_realtime_cache \
  --symbols-file saved_data/intraday_nextday_signals/20260515/effective_watchlist.txt \
  --cutoff-time 14:55
```
