#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="backups/today_scoring_pipeline_fix_${TS}"
mkdir -p "$BACKUP_DIR"

FILES=(
  "pipelines/run_intraday_nextday_signals.py"
  "data_collection/collect_realtime_context.py"
  "data_collection/collect_akshare_l1_cache.py"
)

for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp "$f" "$BACKUP_DIR/$f"
  else
    echo "[ERROR] missing file: $f" >&2
    exit 2
  fi
done

echo "[1/5] Apply scoring/context fixes ..."
"$PYTHON" - <<'PY'
from pathlib import Path
import re

ROOT = Path('.')

# -----------------------------------------------------------------------------
# Patch 1: pipelines/run_intraday_nextday_signals.py
#   - keep live amount/volume/prev_close/pct_chg even when training samples lack cols
#   - compute pct_chg/overnight_ret from snapshot prev_close
#   - merge today's pending minute_bars into intraday segment features
#   - skip sector_range_z20 as a hard realtime-context dependency
# -----------------------------------------------------------------------------
p = ROOT / 'pipelines' / 'run_intraday_nextday_signals.py'
s = p.read_text(encoding='utf-8')
orig = s

# 1A) sector_range_z20 should not be a hard realtime context dependency.
if 'NON_CRITICAL_REALTIME_CONTEXT_FEATURES' not in s:
    anchor = 'REALTIME_CONTEXT_EXACT_FEATURES = {'
    if anchor not in s:
        raise SystemExit('[ERROR] cannot find REALTIME_CONTEXT_EXACT_FEATURES anchor')
    s = s.replace(
        anchor,
        'NON_CRITICAL_REALTIME_CONTEXT_FEATURES = {"sector_range_z20"}\n\n' + anchor,
        1,
    )

# Insert skip inside context_dependencies_for_model_features loop if not present.
if 'if col in NON_CRITICAL_REALTIME_CONTEXT_FEATURES:' not in s:
    s = s.replace(
        '    for col in cols:\n'
        '        if not is_realtime_context_feature(col):\n',
        '    for col in cols:\n'
        '        if col in NON_CRITICAL_REALTIME_CONTEXT_FEATURES:\n'
        '            # sector_range_z20 needs a 20-day historical sector_range_pct window.\n'
        '            # The realtime THS summary snapshot alone cannot compute it correctly,\n'
        '            # so do not let this single feature hard-block sector models.\n'
        '            continue\n'
        '        if not is_realtime_context_feature(col):\n',
        1,
    )

# 1B) Replace live_daily_from_snapshots.
new_live_daily = r'''def live_daily_from_snapshots(stock_code: str, trade_date: str, cache_dir: Path, cutoff_time: Optional[str]) -> dict:
    """Return latest live daily fields from pending snapshots.

    Important fixes:
      - keep true cumulative amount/volume for liquidity gates;
      - use vendor daily open/high/low when available instead of only sampled px range;
      - expose prev_close, pct_chg and overnight_ret for same-day scoring features.
    """
    snap_path = cache_symbol_dir(cache_dir, trade_date, stock_code) / "snapshot_5level.csv"
    if not snap_path.exists():
        return {}
    try:
        snap = pd.read_csv(snap_path, encoding="utf-8-sig")
    except Exception:
        return {}
    if snap.empty:
        return {}

    if "datetime" in snap.columns:
        snap["datetime"] = pd.to_datetime(snap["datetime"], errors="coerce")
    else:
        snap["datetime"] = pd.to_datetime(
            snap.get("trade_date", "").astype(str) + snap.get("trade_time", "").astype(str).str.zfill(6),
            errors="coerce",
        )
    snap = snap.dropna(subset=["datetime"]).sort_values("datetime")
    cutoff_dt = parse_cutoff_dt(trade_date, cutoff_time)
    if cutoff_dt is not None:
        snap = snap[snap["datetime"] <= cutoff_dt].copy()
    if snap.empty:
        return {"core_complete": False, "missing_core_fields": "no_snapshot_before_cutoff"}

    def last_num(col: str):
        if col not in snap.columns:
            return np.nan
        ss = pd.to_numeric(snap[col], errors="coerce").dropna()
        return float(ss.iloc[-1]) if len(ss) else np.nan

    px = pd.to_numeric(snap.get("last_price"), errors="coerce") if "last_price" in snap.columns else pd.Series(dtype=float)
    valid_px = px.dropna()
    if valid_px.empty:
        return {"core_complete": False, "missing_core_fields": "close"}

    last = snap.iloc[-1]
    close = float(valid_px.iloc[-1])
    open_v = last_num("open")
    high_v = last_num("high")
    low_v = last_num("low")
    if not np.isfinite(open_v) or open_v <= 0:
        open_v = float(valid_px.iloc[0])
    if not np.isfinite(high_v) or high_v <= 0:
        high_v = float(valid_px.max())
    if not np.isfinite(low_v) or low_v <= 0:
        low_v = float(valid_px.min())

    volume = last_num("volume")
    amount = last_num("amount")
    prev_close = last_num("prev_close")
    pct_chg = last_num("pct_chg")
    # Vendors differ: some pct_chg are percent points, some are ratio.  Keep ratio.
    if np.isfinite(pct_chg) and abs(pct_chg) > 1.0:
        pct_chg = pct_chg / 100.0
    if (not np.isfinite(pct_chg)) and np.isfinite(prev_close) and prev_close > 0:
        pct_chg = close / prev_close - 1.0

    row = {
        "open": open_v,
        "high": high_v,
        "low": low_v,
        "close": close,
        "volume": volume,
        "amount": amount,
        "prev_close": prev_close,
        "pct_chg": pct_chg,
        "snapshots": int(len(snap)),
        "snapshot_time": str(last["datetime"]),
        "source_used": str(last.get("spot_source_used") or last.get("quote_source") or ""),
    }
    if np.isfinite(amount) and np.isfinite(volume) and volume > 0:
        row["daily_vwap"] = amount / volume
    if np.isfinite(prev_close) and prev_close > 0 and np.isfinite(open_v):
        row["overnight_ret"] = open_v / prev_close - 1.0

    missing = []
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        v = row.get(field)
        if v is None or not np.isfinite(v) or float(v) <= 0:
            missing.append(field)
    row["core_complete"] = len(missing) == 0
    row["missing_core_fields"] = ",".join(missing)
    return row

'''
pat = r'def live_daily_from_snapshots\(stock_code: str, trade_date: str, cache_dir: Path, cutoff_time: Optional\[str\]\) -> dict:\n.*?\n(?=def overlay_current_day_from_cache)'
s2, n = re.subn(pat, new_live_daily, s, flags=re.S)
if n != 1:
    raise SystemExit(f'[ERROR] failed to replace live_daily_from_snapshots, replacements={n}')
s = s2

# 1C) Replace overlay_current_day_from_cache.
new_overlay = r'''def overlay_current_day_from_cache(df: pd.DataFrame, stock_code: str, trade_date: str, cache_dir: Path, cutoff_time: Optional[str] = None) -> pd.DataFrame:
    """Overlay current-day live daily fields onto the last scoring row.

    Fix: do not require live fields to already exist in the historical training
    sample.  Liquidity gates and diagnostics need amount/volume even if the
    saved model feature set never used them.
    """
    live = live_daily_from_snapshots(stock_code, trade_date, cache_dir, cutoff_time)
    if not live:
        daily_path = cache_symbol_dir(cache_dir, trade_date, stock_code) / "daily_features.csv"
        if not daily_path.exists():
            return df
        daily = pd.read_csv(daily_path)
        if daily.empty:
            return df
        current = daily.iloc[-1].to_dict()
        if cutoff_time and "last_time" in current:
            try:
                last_time = pd.to_datetime(current["last_time"]).time()
                hh, mm = cutoff_time.split(":", 1)
                if last_time > dtime(int(hh), int(mm)):
                    return df
            except Exception:
                pass
        live = dict(current)

    out = df.sort_values("date").copy()
    trade_ts = pd.to_datetime(yyyymmdd_to_iso(trade_date))

    # Ensure diagnostics/liquidity/raw live columns always exist.
    live_columns = {
        "open", "high", "low", "close", "volume", "amount", "daily_vwap",
        "daily_vwap_volume", "daily_vwap_pv", "n_intraday_bars",
        "prev_close", "pct_chg", "overnight_ret",
        "_snapshot_time", "_source_used", "_core_complete", "_missing_core_fields",
    }
    for c in live_columns:
        if c not in out.columns:
            out[c] = np.nan

    row = out.iloc[-1].copy()
    row["date"] = trade_ts
    mapping = [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("amount", "amount"),
        ("daily_vwap", "daily_vwap"),
        ("volume", "daily_vwap_volume"),
        ("amount", "daily_vwap_pv"),
        ("snapshots", "n_intraday_bars"),
        ("prev_close", "prev_close"),
        ("pct_chg", "pct_chg"),
        ("overnight_ret", "overnight_ret"),
    ]
    for src, dst in mapping:
        if src in live:
            row[dst] = pd.to_numeric(live[src], errors="coerce")

    # If prev_close exists but pct/overnight were not supplied, derive them.
    try:
        if pd.notna(row.get("prev_close")) and float(row["prev_close"]) > 0:
            if pd.isna(row.get("pct_chg")) and pd.notna(row.get("close")):
                row["pct_chg"] = float(row["close"]) / float(row["prev_close"]) - 1.0
            if pd.isna(row.get("overnight_ret")) and pd.notna(row.get("open")):
                row["overnight_ret"] = float(row["open"]) / float(row["prev_close"]) - 1.0
    except Exception:
        pass

    row["_snapshot_time"] = live.get("snapshot_time", live.get("last_time", ""))
    row["_source_used"] = live.get("source_used", live.get("source", ""))
    row["_core_complete"] = bool(live.get("core_complete", False))
    row["_missing_core_fields"] = live.get("missing_core_fields", "")

    out = out[pd.to_datetime(out["date"], errors="coerce").dt.normalize() != trade_ts]
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)

'''
pat = r'def overlay_current_day_from_cache\(df: pd\.DataFrame, stock_code: str, trade_date: str, cache_dir: Path, cutoff_time: Optional\[str\] = None\) -> pd\.DataFrame:\n.*?\n(?=def \w)'
s2, n = re.subn(pat, new_overlay, s, flags=re.S)
if n != 1:
    raise SystemExit(f'[ERROR] failed to replace overlay_current_day_from_cache, replacements={n}')
s = s2

# 1D) Replace add_scoring_features to include pending minute bars.
new_add_scoring = r'''def _normalize_realtime_minute_bars(bars: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    out = bars.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif {"trade_date", "trade_time"}.issubset(out.columns):
        out["datetime"] = pd.to_datetime(
            out["trade_date"].astype(str) + out["trade_time"].astype(str).str.zfill(6),
            errors="coerce",
        )
    else:
        # Last resort: use date plus row order minutes; caller will likely drop if invalid.
        out["datetime"] = pd.NaT
    for c in ["open", "high", "low", "close", "volume", "amount", "bar_volume", "bar_amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # If minute bars store cumulative volume/amount, convert to bar increments when available.
    if "bar_volume" in out.columns and out["bar_volume"].notna().any():
        out["volume"] = out["bar_volume"].fillna(0)
    if "bar_amount" in out.columns and out["bar_amount"].notna().any():
        out["amount"] = out["bar_amount"].fillna(0)
    return out.dropna(subset=["datetime", "open", "high", "low", "close"])


def _load_pending_intraday_rows(cache_dir: Path, stock_code: str, trade_date: str) -> pd.DataFrame:
    base = cache_symbol_dir(cache_dir, trade_date, stock_code)
    candidates = [base / "minute_bars_1min.csv", base / "minute_bars_5min.csv"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            bars = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if bars.empty:
            continue
        bars = _normalize_realtime_minute_bars(bars, trade_date)
        if bars.empty:
            continue
        rows = build_intraday_rows(bars)
        if not rows.empty:
            return rows
    return pd.DataFrame()


def add_scoring_features(df: pd.DataFrame, intraday_path: Optional[Path], cache_dir: Path, stock_code: str) -> pd.DataFrame:
    out = add_reversal_daily_features(df)

    intra = load_intraday_feature_cache(intraday_path, cache_dir, stock_code)
    # Merge current-day pending minute bars.  The historical feature cache is
    # normally only complete up to T-1, so without this the same-day row misses
    # morning/afternoon/last-30m segment features and triggers filled_features_gt_*.
    try:
        if "date" in out.columns and not out.empty:
            trade_date = pd.to_datetime(out["date"].max()).strftime("%Y%m%d")
            pending_intra = _load_pending_intraday_rows(cache_dir, stock_code, trade_date)
            if not pending_intra.empty:
                if intra.empty:
                    intra = pending_intra
                else:
                    intra = pd.concat([intra, pending_intra], ignore_index=True)
                    intra["date"] = pd.to_datetime(intra["date"], errors="coerce")
                    intra = intra.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    except Exception as exc:
        print(f"[WARN] failed to merge pending intraday bars for {stock_code}: {type(exc).__name__}: {exc}", flush=True)

    if not intra.empty:
        out = out.merge(intra, on="date", how="left")
        for col in ["morning_vwap", "afternoon_vwap", "last_30m_vwap"]:
            if col in out.columns and "close" in out.columns:
                out[f"{col}_to_close"] = out["close"] / out[col].replace(0, np.nan) - 1.0
        if {"morning_ret", "afternoon_ret"}.issubset(out.columns):
            out["morning_afternoon_reversal"] = -out["morning_ret"] * out["afternoon_ret"]
        if {"first_60m_ret", "last_30m_ret"}.issubset(out.columns):
            out["first60_last30_reversal"] = -out["first_60m_ret"] * out["last_30m_ret"]
    return add_market_state_features(out)

'''
pat = r'def add_scoring_features\(df: pd\.DataFrame, intraday_path: Optional\[Path\], cache_dir: Path, stock_code: str\) -> pd\.DataFrame:\n.*?\n(?=def parse_cutoff_dt)'
s2, n = re.subn(pat, new_add_scoring, s, flags=re.S)
if n != 1:
    raise SystemExit(f'[ERROR] failed to replace add_scoring_features, replacements={n}')
s = s2

# 1E) Best-effort insertion of missing_feature_names in diagnostics.
# Current project variants differ here, so patch common patterns but do not hard-fail.
if 'missing_feature_names' not in s:
    # Pattern A: after missing_feature_count is computed from current row.
    s = s.replace(
        'missing_feature_count = int(score_row[feature_cols].isna().sum())',
        'missing_feature_names = [c for c in feature_cols if pd.isna(score_row.get(c))]\n'
        '        missing_feature_count = int(len(missing_feature_names))'
    )
    s = s.replace(
        '"missing_feature_count": missing_feature_count,',
        '"missing_feature_count": missing_feature_count,\n'
        '            "missing_feature_names": ",".join(missing_feature_names) if "missing_feature_names" in locals() else "",'
    )
    s = s.replace(
        '"missing_feature_count": int(missing_feature_count),',
        '"missing_feature_count": int(missing_feature_count),\n'
        '            "missing_feature_names": ",".join(missing_feature_names) if "missing_feature_names" in locals() else "",'
    )

if s == orig:
    raise SystemExit('[ERROR] run_intraday_nextday_signals.py was not modified')
p.write_text(s, encoding='utf-8')

# -----------------------------------------------------------------------------
# Patch 2: data_collection/collect_realtime_context.py
#   - do not hard-require sector_range_z20 from realtime context plan.
# -----------------------------------------------------------------------------
p = ROOT / 'data_collection' / 'collect_realtime_context.py'
s = p.read_text(encoding='utf-8')
orig = s
if 'NON_CRITICAL_REALTIME_CONTEXT_FEATURES' not in s:
    marker = 'METADATA_COLS = {'
    if marker not in s:
        raise SystemExit('[ERROR] cannot find METADATA_COLS anchor in collect_realtime_context.py')
    s = s.replace(marker, 'NON_CRITICAL_REALTIME_CONTEXT_FEATURES = {"sector_range_z20"}\n\n' + marker, 1)
if 'if feat in NON_CRITICAL_REALTIME_CONTEXT_FEATURES:' not in s:
    s = s.replace(
        '    for feat in features:\n'
        '        if not feat:\n',
        '    for feat in features:\n'
        '        if not feat:\n'
        '            continue\n'
        '        if feat in NON_CRITICAL_REALTIME_CONTEXT_FEATURES:\n'
        '            # Needs historical 20-day sector range.  Realtime THS summary\n'
        '            # alone cannot compute it correctly, so do not hard-block.\n'
        '            continue\n',
        1,
    )
if s == orig:
    print('[WARN] collect_realtime_context.py unchanged; maybe already patched')
p.write_text(s, encoding='utf-8')

# -----------------------------------------------------------------------------
# Patch 3: data_collection/collect_akshare_l1_cache.py
#   - core completeness aliases should understand English normalized keys too.
# -----------------------------------------------------------------------------
p = ROOT / 'data_collection' / 'collect_akshare_l1_cache.py'
s = p.read_text(encoding='utf-8')
orig = s
# Patch CORE_FIELD_ALIASES conservatively.
repls = {
    '"close": ["最新", "最新价", "现价", "最新报价"],': '"close": ["最新", "最新价", "现价", "最新报价", "last_price", "close"],',
    '"last_price": ["最新", "最新价", "现价", "最新报价"],': '"last_price": ["最新", "最新价", "现价", "最新报价", "last_price", "close"],',
    '"open": ["今开", "开盘", "开盘价"],': '"open": ["今开", "开盘", "开盘价", "open"],',
    '"high": ["最高", "最高价"],': '"high": ["最高", "最高价", "high"],',
    '"low": ["最低", "最低价"],': '"low": ["最低", "最低价", "low"],',
    '"volume": ["成交量", "总手", "总量"],': '"volume": ["成交量", "总手", "总量", "volume"],',
    '"amount": ["成交额", "金额"],': '"amount": ["成交额", "金额", "amount"],',
}
for a, b in repls.items():
    if a in s:
        s = s.replace(a, b, 1)
if s == orig:
    print('[WARN] collect_akshare_l1_cache.py unchanged; maybe aliases already patched')
p.write_text(s, encoding='utf-8')

print('[OK] patch applied')
PY

echo "[2/5] Syntax check ..."
"$PYTHON" -m py_compile \
  pipelines/run_intraday_nextday_signals.py \
  data_collection/collect_realtime_context.py \
  data_collection/collect_akshare_l1_cache.py

echo "[3/5] Verify patch markers ..."
grep -n "NON_CRITICAL_REALTIME_CONTEXT_FEATURES" pipelines/run_intraday_nextday_signals.py data_collection/collect_realtime_context.py
grep -n "_load_pending_intraday_rows" pipelines/run_intraday_nextday_signals.py
grep -n "missing_feature_names" pipelines/run_intraday_nextday_signals.py || true

echo "[4/5] Write replay helper ..."
cat > scripts/replay_today_after_scoring_fix.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
DATE="${DATE:-$(date +%Y%m%d)}"
PYTHON="${PYTHON:-python3}"
WATCHLIST="${WATCHLIST:-selected_watchlist.txt}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
MODEL_POLICY="${MODEL_POLICY:-all}"
MAX_MISSING_FEATURES="${MAX_MISSING_FEATURES:-5}"
MIN_AMOUNT_YUAN="${MIN_AMOUNT_YUAN:-50000000}"
OUT_DIR="${OUT_DIR:-saved_data/intraday_nextday_signals/${DATE}_replay_fixed}"

echo "[REPLAY] DATE=$DATE OUT_DIR=$OUT_DIR"
mkdir -p "$OUT_DIR"

# Re-score using existing pending cache and context outputs.  This assumes your
# main pipeline has already collected/built today's cache.  We run score-now mode
# through the same public pipeline entry when available.
$PYTHON pipelines/run_intraday_nextday_signals.py \
  --watchlist "$WATCHLIST" \
  --context-config "$CONTEXT_CONFIG" \
  --signal-out-dir saved_data/intraday_nextday_signals \
  --model-policy "$MODEL_POLICY" \
  --trade-date "$DATE" \
  --cutoff-time 14:55 \
  --score-time 00:00 \
  --max-missing-features "$MAX_MISSING_FEATURES" \
  --min-amount-yuan "$MIN_AMOUNT_YUAN" \
  --score-now-only || {
    echo "[WARN] score-now-only mode is not supported by your current script."
    echo "Run the normal trading-day pipeline after applying the patch, or use your existing replay wrapper."
    exit 1
  }
EOS
chmod +x scripts/replay_today_after_scoring_fix.sh

echo "[5/5] Done. Backup: $BACKUP_DIR"
echo "Next normal run: bash scripts/run_trading_day_signal_and_portfolio_all_models.sh"
echo "If your local run_intraday_nextday_signals.py supports score-now-only, replay with:"
echo "  DATE=20260513 bash scripts/replay_today_after_scoring_fix.sh"
