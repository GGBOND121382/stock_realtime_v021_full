#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare today's local realtime collected bars with BaoStock 5-minute bars.

Inputs:
  saved_data/akshare_realtime_cache/pending/<DATE>/<SYMBOL>/minute_bars_5min.csv
  saved_data/akshare_realtime_cache/pending/<DATE>/<SYMBOL>/daily_features.csv

Outputs:
  saved_data/baostock_compare/<DATE>/comparison_summary.csv
  saved_data/baostock_compare/<DATE>/comparison_summary.json
  saved_data/baostock_compare/<DATE>/<SYMBOL>_aligned_diff.csv
  saved_data/baostock_compare/<DATE>/<SYMBOL>_baostock_5m.csv
  saved_data/baostock_compare/<DATE>/<SYMBOL>_missing_times.csv
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS = ["volume", "amount"]
ALL_COMPARE_COLS = PRICE_COLS + VOL_COLS


def empty_bar_frame(source: str | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"datetime": pd.Series(dtype="datetime64[ns]")})
    for col in ALL_COMPARE_COLS:
        out[col] = pd.Series(dtype="float64")
    if source is not None:
        out["source"] = pd.Series(dtype="object")
    return out


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper().replace("_", ".")
    if not s:
        return ""
    if "." in s:
        a, b = s.split(".", 1)
        if a in {"SH", "SZ"}:
            market, code = a, b
        else:
            code, market = a, b
        return f"{code.zfill(6)}.{market}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def baostock_code(symbol: str) -> str:
    s = normalize_symbol(symbol)
    code, market = s.split(".", 1)
    return f"{market.lower()}.{code}"


def yyyymmdd_to_dash(value: str) -> str:
    value = str(value)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def parse_hhmm(value: str | None) -> Optional[dtime]:
    if not value:
        return None
    hh, mm = str(value).split(":", 1)
    return dtime(int(hh), int(mm))


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_datetime_col(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty and "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        return out
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif {"date", "time"}.issubset(out.columns):
        # BaoStock time sometimes looks like 20260513093500000.
        date_s = out["date"].astype(str).str.replace("-", "", regex=False)
        time_s = out["time"].astype(str).str.replace(":", "", regex=False)
        dt_vals = []
        for d, t in zip(date_s, time_s):
            t = str(t)
            if len(t) >= 14 and t[:8].isdigit():
                raw = t[:14]
            elif len(t) >= 6:
                raw = str(d)[:8] + t[:6]
            else:
                raw = str(d)[:8] + t.zfill(6)
            dt_vals.append(raw)
        out["datetime"] = pd.to_datetime(dt_vals, format="%Y%m%d%H%M%S", errors="coerce")
    elif "trade_time" in out.columns:
        t = out["trade_time"].astype(str).str.replace(":", "", regex=False).str.zfill(6)
        out["datetime"] = pd.to_datetime(str(date) + t, format="%Y%m%d%H%M%S", errors="coerce")
    else:
        raise ValueError(f"cannot infer datetime column; columns={list(out.columns)}")

    out = out.dropna(subset=["datetime"]).copy()
    # Normalize to minute resolution.
    out["datetime"] = out["datetime"].dt.floor("min")
    return out


def normalize_bar_df(df: pd.DataFrame, date: str, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        out = empty_bar_frame(source)
        out["source"] = source
        return out
    out = ensure_datetime_col(df, date)
    rename_map = {}
    aliases = {
        "open": ["open", "开盘", "开盘价"],
        "high": ["high", "最高", "最高价"],
        "low": ["low", "最低", "最低价"],
        "close": ["close", "收盘", "收盘价"],
        "volume": ["volume", "成交量", "vol"],
        "amount": ["amount", "成交额"],
    }
    for canon, cands in aliases.items():
        if canon in out.columns:
            continue
        for c in cands:
            if c in out.columns:
                rename_map[c] = canon
                break
    if rename_map:
        out = out.rename(columns=rename_map)

    keep = ["datetime"] + [c for c in ALL_COMPARE_COLS if c in out.columns]
    out = out[keep].copy()
    for c in ALL_COMPARE_COLS:
        if c in out.columns:
            out[c] = safe_num(out[c])
        else:
            out[c] = np.nan
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    out["source"] = source
    return out


def filter_cutoff(df: pd.DataFrame, cutoff: Optional[str]) -> pd.DataFrame:
    if not cutoff:
        return df
    t = parse_hhmm(cutoff)
    if t is None:
        return df
    if df.empty:
        return df.copy()
    if "datetime" not in df.columns:
        return df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
    if df.empty:
        return df.copy()
    return df[df["datetime"].dt.time <= t].copy()


def query_baostock_5m(symbol: str, date: str, adjustflag: str = "3") -> pd.DataFrame:
    try:
        import baostock as bs
    except Exception as exc:
        raise RuntimeError(
            "baostock is not installed. Install it in your project environment: "
            "python3 -m pip install baostock"
        ) from exc

    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")

    try:
        bs_code = baostock_code(symbol)
        date_dash = yyyymmdd_to_dash(date)
        fields = "date,time,code,open,high,low,close,volume,amount"
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=date_dash,
            end_date=date_dash,
            frequency="5",
            adjustflag=str(adjustflag),
        )
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(f"BaoStock query failed for {bs_code}: {rs.error_code} {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        if df.empty:
            return empty_bar_frame("baostock")
        return normalize_bar_df(df, date, "baostock")
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def read_local_collected(cache_dir: Path, date: str, symbol: str) -> tuple[pd.DataFrame, Optional[pd.Series], Path]:
    sym = normalize_symbol(symbol)
    sym_dir = cache_dir / "pending" / date / sym
    bar_path = sym_dir / "minute_bars_5min.csv"
    daily_path = sym_dir / "daily_features.csv"

    if not bar_path.exists():
        return empty_bar_frame("collected"), None, sym_dir

    bars = pd.read_csv(bar_path)
    bars = normalize_bar_df(bars, date, "collected")

    daily_row = None
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        if not daily.empty:
            daily_row = daily.iloc[-1]
    return bars, daily_row, sym_dir


def discover_symbols(cache_dir: Path, date: str, explicit: str | None) -> list[str]:
    if explicit:
        out = []
        for x in explicit.replace(";", ",").split(","):
            x = normalize_symbol(x)
            if x:
                out.append(x)
        return list(dict.fromkeys(out))
    day_dir = cache_dir / "pending" / date
    if not day_dir.exists():
        raise FileNotFoundError(f"day cache dir not found: {day_dir}")
    syms = [p.name for p in day_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    return sorted(normalize_symbol(s) for s in syms)


def rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.maximum(np.abs(b), 1e-12)
    return (a - b) / den


def price_bps(a, b):
    return rel_diff(a, b) * 10000.0


def calc_summary(symbol: str, collected: pd.DataFrame, bao: pd.DataFrame, daily_row: Optional[pd.Series], cutoff: Optional[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    c = filter_cutoff(collected, cutoff)
    b = filter_cutoff(bao, cutoff)

    merged = c.merge(b, on="datetime", how="outer", suffixes=("_collected", "_baostock"), indicator=True)
    exact = merged[merged["_merge"] == "both"].copy()

    for col in ALL_COMPARE_COLS:
        lc = f"{col}_collected"
        rb = f"{col}_baostock"
        if lc in exact.columns and rb in exact.columns:
            exact[f"{col}_abs_diff"] = exact[lc] - exact[rb]
            exact[f"{col}_rel_diff"] = rel_diff(exact[lc], exact[rb])
            if col in PRICE_COLS:
                exact[f"{col}_diff_bps"] = price_bps(exact[lc], exact[rb])

    missing_times = merged.loc[merged["_merge"] != "both", ["datetime", "_merge"]].copy()
    missing_times["_merge"] = missing_times["_merge"].map({
        "left_only": "only_in_collected",
        "right_only": "only_in_baostock",
        "both": "both",
    })

    row: dict = {
        "symbol": symbol,
        "cutoff": cutoff or "",
        "collected_bars": int(len(c)),
        "baostock_bars": int(len(b)),
        "aligned_bars": int(len(exact)),
        "only_in_collected": int((merged["_merge"] == "left_only").sum()),
        "only_in_baostock": int((merged["_merge"] == "right_only").sum()),
        "collected_start": str(c["datetime"].min()) if not c.empty else "",
        "collected_end": str(c["datetime"].max()) if not c.empty else "",
        "baostock_start": str(b["datetime"].min()) if not b.empty else "",
        "baostock_end": str(b["datetime"].max()) if not b.empty else "",
    }

    for col in PRICE_COLS:
        diff_col = f"{col}_diff_bps"
        if diff_col in exact.columns and not exact[diff_col].dropna().empty:
            s = exact[diff_col].dropna()
            row[f"{col}_mean_diff_bps"] = float(s.mean())
            row[f"{col}_max_abs_diff_bps"] = float(s.abs().max())
            row[f"{col}_p95_abs_diff_bps"] = float(s.abs().quantile(0.95))
        else:
            row[f"{col}_mean_diff_bps"] = np.nan
            row[f"{col}_max_abs_diff_bps"] = np.nan
            row[f"{col}_p95_abs_diff_bps"] = np.nan

    for col in VOL_COLS:
        lc = f"{col}_collected"
        rb = f"{col}_baostock"
        if lc in exact.columns and rb in exact.columns:
            row[f"{col}_sum_collected_aligned"] = float(pd.to_numeric(exact[lc], errors="coerce").sum())
            row[f"{col}_sum_baostock_aligned"] = float(pd.to_numeric(exact[rb], errors="coerce").sum())
            den = row[f"{col}_sum_baostock_aligned"]
            row[f"{col}_sum_rel_diff"] = float((row[f"{col}_sum_collected_aligned"] - den) / den) if abs(den) > 1e-12 else np.nan

    # Compare daily_features cumulative amount/volume/close with BaoStock 5m aggregate before cutoff.
    if daily_row is not None:
        for col in ["close", "volume", "amount", "daily_vwap"]:
            row[f"daily_{col}"] = float(pd.to_numeric(pd.Series([daily_row.get(col)]), errors="coerce").iloc[0]) if col in daily_row.index else np.nan
        if not b.empty:
            row["baostock_last_close_before_cutoff"] = float(b["close"].dropna().iloc[-1]) if b["close"].dropna().size else np.nan
            row["baostock_sum_volume_before_cutoff"] = float(b["volume"].sum(skipna=True))
            row["baostock_sum_amount_before_cutoff"] = float(b["amount"].sum(skipna=True))
            if row["baostock_sum_volume_before_cutoff"] and np.isfinite(row["baostock_sum_volume_before_cutoff"]):
                row["baostock_vwap_before_cutoff"] = row["baostock_sum_amount_before_cutoff"] / row["baostock_sum_volume_before_cutoff"]
            for col in ["close"]:
                dv = row.get(f"daily_{col}", np.nan)
                bv = row.get(f"baostock_last_{col}_before_cutoff", np.nan)
                row[f"daily_vs_baostock_{col}_diff_bps"] = float((dv / bv - 1) * 10000) if np.isfinite(dv) and np.isfinite(bv) and bv != 0 else np.nan
            for col in ["volume", "amount"]:
                dv = row.get(f"daily_{col}", np.nan)
                bv = row.get(f"baostock_sum_{col}_before_cutoff", np.nan)
                row[f"daily_vs_baostock_{col}_rel_diff"] = float((dv - bv) / bv) if np.isfinite(dv) and np.isfinite(bv) and bv != 0 else np.nan
            dvwap = row.get("daily_daily_vwap", np.nan)
            bvwap = row.get("baostock_vwap_before_cutoff", np.nan)
            row["daily_vs_baostock_vwap_diff_bps"] = float((dvwap / bvwap - 1) * 10000) if np.isfinite(dvwap) and np.isfinite(bvwap) and bvwap != 0 else np.nan

    # Simple severity label.
    max_close_bps = row.get("close_max_abs_diff_bps", np.nan)
    missing_bao = row.get("only_in_baostock", 0)
    if missing_bao > 0:
        row["severity"] = "missing_collected_bars"
    elif np.isfinite(max_close_bps) and max_close_bps > 10:
        row["severity"] = "large_price_diff"
    else:
        row["severity"] = "ok"

    return row, exact.sort_values("datetime"), missing_times.sort_values("datetime")


def run_premarket_update(args: argparse.Namespace, symbols: list[str]) -> None:
    cmd = [
        args.python,
        "pipelines/run_premarket_history_update.py",
        "--models-dir", args.models_dir,
        "--saved-data-dir", args.saved_data_dir,
        "--context-config", args.context_config,
        "--end-date", yyyymmdd_to_dash(args.date),
        "--keep-going",
    ]
    if symbols:
        cmd.extend(["--symbols", ",".join(symbols)])
    if args.premarket_resume:
        cmd.append("--resume")
    if args.premarket_cache_mode:
        cmd.extend(["--cache-mode", args.premarket_cache_mode])
    if args.premarket_feature_cache_mode:
        cmd.extend(["--feature-cache-mode", args.premarket_feature_cache_mode])
    print("[PREMARKET]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def resolve_pipeline_out(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    parts = list(path.resolve().parts)
    for i, part in enumerate(parts):
        if "_pipeline_out" in part:
            return Path(*parts[: i + 1])
    return None


def read_csv_meta(path: Path, date: str) -> dict:
    row = {
        "path": str(path),
        "exists": path.exists(),
        "kind": path.suffix.lower().lstrip("."),
        "rows": np.nan,
        "columns": "",
        "has_date": False,
        "has_target_date": False,
        "max_date": "",
        "error": "",
    }
    if not path.exists() or path.suffix.lower() != ".csv":
        return row
    try:
        df = pd.read_csv(path, nrows=200000)
        row["rows"] = int(len(df))
        row["columns"] = ",".join(map(str, df.columns[:50]))
        date_col = "date" if "date" in df.columns else ("datetime" if "datetime" in df.columns else None)
        if date_col:
            ds = pd.to_datetime(df[date_col], errors="coerce")
            row["has_date"] = True
            row["has_target_date"] = bool((ds.dt.strftime("%Y%m%d") == date).any()) if ds.notna().any() else False
            row["max_date"] = str(ds.max()) if ds.notna().any() else ""
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def pipeline_file_inventory(pipeline_dirs: Iterable[Path], date: str) -> pd.DataFrame:
    rows = []
    seen = set()
    for pdir in pipeline_dirs:
        if pdir is None or not pdir.exists():
            continue
        for path in sorted(pdir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            meta = read_csv_meta(path, date) if path.suffix.lower() == ".csv" else {
                "path": str(path), "exists": True, "kind": "json", "rows": np.nan,
                "columns": "", "has_date": False, "has_target_date": False, "max_date": "", "error": "",
            }
            rows.append(meta)
    return pd.DataFrame(rows)


def compare_numeric_rows(left: dict, right: dict, keys: Iterable[str], left_name: str, right_name: str) -> list[dict]:
    rows = []
    for key in keys:
        lv = pd.to_numeric(pd.Series([left.get(key, np.nan)]), errors="coerce").iloc[0]
        rv = pd.to_numeric(pd.Series([right.get(key, np.nan)]), errors="coerce").iloc[0]
        abs_diff = float(lv - rv) if pd.notna(lv) and pd.notna(rv) else np.nan
        rel = float(abs_diff / rv) if pd.notna(abs_diff) and pd.notna(rv) and abs(rv) > 1e-12 else np.nan
        rows.append({
            "field": key,
            left_name: lv,
            right_name: rv,
            "abs_diff": abs_diff,
            "rel_diff": rel,
            "diff_bps": rel * 10000.0 if pd.notna(rel) else np.nan,
        })
    return rows


def compare_pipeline_vs_collected_data(symbols: list[str], artifacts: list, cache_dir: Path, date: str, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    import pipelines.run_intraday_nextday_signals as rt

    daily_rows = []
    bar_rows = []
    pipeline_dirs = []
    seen_symbols = set(symbols)
    for art in artifacts:
        seen_symbols.add(art.stock_code)
    for sym in sorted(seen_symbols):
        arts = [a for a in artifacts if a.stock_code == sym]
        pdir = None
        if arts:
            sp = rt.resolve_repo_path(arts[0].metadata.get("samples"), sym)
            pdir = resolve_pipeline_out(sp)
            if pdir:
                pipeline_dirs.append(pdir)
        code6 = normalize_symbol(sym).split(".", 1)[0]
        collected, daily_row, _ = read_local_collected(cache_dir, date, sym)
        daily_dict = daily_row.to_dict() if daily_row is not None else {}

        pipeline_daily = {}
        if pdir:
            for cand in [pdir / "00_base" / "daily_features.csv", pdir / "00_base" / f"{code6}_daily.csv", pdir / "00_base" / "raw_cache" / f"{code6}_daily_raw.csv"]:
                if cand.exists():
                    try:
                        df = pd.read_csv(cand)
                        dcol = "date" if "date" in df.columns else None
                        if dcol:
                            ds = pd.to_datetime(df[dcol], errors="coerce")
                            part = df[ds.dt.strftime("%Y%m%d") == date]
                            if not part.empty:
                                pipeline_daily = part.iloc[-1].to_dict()
                                pipeline_daily["_pipeline_daily_path"] = str(cand)
                                break
                    except Exception:
                        pass
        for r in compare_numeric_rows(pipeline_daily, daily_dict, ALL_COMPARE_COLS + ["daily_vwap"], "pipeline", "collected"):
            r.update({"symbol": sym, "pipeline_path": pipeline_daily.get("_pipeline_daily_path", "")})
            daily_rows.append(r)

        pipeline_5m = empty_bar_frame("pipeline")
        raw_path = None
        if arts:
            raw_path = rt.resolve_intraday_path(arts[0], date, cache_dir)
        if raw_path and raw_path.exists():
            try:
                pipeline_5m = normalize_bar_df(pd.read_csv(raw_path), date, "pipeline")
                pipeline_5m = pipeline_5m[pipeline_5m["datetime"].dt.strftime("%Y%m%d") == date].copy()
                pipeline_5m = filter_cutoff(pipeline_5m, cutoff)
            except Exception:
                pipeline_5m = empty_bar_frame("pipeline")
        c = filter_cutoff(collected, cutoff)
        merged = c.merge(pipeline_5m, on="datetime", how="outer", suffixes=("_collected", "_pipeline"), indicator=True)
        exact = merged[merged["_merge"] == "both"]
        row = {
            "symbol": sym,
            "pipeline_5m_path": str(raw_path or ""),
            "collected_bars": int(len(c)),
            "pipeline_bars": int(len(pipeline_5m)),
            "aligned_bars": int(len(exact)),
            "only_in_collected": int((merged["_merge"] == "left_only").sum()) if "_merge" in merged else 0,
            "only_in_pipeline": int((merged["_merge"] == "right_only").sum()) if "_merge" in merged else 0,
        }
        for col in ALL_COMPARE_COLS:
            lc, rp = f"{col}_collected", f"{col}_pipeline"
            if lc in exact.columns and rp in exact.columns and not exact.empty:
                diff = pd.to_numeric(exact[lc], errors="coerce") - pd.to_numeric(exact[rp], errors="coerce")
                row[f"{col}_max_abs_diff"] = float(diff.abs().max()) if diff.notna().any() else np.nan
                if col in PRICE_COLS:
                    den = pd.to_numeric(exact[rp], errors="coerce").replace(0, np.nan)
                    bps = diff / den * 10000.0
                    row[f"{col}_max_abs_diff_bps"] = float(bps.abs().max()) if bps.notna().any() else np.nan
        bar_rows.append(row)
    return pd.DataFrame(daily_rows), pd.DataFrame(bar_rows), pipeline_dirs


def safe_predict_score(model, x: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[:, 1][0])
    return float(model.predict(x)[0])


def build_prediction_comparison(artifacts: list, cache_dir: Path, context_dir: Path, date: str, cutoff: str, benchmark_symbols: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import joblib
    import argparse as _argparse
    import pipelines.run_intraday_nextday_signals as rt

    feature_rows = []
    signal_rows = []
    args = _argparse.Namespace(
        context_dir=str(context_dir),
        cutoff_time=cutoff,
        benchmark_symbols=benchmark_symbols,
    )
    target_date = pd.to_datetime(yyyymmdd_to_dash(date)).normalize()
    for art in artifacts:
        try:
            samples_path = rt.resolve_repo_path(art.metadata.get("samples"), art.stock_code)
            if samples_path is None or not samples_path.exists():
                signal_rows.append({"stock_code": art.stock_code, "artifact_name": art.artifact_name, "error": "samples_missing"})
                continue
            cols = [x.strip() for x in (art.artifact_dir / "feature_columns.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
            med = pd.read_csv(art.artifact_dir / "feature_median.csv", index_col=0)["median"]
            samples = pd.read_csv(samples_path, parse_dates=["date"])
            pipe_df = samples.copy()
            live_df = rt.overlay_current_day_from_cache(samples, art.stock_code, date, cache_dir, cutoff_time=cutoff)
            intraday_path = rt.resolve_intraday_path(art, date, cache_dir)
            live_df = rt.add_scoring_features(
                live_df, intraday_path, cache_dir, art.stock_code,
                cutoff_time=cutoff, scoring_trade_date=date, benchmark_symbols=benchmark_symbols,
            )
            req = rt.infer_runtime_requirement(art)
            live_df, context_meta = rt.apply_realtime_context_to_df(live_df, art, date, req, args)
            live_df, lagged_filled, lagged_missing = rt.fill_lagged_daily_features_from_current_sample(live_df, samples, date, cols)

            def pick_day(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
                if "date" not in df.columns or df.empty:
                    return pd.DataFrame(), "missing"
                mask = pd.to_datetime(df["date"], errors="coerce").dt.normalize() == target_date
                if mask.any():
                    return df.loc[mask].tail(1).copy(), "exact"
                return df.tail(1).copy(), "fallback_latest"

            pipe_day, pipe_date_status = pick_day(pipe_df)
            live_day, live_date_status = pick_day(live_df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True))
            for c in cols:
                if c not in pipe_day.columns:
                    pipe_day[c] = np.nan
                if c not in live_day.columns:
                    live_day[c] = np.nan
            pipe_x_raw = pipe_day[cols].apply(pd.to_numeric, errors="coerce")
            live_x_raw = live_day[cols].apply(pd.to_numeric, errors="coerce")
            pipe_x = pipe_x_raw.fillna(med)
            live_x = live_x_raw.fillna(med)
            model = joblib.load(art.artifact_dir / "model.joblib")
            pipe_score = safe_predict_score(model, pipe_x)
            live_score = safe_predict_score(model, live_x)
            threshold = float(art.metadata.get("threshold", np.nan))
            feature_diff_count = 0
            max_abs_diff = 0.0
            for col in cols:
                pv = pipe_x_raw[col].iloc[-1]
                lv = live_x_raw[col].iloc[-1]
                if pd.isna(pv) and pd.isna(lv):
                    continue
                abs_diff = float(abs((lv if pd.notna(lv) else np.nan) - (pv if pd.notna(pv) else np.nan))) if pd.notna(pv) and pd.notna(lv) else np.nan
                changed = (pd.isna(pv) != pd.isna(lv)) or (pd.notna(abs_diff) and abs_diff > 1e-12)
                if changed:
                    feature_diff_count += 1
                    if pd.notna(abs_diff):
                        max_abs_diff = max(max_abs_diff, abs_diff)
                    feature_rows.append({
                        "stock_code": art.stock_code,
                        "artifact_name": art.artifact_name,
                        "feature": col,
                        "pipeline_value": pv,
                        "collected_value": lv,
                        "abs_diff": abs_diff,
                        "pipeline_missing": bool(pd.isna(pv)),
                        "collected_missing": bool(pd.isna(lv)),
                    })
            signal_rows.append({
                "stock_code": art.stock_code,
                "artifact_name": art.artifact_name,
                "pipeline_date_status": pipe_date_status,
                "collected_date_status": live_date_status,
                "pipeline_missing_features": int(pipe_x_raw.isna().sum(axis=1).iloc[-1]) if not pipe_x_raw.empty else len(cols),
                "collected_missing_features": int(live_x_raw.isna().sum(axis=1).iloc[-1]) if not live_x_raw.empty else len(cols),
                "feature_diff_count": feature_diff_count,
                "feature_max_abs_diff": max_abs_diff,
                "pipeline_score": pipe_score,
                "collected_score": live_score,
                "score_diff": live_score - pipe_score,
                "threshold": threshold,
                "pipeline_raw_pass": bool(pipe_score >= threshold) if np.isfinite(threshold) else "",
                "collected_raw_pass": bool(live_score >= threshold) if np.isfinite(threshold) else "",
                "context_status": context_meta.get("context_status", ""),
                "lagged_daily_missing_features": ",".join(lagged_missing),
            })
        except Exception as exc:
            signal_rows.append({"stock_code": getattr(art, "stock_code", ""), "artifact_name": getattr(art, "artifact_name", ""), "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(feature_rows), pd.DataFrame(signal_rows)


def compare_signal_and_portfolio(date: str, signal_dir: Path, portfolio_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_dir = signal_dir / date
    date_dash = yyyymmdd_to_dash(date)
    files = {
        "all_scores": day_dir / "all_scores.csv",
        "buy_signals": day_dir / "buy_signals.csv",
        "rejected_scores": day_dir / "rejected_scores.csv",
        "portfolio_orders": portfolio_dir / f"daily_portfolio_orders_{date_dash}.csv",
        "portfolio_selected": portfolio_dir / f"daily_portfolio_selected_{date_dash}.csv",
        "portfolio_rejected": portfolio_dir / f"daily_portfolio_rejected_{date_dash}.csv",
        "portfolio_report": portfolio_dir / f"daily_portfolio_report_{date_dash}.json",
    }
    inv = []
    for name, path in files.items():
        inv.append({"name": name, "path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0})

    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    buy = read(files["buy_signals"])
    selected = read(files["portfolio_selected"])
    orders = read(files["portfolio_orders"])
    rows = []
    stock_cols = ["stock_code", "symbol", "code"]
    def stock_set(df: pd.DataFrame) -> set[str]:
        for c in stock_cols:
            if c in df.columns:
                return set(df[c].astype(str))
        return set()
    buy_set = stock_set(buy)
    sel_set = stock_set(selected) or stock_set(orders)
    for stock in sorted(buy_set | sel_set):
        rows.append({
            "stock_code": stock,
            "in_buy_signals": stock in buy_set,
            "in_portfolio": stock in sel_set,
            "status": "ok" if (stock in buy_set and stock in sel_set) else ("portfolio_without_signal" if stock in sel_set else "signal_not_selected"),
        })
    return pd.DataFrame(inv), pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD, e.g. 20260513")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols; default: auto-discover under cache")
    ap.add_argument("--cache-dir", default="saved_data/akshare_realtime_cache")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cutoff-time", default="14:55")
    ap.add_argument("--adjustflag", default="3", help="BaoStock adjustflag: 3=none")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--saved-data-dir", default="saved_data")
    ap.add_argument("--watchlist", default="selected_watchlist.txt")
    ap.add_argument("--model-policy", choices=["preferred", "all"], default="all")
    ap.add_argument("--context-dir", default="saved_data/realtime_context")
    ap.add_argument("--context-config", default="configs/realtime_context_sources.toml")
    ap.add_argument("--signal-dir", default="saved_data/intraday_nextday_signals")
    ap.add_argument("--portfolio-dir", default="portfolio_reports")
    ap.add_argument("--benchmark-symbols", default="000300.SH,000001.SH,399001.SZ,399006.SZ")
    ap.add_argument("--run-premarket-update", action="store_true", help="Run run_premarket_history_update.py before comparing.")
    ap.add_argument("--premarket-resume", action="store_true")
    ap.add_argument("--premarket-cache-mode", default=None)
    ap.add_argument("--premarket-feature-cache-mode", default=None)
    ap.add_argument("--skip-baostock-query", action="store_true", help="Skip direct BaoStock 5m query; compare local pipeline/cache artifacts only.")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir or f"saved_data/baostock_compare/{args.date}")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = discover_symbols(cache_dir, args.date, args.symbols)
    if args.run_premarket_update:
        run_premarket_update(args, symbols)

    import pipelines.run_intraday_nextday_signals as rt
    watchlist = set(rt.read_watchlist(Path(args.watchlist))) if Path(args.watchlist).exists() else set(symbols)
    artifacts = rt.load_artifacts(Path(args.models_dir), watchlist, args.model_policy)
    if symbols:
        symbol_set = set(symbols)
        artifacts = [a for a in artifacts if a.stock_code in symbol_set]
    print(f"[INFO] date={args.date} symbols={len(symbols)} out={out_dir}")

    rows = []
    errors = []
    for sym in symbols:
        print(f"[COMPARE] {sym}", flush=True)
        try:
            collected, daily_row, sym_dir = read_local_collected(cache_dir, args.date, sym)
            bao = empty_bar_frame("baostock") if args.skip_baostock_query else query_baostock_5m(sym, args.date, adjustflag=args.adjustflag)

            bao.to_csv(out_dir / f"{sym}_baostock_5m.csv", index=False, encoding="utf-8-sig")
            collected.to_csv(out_dir / f"{sym}_collected_5m_normalized.csv", index=False, encoding="utf-8-sig")

            row, aligned, missing = calc_summary(sym, collected, bao, daily_row, args.cutoff_time)
            aligned.to_csv(out_dir / f"{sym}_aligned_diff.csv", index=False, encoding="utf-8-sig")
            missing.to_csv(out_dir / f"{sym}_missing_times.csv", index=False, encoding="utf-8-sig")
            rows.append(row)
        except Exception as exc:
            err = {"symbol": sym, "error": f"{type(exc).__name__}: {exc}"}
            print("[ERROR]", err, file=sys.stderr)
            errors.append(err)
            if args.fail_fast:
                raise

    try:
        daily_cmp, bar_cmp, pipeline_dirs = compare_pipeline_vs_collected_data(symbols, artifacts, cache_dir, args.date, args.cutoff_time)
        inventory = pipeline_file_inventory(pipeline_dirs, args.date)
        feature_diff, signal_diff = build_prediction_comparison(
            artifacts,
            cache_dir,
            Path(args.context_dir),
            args.date,
            args.cutoff_time,
            args.benchmark_symbols,
        )
        portfolio_files, portfolio_signal_diff = compare_signal_and_portfolio(args.date, Path(args.signal_dir), Path(args.portfolio_dir))
        daily_cmp.to_csv(out_dir / "pipeline_vs_collected_daily.csv", index=False, encoding="utf-8-sig")
        bar_cmp.to_csv(out_dir / "pipeline_vs_collected_5m.csv", index=False, encoding="utf-8-sig")
        inventory.to_csv(out_dir / "pipeline_file_inventory.csv", index=False, encoding="utf-8-sig")
        feature_diff.to_csv(out_dir / "prediction_feature_diff.csv", index=False, encoding="utf-8-sig")
        signal_diff.to_csv(out_dir / "prediction_signal_diff.csv", index=False, encoding="utf-8-sig")
        portfolio_files.to_csv(out_dir / "portfolio_file_inventory.csv", index=False, encoding="utf-8-sig")
        portfolio_signal_diff.to_csv(out_dir / "portfolio_signal_diff.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        err = {"stage": "pipeline_collected_prediction_portfolio_compare", "error": f"{type(exc).__name__}: {exc}"}
        print("[ERROR]", err, file=sys.stderr)
        errors.append(err)
        if args.fail_fast:
            raise

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["severity", "symbol"])
    summary.to_csv(out_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "date": args.date,
        "cutoff_time": args.cutoff_time,
        "symbols": symbols,
        "n_symbols": len(symbols),
        "n_ok": int(len(rows)),
        "n_error": int(len(errors)),
        "errors": errors,
        "summary_rows": rows,
        "outputs": {
            "comparison_summary_csv": str(out_dir / "comparison_summary.csv"),
            "pipeline_vs_collected_daily_csv": str(out_dir / "pipeline_vs_collected_daily.csv"),
            "pipeline_vs_collected_5m_csv": str(out_dir / "pipeline_vs_collected_5m.csv"),
            "pipeline_file_inventory_csv": str(out_dir / "pipeline_file_inventory.csv"),
            "prediction_feature_diff_csv": str(out_dir / "prediction_feature_diff.csv"),
            "prediction_signal_diff_csv": str(out_dir / "prediction_signal_diff.csv"),
            "portfolio_signal_diff_csv": str(out_dir / "portfolio_signal_diff.csv"),
            "out_dir": str(out_dir),
        },
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"n_ok": len(rows), "n_error": len(errors), "summary": str(out_dir / "comparison_summary.csv")}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
