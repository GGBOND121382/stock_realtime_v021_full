#!/usr/bin/env python3
"""Build a ML4T-compatible assets.h5 from Yahoo Finance data.

This is a compatibility builder for running the ML4T Chapter 7 notebooks when
the original Quandl WIKI/PRICES CSV is unavailable. It writes the HDF keys that
Chapter 7 expects:

  /quandl/wiki/prices
  /quandl/wiki/stocks
  /us_equities/stocks

The price source is Yahoo Finance, not the original Quandl WIKI dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def ticker_batches(tickers: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(tickers), batch_size):
        yield tickers[i : i + batch_size]


def pick_tickers(data_dir: Path, max_tickers: int) -> pd.DataFrame:
    wiki = pd.read_csv(data_dir / "wiki_stocks.csv")
    wiki["ticker"] = wiki["code"].map(normalize_ticker)
    meta = pd.read_csv(data_dir / "us_equities_meta_data.csv")
    meta["ticker"] = meta["ticker"].map(normalize_ticker)
    merged = wiki[["ticker", "name"]].merge(
        meta.drop(columns=["name"], errors="ignore"),
        on="ticker",
        how="left",
    )
    merged["marketcap"] = pd.to_numeric(merged.get("marketcap"), errors="coerce")
    merged = merged.drop_duplicates("ticker")
    merged = merged[merged["ticker"].str.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", na=False)]
    merged = merged.sort_values(["marketcap", "ticker"], ascending=[False, True], na_position="last")
    if max_tickers > 0:
        merged = merged.head(max_tickers)
    return merged.reset_index(drop=True)


def yahoo_to_wiki_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        tickers = raw.columns.get_level_values(0).unique()
        for ticker in tickers:
            part = raw[ticker].copy()
            if part.dropna(how="all").empty:
                continue
            part["ticker"] = ticker
            frames.append(part)
    else:
        part = raw.copy()
        part["ticker"] = ""
        frames.append(part)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames)
    data.index.name = "date"
    data = data.reset_index()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    data = data.dropna(subset=["date", "ticker", "close", "adj_close"])
    data = data[data["close"] > 0]
    ratio = data["adj_close"] / data["close"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(data["date"]).dt.normalize(),
            "ticker": data["ticker"].map(normalize_ticker),
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "ex-dividend": 0.0,
            "split_ratio": 1.0,
            "adj_open": data["open"] * ratio,
            "adj_high": data["high"] * ratio,
            "adj_low": data["low"] * ratio,
            "adj_close": data["adj_close"],
            "adj_volume": data["volume"] / ratio.replace(0, np.nan),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["adj_close"])
    return out.sort_values(["date", "ticker"])


def download_prices(tickers: List[str], start: str, end: str, batch_size: int, pause: float) -> pd.DataFrame:
    parts = []
    failures = []
    for batch_id, batch in enumerate(ticker_batches(tickers, batch_size), start=1):
        label = ",".join(batch[:5]) + ("..." if len(batch) > 5 else "")
        print(f"[DOWNLOAD] batch={batch_id} tickers={len(batch)} {label}", flush=True)
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            part = yahoo_to_wiki_frame(raw)
            if not part.empty:
                parts.append(part)
        except Exception as exc:
            failures.extend({"ticker": t, "error": f"{type(exc).__name__}: {exc}"} for t in batch)
        if pause > 0:
            time.sleep(pause)
    prices = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if prices.empty:
        raise RuntimeError("No Yahoo price data downloaded.")
    present = set(prices["ticker"].unique())
    for ticker in tickers:
        if ticker not in present:
            failures.append({"ticker": ticker, "error": "no_price_rows"})
    return prices, pd.DataFrame(failures).drop_duplicates() if failures else pd.DataFrame(columns=["ticker", "error"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="machine-learning-for-trading/data")
    ap.add_argument("--assets", default="")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--max-tickers", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--pause", type=float, default=0.5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    assets_path = Path(args.assets).resolve() if args.assets else data_dir / "assets.h5"
    data_dir.mkdir(parents=True, exist_ok=True)

    meta = pick_tickers(data_dir, int(args.max_tickers))
    tickers = meta["ticker"].tolist()
    print(f"[INFO] selected_tickers={len(tickers)} assets={assets_path}", flush=True)
    prices, failures = download_prices(tickers, args.start, args.end, int(args.batch_size), float(args.pause))
    keep = sorted(set(prices["ticker"].unique()))
    meta = meta[meta["ticker"].isin(keep)].copy()
    if "sector" not in meta.columns:
        meta["sector"] = "unknown"
    meta["sector"] = meta["sector"].fillna("unknown")
    if "ipoyear" not in meta.columns:
        meta["ipoyear"] = np.nan
    if "marketcap" not in meta.columns:
        meta["marketcap"] = np.nan
    stocks = meta.rename(columns={"ticker": "ticker"}).set_index("ticker")

    if assets_path.exists():
        assets_path.unlink()
    with pd.HDFStore(assets_path) as store:
        store.put("quandl/wiki/prices", prices.set_index(["date", "ticker"]).sort_index())
        store.put("quandl/wiki/stocks", meta.rename(columns={"ticker": "code"}))
        store.put("us_equities/stocks", stocks)

    failures.to_csv(data_dir / "yahoo_assets_failures.csv", index=False)
    summary = {
        "source": "Yahoo Finance compatibility data, not original Quandl WIKI",
        "assets": str(assets_path),
        "start": args.start,
        "end": args.end,
        "requested_tickers": len(tickers),
        "tickers_with_prices": len(keep),
        "price_rows": int(len(prices)),
        "failures": int(len(failures)),
    }
    (data_dir / "yahoo_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
