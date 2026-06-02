#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def normalize_stock_code(x: Any) -> str:
    s = str(x or "").strip().upper()
    if not s:
        return s
    if s.isdigit() and len(s) == 6:
        return f"{s}.SH" if s.startswith(("6", "9")) else f"{s}.SZ"
    if s.startswith("SH."):
        return f"{s[3:]}.SH"
    if s.startswith("SZ."):
        return f"{s[3:]}.SZ"
    return s


def compact_code(code: str) -> str:
    return normalize_stock_code(code).split(".")[0]


def load_signal_codes(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    if path.is_dir():
        candidates = [path / "buy_signals.csv", path / "all_scores.csv"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return set()
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "stock_code" not in df.columns:
        return set()
    return {normalize_stock_code(x) for x in df["stock_code"].dropna().unique()}


def load_account_codes(path: Optional[Path]) -> Set[str]:
    if path is None or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    holdings = data.get("holdings", {})
    out: Set[str] = set()
    if isinstance(holdings, dict):
        for code, h in holdings.items():
            if not isinstance(h, dict):
                continue
            shares = pd.to_numeric(pd.Series([h.get("shares", 0)]), errors="coerce").iloc[0]
            mv = pd.to_numeric(pd.Series([h.get("market_value", 0)]), errors="coerce").iloc[0]
            if pd.notna(shares) and shares > 0 or pd.notna(mv) and mv > 0:
                out.add(normalize_stock_code(code))
    elif isinstance(holdings, list):
        for h in holdings:
            if not isinstance(h, dict):
                continue
            code = h.get("stock_code")
            shares = pd.to_numeric(pd.Series([h.get("shares", 0)]), errors="coerce").iloc[0]
            mv = pd.to_numeric(pd.Series([h.get("market_value", 0)]), errors="coerce").iloc[0]
            if code and (pd.notna(shares) and shares > 0 or pd.notna(mv) and mv > 0):
                out.add(normalize_stock_code(code))
    return out


def _walk_json_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json_strings(v)
    elif isinstance(obj, str):
        yield obj


def sample_paths_from_metadata(saved_models: Path) -> Dict[str, list[Path]]:
    out: Dict[str, list[Path]] = {}
    if not saved_models.exists():
        return out
    for meta in saved_models.glob("*/*/metadata.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        stock = normalize_stock_code(data.get("stock_code") or meta.parents[1].name)
        paths: list[Path] = []
        for s in _walk_json_strings(data):
            low = s.lower()
            if "sample" not in low or not low.endswith(".csv"):
                continue
            p = Path(s)
            if not p.is_absolute():
                p = (PROJECT_DIR / p).resolve()
            if p.exists():
                paths.append(p)
        if paths:
            out.setdefault(stock, []).extend(paths)
    return out


def candidate_pipeline_csvs(saved_data_dir: Path, stock: str) -> list[Path]:
    code = compact_code(stock)
    paths: list[Path] = []
    root = saved_data_dir / f"{code}_pipeline_out"
    if root.exists():
        paths.extend([
            root / "01_samples" / "training_samples.csv",
            root / "00_base" / "daily_features.csv",
            root / f"{code}_5m.csv",
            root / "00_base" / f"{code}_5m.csv",
        ])
    return [p for p in paths if p.exists()]


def read_close_series(path: Path) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    date_col = None
    for c in ["date", "trade_date", "datetime", "time"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return None

    close_col = None
    for c in ["close", "stock_close", "daily_close", "收盘", "close_price"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        return None

    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    close = pd.to_numeric(df[close_col], errors="coerce")
    s = pd.Series(close.values, index=dates)
    s = s[s.index.notna()].dropna()
    if s.empty:
        return None
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def build_history(
    saved_models: Path,
    saved_data_dir: Path,
    codes: Set[str],
    cutoff_date: Optional[str],
    include_current_date: bool,
    min_rows: int,
) -> pd.DataFrame:
    meta_paths = sample_paths_from_metadata(saved_models)
    all_codes = set(codes) or set(meta_paths.keys())

    if not all_codes and saved_data_dir.exists():
        for p in saved_data_dir.glob("*_pipeline_out"):
            m = re.match(r"(\d{6})_pipeline_out", p.name)
            if m:
                all_codes.add(normalize_stock_code(m.group(1)))

    series_by_code: Dict[str, pd.Series] = {}
    for stock in sorted(all_codes):
        paths: list[Path] = []
        paths.extend(meta_paths.get(stock, []))
        paths.extend(candidate_pipeline_csvs(saved_data_dir, stock))

        best: Optional[pd.Series] = None
        for p in paths:
            s = read_close_series(p)
            if s is None or len(s) < min_rows:
                continue
            if best is None or len(s.dropna()) > len(best.dropna()):
                best = s
        if best is not None and len(best.dropna()) >= min_rows:
            series_by_code[stock] = best.rename(stock)

    if not series_by_code:
        return pd.DataFrame()

    hist = pd.concat(series_by_code.values(), axis=1).sort_index()
    hist = hist.dropna(axis=1, how="all")

    if cutoff_date:
        cutoff = pd.Timestamp(cutoff_date).normalize()
        if include_current_date:
            hist = hist.loc[hist.index <= cutoff]
        else:
            hist = hist.loc[hist.index < cutoff]

    hist = hist.dropna(axis=1, how="all")
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-models", default="saved_models")
    ap.add_argument("--saved-data-dir", default="saved_data")
    ap.add_argument("--signals", default=None, help="Signal CSV or signal directory. Used to restrict stock universe.")
    ap.add_argument("--account", default=None, help="Optional account JSON. Existing holdings are included in risk history.")
    ap.add_argument("--date", default=None, help="Decision date. Live mode should exclude this date.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--include-current-date", action="store_true", help="Include rows <= date. Default excludes date for live intraday use.")
    args = ap.parse_args()

    signals = Path(args.signals) if args.signals else None
    codes = load_signal_codes(signals)
    if args.account:
        codes |= load_account_codes(Path(args.account))

    hist = build_history(
        saved_models=Path(args.saved_models),
        saved_data_dir=Path(args.saved_data_dir),
        codes=codes,
        cutoff_date=args.date,
        include_current_date=bool(args.include_current_date),
        min_rows=int(args.min_rows),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if hist.empty:
        print(f"[ERROR] risk history is empty; refusing to write unusable history: {out}")
        return 2

    export = hist.reset_index().rename(columns={"index": "date"})
    if export.columns[0] != "date":
        export = export.rename(columns={export.columns[0]: "date"})
    export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if len(export) < int(args.min_rows) or len(export.columns) < 2:
        print(f"[ERROR] risk history invalid: shape={export.shape}, min_rows={args.min_rows}, out={out}")
        return 2
    export.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote risk history: {out} rows={len(export)} cols={len(export.columns)-1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
