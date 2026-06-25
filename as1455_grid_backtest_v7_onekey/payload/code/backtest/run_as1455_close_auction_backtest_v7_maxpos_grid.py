#!/usr/bin/env python3
"""AS1455 close-auction / last-5min execution backtest.

This script is intentionally independent from the older next-open/daily backtest
code.  It uses AS1455 predictions as signals at date T and simulates execution
at the same day's 15:00 close price, which is the closest historical proxy for
participating in the closing call auction after a 14:55 signal.

V7 scope:
- long-only close-auction execution using same-day 15:00 close
- full-universe ranking, trade mainboard-only by default
- max-positions / buy-candidate-rank / sell-rank hysteresis
- T+1 sell restriction
- close_auction_simple and close_auction_skip_limit execution profiles
- 100-share lot rounding for buys/sells
- optional rebalance interval/offset and detailed trade/win-rate/drawdown/amount logs
- optional last-5min volume/amount capacity constraint and partial fills
- optional date-specific historical ST status CSV
- optional corporate-action CSV for exact cash-dividend/share-multiplier handling
- default synthetic share-factor adjustment inferred from raw preclose/previous
  close to keep holding value continuous across ex-dividend/ex-right dates

Important limitation:
- Exact dividend/split accounting requires an explicit corporate-action file.
  The default synthetic share-factor mode preserves total-return continuity from
  preclose/previous-close, but it cannot distinguish cash dividends from share
  splits/bonus shares or reconstruct the true broker statement.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PRED_KEY = "predictions"
DEFAULT_INITIAL_CASH = 200_000.0
DEFAULT_BUY_CANDIDATE_RANK = 75
DEFAULT_SELL_RANK = 75
DEFAULT_COMMISSION_RATE = 0.000085
DEFAULT_STAMP_TAX_RATE = 0.0005
DEFAULT_SLIPPAGE_BPS = 0.0
DEFAULT_LOT_SIZE = 100
DEFAULT_MIN_COMMISSION = 5.0
DEFAULT_TRANSFER_FEE_RATE = 0.00001
DEFAULT_MAX_POSITIONS = 25
DEFAULT_REBALANCE_EVERY = 1
DEFAULT_REBALANCE_OFFSET = 0
DEFAULT_PARTICIPATION_RATE = 0.05


@dataclass(frozen=True)
class TradeConfig:
    # Portfolio semantics:
    # - max_positions controls the portfolio capacity.
    # - buy_candidate_rank controls the candidate pool used only when the
    #   portfolio has empty slots.
    # - sell_rank controls when existing positions are sold.
    max_positions: int
    buy_candidate_rank: int
    sell_rank: int
    rebalance_every: int
    rebalance_offset: int
    initial_cash: float
    commission_rate: float
    stamp_tax_rate: float
    transfer_fee_rate: float
    slippage_bps: float
    profile: str
    mainboard_only: bool
    min_price: float
    limit_eps: float
    lot_size: int
    min_commission: float
    exclude_st: bool
    capacity_mode: str
    participation_rate: float
    corporate_action_mode: str
    corporate_action_threshold: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def normalize_symbol(value: object) -> str:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")
    s = s.replace("sz", "").replace("sh", "") if re.fullmatch(r"(?i)(sz|sh)\d{6}", s) else s
    m = re.search(r"(\d{6})", s)
    if m:
        code = m.group(1)
    elif re.fullmatch(r"\d{1,6}", s):
        code = s.zfill(6)
    else:
        return s.upper()
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def compact_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d{1,6}", s):
        return s.zfill(6)
    return s


def infer_board(symbol: str) -> str:
    code = compact_symbol(symbol)
    if code.startswith(("600", "601", "603", "605")):
        return "sh_mainboard"
    if code.startswith(("000", "001", "002", "003")):
        return "sz_mainboard"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("8", "4", "920")):
        return "bse"
    return "unknown"


def is_mainboard_symbol(symbol: str) -> bool:
    return infer_board(symbol) in {"sh_mainboard", "sz_mainboard"}


def load_predictions(path: Path, key: str | None, score_col: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf", ".hdf5"}:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
            if not keys:
                raise ValueError(f"empty HDF: {path}")
            if key:
                k = key if key.startswith("/") else f"/{key}"
                if k not in keys:
                    raise KeyError(f"key {k!r} not found in {path}; available={keys}")
            else:
                k = "/predictions" if "/predictions" in keys else keys[0]
            df = store[k]
    else:
        df = pd.read_csv(path)

    # Convert common index layouts to columns.
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if "symbol" in names or "date" in names:
            df = df.reset_index()
        else:
            # Common training output is MultiIndex(symbol, date).
            df = df.reset_index()
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: "symbol", df.columns[1]: "date"})
    else:
        if df.index.name in {"symbol", "date"}:
            df = df.reset_index()

    if "symbol" not in df.columns:
        for c in ["ticker", "code", "asset"]:
            if c in df.columns:
                df = df.rename(columns={c: "symbol"})
                break
    if "date" not in df.columns:
        for c in ["trade_date", "datetime", "dt"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    if "symbol" not in df.columns or "date" not in df.columns:
        raise ValueError(f"predictions need symbol/date columns or index; columns={list(df.columns)} index={df.index.names}")

    if score_col is not None:
        if score_col not in df.columns:
            # Allow numeric column names from HDF to be passed as strings.
            matches = [c for c in df.columns if str(c) == str(score_col)]
            if not matches:
                raise KeyError(f"score column {score_col!r} not found; columns={list(df.columns)}")
            score_col_actual = matches[0]
        else:
            score_col_actual = score_col
    else:
        candidates = [c for c in df.columns if c not in {"symbol", "date"}]
        numeric_candidates = []
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                numeric_candidates.append(c)
        if not numeric_candidates:
            raise ValueError("no numeric score column found in predictions")
        score_col_actual = numeric_candidates[0]

    out = pd.DataFrame(
        {
            "symbol": df["symbol"].map(normalize_symbol),
            "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "score": pd.to_numeric(df[score_col_actual], errors="coerce"),
        }
    )
    out = out.dropna(subset=["symbol", "date", "score"])
    out = out[out["symbol"].astype(str).str.len() > 0]
    out = out.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])
    if out.empty:
        raise ValueError("predictions are empty after normalization")
    return out.reset_index(drop=True)


def read_universe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol", "board", "is_mainboard"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        for c in ["code", "ticker", "asset"]:
            if c in df.columns:
                df = df.rename(columns={c: "symbol"})
                break
    if "symbol" not in df.columns:
        raise ValueError(f"universe file has no symbol/code column: {path}")
    out = pd.DataFrame({"symbol": df["symbol"].map(normalize_symbol)})
    if "board" in df.columns:
        out["board"] = df["board"].astype(str)
    else:
        out["board"] = out["symbol"].map(infer_board)
    out["is_mainboard"] = out["symbol"].map(is_mainboard_symbol)
    if "name" in df.columns:
        out["name"] = df["name"].astype(str)
    return out.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def load_st_symbols(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        for c in ["code", "ticker"]:
            if c in df.columns:
                df = df.rename(columns={c: "symbol"})
                break
    if "symbol" not in df.columns:
        return set()
    return set(df["symbol"].map(normalize_symbol).dropna().astype(str))


def load_st_status(path: Path | None) -> pd.DataFrame:
    """Load optional date-specific ST status.

    Accepted columns:
    - symbol/code/ticker
    - date/trade_date
    - is_st, or name containing ST/*ST

    If no valid file is supplied, returns an empty DataFrame.
    """
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date", "symbol", "is_st_hist"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        for c in ["code", "ticker"]:
            if c in df.columns:
                df = df.rename(columns={c: "symbol"})
                break
    if "date" not in df.columns:
        for c in ["trade_date", "dt"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    if "symbol" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "symbol", "is_st_hist"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
        "symbol": df["symbol"].map(normalize_symbol),
    })
    if "is_st" in df.columns:
        v = df["is_st"]
        if v.dtype == bool:
            out["is_st_hist"] = v
        else:
            out["is_st_hist"] = v.astype(str).str.lower().isin(["1", "true", "yes", "y", "st", "*st"])
    elif "name" in df.columns:
        name = df["name"].astype(str).str.upper()
        out["is_st_hist"] = name.str.contains("ST", regex=False)
    else:
        out["is_st_hist"] = True
    return out.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def load_corporate_actions(path: Path | None) -> pd.DataFrame:
    """Load optional exact corporate actions.

    Expected columns: symbol, ex_date/date, cash_dividend_per_share, share_multiplier.
    cash_dividend_per_share is in raw currency per pre-action share.
    share_multiplier is post-action shares per old share, e.g. 1.4 for 10-for-4
    bonus/transfer/split; default 1.0.
    """
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date", "symbol", "cash_dividend_per_share", "share_multiplier"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        for c in ["code", "ticker"]:
            if c in df.columns:
                df = df.rename(columns={c: "symbol"})
                break
    if "date" not in df.columns:
        for c in ["ex_date", "trade_date"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    if "symbol" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "symbol", "cash_dividend_per_share", "share_multiplier"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
        "symbol": df["symbol"].map(normalize_symbol),
        "cash_dividend_per_share": pd.to_numeric(df.get("cash_dividend_per_share", 0.0), errors="coerce").fillna(0.0),
        "share_multiplier": pd.to_numeric(df.get("share_multiplier", 1.0), errors="coerce").fillna(1.0),
    })
    return out.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def _parse_intraday_time(value: object) -> str:
    """Parse intraday time to HHMMSS.

    BaoStock/cache variants observed in this project include:
    - 20200102145500000 / 20200102145500: full date + HHMMSS + millis
    - 145500000 / 150000000: HHMMSS + millis
    - 145500: HHMMSS
    - 1455: HHMM
    """
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""

    # Full date + time, e.g. YYYYMMDDHHMMSS or YYYYMMDDHHMMSSmmm.
    if len(digits) >= 14 and 1900 <= int(digits[:4]) <= 2100:
        return digits[8:14]

    # HHMMSS plus milliseconds/extra precision, e.g. 145500000.
    if len(digits) >= 8:
        return digits[:6]

    if len(digits) == 6:
        return digits
    if len(digits) == 4:
        return digits + "00"
    return ""


def read_last5_from_5m_file(cache_dir: Path, symbol: str) -> pd.DataFrame:
    code = compact_symbol(symbol)
    candidates = [
        cache_dir / f"{code}_5m_raw.csv",
        cache_dir / f"{symbol}_5m_raw.csv",
        cache_dir / f"{code}.csv",
        cache_dir / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
    if path is None:
        # Some caches are nested. Fall back to a recursive lookup using common
        # symbol spellings, but keep exact-name candidates preferred above.
        patterns = [f"**/{code}_5m_raw.csv", f"**/{symbol}_5m_raw.csv", f"**/*{code}*_5m_raw.csv"]
        hits = []
        for pattern in patterns:
            hits.extend([p for p in cache_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0])
        if hits:
            path = sorted(set(hits), key=lambda x: len(str(x)))[0]
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_col = None
    for c in ["date", "trade_date"]:
        if c in df.columns:
            date_col = c
            break
    time_col = "time" if "time" in df.columns else None
    if time_col is None:
        for c in ["datetime", "trade_time"]:
            if c in df.columns:
                time_col = c
                break
    if date_col is None and "datetime" in df.columns:
        date_series = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    elif date_col is not None:
        date_series = pd.to_datetime(df[date_col].astype(str), errors="coerce").dt.normalize()
    else:
        return pd.DataFrame()
    if time_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": date_series,
        "symbol": normalize_symbol(symbol),
        "bar_time": df[time_col].map(_parse_intraday_time),
        "last5_volume": pd.to_numeric(df.get("volume", np.nan), errors="coerce"),
        "last5_amount": pd.to_numeric(df.get("amount", np.nan), errors="coerce"),
    })
    # Last 5 minutes after AS1455 cutoff: prefer bar(s) after 14:55 and <=15:00.
    out = out[(out["bar_time"] > "145500") & (out["bar_time"] <= "150000")]
    if out.empty:
        return pd.DataFrame()
    out = out.groupby(["date", "symbol"], as_index=False).agg(
        last5_volume=("last5_volume", "sum"),
        last5_amount=("last5_amount", "sum"),
    )
    return out


def load_last5_panel(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["date", "symbol", "last5_volume", "last5_amount"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        for c in ["code", "ticker"]:
            if c in df.columns:
                df = df.rename(columns={c: "symbol"})
                break
    if "date" not in df.columns:
        for c in ["trade_date", "dt"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break
    if "symbol" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "symbol", "last5_volume", "last5_amount"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
        "symbol": df["symbol"].map(normalize_symbol),
        "last5_volume": pd.to_numeric(df.get("last5_volume", df.get("volume", np.nan)), errors="coerce"),
        "last5_amount": pd.to_numeric(df.get("last5_amount", df.get("amount", np.nan)), errors="coerce"),
    })
    return out.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def read_raw_daily_file(cache_dir: Path, symbol: str) -> pd.DataFrame:
    code = compact_symbol(symbol)
    candidates = [
        cache_dir / f"{code}_daily_raw.csv",
        cache_dir / f"{symbol}_daily_raw.csv",
        cache_dir / f"{code}.csv",
        cache_dir / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
    if path is None:
        # Some caches are nested. Fall back to a recursive lookup using common
        # symbol spellings, but keep exact-name candidates preferred above.
        patterns = [f"**/{code}_5m_raw.csv", f"**/{symbol}_5m_raw.csv", f"**/*{code}*_5m_raw.csv"]
        hits = []
        for pattern in patterns:
            hits.extend([p for p in cache_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0])
        if hits:
            path = sorted(set(hits), key=lambda x: len(str(x)))[0]
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce").dt.normalize()})
    out["symbol"] = normalize_symbol(symbol)
    out["raw_close_1500"] = pd.to_numeric(df["close"], errors="coerce")
    if "preclose" in df.columns:
        out["raw_preclose"] = pd.to_numeric(df["preclose"], errors="coerce")
    else:
        out["raw_preclose"] = out["raw_close_1500"].shift(1)
    if "tradestatus" in df.columns:
        out["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
    else:
        vol = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else np.nan
        out["tradestatus"] = np.where((out["raw_close_1500"] > 0) & (pd.Series(vol).fillna(1) >= 0), 1, np.nan)
    if "volume" in df.columns:
        out["daily_volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        out["daily_volume"] = np.nan
    if "amount" in df.columns:
        out["daily_amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        out["daily_amount"] = np.nan
    out = out.dropna(subset=["date", "raw_close_1500", "raw_preclose"])
    out = out[(out["raw_close_1500"] > 0) & (out["raw_preclose"] > 0)]
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def compute_current_front_factor(raw_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute current/front-adjustment factor from raw preclose.

    event_ratio_d = preclose_d / close_{d-1}.  Current/front factor for date t
    is the product of event ratios after t.
    """
    df = raw_daily.sort_values("date").copy().reset_index(drop=True)
    prev_close = df["raw_close_1500"].shift(1)
    df["prev_raw_close_1500"] = prev_close
    event_ratio = df["raw_preclose"].div(prev_close)
    event_ratio = event_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    future_ratio = event_ratio.shift(-1).fillna(1.0)
    df["qfq_factor_1500"] = future_ratio.iloc[::-1].cumprod().iloc[::-1].to_numpy()
    df["qfq_close_1500"] = df["raw_close_1500"] * df["qfq_factor_1500"]
    df["event_ratio"] = event_ratio
    df["factor_event_abs_pct"] = (event_ratio - 1.0).abs() * 100.0
    return df


def round_cent(x: pd.Series | float) -> pd.Series | float:
    return np.round(x, 2)


def build_execution_panel(
    symbols: Iterable[str],
    raw_daily_cache_dir: Path,
    universe: pd.DataFrame,
    st_symbols: set[str],
    st_status: pd.DataFrame | None = None,
    last5_panel: pd.DataFrame | None = None,
    raw_5m_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_map = universe.set_index("symbol") if not universe.empty else pd.DataFrame()
    frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []
    st_status = st_status if st_status is not None else pd.DataFrame()
    last5_panel = last5_panel if last5_panel is not None else pd.DataFrame()

    for symbol in sorted(set(map(normalize_symbol, symbols))):
        raw = read_raw_daily_file(raw_daily_cache_dir, symbol)
        if raw.empty:
            report_rows.append({"symbol": symbol, "status": "missing_raw_daily", "rows": 0})
            continue
        df = compute_current_front_factor(raw)
        if not universe_map.empty and symbol in universe_map.index:
            board = str(universe_map.loc[symbol].get("board", infer_board(symbol)))
            is_mainboard = bool(universe_map.loc[symbol].get("is_mainboard", is_mainboard_symbol(symbol)))
            name = str(universe_map.loc[symbol].get("name", "")) if "name" in universe_map.columns else ""
        else:
            board = infer_board(symbol)
            is_mainboard = is_mainboard_symbol(symbol)
            name = ""

        df["board"] = board
        df["is_mainboard"] = is_mainboard
        static_is_st = bool(symbol in st_symbols or ("ST" in name.upper()))
        df["is_st"] = static_is_st

        if not st_status.empty:
            ss = st_status[st_status["symbol"].eq(symbol)][["date", "is_st_hist"]]
            if not ss.empty:
                df = df.merge(ss, on="date", how="left")
                df["is_st"] = df["is_st_hist"].fillna(df["is_st"]).astype(bool)
                df = df.drop(columns=["is_st_hist"])

        df["limit_pct"] = np.where(df["is_st"].astype(bool), 0.05, 0.10)
        df["up_limit"] = round_cent(df["raw_preclose"] * (1.0 + df["limit_pct"]))
        df["down_limit"] = round_cent(df["raw_preclose"] * (1.0 - df["limit_pct"]))
        df["tradable"] = df["tradestatus"].fillna(1).eq(1) & df["raw_close_1500"].gt(0)

        # Optional last-5min capacity data, either from a prebuilt panel or from raw 5m cache.
        last5_sym = pd.DataFrame()
        if not last5_panel.empty:
            last5_sym = last5_panel[last5_panel["symbol"].eq(symbol)][["date", "last5_volume", "last5_amount"]]
        elif raw_5m_cache_dir is not None and raw_5m_cache_dir.exists():
            last5_sym = read_last5_from_5m_file(raw_5m_cache_dir, symbol)
        if not last5_sym.empty:
            df = df.merge(last5_sym, on=["date", "symbol"], how="left")
        else:
            df["last5_volume"] = np.nan
            df["last5_amount"] = np.nan

        frames.append(df)
        report_rows.append(
            {
                "symbol": symbol,
                "status": "ok",
                "rows": int(len(df)),
                "first_date": df["date"].min(),
                "last_date": df["date"].max(),
                "board": board,
                "is_mainboard": is_mainboard,
                "static_is_st": static_is_st,
                "st_rows": int(df["is_st"].sum()),
                "last5_rows": int(df["last5_volume"].notna().sum()) if "last5_volume" in df.columns else 0,
            }
        )
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report = pd.DataFrame(report_rows)
    return panel, report


def apply_date_filters(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = df.copy()
    if start_date:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return out

def build_capacity_precheck(exec_panel: pd.DataFrame, preds: pd.DataFrame, capacity_mode: str) -> dict[str, object]:
    """Summarize last-5min capacity availability on prediction dates.

    Capacity constraints are meaningful only if last5_volume/last5_amount are
    actually populated for the dates being backtested.  Failing fast here avoids
    the misleading all-cash result where every buy is rejected as
    capacity_zero_missing_last5_*.
    """
    if exec_panel.empty or preds.empty:
        return {
            "capacity_mode": capacity_mode,
            "checked_rows": 0,
            "required_columns": [],
            "coverage_rate": 0.0,
            "passed": False,
            "reason": "empty_exec_panel_or_predictions",
        }
    pred_dates = set(pd.to_datetime(preds["date"], errors="coerce").dropna().dt.normalize())
    ep = exec_panel[exec_panel["date"].isin(pred_dates)].copy()
    required: list[str] = []
    if capacity_mode in {"last5_volume", "last5_both"}:
        required.append("last5_volume")
    if capacity_mode in {"last5_amount", "last5_both"}:
        required.append("last5_amount")
    if not required:
        return {
            "capacity_mode": capacity_mode,
            "checked_rows": int(len(ep)),
            "required_columns": [],
            "coverage_rate": 1.0,
            "passed": True,
            "reason": "capacity_disabled",
        }
    coverage_by_col: dict[str, float] = {}
    positive_by_col: dict[str, float] = {}
    for col in required:
        if col not in ep.columns or ep.empty:
            coverage_by_col[col] = 0.0
            positive_by_col[col] = 0.0
        else:
            vals = pd.to_numeric(ep[col], errors="coerce")
            coverage_by_col[col] = float(vals.notna().mean()) if len(vals) else 0.0
            positive_by_col[col] = float(vals.gt(0).mean()) if len(vals) else 0.0
    coverage_rate = min(coverage_by_col.values()) if coverage_by_col else 1.0
    positive_rate = min(positive_by_col.values()) if positive_by_col else 1.0
    return {
        "capacity_mode": capacity_mode,
        "checked_rows": int(len(ep)),
        "prediction_dates": int(len(pred_dates)),
        "required_columns": required,
        "coverage_by_column": coverage_by_col,
        "positive_rate_by_column": positive_by_col,
        "coverage_rate": float(coverage_rate),
        "positive_rate": float(positive_rate),
        "passed": bool(coverage_rate > 0.0 and positive_rate > 0.0),
        "reason": "ok" if (coverage_rate > 0.0 and positive_rate > 0.0) else "missing_or_zero_last5_capacity_data",
    }


def can_buy(row: pd.Series, cfg: TradeConfig) -> tuple[bool, str]:
    if row is None or row.empty:
        return False, "missing_execution_row"
    if cfg.mainboard_only and not bool(row.get("is_mainboard", False)):
        return False, "not_mainboard"
    if cfg.exclude_st and bool(row.get("is_st", False)):
        return False, "is_st"
    if not bool(row.get("tradable", False)):
        return False, "not_tradable"
    price = float(row.get("raw_close_1500", np.nan))
    if not np.isfinite(price) or price <= cfg.min_price:
        return False, "bad_price"
    if cfg.profile == "close_auction_skip_limit":
        up = float(row.get("up_limit", np.nan))
        if np.isfinite(up) and price >= up - cfg.limit_eps:
            return False, "buy_blocked_limit_up"
    return True, "ok"


def can_sell(row: pd.Series, cfg: TradeConfig) -> tuple[bool, str]:
    if row is None or row.empty:
        return False, "missing_execution_row"
    if not bool(row.get("tradable", False)):
        return False, "not_tradable"
    price = float(row.get("raw_close_1500", np.nan))
    if not np.isfinite(price) or price <= cfg.min_price:
        return False, "bad_price"
    if cfg.profile == "close_auction_skip_limit":
        down = float(row.get("down_limit", np.nan))
        if np.isfinite(down) and price <= down + cfg.limit_eps:
            return False, "sell_blocked_limit_down"
    return True, "ok"


def trade_commission(notional: float, rate: float, min_commission: float) -> float:
    if notional <= 0:
        return 0.0
    return max(notional * rate, min_commission)


def trade_fee_components(notional: float, side: str, cfg: TradeConfig) -> dict[str, float]:
    """Return explicit A-share fee components for a filled trade.

    commission: broker commission, both sides, with minimum commission.
    stamp_tax: sell side only.
    transfer_fee: both sides, configurable.

    The script treats commission_rate as the broker statement rate.  Do not add
    exchange handling fees/regulatory fees separately unless you intentionally
    want to model them outside the quoted commission rate.
    """
    amount = float(notional or 0.0)
    if amount <= 0:
        return {"commission": 0.0, "stamp_tax": 0.0, "transfer_fee": 0.0, "total_fee": 0.0}
    commission = trade_commission(amount, cfg.commission_rate, cfg.min_commission)
    stamp_tax = amount * cfg.stamp_tax_rate if str(side).lower() == "sell" else 0.0
    transfer_fee = amount * cfg.transfer_fee_rate
    total_fee = commission + stamp_tax + transfer_fee
    return {
        "commission": float(commission),
        "stamp_tax": float(stamp_tax),
        "transfer_fee": float(transfer_fee),
        "total_fee": float(total_fee),
    }


def is_rebalance_day_index(day_index: int, cfg: TradeConfig) -> bool:
    every = max(1, int(cfg.rebalance_every))
    offset = int(cfg.rebalance_offset) % every
    return (int(day_index) - offset) % every == 0


def floor_to_lot(shares: float, lot_size: int) -> int:
    lot = max(1, int(lot_size))
    if not np.isfinite(shares) or shares <= 0:
        return 0
    return int(math.floor(shares / lot) * lot)


def apply_corporate_actions_for_date(
    date: pd.Timestamp,
    positions: dict[str, dict[str, object]],
    exec_t: pd.DataFrame,
    exact_actions: pd.DataFrame,
    cfg: TradeConfig,
) -> tuple[float, list[dict[str, object]]]:
    """Apply corporate actions before trading on `date`.

    Exact mode uses cash_dividend_per_share/share_multiplier if supplied.
    Synthetic share-factor mode is the default fallback: for detected factor
    events it multiplies shares by previous_close / preclose so that
    shares_before * previous_close ~= shares_after * preclose.  This preserves
    total-return continuity without requiring corporate-action details, but it
    is not a true broker-statement reconstruction.

    Synthetic cash mode is retained for comparison/backward compatibility.
    """
    cash_delta = 0.0
    rows: list[dict[str, object]] = []
    if not positions:
        return cash_delta, rows

    exact_map = {}
    if exact_actions is not None and not exact_actions.empty:
        sub = exact_actions[exact_actions["date"].eq(date)]
        exact_map = {r["symbol"]: r for _, r in sub.iterrows()}

    for sym, pos in list(positions.items()):
        shares = float(pos.get("shares", 0.0))
        if shares <= 0 or sym not in exec_t.index:
            continue
        row = exec_t.loc[sym]
        used = False
        if sym in exact_map:
            action = exact_map[sym]
            cash_div = float(action.get("cash_dividend_per_share", 0.0) or 0.0)
            multiplier = float(action.get("share_multiplier", 1.0) or 1.0)
            if cash_div != 0.0:
                cd = shares * cash_div
                cash_delta += cd
            else:
                cd = 0.0
            old_shares = shares
            if np.isfinite(multiplier) and multiplier > 0 and abs(multiplier - 1.0) > 1e-12:
                new_shares = shares * multiplier
                positions[sym]["shares"] = float(new_shares)
            else:
                new_shares = shares
            rows.append({
                "date": date,
                "symbol": sym,
                "mode": "exact",
                "old_shares": old_shares,
                "new_shares": new_shares,
                "cash_dividend_per_share": cash_div,
                "share_multiplier": multiplier,
                "cash_delta": cd,
                "prev_raw_close_1500": row.get("prev_raw_close_1500", np.nan),
                "raw_preclose": row.get("raw_preclose", np.nan),
                "event_ratio": row.get("event_ratio", np.nan),
            })
            used = True
        if used or cfg.corporate_action_mode == "none":
            continue
        event_ratio = float(row.get("event_ratio", 1.0) or 1.0)
        prev_close = float(row.get("prev_raw_close_1500", np.nan))
        preclose = float(row.get("raw_preclose", np.nan))
        valid_event = (
            np.isfinite(event_ratio)
            and abs(event_ratio - 1.0) > cfg.corporate_action_threshold
            and np.isfinite(prev_close)
            and np.isfinite(preclose)
            and prev_close > 0
            and preclose > 0
        )
        if not valid_event:
            continue
        if cfg.corporate_action_mode == "synthetic_share_factor_from_preclose":
            share_multiplier = prev_close / preclose
            old_shares = shares
            new_shares = shares * share_multiplier
            positions[sym]["shares"] = float(new_shares)
            rows.append({
                "date": date,
                "symbol": sym,
                "mode": "synthetic_share_factor_from_preclose",
                "old_shares": old_shares,
                "new_shares": new_shares,
                "cash_dividend_per_share": 0.0,
                "share_multiplier": share_multiplier,
                "cash_delta": 0.0,
                "prev_raw_close_1500": prev_close,
                "raw_preclose": preclose,
                "event_ratio": event_ratio,
                "value_before_at_prev_close": old_shares * prev_close,
                "value_after_at_preclose": new_shares * preclose,
            })
        elif cfg.corporate_action_mode == "synthetic_cash_from_preclose":
            per_share = prev_close - preclose
            cd = shares * per_share
            cash_delta += cd
            rows.append({
                "date": date,
                "symbol": sym,
                "mode": "synthetic_cash_from_preclose",
                "old_shares": shares,
                "new_shares": shares,
                "cash_dividend_per_share": per_share,
                "share_multiplier": 1.0,
                "cash_delta": cd,
                "prev_raw_close_1500": prev_close,
                "raw_preclose": preclose,
                "event_ratio": event_ratio,
                "value_before_at_prev_close": shares * prev_close,
                "value_after_at_preclose": shares * preclose + cd,
            })
    return cash_delta, rows


def buy_capacity_notional(row: pd.Series, cfg: TradeConfig) -> tuple[float, str]:
    if cfg.capacity_mode == "none":
        return math.inf, "no_capacity_limit"
    if cfg.capacity_mode in {"last5_amount", "last5_both"}:
        amt = float(row.get("last5_amount", np.nan))
        if not np.isfinite(amt) or amt <= 0:
            return 0.0, "missing_last5_amount"
        return max(0.0, cfg.participation_rate * amt), "last5_amount"
    return math.inf, "unknown_capacity_mode"


def sell_capacity_shares(row: pd.Series, cfg: TradeConfig) -> tuple[int | None, str]:
    if cfg.capacity_mode == "none":
        return None, "no_capacity_limit"
    if cfg.capacity_mode in {"last5_volume", "last5_both", "last5_amount"}:
        vol = float(row.get("last5_volume", np.nan))
        if not np.isfinite(vol) or vol <= 0:
            return 0, "missing_last5_volume"
        return floor_to_lot(cfg.participation_rate * vol, cfg.lot_size), "last5_volume"
    return None, "unknown_capacity_mode"



def _append_rejection(
    reject_rows: list[dict[str, object]],
    *,
    date: pd.Timestamp,
    symbol: str,
    side: str,
    reason: str,
    rank: float,
    score: float | None = None,
    cfg: TradeConfig,
    is_rebalance_day: bool,
    extra: dict[str, object] | None = None,
) -> None:
    row: dict[str, object] = {
        "date": date,
        "symbol": symbol,
        "side": side,
        "reason": reason,
        "rank": rank,
        "score": score if score is not None else np.nan,
        "is_rebalance_day": bool(is_rebalance_day),
        "max_positions": cfg.max_positions,
        "buy_candidate_rank": cfg.buy_candidate_rank,
        "sell_rank": cfg.sell_rank,
        "rebalance_every": cfg.rebalance_every,
        "rebalance_offset": cfg.rebalance_offset,
    }
    if extra:
        row.update(extra)
    reject_rows.append(row)


def _position_value(sym: str, pos: dict[str, object], exec_t: pd.DataFrame) -> float:
    if sym not in exec_t.index:
        return 0.0
    raw_price = float(exec_t.loc[sym, "raw_close_1500"])
    shares = float(pos.get("shares", 0.0))
    return shares * raw_price


def _portfolio_holding_values(positions: dict[str, dict[str, object]], exec_t: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    missing: list[str] = []
    for sym, pos in list(positions.items()):
        if sym not in exec_t.index:
            missing.append(sym)
            values[sym] = 0.0
        else:
            values[sym] = _position_value(sym, pos, exec_t)
    return values, missing


def backtest(
    preds: pd.DataFrame,
    exec_panel: pd.DataFrame,
    cfg: TradeConfig,
    corporate_actions: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict]:
    if preds.empty:
        raise ValueError("empty predictions")
    if exec_panel.empty:
        raise ValueError("empty execution panel")

    pred_dates = pd.DatetimeIndex(preds["date"].unique()).sort_values()
    exec_dates = pd.DatetimeIndex(exec_panel["date"].unique()).sort_values()
    exec_date_set = set(exec_dates)
    dates = [d for d in pred_dates if d in exec_date_set]
    if len(dates) < 2:
        raise ValueError(f"not enough overlapping dates: pred={len(pred_dates)} exec={len(exec_dates)} overlap={len(dates)}")

    exec_by_date = {d: g.set_index("symbol", drop=False) for d, g in exec_panel.groupby("date", sort=True)}
    preds_by_date = {d: g.copy() for d, g in preds.groupby("date", sort=True)}
    corporate_actions = corporate_actions if corporate_actions is not None else pd.DataFrame()

    cash = float(cfg.initial_cash)
    # positions: symbol -> dict(shares, buy_date, avg_entry_price, entry_rank,
    # entry_score, cost_basis_notional, cost_basis_fee)
    positions: dict[str, dict[str, object]] = {}
    nav_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    reject_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    round_trip_rows: list[dict[str, object]] = []

    last_nav = float(cfg.initial_cash)
    round_trip_id = 0

    for day_index, date in enumerate(dates):
        exec_t = exec_by_date[date]
        pred_t = preds_by_date[date].sort_values("score", ascending=False).copy()
        pred_t["rank"] = np.arange(1, len(pred_t) + 1)
        rank_map = pred_t.set_index("symbol")["rank"].to_dict()
        score_map = pred_t.set_index("symbol")["score"].to_dict()
        is_reb = is_rebalance_day_index(day_index, cfg)

        cash_delta, action_log = apply_corporate_actions_for_date(date, positions, exec_t, corporate_actions, cfg)
        if cash_delta:
            cash += cash_delta
        action_rows.extend(action_log)

        holding_values, missing_marks = _portfolio_holding_values(positions, exec_t)
        nav_before_trade = cash + sum(holding_values.values())

        if is_reb:
            # 1) Sell positions whose current rank falls beyond sell_rank.
            for sym in list(positions.keys()):
                rank = rank_map.get(sym, math.inf)
                should_sell = bool(rank > cfg.sell_rank)
                if not should_sell:
                    continue
                buy_date = pd.Timestamp(positions[sym]["buy_date"])
                if date <= buy_date:
                    _append_rejection(
                        reject_rows,
                        date=date,
                        symbol=sym,
                        side="sell",
                        reason="t_plus_1_restriction",
                        rank=rank,
                        score=score_map.get(sym, np.nan),
                        cfg=cfg,
                        is_rebalance_day=is_reb,
                    )
                    continue
                row = exec_t.loc[sym] if sym in exec_t.index else pd.Series(dtype=object)
                ok, reason = can_sell(row, cfg)
                if not ok:
                    _append_rejection(
                        reject_rows,
                        date=date,
                        symbol=sym,
                        side="sell",
                        reason=reason,
                        rank=rank,
                        score=score_map.get(sym, np.nan),
                        cfg=cfg,
                        is_rebalance_day=is_reb,
                    )
                    continue

                held_shares = float(positions[sym].get("shares", 0.0))
                shares_before = held_shares
                cap_shares, cap_reason = sell_capacity_shares(row, cfg)
                if cap_shares is None:
                    sell_shares = held_shares
                elif cap_shares >= held_shares:
                    sell_shares = held_shares
                else:
                    sell_shares = float(floor_to_lot(cap_shares, cfg.lot_size))
                if sell_shares <= 1e-12:
                    _append_rejection(
                        reject_rows,
                        date=date,
                        symbol=sym,
                        side="sell",
                        reason=f"capacity_zero_{cap_reason}",
                        rank=rank,
                        score=score_map.get(sym, np.nan),
                        cfg=cfg,
                        is_rebalance_day=is_reb,
                    )
                    continue

                raw_price = float(row["raw_close_1500"]) * (1.0 - cfg.slippage_bps / 10000.0)
                notional = sell_shares * raw_price
                fees = trade_fee_components(notional, "sell", cfg)
                cash_before = cash
                position_before = held_shares
                cash += notional - fees["total_fee"]

                # Realized trade accounting.  For full sell, ratio=1.  For
                # partial capacity-limited sells, allocate entry notional/fees
                # proportionally and reduce the remaining cost basis.
                pos = positions[sym]
                ratio = min(1.0, sell_shares / shares_before) if shares_before > 0 else 1.0
                entry_notional_alloc = float(pos.get("cost_basis_notional", 0.0)) * ratio
                entry_fee_alloc = float(pos.get("cost_basis_fee", 0.0)) * ratio
                gross_pnl = notional - entry_notional_alloc
                net_pnl = notional - fees["total_fee"] - entry_notional_alloc - entry_fee_alloc
                holding_days = int((pd.Timestamp(date) - pd.Timestamp(pos.get("buy_date"))).days)
                round_trip_id += 1
                round_trip_rows.append(
                    {
                        "round_trip_id": round_trip_id,
                        "symbol": sym,
                        "entry_date": pos.get("buy_date"),
                        "exit_date": date,
                        "holding_days": holding_days,
                        "entry_price": pos.get("avg_entry_price", np.nan),
                        "exit_price": raw_price,
                        "entry_amount": entry_notional_alloc,
                        "exit_amount": notional,
                        "shares": sell_shares,
                        "gross_pnl": gross_pnl,
                        "entry_fee_alloc": entry_fee_alloc,
                        "exit_commission": fees["commission"],
                        "exit_stamp_tax": fees["stamp_tax"],
                        "exit_transfer_fee": fees["transfer_fee"],
                        "exit_total_fee": fees["total_fee"],
                        "total_fee": entry_fee_alloc + fees["total_fee"],
                        "net_pnl": net_pnl,
                        "net_return": net_pnl / max(entry_notional_alloc + entry_fee_alloc, 1e-12),
                        "exit_reason": "rank_gt_sell_rank",
                        "entry_rank": pos.get("entry_rank", np.nan),
                        "exit_rank": rank,
                        "entry_score": pos.get("entry_score", np.nan),
                        "exit_score": score_map.get(sym, np.nan),
                        "partial_exit": bool(sell_shares < shares_before - 1e-12),
                        "max_positions": cfg.max_positions,
                        "buy_candidate_rank": cfg.buy_candidate_rank,
                        "sell_rank": cfg.sell_rank,
                        "rebalance_every": cfg.rebalance_every,
                        "rebalance_offset": cfg.rebalance_offset,
                    }
                )

                positions[sym]["shares"] = shares_before - sell_shares
                positions[sym]["cost_basis_notional"] = float(pos.get("cost_basis_notional", 0.0)) * (1.0 - ratio)
                positions[sym]["cost_basis_fee"] = float(pos.get("cost_basis_fee", 0.0)) * (1.0 - ratio)
                partial = positions[sym]["shares"] > 1e-12
                if positions[sym]["shares"] <= 1e-12:
                    positions.pop(sym, None)
                    holding_values.pop(sym, None)
                else:
                    holding_values[sym] = positions[sym]["shares"] * float(row["raw_close_1500"])
                order_rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "side": "sell",
                        "rank": rank,
                        "score": score_map.get(sym, np.nan),
                        "raw_exec_price": raw_price,
                        "raw_close_1500": float(row["raw_close_1500"]),
                        "qfq_exec_price": float(row["qfq_close_1500"]),
                        "intended_shares": shares_before,
                        "filled_shares": sell_shares,
                        "shares": sell_shares,
                        "intended_amount": shares_before * raw_price,
                        "filled_amount": notional,
                        "notional": notional,
                        "commission": fees["commission"],
                        "stamp_tax": fees["stamp_tax"],
                        "transfer_fee": fees["transfer_fee"],
                        "cost": fees["total_fee"],
                        "total_fee": fees["total_fee"],
                        "cash_before": cash_before,
                        "cash_after": cash,
                        "position_before": position_before,
                        "position_after": float(positions.get(sym, {}).get("shares", 0.0)),
                        "order_status": "filled_partial" if partial else "filled",
                        "capacity_reason": cap_reason,
                        "partial_fill": partial,
                        "reason": "rank_gt_sell_rank",
                        "is_rebalance_day": bool(is_reb),
                        "day_index": day_index,
                        "max_positions": cfg.max_positions,
                        "buy_candidate_rank": cfg.buy_candidate_rank,
                        "sell_rank": cfg.sell_rank,
                        "rebalance_every": cfg.rebalance_every,
                        "rebalance_offset": cfg.rebalance_offset,
                    }
                )

            holding_values, _ = _portfolio_holding_values(positions, exec_t)
            nav_after_sells = cash + sum(holding_values.values())

            # 2) Fill empty slots.  Existing holdings are not replaced just
            # because a new name has a better rank.
            if len(positions) < cfg.max_positions and cash > 0:
                candidate_symbols = pred_t.loc[pred_t["rank"] <= cfg.buy_candidate_rank, "symbol"].tolist()
                for sym in candidate_symbols:
                    if len(positions) >= cfg.max_positions:
                        break
                    if sym in positions:
                        continue
                    row = exec_t.loc[sym] if sym in exec_t.index else pd.Series(dtype=object)
                    ok, reason = can_buy(row, cfg)
                    rank = rank_map.get(sym, math.inf)
                    score = score_map.get(sym, np.nan)
                    if not ok:
                        _append_rejection(
                            reject_rows,
                            date=date,
                            symbol=sym,
                            side="buy",
                            reason=reason,
                            rank=rank,
                            score=score,
                            cfg=cfg,
                            is_rebalance_day=is_reb,
                        )
                        continue
                    slots = max(1, cfg.max_positions - len(positions))
                    base_target = min(nav_after_sells / cfg.max_positions, cash / slots if slots > 1 else cash)
                    cap_notional, cap_reason = buy_capacity_notional(row, cfg)
                    target_notional = min(base_target, cap_notional)
                    raw_price = float(row["raw_close_1500"]) * (1.0 + cfg.slippage_bps / 10000.0)
                    if not np.isfinite(raw_price) or raw_price <= 0:
                        _append_rejection(
                            reject_rows,
                            date=date,
                            symbol=sym,
                            side="buy",
                            reason="bad_raw_price",
                            rank=rank,
                            score=score,
                            cfg=cfg,
                            is_rebalance_day=is_reb,
                        )
                        continue
                    intended_shares = floor_to_lot(target_notional / raw_price, cfg.lot_size)
                    shares = intended_shares
                    while shares > 0:
                        notional = shares * raw_price
                        fees = trade_fee_components(notional, "buy", cfg)
                        total_cash_needed = notional + fees["total_fee"]
                        if total_cash_needed <= cash + 1e-9:
                            break
                        shares -= cfg.lot_size
                    if shares <= 0:
                        reason2 = f"capacity_zero_{cap_reason}" if np.isfinite(cap_notional) and cap_notional <= 0 else "cash_or_lot_too_small"
                        _append_rejection(
                            reject_rows,
                            date=date,
                            symbol=sym,
                            side="buy",
                            reason=reason2,
                            rank=rank,
                            score=score,
                            cfg=cfg,
                            is_rebalance_day=is_reb,
                            extra={"intended_amount": target_notional},
                        )
                        continue
                    notional = shares * raw_price
                    fees = trade_fee_components(notional, "buy", cfg)
                    cash_before = cash
                    cash -= notional + fees["total_fee"]
                    positions[sym] = {
                        "shares": float(shares),
                        "buy_date": date,
                        "avg_entry_price": raw_price,
                        "entry_rank": rank,
                        "entry_score": score,
                        "cost_basis_notional": float(notional),
                        "cost_basis_fee": float(fees["total_fee"]),
                    }
                    holding_values[sym] = float(shares) * float(row["raw_close_1500"])
                    partial = bool(np.isfinite(cap_notional) and cap_notional < base_target - 1e-9)
                    order_rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "side": "buy",
                            "rank": rank,
                            "score": score,
                            "raw_exec_price": raw_price,
                            "raw_close_1500": float(row["raw_close_1500"]),
                            "qfq_exec_price": float(row["qfq_close_1500"]),
                            "intended_shares": intended_shares,
                            "filled_shares": int(shares),
                            "shares": int(shares),
                            "intended_amount": target_notional,
                            "filled_amount": notional,
                            "notional": notional,
                            "commission": fees["commission"],
                            "stamp_tax": fees["stamp_tax"],
                            "transfer_fee": fees["transfer_fee"],
                            "cost": fees["total_fee"],
                            "total_fee": fees["total_fee"],
                            "cash_before": cash_before,
                            "cash_after": cash,
                            "position_before": 0.0,
                            "position_after": float(shares),
                            "order_status": "filled_partial" if partial else "filled",
                            "capacity_reason": cap_reason,
                            "partial_fill": partial,
                            "reason": "fill_empty_slot_rank_le_buy_candidate_rank",
                            "is_rebalance_day": bool(is_reb),
                            "day_index": day_index,
                            "max_positions": cfg.max_positions,
                            "buy_candidate_rank": cfg.buy_candidate_rank,
                            "sell_rank": cfg.sell_rank,
                            "rebalance_every": cfg.rebalance_every,
                            "rebalance_offset": cfg.rebalance_offset,
                        }
                    )

        holding_values, missing_marks = _portfolio_holding_values(positions, exec_t)
        nav = cash + sum(holding_values.values())
        daily_return = nav / last_nav - 1.0 if last_nav > 0 else np.nan
        todays_orders = [r for r in order_rows if pd.Timestamp(r["date"]) == date]
        todays_rejects = [r for r in reject_rows if pd.Timestamp(r["date"]) == date]
        gross_trade_amount = sum(abs(float(r.get("notional", 0.0))) for r in todays_orders)
        turnover = gross_trade_amount / max(nav_before_trade, 1e-12)
        total_fee_today = sum(float(r.get("total_fee", r.get("cost", 0.0))) for r in todays_orders)
        nav_rows.append(
            {
                "date": date,
                "day_index": day_index,
                "is_rebalance_day": bool(is_reb),
                "nav": nav,
                "daily_return": daily_return,
                "cash": cash,
                "cash_ratio": cash / nav if nav > 0 else np.nan,
                "holding_value": sum(holding_values.values()),
                "gross_exposure": sum(holding_values.values()) / nav if nav > 0 else np.nan,
                "n_positions": len(positions),
                "turnover": turnover,
                "gross_trade_amount": gross_trade_amount,
                "total_fee": total_fee_today,
                "nav_before_trade": nav_before_trade,
                "corporate_action_cash_delta": cash_delta,
                "orders": len(todays_orders),
                "buy_orders": sum(1 for r in todays_orders if r.get("side") == "buy"),
                "sell_orders": sum(1 for r in todays_orders if r.get("side") == "sell"),
                "partial_fill_orders": int(sum(bool(r.get("partial_fill", False)) for r in todays_orders)),
                "rejections": len(todays_rejects),
                "missing_marks": ";".join(missing_marks),
                "max_positions": cfg.max_positions,
                "buy_candidate_rank": cfg.buy_candidate_rank,
                "sell_rank": cfg.sell_rank,
                "rebalance_every": cfg.rebalance_every,
                "rebalance_offset": cfg.rebalance_offset,
            }
        )
        for sym, value in sorted(holding_values.items()):
            pos = positions[sym]
            position_rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "rank": rank_map.get(sym, np.nan),
                    "score": score_map.get(sym, np.nan),
                    "shares": float(pos.get("shares", 0.0)),
                    "raw_close_1500": float(exec_t.loc[sym, "raw_close_1500"]) if sym in exec_t.index else np.nan,
                    "value": value,
                    "weight": value / nav if nav > 0 else np.nan,
                    "buy_date": pos.get("buy_date"),
                    "holding_days": int((pd.Timestamp(date) - pd.Timestamp(pos.get("buy_date"))).days) if pos.get("buy_date") is not None else np.nan,
                    "entry_rank": pos.get("entry_rank", np.nan),
                    "entry_score": pos.get("entry_score", np.nan),
                    "avg_entry_price": pos.get("avg_entry_price", np.nan),
                    "cost_basis_notional": pos.get("cost_basis_notional", np.nan),
                    "cost_basis_fee": pos.get("cost_basis_fee", np.nan),
                    "max_positions": cfg.max_positions,
                    "buy_candidate_rank": cfg.buy_candidate_rank,
                    "sell_rank": cfg.sell_rank,
                    "rebalance_every": cfg.rebalance_every,
                    "rebalance_offset": cfg.rebalance_offset,
                }
            )
        last_nav = nav

    nav_df = pd.DataFrame(nav_rows)
    orders_df = pd.DataFrame(order_rows)
    rejects_df = pd.DataFrame(reject_rows)
    positions_df = pd.DataFrame(position_rows)
    actions_df = pd.DataFrame(action_rows)
    round_trips_df = pd.DataFrame(round_trip_rows)
    daily_drawdown_df = build_daily_drawdown(nav_df)
    monthly_summary_df = build_period_summary(nav_df, "M")
    yearly_summary_df = build_period_summary(nav_df, "Y")
    fee_summary_df = build_fee_summary(orders_df, cfg)
    turnover_summary_df = build_turnover_summary(nav_df, orders_df)
    summary = summarize_nav(nav_df, orders_df, rejects_df, cfg, actions_df, round_trips_df, daily_drawdown_df)
    return {
        "nav": nav_df,
        "orders": orders_df,
        "trades": orders_df.copy(),
        "rejections": rejects_df,
        "positions": positions_df,
        "corporate_actions": actions_df,
        "round_trips": round_trips_df,
        "daily_drawdown": daily_drawdown_df,
        "monthly_summary": monthly_summary_df,
        "yearly_summary": yearly_summary_df,
        "fee_summary": fee_summary_df,
        "turnover_summary": turnover_summary_df,
        "summary": summary,
    }


def build_daily_drawdown(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame(columns=["date", "equity", "cummax_equity", "drawdown", "drawdown_pct", "is_new_high", "drawdown_days", "underwater_group_id"])
    out = nav[["date", "nav"]].copy().rename(columns={"nav": "equity"})
    out["cummax_equity"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["cummax_equity"]
    out["drawdown_pct"] = out["equity"] / out["cummax_equity"] - 1.0
    out["is_new_high"] = out["drawdown_pct"].abs() < 1e-12
    group_id = 0
    days = 0
    group_ids: list[int] = []
    drawdown_days: list[int] = []
    in_dd = False
    for is_high in out["is_new_high"].astype(bool).tolist():
        if is_high:
            in_dd = False
            days = 0
            group_ids.append(0)
            drawdown_days.append(0)
        else:
            if not in_dd:
                group_id += 1
                in_dd = True
                days = 1
            else:
                days += 1
            group_ids.append(group_id)
            drawdown_days.append(days)
    out["drawdown_days"] = drawdown_days
    out["underwater_group_id"] = group_ids
    return out


def top_drawdown_periods(daily_drawdown: pd.DataFrame, top_n: int = 5) -> list[dict[str, object]]:
    if daily_drawdown.empty or "underwater_group_id" not in daily_drawdown.columns:
        return []
    periods: list[dict[str, object]] = []
    dd = daily_drawdown.copy()
    for gid, g in dd[dd["underwater_group_id"].gt(0)].groupby("underwater_group_id"):
        if g.empty:
            continue
        bottom_idx = g["drawdown_pct"].idxmin()
        bottom_row = g.loc[bottom_idx]
        start_pos = max(int(g.index.min()) - 1, 0)
        start_date = dd.loc[start_pos, "date"] if start_pos in dd.index else g["date"].iloc[0]
        after = dd.loc[int(g.index.max()) + 1:]
        recovery_candidates = after[after["is_new_high"].astype(bool)] if not after.empty else pd.DataFrame()
        recovery_date = recovery_candidates["date"].iloc[0] if not recovery_candidates.empty else pd.NaT
        periods.append(
            {
                "underwater_group_id": int(gid),
                "start_date": start_date,
                "bottom_date": bottom_row["date"],
                "recovery_date": recovery_date,
                "max_drawdown": float(bottom_row["drawdown_pct"]),
                "duration_days": int(len(g)),
            }
        )
    periods.sort(key=lambda r: r["max_drawdown"])
    return periods[:top_n]


def build_period_summary(nav: pd.DataFrame, freq: str) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    x = nav.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.set_index("date")
    rows: list[dict[str, object]] = []
    for period, g in x.groupby(pd.Grouper(freq=freq)):
        if g.empty:
            continue
        start_nav = float(g["nav"].iloc[0] / (1.0 + g["daily_return"].iloc[0])) if np.isfinite(g["daily_return"].iloc[0]) else float(g["nav"].iloc[0])
        end_nav = float(g["nav"].iloc[-1])
        ret = end_nav / max(start_nav, 1e-12) - 1.0
        rows.append(
            {
                "period": period.strftime("%Y-%m") if freq.upper().startswith("M") else period.strftime("%Y"),
                "start_date": g.index.min(),
                "end_date": g.index.max(),
                "n_days": int(len(g)),
                "period_return": float(ret),
                "win": bool(ret > 0),
                "gross_trade_amount": float(g.get("gross_trade_amount", pd.Series(dtype=float)).sum()),
                "total_fee": float(g.get("total_fee", pd.Series(dtype=float)).sum()),
                "avg_turnover": float(g.get("turnover", pd.Series(dtype=float)).mean()),
                "avg_positions": float(g.get("n_positions", pd.Series(dtype=float)).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_fee_summary(orders: pd.DataFrame, cfg: TradeConfig) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame([{"side": "all", "n_orders": 0, "gross_amount": 0.0, "commission": 0.0, "stamp_tax": 0.0, "transfer_fee": 0.0, "total_fee": 0.0}])
    rows: list[dict[str, object]] = []
    for side, g in list(orders.groupby("side")) + [("all", orders)]:
        rows.append(
            {
                "side": side,
                "n_orders": int(len(g)),
                "gross_amount": float(pd.to_numeric(g["notional"], errors="coerce").abs().sum()),
                "commission": float(pd.to_numeric(g.get("commission", 0.0), errors="coerce").sum()),
                "stamp_tax": float(pd.to_numeric(g.get("stamp_tax", 0.0), errors="coerce").sum()),
                "transfer_fee": float(pd.to_numeric(g.get("transfer_fee", 0.0), errors="coerce").sum()),
                "total_fee": float(pd.to_numeric(g.get("total_fee", g.get("cost", 0.0)), errors="coerce").sum()),
                "commission_rate_config": cfg.commission_rate,
                "min_commission_config": cfg.min_commission,
                "stamp_tax_rate_config": cfg.stamp_tax_rate,
                "transfer_fee_rate_config": cfg.transfer_fee_rate,
            }
        )
    return pd.DataFrame(rows)


def build_turnover_summary(nav: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    gross = float(pd.to_numeric(orders.get("notional", pd.Series(dtype=float)), errors="coerce").abs().sum()) if not orders.empty else 0.0
    return pd.DataFrame(
        [
            {
                "avg_daily_turnover": float(pd.to_numeric(nav.get("turnover", pd.Series(dtype=float)), errors="coerce").mean()) if not nav.empty else np.nan,
                "median_daily_turnover": float(pd.to_numeric(nav.get("turnover", pd.Series(dtype=float)), errors="coerce").median()) if not nav.empty else np.nan,
                "max_daily_turnover": float(pd.to_numeric(nav.get("turnover", pd.Series(dtype=float)), errors="coerce").max()) if not nav.empty else np.nan,
                "total_gross_trade_amount": gross,
                "annualized_turnover": float(pd.to_numeric(nav.get("turnover", pd.Series(dtype=float)), errors="coerce").mean() * 252) if not nav.empty else np.nan,
            }
        ]
    )


def summarize_nav(
    nav: pd.DataFrame,
    orders: pd.DataFrame,
    rejects: pd.DataFrame,
    cfg: TradeConfig,
    actions: pd.DataFrame | None = None,
    round_trips: pd.DataFrame | None = None,
    daily_drawdown: pd.DataFrame | None = None,
) -> dict:
    if nav.empty:
        return {}
    rets = pd.to_numeric(nav["daily_return"], errors="coerce").dropna()
    total_return = float(nav["nav"].iloc[-1] / cfg.initial_cash - 1.0)
    n_days = int(len(nav))
    ann_return = float((1.0 + total_return) ** (252.0 / max(n_days, 1)) - 1.0) if total_return > -1 else np.nan
    ann_vol = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) else np.nan
    sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(252)) if len(rets) and rets.std(ddof=0) > 0 else np.nan
    if daily_drawdown is None or daily_drawdown.empty:
        daily_drawdown = build_daily_drawdown(nav)
    max_dd = float(daily_drawdown["drawdown_pct"].min()) if not daily_drawdown.empty else np.nan
    dd_periods = top_drawdown_periods(daily_drawdown, 5)
    deepest = dd_periods[0] if dd_periods else {}

    monthly = build_period_summary(nav, "M")
    weekly_nav = nav.copy()
    weekly_win_rate = np.nan
    if not weekly_nav.empty:
        weekly_nav["date"] = pd.to_datetime(weekly_nav["date"])
        weekly_nav = weekly_nav.set_index("date")
        week_rets = []
        for _, g in weekly_nav.groupby(pd.Grouper(freq="W-FRI")):
            if len(g):
                start_nav = float(g["nav"].iloc[0] / (1.0 + g["daily_return"].iloc[0])) if np.isfinite(g["daily_return"].iloc[0]) else float(g["nav"].iloc[0])
                week_rets.append(float(g["nav"].iloc[-1] / max(start_nav, 1e-12) - 1.0))
        weekly_win_rate = float(np.mean([r > 0 for r in week_rets])) if week_rets else np.nan

    rt = round_trips if round_trips is not None else pd.DataFrame()
    rt_net = pd.to_numeric(rt.get("net_pnl", pd.Series(dtype=float)), errors="coerce") if not rt.empty else pd.Series(dtype=float)
    wins = rt_net[rt_net > 0]
    losses = rt_net[rt_net < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss_abs = float(-losses.sum()) if len(losses) else 0.0

    total_commission = float(pd.to_numeric(orders.get("commission", pd.Series(dtype=float)), errors="coerce").sum()) if not orders.empty else 0.0
    total_stamp_tax = float(pd.to_numeric(orders.get("stamp_tax", pd.Series(dtype=float)), errors="coerce").sum()) if not orders.empty else 0.0
    total_transfer_fee = float(pd.to_numeric(orders.get("transfer_fee", pd.Series(dtype=float)), errors="coerce").sum()) if not orders.empty else 0.0
    total_fee = float(pd.to_numeric(orders.get("total_fee", orders.get("cost", pd.Series(dtype=float))), errors="coerce").sum()) if not orders.empty else 0.0
    gross_buy_amount = float(pd.to_numeric(orders.loc[orders["side"].eq("buy"), "notional"], errors="coerce").sum()) if not orders.empty else 0.0
    gross_sell_amount = float(pd.to_numeric(orders.loc[orders["side"].eq("sell"), "notional"], errors="coerce").sum()) if not orders.empty else 0.0
    gross_trade_amount = gross_buy_amount + gross_sell_amount

    return {
        "profile": cfg.profile,
        "date_min": nav["date"].min(),
        "date_max": nav["date"].max(),
        "n_days": n_days,
        "initial_cash": cfg.initial_cash,
        "final_nav": float(nav["nav"].iloc[-1]),
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "calmar": float(ann_return / abs(max_dd)) if np.isfinite(ann_return) and np.isfinite(max_dd) and max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "max_drawdown_start": deepest.get("start_date"),
        "max_drawdown_bottom": deepest.get("bottom_date"),
        "max_drawdown_recovery": deepest.get("recovery_date"),
        "max_drawdown_duration_days": deepest.get("duration_days"),
        "top_5_drawdowns": dd_periods,
        "daily_win_rate": float((rets > 0).mean()) if len(rets) else np.nan,
        "weekly_win_rate": weekly_win_rate,
        "monthly_win_rate": float(monthly["win"].mean()) if not monthly.empty and "win" in monthly.columns else np.nan,
        "trade_win_rate": float((rt_net > 0).mean()) if len(rt_net) else np.nan,
        "round_trip_win_rate": float((rt_net > 0).mean()) if len(rt_net) else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "win_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        "profit_factor": float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else np.nan,
        "avg_holding_days": float(pd.to_numeric(rt.get("holding_days", pd.Series(dtype=float)), errors="coerce").mean()) if not rt.empty else np.nan,
        "median_holding_days": float(pd.to_numeric(rt.get("holding_days", pd.Series(dtype=float)), errors="coerce").median()) if not rt.empty else np.nan,
        "avg_turnover": float(nav["turnover"].mean()),
        "median_turnover": float(nav["turnover"].median()),
        "annualized_turnover": float(nav["turnover"].mean() * 252),
        "avg_positions": float(nav["n_positions"].mean()),
        "median_positions": float(nav["n_positions"].median()),
        "max_positions_observed": int(nav["n_positions"].max()),
        "min_positions_observed": int(nav["n_positions"].min()),
        "avg_cash_ratio": float(nav["cash_ratio"].mean()) if "cash_ratio" in nav.columns else np.nan,
        "avg_exposure_ratio": float(nav["gross_exposure"].mean()),
        "n_orders": int(len(orders)),
        "n_buy_orders": int((orders["side"].eq("buy")).sum()) if not orders.empty else 0,
        "n_sell_orders": int((orders["side"].eq("sell")).sum()) if not orders.empty else 0,
        "n_partial_fill_orders": int((orders["partial_fill"].fillna(False).astype(bool)).sum()) if (not orders.empty and "partial_fill" in orders.columns) else 0,
        "n_rejections": int(len(rejects)),
        "rejection_reason_counts": rejects["reason"].value_counts().to_dict() if not rejects.empty else {},
        "n_round_trips": int(len(rt)),
        "gross_buy_amount": gross_buy_amount,
        "gross_sell_amount": gross_sell_amount,
        "gross_trade_amount": gross_trade_amount,
        "avg_order_amount": float(pd.to_numeric(orders.get("notional", pd.Series(dtype=float)), errors="coerce").abs().mean()) if not orders.empty else np.nan,
        "median_order_amount": float(pd.to_numeric(orders.get("notional", pd.Series(dtype=float)), errors="coerce").abs().median()) if not orders.empty else np.nan,
        "max_order_amount": float(pd.to_numeric(orders.get("notional", pd.Series(dtype=float)), errors="coerce").abs().max()) if not orders.empty else np.nan,
        "total_commission": total_commission,
        "total_stamp_tax": total_stamp_tax,
        "total_transfer_fee": total_transfer_fee,
        "total_fee": total_fee,
        "fee_to_initial_cash": float(total_fee / cfg.initial_cash) if cfg.initial_cash else np.nan,
        "fee_to_gross_trade_amount": float(total_fee / gross_trade_amount) if gross_trade_amount > 0 else np.nan,
        "n_corporate_action_adjustments": int(len(actions)) if actions is not None else 0,
        "max_positions": cfg.max_positions,
        "buy_candidate_rank": cfg.buy_candidate_rank,
        "sell_rank": cfg.sell_rank,
        "rebalance_every": cfg.rebalance_every,
        "rebalance_offset": cfg.rebalance_offset,
        "commission_rate": cfg.commission_rate,
        "stamp_tax_rate": cfg.stamp_tax_rate,
        "transfer_fee_rate": cfg.transfer_fee_rate,
        "slippage_bps": cfg.slippage_bps,
        "lot_size": cfg.lot_size,
        "min_commission": cfg.min_commission,
        "mainboard_only": cfg.mainboard_only,
        "exclude_st": cfg.exclude_st,
        "capacity_mode": cfg.capacity_mode,
        "participation_rate": cfg.participation_rate,
        "corporate_action_mode": cfg.corporate_action_mode,
        "corporate_action_threshold": cfg.corporate_action_threshold,
        "notes": [
            "Signals are AS1455/14:55 scores; executions are approximated at same-day 15:00 close.",
            "max_positions is portfolio capacity; buy_candidate_rank is only the candidate pool used to fill empty slots.",
            "Existing holdings are not replaced solely because new symbols have better ranks; sells are triggered by rank > sell_rank on rebalance days.",
            "Buys use raw 100-share board lots; synthetic corporate actions may create fractional/economic shares, and full liquidation can sell them.",
            "Commission is modeled as both-side broker commission with minimum commission; stamp tax is sell-side only; transfer fee is both-side.",
            "Exact dividend/split handling requires --corporate-actions; default synthetic share-factor mode preserves total-return continuity but is not a broker-statement reconstruction.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AS1455 close-auction execution backtest v7 with max-position grid semantics")
    parser.add_argument("--predictions", required=True, help="NN predictions HDF/CSV; HDF default key /predictions")
    parser.add_argument("--prediction-key", default=None, help="HDF key for predictions; default /predictions if present")
    parser.add_argument("--score-col", default=None, help="Score column; default first numeric non-symbol/date column")
    parser.add_argument("--raw-daily-cache-dir", default="saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache")
    parser.add_argument("--raw-5m-cache-dir", default=None, help="Optional raw 5m cache dir used to derive last-5min volume/amount capacity")
    parser.add_argument("--last5-panel", default=None, help="Optional CSV with date,symbol,last5_volume,last5_amount")
    parser.add_argument("--universe", default=None, help="Optional universe CSV with symbol/board/name")
    parser.add_argument("--st-symbols", default=None, help="Optional static CSV listing symbols to treat as ST")
    parser.add_argument("--st-status", default=None, help="Optional date-specific ST CSV with date,symbol,is_st/name")
    parser.add_argument("--corporate-actions", default=None, help="Optional CSV with date/ex_date,symbol,cash_dividend_per_share,share_multiplier")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--profile", choices=["close_auction_simple", "close_auction_skip_limit"], default="close_auction_skip_limit")
    parser.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS, help="Portfolio capacity, e.g. 5/10/15/20/25")
    parser.add_argument("--buy-candidate-rank", type=int, default=None, help="Candidate pool threshold used to fill empty slots. Default equals --sell-rank.")
    parser.add_argument("--buy-rank", type=int, default=None, help="Deprecated alias for --buy-candidate-rank for compatibility only.")
    parser.add_argument("--sell-rank", type=int, default=DEFAULT_SELL_RANK)
    parser.add_argument("--rebalance-every", type=int, default=DEFAULT_REBALANCE_EVERY, help="Rebalance every K overlapping prediction/execution trading days")
    parser.add_argument("--rebalance-offset", type=int, default=DEFAULT_REBALANCE_OFFSET, help="Offset in [0,K-1] for rebalance schedule")
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--commission-rate", type=float, default=DEFAULT_COMMISSION_RATE, help="Broker commission rate, default 0.000085 = 万0.85")
    parser.add_argument("--stamp-tax-rate", type=float, default=DEFAULT_STAMP_TAX_RATE, help="Sell-side stamp tax rate, default 0.0005")
    parser.add_argument("--transfer-fee-rate", type=float, default=DEFAULT_TRANSFER_FEE_RATE, help="Both-side transfer fee rate, default 0.00001 = 0.01‰")
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--min-commission", type=float, default=DEFAULT_MIN_COMMISSION)
    parser.add_argument("--lot-size", type=int, default=DEFAULT_LOT_SIZE)
    parser.add_argument("--allow-non-mainboard", action="store_true", help="Disable mainboard-only trading filter")
    parser.add_argument("--allow-st", action="store_true", help="Allow ST stocks to be traded; otherwise ST rows are excluded from buys")
    parser.add_argument("--capacity-mode", choices=["none", "last5_amount", "last5_volume", "last5_both"], default="none")
    parser.add_argument(
        "--capacity-missing-policy",
        choices=["fail", "reject", "disable"],
        default="fail",
        help=(
            "What to do when a last-5min capacity mode is requested but last5 data "
            "is missing on prediction dates. fail=stop before backtest; reject=old behavior "
            "that rejects affected orders; disable=turn capacity_mode to none with a warning."
        ),
    )
    parser.add_argument("--min-last5-coverage", type=float, default=0.95, help="Minimum coverage required when capacity-missing-policy=fail")
    parser.add_argument("--participation-rate", type=float, default=DEFAULT_PARTICIPATION_RATE)
    parser.add_argument(
        "--corporate-action-mode",
        choices=["none", "synthetic_share_factor_from_preclose", "synthetic_cash_from_preclose"],
        default="synthetic_share_factor_from_preclose",
        help=(
            "Company-action handling for held positions. Default uses preclose/previous-close "
            "to rescale shares and keep holding value continuous; cash mode is retained for comparison."
        ),
    )
    parser.add_argument("--corporate-action-threshold", type=float, default=1e-3)
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--limit-eps", type=float, default=1e-6)
    parser.add_argument("--write-execution-panel", action="store_true", help="Also write execution_panel.csv.gz")
    args = parser.parse_args()

    if args.rebalance_every <= 0:
        raise SystemExit("--rebalance-every must be >= 1")
    if args.rebalance_offset < 0:
        raise SystemExit("--rebalance-offset must be >= 0")
    if args.rebalance_offset >= args.rebalance_every:
        raise SystemExit("--rebalance-offset must be smaller than --rebalance-every")
    if args.max_positions <= 0:
        raise SystemExit("--max-positions must be >= 1")

    buy_candidate_rank = args.buy_candidate_rank
    if buy_candidate_rank is None and args.buy_rank is not None:
        buy_candidate_rank = args.buy_rank
    if buy_candidate_rank is None:
        buy_candidate_rank = args.sell_rank
    if buy_candidate_rank <= 0:
        raise SystemExit("--buy-candidate-rank/--buy-rank must be >= 1")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    preds = load_predictions(Path(args.predictions), args.prediction_key, args.score_col)
    preds = apply_date_filters(preds, args.start_date, args.end_date)
    if preds.empty:
        raise SystemExit("predictions empty after date filters")

    universe = read_universe(Path(args.universe) if args.universe else None)
    st_symbols = load_st_symbols(Path(args.st_symbols) if args.st_symbols else None)
    st_status = load_st_status(Path(args.st_status) if args.st_status else None)
    last5_panel = load_last5_panel(Path(args.last5_panel) if args.last5_panel else None)
    corporate_actions = load_corporate_actions(Path(args.corporate_actions) if args.corporate_actions else None)
    symbols = preds["symbol"].unique().tolist()
    exec_panel, exec_report = build_execution_panel(
        symbols,
        Path(args.raw_daily_cache_dir),
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=last5_panel,
        raw_5m_cache_dir=Path(args.raw_5m_cache_dir) if args.raw_5m_cache_dir else None,
    )
    exec_panel = apply_date_filters(exec_panel, args.start_date, args.end_date)

    capacity_mode_effective = str(args.capacity_mode)
    capacity_precheck = build_capacity_precheck(exec_panel, preds, capacity_mode_effective)
    if capacity_mode_effective != "none":
        with open(out_dir / "capacity_precheck_report.json", "w", encoding="utf-8") as f:
            json.dump(capacity_precheck, f, ensure_ascii=False, indent=2)
        coverage = float(capacity_precheck.get("coverage_rate", 0.0))
        positive = float(capacity_precheck.get("positive_rate", 0.0))
        if coverage < float(args.min_last5_coverage) or positive <= 0.0:
            msg = (
                f"capacity_mode={capacity_mode_effective} requested, but last-5min capacity data "
                f"coverage is insufficient on prediction dates: coverage={coverage:.6f}, "
                f"positive={positive:.6f}. See {out_dir / 'capacity_precheck_report.json'}."
            )
            if args.capacity_missing_policy == "fail":
                raise SystemExit(msg + " Rerun with --capacity-mode none, provide --last5-panel, "
                                 "or use --capacity-missing-policy reject/disable explicitly.")
            if args.capacity_missing_policy == "disable":
                print("[WARN] " + msg + " Disabling capacity constraints for this run.")
                capacity_mode_effective = "none"
                capacity_precheck["policy_action"] = "disabled_capacity_mode"
            else:
                print("[WARN] " + msg + " Keeping old reject-on-missing behavior.")
                capacity_precheck["policy_action"] = "reject_on_missing_capacity"
            with open(out_dir / "capacity_precheck_report.json", "w", encoding="utf-8") as f:
                json.dump(capacity_precheck, f, ensure_ascii=False, indent=2)

    exec_report.to_csv(out_dir / "execution_panel_build_report.csv", index=False, encoding="utf-8-sig")
    if args.write_execution_panel:
        exec_panel.to_csv(out_dir / "execution_panel.csv.gz", index=False, compression="gzip")

    cfg = TradeConfig(
        max_positions=int(args.max_positions),
        buy_candidate_rank=int(buy_candidate_rank),
        sell_rank=int(args.sell_rank),
        rebalance_every=int(args.rebalance_every),
        rebalance_offset=int(args.rebalance_offset),
        initial_cash=float(args.initial_cash),
        commission_rate=float(args.commission_rate),
        stamp_tax_rate=float(args.stamp_tax_rate),
        transfer_fee_rate=float(args.transfer_fee_rate),
        slippage_bps=float(args.slippage_bps),
        profile=str(args.profile),
        mainboard_only=not bool(args.allow_non_mainboard),
        min_price=float(args.min_price),
        limit_eps=float(args.limit_eps),
        lot_size=int(args.lot_size),
        min_commission=float(args.min_commission),
        exclude_st=not bool(args.allow_st),
        capacity_mode=capacity_mode_effective,
        participation_rate=float(args.participation_rate),
        corporate_action_mode=str(args.corporate_action_mode),
        corporate_action_threshold=float(args.corporate_action_threshold),
    )

    result = backtest(preds, exec_panel, cfg, corporate_actions=corporate_actions)
    output_map = {
        "close_auction_nav.csv": result["nav"],
        "close_auction_orders.csv": result["orders"],
        "close_auction_trades.csv": result["trades"],
        "close_auction_rejections.csv": result["rejections"],
        "close_auction_positions.csv": result["positions"],
        "close_auction_corporate_actions.csv": result.get("corporate_actions", pd.DataFrame()),
        "daily_drawdown.csv": result.get("daily_drawdown", pd.DataFrame()),
        "round_trips.csv": result.get("round_trips", pd.DataFrame()),
        "monthly_summary.csv": result.get("monthly_summary", pd.DataFrame()),
        "yearly_summary.csv": result.get("yearly_summary", pd.DataFrame()),
        "fee_summary.csv": result.get("fee_summary", pd.DataFrame()),
        "turnover_summary.csv": result.get("turnover_summary", pd.DataFrame()),
    }
    for name, df in output_map.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    summary = result["summary"]
    run_meta = {
        "predictions": str(args.predictions),
        "raw_daily_cache_dir": str(args.raw_daily_cache_dir),
        "raw_5m_cache_dir": str(args.raw_5m_cache_dir) if args.raw_5m_cache_dir else None,
        "last5_panel": str(args.last5_panel) if args.last5_panel else None,
        "capacity_precheck": capacity_precheck,
        "capacity_missing_policy": str(args.capacity_missing_policy),
        "min_last5_coverage": float(args.min_last5_coverage),
        "universe": str(args.universe) if args.universe else None,
        "st_status": str(args.st_status) if args.st_status else None,
        "corporate_actions": str(args.corporate_actions) if args.corporate_actions else None,
        "n_prediction_rows": int(len(preds)),
        "n_prediction_symbols": int(preds["symbol"].nunique()),
        "n_prediction_dates": int(preds["date"].nunique()),
        "n_execution_rows": int(len(exec_panel)),
        "n_execution_symbols": int(exec_panel["symbol"].nunique()) if not exec_panel.empty else 0,
        "n_execution_dates": int(exec_panel["date"].nunique()) if not exec_panel.empty else 0,
        "config": cfg.__dict__,
        "summary": summary,
        "output_files": sorted(output_map.keys()),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2, default=json_default)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)
    with open(out_dir / "close_auction_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2, default=json_default)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    print(f"[OK] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
