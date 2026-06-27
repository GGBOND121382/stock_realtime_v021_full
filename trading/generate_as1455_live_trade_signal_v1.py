#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate AS1455 live trade signal with backtest-like execution semantics.

This is a live single-day planner.  It does not replay the full historical
backtest, but it mirrors the backtest's same-day decision rules as closely as a
single-day live state permits:
- rebalance-day gate;
- sell rank > sell_rank first;
- T+1 sell restriction when buy_date is available;
- can_buy/can_sell filters for tradable, ST, mainboard, bad price, limit up/down;
- fill empty slots only; existing holdings are not replaced by higher-rank names;
- cash/fee/lot-size sizing using the same base_target formula as the backtest.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRICE_COLUMNS_PRIORITY = [
    "raw_close_1500", "raw_close_as1455", "close_as1455", "latest_price", "last_price", "price",
    "close", "last", "new_price", "最新价", "现价",
]
UP_LIMIT_COLUMNS = ["up_limit", "open_limit_up", "close_limit_up", "limit_up", "涨停价", "涨停"]
DOWN_LIMIT_COLUMNS = ["down_limit", "open_limit_down", "close_limit_down", "limit_down", "跌停价", "跌停"]
TRADESTATUS_COLUMNS = ["tradestatus", "trade_status", "tradable"]
IS_ST_COLUMNS = ["is_st", "isST", "is_st_hist", "st"]
IS_MAINBOARD_COLUMNS = ["is_mainboard", "trade_allowed_mainboard"]
LAST5_AMOUNT_COLUMNS = ["last5_amount", "last_5m_amount", "amount_1455_1500"]
LAST5_VOLUME_COLUMNS = ["last5_volume", "last_5m_volume", "volume_1455_1500"]


def normalize_symbol(value: object) -> str:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")
    m = re.search(r"(\d{6})", s)
    if m:
        code = m.group(1)
    elif re.fullmatch(r"\d{1,6}", s):
        code = s.zfill(6)
    else:
        return s.upper()
    return f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"


def compact_symbol(symbol: str) -> str:
    m = re.search(r"(\d{6})", str(symbol))
    return m.group(1) if m else str(symbol)


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


def json_default(o: Any) -> Any:
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return default
    return s in {"1", "true", "yes", "y", "t", "ok", "交易", "正常"}


def parse_float(value: object, default: float = np.nan) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return pd.NaT
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{8}", digits):
        return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce").normalize()


def date_str(value: object) -> str:
    ts = parse_date(value)
    if pd.isna(ts):
        return str(value)
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "shares"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        else:
            raise RuntimeError(f"positions file has no symbol/code: {path}")
    if "shares" not in df.columns:
        df["shares"] = 0
    out = df.copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0.0)
    for c in ["buy_date", "entry_date", "date"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: date_str(x) if not pd.isna(parse_date(x)) else "")
    for c in ["market_value", "cost_price", "last_price", "price", "avg_entry_price"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[(out["symbol"].astype(str).str.len() > 0) & (out["shares"] > 0)]
    return out.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def choose_price_file(live_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for p in [
        live_dir / "08_live_raw_row_as1455.csv",
        live_dir / "09_live_qfq_row_as1455.csv",
        live_dir / "07_live_snapshot_asof1455.csv",
        live_dir / "15_live_rank.csv",
    ]:
        if p.exists():
            return p
    return None


def load_execution_sidecar(path: Path | None, price_column: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"price_file": str(path) if path else None, "price_found": False}
    if path is None or not path.exists():
        meta["price_error"] = "price file missing"
        return pd.DataFrame(columns=["symbol"]), meta
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        for c in ["code", "ticker", "asset"]:
            if c in df.columns:
                df["symbol"] = df[c]
                break
    if "symbol" not in df.columns:
        meta["price_error"] = f"no symbol/code column in {path}"
        return pd.DataFrame(columns=["symbol"]), meta
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"].astype(str).str.len() > 0]

    out = pd.DataFrame({"symbol": df["symbol"]})
    col = None
    if price_column:
        if price_column not in df.columns:
            meta["price_error"] = f"explicit price column {price_column!r} not found; columns={list(df.columns)}"
        else:
            col = price_column
    if col is None:
        col = find_first_col(df, PRICE_COLUMNS_PRIORITY)
    if col is not None:
        out["order_price"] = pd.to_numeric(df[col], errors="coerce")
        out["raw_close_1500"] = out["order_price"]
        meta.update({"price_found": True, "price_column": col})
    else:
        out["order_price"] = np.nan
        out["raw_close_1500"] = np.nan
        meta["price_error"] = f"cannot infer price column; columns={list(df.columns)}"

    up_col = find_first_col(df, UP_LIMIT_COLUMNS)
    down_col = find_first_col(df, DOWN_LIMIT_COLUMNS)
    out["up_limit"] = pd.to_numeric(df[up_col], errors="coerce") if up_col else np.nan
    out["down_limit"] = pd.to_numeric(df[down_col], errors="coerce") if down_col else np.nan
    meta["up_limit_column"] = up_col
    meta["down_limit_column"] = down_col

    status_col = find_first_col(df, TRADESTATUS_COLUMNS)
    if status_col:
        if status_col == "tradable":
            out["tradable"] = df[status_col].map(lambda x: parse_bool(x, default=False))
        else:
            s = pd.to_numeric(df[status_col], errors="coerce")
            out["tradable"] = s.eq(1) | s.gt(0)
    else:
        out["tradable"] = True
    meta["tradestatus_column"] = status_col

    st_col = find_first_col(df, IS_ST_COLUMNS)
    if st_col:
        out["is_st"] = df[st_col].map(lambda x: parse_bool(x, default=False))
    elif "name" in df.columns:
        out["is_st"] = df["name"].astype(str).str.upper().str.contains("ST", regex=False)
    else:
        out["is_st"] = False
    meta["is_st_column"] = st_col

    main_col = find_first_col(df, IS_MAINBOARD_COLUMNS)
    if main_col:
        out["is_mainboard"] = df[main_col].map(lambda x: parse_bool(x, default=False))
    elif "board" in df.columns:
        out["is_mainboard"] = df["board"].astype(str).isin(["sh_mainboard", "sz_mainboard"])
    else:
        out["is_mainboard"] = out["symbol"].map(is_mainboard_symbol)
    meta["is_mainboard_column"] = main_col

    amt_col = find_first_col(df, LAST5_AMOUNT_COLUMNS)
    vol_col = find_first_col(df, LAST5_VOLUME_COLUMNS)
    out["last5_amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else np.nan
    out["last5_volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan
    meta["last5_amount_column"] = amt_col
    meta["last5_volume_column"] = vol_col

    out = out.drop_duplicates("symbol", keep="last")
    meta["execution_rows"] = int(len(out))
    return out, meta


def trade_fee_components(notional: float, side: str, commission_rate: float, stamp_tax_rate: float, transfer_fee_rate: float, min_commission: float) -> dict[str, float]:
    amount = float(notional or 0.0)
    if amount <= 0:
        return {"commission": 0.0, "stamp_tax": 0.0, "transfer_fee": 0.0, "total_fee": 0.0}
    commission = max(amount * commission_rate, min_commission)
    stamp = amount * stamp_tax_rate if str(side).lower() == "sell" else 0.0
    transfer = amount * transfer_fee_rate
    return {"commission": commission, "stamp_tax": stamp, "transfer_fee": transfer, "total_fee": commission + stamp + transfer}


def floor_to_lot(shares: float, lot_size: int) -> int:
    lot = max(1, int(lot_size))
    if not math.isfinite(shares) or shares <= 0:
        return 0
    return int(math.floor(shares / lot) * lot)


def can_buy(row: pd.Series, args: argparse.Namespace) -> tuple[bool, str]:
    if row is None or row.empty:
        return False, "missing_execution_row"
    if args.mainboard_only and not parse_bool(row.get("is_mainboard"), False):
        return False, "not_mainboard"
    if args.exclude_st and parse_bool(row.get("is_st"), False):
        return False, "is_st"
    if not parse_bool(row.get("tradable"), False):
        return False, "not_tradable"
    price = parse_float(row.get("order_price"), np.nan)
    if not math.isfinite(price) or price <= args.min_price:
        return False, "bad_price"
    if args.profile == "close_auction_skip_limit":
        up = parse_float(row.get("up_limit"), np.nan)
        if math.isfinite(up) and price >= up - args.limit_eps:
            return False, "buy_blocked_limit_up"
    return True, "ok"


def can_sell(row: pd.Series, args: argparse.Namespace) -> tuple[bool, str]:
    if row is None or row.empty:
        return False, "missing_execution_row"
    if not parse_bool(row.get("tradable"), False):
        return False, "not_tradable"
    price = parse_float(row.get("order_price"), np.nan)
    if not math.isfinite(price) or price <= args.min_price:
        return False, "bad_price"
    if args.profile == "close_auction_skip_limit":
        down = parse_float(row.get("down_limit"), np.nan)
        if math.isfinite(down) and price <= down + args.limit_eps:
            return False, "sell_blocked_limit_down"
    return True, "ok"


def buy_capacity_notional(row: pd.Series, args: argparse.Namespace) -> tuple[float, str]:
    if args.capacity_mode == "none":
        return math.inf, "no_capacity_limit"
    if args.capacity_mode in {"last5_amount", "last5_both"}:
        amt = parse_float(row.get("last5_amount"), np.nan)
        if not math.isfinite(amt) or amt <= 0:
            return 0.0, "missing_last5_amount"
        return max(0.0, args.participation_rate * amt), "last5_amount"
    return math.inf, "unknown_capacity_mode"


def sell_capacity_shares(row: pd.Series, args: argparse.Namespace) -> tuple[int | None, str]:
    if args.capacity_mode == "none":
        return None, "no_capacity_limit"
    if args.capacity_mode in {"last5_volume", "last5_both", "last5_amount"}:
        vol = parse_float(row.get("last5_volume"), np.nan)
        if not math.isfinite(vol) or vol <= 0:
            return 0, "missing_last5_volume"
        return floor_to_lot(args.participation_rate * vol, args.lot_size), "last5_volume"
    return None, "unknown_capacity_mode"


def compute_rebalance(args: argparse.Namespace, trade_date: pd.Timestamp, rank: pd.DataFrame) -> dict[str, Any]:
    every = max(1, int(args.rebalance_every))
    offset = int(args.rebalance_offset) % every
    if args.force_rebalance:
        return {"is_rebalance_day": True, "reason": "force_rebalance", "day_index": None, "calendar_exact": False}
    if args.skip_rebalance:
        return {"is_rebalance_day": False, "reason": "skip_rebalance", "day_index": None, "calendar_exact": False}
    if args.day_index is not None:
        idx = int(args.day_index)
        return {"is_rebalance_day": (idx - offset) % every == 0, "reason": "explicit_day_index", "day_index": idx, "calendar_exact": True}
    if args.rebalance_calendar:
        p = Path(args.rebalance_calendar)
        if p.exists():
            cal = pd.read_csv(p)
            if "date" not in cal.columns:
                for c in ["trade_date", "dt"]:
                    if c in cal.columns:
                        cal["date"] = cal[c]
                        break
            if "date" in cal.columns:
                dates = pd.DatetimeIndex(cal["date"].map(parse_date).dropna().unique()).sort_values()
                hits = np.where(dates == trade_date)[0]
                if len(hits):
                    idx = int(hits[0])
                    return {"is_rebalance_day": (idx - offset) % every == 0, "reason": f"calendar:{p}", "day_index": idx, "calendar_exact": True}
                return {"is_rebalance_day": False, "reason": f"trade_date_not_in_calendar:{p}", "day_index": None, "calendar_exact": True}
    # If rank has historical dates, use them.  Usually live rank has one date only.
    if rank["date_ts"].nunique() > 1:
        dates = pd.DatetimeIndex(rank["date_ts"].dropna().unique()).sort_values()
        hits = np.where(dates == trade_date)[0]
        if len(hits):
            idx = int(hits[0])
            return {"is_rebalance_day": (idx - offset) % every == 0, "reason": "rank_file_dates", "day_index": idx, "calendar_exact": True}
    policy = str(args.calendar_unknown_policy).lower()
    if policy == "fail":
        raise RuntimeError("cannot determine rebalance day: provide --day-index, --rebalance-calendar, --force-rebalance, or --skip-rebalance")
    if policy == "skip":
        return {"is_rebalance_day": False, "reason": "calendar_unknown_skip", "day_index": None, "calendar_exact": False}
    # Backward-compatible default: still produce a signal, but mark that calendar parity is not exact.
    return {"is_rebalance_day": True, "reason": "calendar_unknown_force", "day_index": None, "calendar_exact": False}


def pick_buy_date(row: pd.Series) -> pd.Timestamp | pd.NaT:
    for c in ["buy_date", "entry_date", "date_bought", "open_date"]:
        if c in row.index:
            ts = parse_date(row.get(c))
            if not pd.isna(ts):
                return pd.Timestamp(ts)
    return pd.NaT


def row_dict(**kwargs: Any) -> dict[str, Any]:
    base = {
        "date": None, "symbol": None, "action": None, "order_side": "", "order_shares": 0,
        "order_price": np.nan, "order_amount_est": np.nan, "target_value": np.nan,
        "intended_shares": 0, "filled_shares": 0, "commission": 0.0, "stamp_tax": 0.0,
        "transfer_fee": 0.0, "total_fee": 0.0, "rank": np.nan, "pred_score": np.nan,
        "shares_current": 0.0, "reason": "", "order_status": "", "is_rebalance_day": False,
        "day_index": np.nan, "capacity_reason": "", "partial_fill": False,
    }
    base.update(kwargs)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", required=True)
    ap.add_argument("--rank-file", default=None)
    ap.add_argument("--positions-file", default=None)
    ap.add_argument("--price-file", default=None)
    ap.add_argument("--price-column", default=None)
    ap.add_argument("--out-signal", default=None)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--max-positions", type=int, default=15)
    ap.add_argument("--sell-rank", type=int, default=300)
    ap.add_argument("--buy-candidate-rank", type=int, default=300)
    ap.add_argument("--rebalance-every", type=int, default=3)
    ap.add_argument("--rebalance-offset", type=int, default=0)
    ap.add_argument("--force-rebalance", action="store_true")
    ap.add_argument("--skip-rebalance", action="store_true")
    ap.add_argument("--day-index", type=int, default=None)
    ap.add_argument("--rebalance-calendar", default=None)
    ap.add_argument("--calendar-unknown-policy", choices=["force", "skip", "fail"], default="force")
    ap.add_argument("--unknown-buy-date-policy", choices=["allow", "block"], default="allow")
    ap.add_argument("--cash", type=float, default=None, help="available cash before planned sells")
    ap.add_argument("--portfolio-value", type=float, default=None, help="portfolio value; if cash absent, cash is inferred as value - current holdings")
    ap.add_argument("--buy-cash-per-position", type=float, default=None, help="override backtest sizing with fixed cash per BUY; not backtest-identical")
    ap.add_argument("--cash-buffer-pct", type=float, default=0.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--commission-rate", type=float, default=0.000085)
    ap.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    ap.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--profile", default="close_auction_skip_limit", choices=["close_auction_simple", "close_auction_skip_limit"])
    ap.add_argument("--mainboard-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min-price", type=float, default=0.0)
    ap.add_argument("--limit-eps", type=float, default=1e-6)
    ap.add_argument("--capacity-mode", default="none", choices=["none", "last5_amount", "last5_volume", "last5_both"])
    ap.add_argument("--participation-rate", type=float, default=0.05)
    args = ap.parse_args()

    start = time.time()
    live_dir = Path(args.live_dir)
    rank_file = Path(args.rank_file) if args.rank_file else live_dir / "15_live_rank.csv"
    pos_file = Path(args.positions_file) if args.positions_file else live_dir.parent / "current_positions.csv"
    price_file = choose_price_file(live_dir, args.price_file)
    out_signal = Path(args.out_signal) if args.out_signal else live_dir / "16_live_trade_signal.csv"
    out_report = Path(args.out_report) if args.out_report else live_dir / "16_live_trade_signal_report.json"

    report: dict[str, Any] = {
        "passed": False,
        "mode": "backtest_like_single_day_v5",
        "rank_file": str(rank_file),
        "positions_file": str(pos_file),
        "price_file": str(price_file) if price_file else None,
        "strategy": {
            "max_positions": args.max_positions,
            "sell_rank": args.sell_rank,
            "buy_candidate_rank": args.buy_candidate_rank,
            "rebalance_every": args.rebalance_every,
            "rebalance_offset": args.rebalance_offset,
            "profile": args.profile,
            "capacity_mode": args.capacity_mode,
            "mainboard_only": args.mainboard_only,
            "exclude_st": args.exclude_st,
            "lot_size": args.lot_size,
            "commission_rate": args.commission_rate,
            "stamp_tax_rate": args.stamp_tax_rate,
            "transfer_fee_rate": args.transfer_fee_rate,
            "min_commission": args.min_commission,
            "slippage_bps": args.slippage_bps,
        },
        "cash_input": args.cash,
        "portfolio_value_input": args.portfolio_value,
        "buy_cash_per_position": args.buy_cash_per_position,
        "cash_buffer_pct": args.cash_buffer_pct,
    }

    try:
        if not rank_file.exists():
            raise FileNotFoundError(rank_file)
        rank = pd.read_csv(rank_file)
        for c in ["date", "symbol", "rank", "pred_score"]:
            if c not in rank.columns:
                raise RuntimeError(f"rank file missing {c}")
        rank = rank.copy()
        rank["symbol"] = rank["symbol"].map(normalize_symbol)
        rank["date_ts"] = rank["date"].map(parse_date)
        rank["rank"] = pd.to_numeric(rank["rank"], errors="coerce")
        rank["pred_score"] = pd.to_numeric(rank["pred_score"], errors="coerce")
        rank = rank.dropna(subset=["symbol", "date_ts", "rank", "pred_score"])
        if rank.empty:
            raise RuntimeError("rank file empty after normalization")
        trade_date_ts = pd.Timestamp(rank["date_ts"].max()).normalize()
        trade_date = trade_date_ts.strftime("%Y-%m-%d")
        rday = rank[rank["date_ts"].eq(trade_date_ts)].copy().sort_values("rank")

        exec_side, exec_meta = load_execution_sidecar(price_file, args.price_column)
        report["execution_meta"] = exec_meta
        # Merge rank with execution data.  Execution sidecar wins for overlapping execution fields.
        rday = rday.merge(exec_side, on="symbol", how="left", suffixes=("", "_exec"))
        if "order_price_exec" in rday.columns and "order_price" in rday.columns:
            rday["order_price"] = rday["order_price_exec"].combine_first(rday["order_price"])
        elif "order_price_exec" in rday.columns:
            rday["order_price"] = rday["order_price_exec"]
        elif "order_price" not in rday.columns:
            # Final fallback: rank file may contain a price column.
            price_col = find_first_col(rday, PRICE_COLUMNS_PRIORITY)
            rday["order_price"] = pd.to_numeric(rday[price_col], errors="coerce") if price_col else np.nan
        for c, default in [("tradable", True), ("is_st", False), ("is_mainboard", None)]:
            if c not in rday.columns:
                rday[c] = default if default is not None else rday["symbol"].map(is_mainboard_symbol)
        if "up_limit" not in rday.columns:
            rday["up_limit"] = np.nan
        if "down_limit" not in rday.columns:
            rday["down_limit"] = np.nan
        if "last5_amount" not in rday.columns:
            rday["last5_amount"] = np.nan
        if "last5_volume" not in rday.columns:
            rday["last5_volume"] = np.nan
        if "trade_allowed_mainboard" in rday.columns:
            rday["is_mainboard"] = rday["trade_allowed_mainboard"].map(lambda x: parse_bool(x, True)).combine_first(rday["is_mainboard"])
        rday["is_mainboard"] = rday.apply(lambda x: parse_bool(x.get("is_mainboard"), is_mainboard_symbol(x["symbol"])), axis=1)
        rday["tradable"] = rday["tradable"].map(lambda x: parse_bool(x, True))
        rday["is_st"] = rday["is_st"].map(lambda x: parse_bool(x, False))
        rday["order_price"] = pd.to_numeric(rday["order_price"], errors="coerce")
        rday = rday.drop_duplicates("symbol", keep="last")
        exec_by_sym = {row["symbol"]: row for _, row in rday.iterrows()}

        positions = load_positions(pos_file)
        pos_by_sym = {row["symbol"]: row for _, row in positions.iterrows()}
        reb = compute_rebalance(args, trade_date_ts, rank)
        is_reb = bool(reb["is_rebalance_day"])
        report["rebalance"] = reb

        # Initial marking.
        holding_values: dict[str, float] = {}
        missing_marks: list[str] = []
        for sym, pos in pos_by_sym.items():
            row = exec_by_sym.get(sym)
            price = parse_float(row.get("order_price") if row is not None else np.nan, np.nan)
            if not math.isfinite(price) or price <= 0:
                missing_marks.append(sym)
                value = 0.0
            else:
                value = float(pos.get("shares", 0.0)) * price
            holding_values[sym] = value
        holding_value_before = sum(holding_values.values())
        if args.cash is not None and args.cash >= 0:
            cash = float(args.cash)
            cash_source = "cash"
        elif args.portfolio_value is not None and args.portfolio_value >= 0:
            cash = max(0.0, float(args.portfolio_value) - holding_value_before)
            cash_source = "portfolio_value_minus_marked_holdings"
        else:
            cash = 0.0
            cash_source = "missing_cash_default_zero"
        cash *= (1.0 - min(max(float(args.cash_buffer_pct), 0.0), 0.95))
        report["cash_source"] = cash_source
        report["cash_after_buffer_before_sells"] = cash
        report["holding_value_before"] = holding_value_before

        rows: list[dict[str, Any]] = []
        remaining_positions: dict[str, pd.Series] = dict(pos_by_sym)
        rank_map = rday.set_index("symbol").to_dict("index")
        score_map = rday.set_index("symbol")["pred_score"].to_dict()
        rank_num_map = rday.set_index("symbol")["rank"].to_dict()

        # Non-rebalance day: emit HOLD/WATCH only.
        if not is_reb:
            for sym, pos in sorted(pos_by_sym.items()):
                info = rank_map.get(sym, {})
                rows.append(row_dict(
                    date=trade_date, symbol=sym, action="HOLD", shares_current=float(pos.get("shares", 0.0)),
                    rank=rank_num_map.get(sym, np.nan), pred_score=score_map.get(sym, np.nan),
                    order_price=parse_float(info.get("order_price"), np.nan) if info else np.nan,
                    reason="not_rebalance_day", is_rebalance_day=False, day_index=reb.get("day_index"),
                ))
            for _, c in rday[rday["rank"] <= args.buy_candidate_rank].head(args.max_positions).iterrows():
                if c["symbol"] in pos_by_sym:
                    continue
                rows.append(row_dict(
                    date=trade_date, symbol=c["symbol"], action="WATCH", rank=int(c["rank"]),
                    pred_score=float(c["pred_score"]), order_price=parse_float(c.get("order_price"), np.nan),
                    reason="not_rebalance_day_candidate", is_rebalance_day=False, day_index=reb.get("day_index"),
                ))
        else:
            # 1) Sell positions whose current rank falls beyond sell_rank.
            for sym, pos in list(pos_by_sym.items()):
                info = rank_map.get(sym)
                rank_val = float(rank_num_map.get(sym, math.inf))
                score_val = score_map.get(sym, np.nan)
                shares_current = float(pos.get("shares", 0.0))
                row = pd.Series(info) if info is not None else pd.Series(dtype=object)
                price = parse_float(row.get("order_price"), np.nan)
                should_sell = bool(rank_val > args.sell_rank)
                if not should_sell:
                    rows.append(row_dict(
                        date=trade_date, symbol=sym, action="HOLD", shares_current=shares_current,
                        rank=(int(rank_val) if math.isfinite(rank_val) else np.nan), pred_score=score_val,
                        order_price=price, reason=f"rank<={args.sell_rank}", is_rebalance_day=True, day_index=reb.get("day_index"),
                    ))
                    continue
                buy_date = pick_buy_date(pos)
                if pd.isna(buy_date):
                    if args.unknown_buy_date_policy == "block":
                        rows.append(row_dict(
                            date=trade_date, symbol=sym, action="SELL_BLOCKED", shares_current=shares_current,
                            rank=(int(rank_val) if math.isfinite(rank_val) else np.nan), pred_score=score_val,
                            order_price=price, reason="t_plus_1_unknown_buy_date_blocked", is_rebalance_day=True, day_index=reb.get("day_index"),
                        ))
                        continue
                    tplus_reason = "t_plus_1_unknown_buy_date_allowed"
                elif trade_date_ts <= pd.Timestamp(buy_date).normalize():
                    rows.append(row_dict(
                        date=trade_date, symbol=sym, action="SELL_BLOCKED", shares_current=shares_current,
                        rank=(int(rank_val) if math.isfinite(rank_val) else np.nan), pred_score=score_val,
                        order_price=price, reason="t_plus_1_restriction", is_rebalance_day=True, day_index=reb.get("day_index"),
                    ))
                    continue
                else:
                    tplus_reason = ""
                ok, reason = can_sell(row, args)
                if not ok:
                    rows.append(row_dict(
                        date=trade_date, symbol=sym, action="SELL_BLOCKED", shares_current=shares_current,
                        rank=(int(rank_val) if math.isfinite(rank_val) else np.nan), pred_score=score_val,
                        order_price=price, reason=reason, is_rebalance_day=True, day_index=reb.get("day_index"),
                    ))
                    continue
                cap_shares, cap_reason = sell_capacity_shares(row, args)
                if cap_shares is None:
                    sell_shares = shares_current
                else:
                    sell_shares = min(shares_current, float(floor_to_lot(cap_shares, args.lot_size)))
                if sell_shares <= 0:
                    rows.append(row_dict(
                        date=trade_date, symbol=sym, action="SELL_BLOCKED", shares_current=shares_current,
                        rank=(int(rank_val) if math.isfinite(rank_val) else np.nan), pred_score=score_val,
                        order_price=price, reason=f"capacity_zero_{cap_reason}", is_rebalance_day=True, day_index=reb.get("day_index"),
                    ))
                    continue
                raw_exec_price = price * (1.0 - args.slippage_bps / 10000.0)
                notional = sell_shares * raw_exec_price
                fees = trade_fee_components(notional, "sell", args.commission_rate, args.stamp_tax_rate, args.transfer_fee_rate, args.min_commission)
                cash += notional - fees["total_fee"]
                remaining_positions.pop(sym, None)
                holding_values.pop(sym, None)
                rows.append(row_dict(
                    date=trade_date, symbol=sym, action="SELL", order_side="SELL", order_shares=int(sell_shares),
                    order_price=raw_exec_price, order_amount_est=notional, target_value=0.0, intended_shares=int(shares_current),
                    filled_shares=int(sell_shares), commission=fees["commission"], stamp_tax=fees["stamp_tax"],
                    transfer_fee=fees["transfer_fee"], total_fee=fees["total_fee"], rank=(int(rank_val) if math.isfinite(rank_val) else np.nan),
                    pred_score=score_val, shares_current=shares_current,
                    reason="rank_gt_sell_rank" + (f";{tplus_reason}" if tplus_reason else ""), order_status="planned",
                    is_rebalance_day=True, day_index=reb.get("day_index"), capacity_reason=cap_reason,
                    partial_fill=bool(sell_shares < shares_current - 1e-12),
                ))

            holding_value_after_sells = sum(holding_values.get(sym, 0.0) for sym in remaining_positions)
            nav_after_sells = cash + holding_value_after_sells
            report["cash_after_planned_sells_before_buys"] = cash
            report["nav_after_planned_sells"] = nav_after_sells

            # 2) Fill empty slots. Existing holdings are not replaced by higher-rank names.
            if len(remaining_positions) < args.max_positions and cash > 0:
                candidate_symbols = rday.loc[rday["rank"] <= args.buy_candidate_rank, "symbol"].tolist()
                for sym in candidate_symbols:
                    if len(remaining_positions) >= args.max_positions:
                        break
                    if sym in remaining_positions:
                        continue
                    row = pd.Series(rank_map.get(sym, {}))
                    rank_val = float(rank_num_map.get(sym, math.inf))
                    score_val = score_map.get(sym, np.nan)
                    price = parse_float(row.get("order_price"), np.nan)
                    ok, reason = can_buy(row, args)
                    if not ok:
                        rows.append(row_dict(
                            date=trade_date, symbol=sym, action="BUY_BLOCKED", rank=int(rank_val), pred_score=score_val,
                            order_price=price, reason=reason, is_rebalance_day=True, day_index=reb.get("day_index"),
                        ))
                        continue
                    slots = max(1, args.max_positions - len(remaining_positions))
                    if args.buy_cash_per_position is not None and args.buy_cash_per_position > 0:
                        base_target = float(args.buy_cash_per_position)
                        sizing_reason = "buy_cash_per_position_override_not_backtest_exact"
                    else:
                        base_target = min(nav_after_sells / args.max_positions, cash / slots if slots > 1 else cash)
                        sizing_reason = "backtest_base_target"
                    cap_notional, cap_reason = buy_capacity_notional(row, args)
                    target_notional = min(base_target, cap_notional)
                    raw_exec_price = price * (1.0 + args.slippage_bps / 10000.0)
                    intended_shares = floor_to_lot(target_notional / raw_exec_price, args.lot_size)
                    shares = intended_shares
                    while shares > 0:
                        notional = shares * raw_exec_price
                        fees = trade_fee_components(notional, "buy", args.commission_rate, args.stamp_tax_rate, args.transfer_fee_rate, args.min_commission)
                        if notional + fees["total_fee"] <= cash + 1e-9:
                            break
                        shares -= args.lot_size
                    if shares <= 0:
                        reason2 = f"capacity_zero_{cap_reason}" if math.isfinite(cap_notional) and cap_notional <= 0 else "cash_or_lot_too_small"
                        rows.append(row_dict(
                            date=trade_date, symbol=sym, action="BUY_BLOCKED", rank=int(rank_val), pred_score=score_val,
                            order_price=raw_exec_price, target_value=target_notional, intended_shares=int(intended_shares),
                            reason=f"{reason2};{sizing_reason}", is_rebalance_day=True, day_index=reb.get("day_index"),
                            capacity_reason=cap_reason,
                        ))
                        continue
                    notional = shares * raw_exec_price
                    fees = trade_fee_components(notional, "buy", args.commission_rate, args.stamp_tax_rate, args.transfer_fee_rate, args.min_commission)
                    cash -= notional + fees["total_fee"]
                    remaining_positions[sym] = pd.Series({"symbol": sym, "shares": float(shares), "buy_date": trade_date})
                    holding_values[sym] = float(shares) * price
                    rows.append(row_dict(
                        date=trade_date, symbol=sym, action="BUY", order_side="BUY", order_shares=int(shares),
                        order_price=raw_exec_price, order_amount_est=notional, target_value=target_notional,
                        intended_shares=int(intended_shares), filled_shares=int(shares), commission=fees["commission"],
                        stamp_tax=fees["stamp_tax"], transfer_fee=fees["transfer_fee"], total_fee=fees["total_fee"],
                        rank=int(rank_val), pred_score=score_val, shares_current=0.0,
                        reason=f"fill_empty_slot_rank_le_buy_candidate_rank;{sizing_reason}", order_status="planned",
                        is_rebalance_day=True, day_index=reb.get("day_index"), capacity_reason=cap_reason,
                        partial_fill=bool(math.isfinite(cap_notional) and cap_notional < base_target - 1e-9),
                    ))

            # Add WATCH rows for top candidates not otherwise emitted, useful for review.
            emitted = {r["symbol"] for r in rows if r.get("symbol")}
            for _, c in rday[rday["rank"] <= args.buy_candidate_rank].head(args.max_positions * 2).iterrows():
                sym = c["symbol"]
                if sym in emitted:
                    continue
                rows.append(row_dict(
                    date=trade_date, symbol=sym, action="WATCH", rank=int(c["rank"]), pred_score=float(c["pred_score"]),
                    order_price=parse_float(c.get("order_price"), np.nan), reason="candidate_no_empty_slot_or_not_reached",
                    is_rebalance_day=True, day_index=reb.get("day_index"),
                ))

        out = pd.DataFrame(rows)
        wanted = [
            "date", "symbol", "action", "order_side", "order_shares", "order_price", "order_amount_est",
            "target_value", "intended_shares", "filled_shares", "commission", "stamp_tax", "transfer_fee",
            "total_fee", "rank", "pred_score", "shares_current", "reason", "order_status", "is_rebalance_day",
            "day_index", "capacity_reason", "partial_fill",
        ]
        if out.empty:
            out = pd.DataFrame(columns=wanted)
        else:
            for c in wanted:
                if c not in out.columns:
                    out[c] = np.nan
            out = out[wanted]
        out_signal.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_signal, index=False, encoding="utf-8-sig")

        report.update({
            "passed": True,
            "signal_file": str(out_signal),
            "trade_date": trade_date,
            "positions_found": int(len(positions)),
            "rows": int(len(out)),
            "action_counts": out["action"].value_counts(dropna=False).to_dict(),
            "order_side_counts": out["order_side"].replace("", np.nan).value_counts(dropna=False).to_dict(),
            "buy_order_shares_sum": int(pd.to_numeric(out.loc[out["order_side"].eq("BUY"), "order_shares"], errors="coerce").fillna(0).sum()),
            "sell_order_shares_sum": int(pd.to_numeric(out.loc[out["order_side"].eq("SELL"), "order_shares"], errors="coerce").fillna(0).sum()),
            "buy_order_amount_est_sum": float(pd.to_numeric(out.loc[out["order_side"].eq("BUY"), "order_amount_est"], errors="coerce").fillna(0).sum()),
            "sell_order_amount_est_sum": float(pd.to_numeric(out.loc[out["order_side"].eq("SELL"), "order_amount_est"], errors="coerce").fillna(0).sum()),
            "cash_after_planned_orders": float(cash),
            "remaining_positions_after_plan": int(len(remaining_positions)),
            "missing_marks": missing_marks,
            "parity_notes": [
                "single-day live planner cannot replay full historical NAV/corporate-action state",
                "rebalance parity is exact only when day_index or rebalance_calendar is supplied, or force_rebalance is intentionally used",
                "T+1 parity requires buy_date/entry_date in current_positions.csv; missing dates follow unknown_buy_date_policy",
            ],
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        raise
    finally:
        report["elapsed_seconds"] = round(time.time() - start, 3)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    print(json.dumps({
        "passed": report["passed"],
        "signal_file": report.get("signal_file"),
        "is_rebalance_day": report.get("rebalance", {}).get("is_rebalance_day"),
        "rebalance_reason": report.get("rebalance", {}).get("reason"),
        "action_counts": report.get("action_counts"),
        "buy_order_shares_sum": report.get("buy_order_shares_sum"),
        "sell_order_shares_sum": report.get("sell_order_shares_sum"),
        "cash_after_planned_orders": report.get("cash_after_planned_orders"),
    }, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
