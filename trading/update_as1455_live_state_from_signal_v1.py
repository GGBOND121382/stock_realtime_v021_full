#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update AS1455 paper/live state from planned single-day trade signal.

This helper persists planned state for multi-day live-signal replay. It is not a
broker reconciliation tool: it assumes the planned BUY/SELL rows in
16_live_trade_signal.csv are filled exactly as written.
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


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return pd.NaT
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{8}", digits):
        return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def date_str(value: object) -> str:
    ts = parse_date(value)
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


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


def load_positions(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["symbol", "shares", "buy_date", "avg_entry_price"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        if "code" in df.columns:
            df["symbol"] = df["code"]
        else:
            raise RuntimeError(f"positions file has no symbol/code column: {path}")
    if "shares" not in df.columns:
        df["shares"] = 0
    out = df.copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0.0)
    if "buy_date" not in out.columns:
        for c in ["entry_date", "date_bought", "open_date", "date"]:
            if c in out.columns:
                out["buy_date"] = out[c]
                break
    if "buy_date" not in out.columns:
        out["buy_date"] = ""
    out["buy_date"] = out["buy_date"].map(date_str)
    if "avg_entry_price" not in out.columns:
        for c in ["cost_price", "entry_price", "price", "last_price"]:
            if c in out.columns:
                out["avg_entry_price"] = out[c]
                break
    if "avg_entry_price" not in out.columns:
        out["avg_entry_price"] = np.nan
    out["avg_entry_price"] = pd.to_numeric(out["avg_entry_price"], errors="coerce")
    out = out[(out["symbol"].astype(str).str.len() > 0) & (out["shares"] > 0)]
    return out[["symbol", "shares", "buy_date", "avg_entry_price"]].drop_duplicates("symbol", keep="last").reset_index(drop=True)


def load_state_positions(state_file: Path) -> tuple[pd.DataFrame, float | None, dict[str, Any]]:
    if not state_file.exists():
        return pd.DataFrame(columns=["symbol", "shares", "buy_date", "avg_entry_price"]), None, {}
    data = json.loads(state_file.read_text(encoding="utf-8"))
    pos = pd.DataFrame(data.get("positions", []))
    tmp = state_file.with_suffix(".positions.tmp.csv")
    if pos.empty:
        pos = pd.DataFrame(columns=["symbol", "shares", "buy_date", "avg_entry_price"])
    pos.to_csv(tmp, index=False, encoding="utf-8-sig")
    return load_positions(tmp), data.get("cash"), data


def positions_to_records(pos: pd.DataFrame) -> list[dict[str, Any]]:
    if pos.empty:
        return []
    out = pos.copy()
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0.0)
    out = out[out["shares"] > 0]
    rows = []
    for _, r in out.iterrows():
        rows.append({
            "symbol": str(r["symbol"]),
            "shares": float(r["shares"]),
            "buy_date": str(r.get("buy_date", "") or ""),
            "avg_entry_price": (None if pd.isna(r.get("avg_entry_price")) else float(r.get("avg_entry_price"))),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--positions-before", required=True)
    ap.add_argument("--signal-file", required=True)
    ap.add_argument("--signal-report", required=True)
    ap.add_argument("--out-positions-csv", required=True)
    ap.add_argument("--out-cash-file", required=True)
    ap.add_argument("--assume-filled", action="store_true", default=True)
    args = ap.parse_args()

    start = time.time()
    state_file = Path(args.state_file)
    positions_before = Path(args.positions_before)
    signal_file = Path(args.signal_file)
    signal_report = Path(args.signal_report)
    out_positions_csv = Path(args.out_positions_csv)
    out_cash_file = Path(args.out_cash_file)

    if not signal_file.exists():
        raise FileNotFoundError(signal_file)
    if not signal_report.exists():
        raise FileNotFoundError(signal_report)

    pos = load_positions(positions_before)
    pos_map: dict[str, dict[str, Any]] = {}
    for _, r in pos.iterrows():
        pos_map[str(r["symbol"])] = {
            "symbol": str(r["symbol"]),
            "shares": float(r["shares"]),
            "buy_date": str(r.get("buy_date", "") or ""),
            "avg_entry_price": (None if pd.isna(r.get("avg_entry_price")) else float(r.get("avg_entry_price"))),
        }

    sig = pd.read_csv(signal_file)
    if sig.empty:
        orders = sig
        trade_date = ""
    else:
        sig["symbol"] = sig["symbol"].map(normalize_symbol)
        sig["order_side"] = sig.get("order_side", "").astype(str).str.upper()
        sig["order_shares"] = pd.to_numeric(sig.get("order_shares", 0), errors="coerce").fillna(0.0)
        sig["order_price"] = pd.to_numeric(sig.get("order_price", np.nan), errors="coerce")
        orders = sig[sig["order_side"].isin(["BUY", "SELL"]) & (sig["order_shares"] > 0)].copy()
        trade_date = date_str(sig["date"].iloc[0]) if "date" in sig.columns and len(sig) else ""

    applied_orders = []
    for _, o in orders.iterrows():
        sym = str(o["symbol"])
        shares = float(o["order_shares"])
        side = str(o["order_side"]).upper()
        price = float(o["order_price"]) if math.isfinite(float(o["order_price"])) else None
        if side == "SELL":
            cur = pos_map.get(sym)
            before = float(cur.get("shares", 0.0)) if cur else 0.0
            after = max(0.0, before - shares)
            if cur and after > 1e-9:
                cur["shares"] = after
                pos_map[sym] = cur
            else:
                pos_map.pop(sym, None)
            applied_orders.append({"side": side, "symbol": sym, "shares": shares, "before": before, "after": after})
        elif side == "BUY":
            cur = pos_map.get(sym)
            if cur:
                before = float(cur.get("shares", 0.0))
                old_price = cur.get("avg_entry_price")
                after = before + shares
                if price is not None and old_price is not None and before > 0:
                    cur["avg_entry_price"] = (before * float(old_price) + shares * price) / after
                elif price is not None:
                    cur["avg_entry_price"] = price
                cur["shares"] = after
                # Keep older buy_date for T+1 if adding to an existing position.
                if not cur.get("buy_date"):
                    cur["buy_date"] = trade_date
                pos_map[sym] = cur
            else:
                pos_map[sym] = {"symbol": sym, "shares": shares, "buy_date": trade_date, "avg_entry_price": price}
                before = 0.0
                after = shares
            applied_orders.append({"side": side, "symbol": sym, "shares": shares, "before": before, "after": after})

    report = json.loads(signal_report.read_text(encoding="utf-8"))
    cash_after = report.get("cash_after_planned_orders")
    if cash_after is None:
        raise RuntimeError(f"signal report missing cash_after_planned_orders: {signal_report}")
    cash_after = float(cash_after)

    final_pos = pd.DataFrame(list(pos_map.values()))
    if final_pos.empty:
        final_pos = pd.DataFrame(columns=["symbol", "shares", "buy_date", "avg_entry_price"])
    else:
        final_pos = final_pos[["symbol", "shares", "buy_date", "avg_entry_price"]].sort_values("symbol")

    out_positions_csv.parent.mkdir(parents=True, exist_ok=True)
    final_pos.to_csv(out_positions_csv, index=False, encoding="utf-8-sig")
    out_cash_file.parent.mkdir(parents=True, exist_ok=True)
    out_cash_file.write_text(str(cash_after), encoding="utf-8")

    state = {
        "state_type": "planned_paper_state",
        "assumption": "orders in 16_live_trade_signal.csv are treated as filled exactly as planned",
        "updated_trade_date": trade_date,
        "cash": cash_after,
        "positions": positions_to_records(final_pos),
        "n_positions": int(len(final_pos)),
        "source_signal_file": str(signal_file),
        "source_signal_report": str(signal_report),
        "positions_before_file": str(positions_before),
        "positions_csv": str(out_positions_csv),
        "cash_file": str(out_cash_file),
        "applied_orders": applied_orders,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({
        "passed": True,
        "state_file": str(state_file),
        "positions_csv": str(out_positions_csv),
        "cash": cash_after,
        "n_positions": int(len(final_pos)),
        "n_applied_orders": len(applied_orders),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
