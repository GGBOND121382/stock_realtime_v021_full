#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate AS1455 model_data contract.

This is the hard gate that prevents raw-price model_data from entering
weekly retrain/backtest or live feature alignment.

The intended clean contract is:
  raw 5m cache / raw daily close+preclose
  -> manual front-adjustment factor from raw daily preclose / previous raw close
  -> adj_open/high/low/close_as1455
  -> ML4T-style per-symbol observation horizon features and forward labels
  -> model_data_as1455.h5

The horizon intentionally follows the original ML4T implementation:
per-symbol effective observation sequence, not a forced exchange-calendar reindex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "dollar_vol",
    "dollar_vol_rank",
    "rsi",
    "bb_high",
    "bb_low",
    "NATR",
    "ATR",
    "PPO",
    "MACD",
    "sector",
    "r01",
    "r05",
    "r10",
    "r21",
    "r42",
    "r63",
    "r01dec",
    "r05dec",
    "r10dec",
    "r21dec",
    "r42dec",
    "r63dec",
    "r01q_sector",
    "r05q_sector",
    "r10q_sector",
    "r21q_sector",
    "r42q_sector",
    "r63q_sector",
    "r01_fwd",
    "r05_fwd",
    "r21_fwd",
    "year",
    "month",
    "weekday",
]
OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]
RET_WINDOWS = [1, 5, 10, 21, 42, 63]
FWD_WINDOWS = [1, 5, 21]

CONTRACT_NAME = "model_data_contract.json"
AUDIT_NAME = "model_data_contract_validation.json"
PRICE_BASIS = "manual_qfq_from_raw_daily_preclose"
BUILDER = "scripts/build_ashare_ch12_as1455_model_data.py"
HORIZON = "ml4t_per_symbol_observation_horizon"


def normalize_symbol(value: Any) -> str:
    s = str(value).strip().upper()
    if "." in s:
        a, b = s.split(".", 1)
        if a.isalpha():
            code = "".join(ch for ch in b if ch.isdigit())[:6].zfill(6)
            market = a
        else:
            code = "".join(ch for ch in a if ch.isdigit())[:6].zfill(6)
            market = b
        if market in {"XSHE", "SZSE"}:
            market = "SZ"
        if market in {"XSHG", "SSE"}:
            market = "SH"
        if market not in {"SH", "SZ"}:
            market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code}.{market}"
    digits = "".join(ch for ch in s if ch.isdigit())[:6].zfill(6)
    market = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{market}"


def code6(symbol: str) -> str:
    return normalize_symbol(symbol).split(".", 1)[0]


def is_mainboard_code(code: str) -> bool:
    return str(code).startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def load_hdf(path: Path, key: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty HDF: {path}")
    try:
        return pd.read_hdf(path, key)
    except Exception as exc:
        raise SystemExit(f"failed to read {path}:{key}: {type(exc).__name__}: {exc}") from exc


def align_symbol_date_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.MultiIndex):
        raise SystemExit(f"{name} must have MultiIndex")
    names = list(out.index.names)
    if names == ["date", "symbol"]:
        out = out.swaplevel("date", "symbol")
    elif names != ["symbol", "date"]:
        raise SystemExit(f"{name} index names must be ['symbol','date'] or ['date','symbol'], got {names}")
    symbols = [normalize_symbol(x) for x in out.index.get_level_values("symbol")]
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce").normalize()
    if pd.isna(dates).any():
        raise SystemExit(f"{name} contains invalid dates in index")
    out.index = pd.MultiIndex.from_arrays([symbols, dates], names=["symbol", "date"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def max_abs_diff(a: pd.Series, b: pd.Series) -> tuple[int, int, float | None]:
    both = ~(a.isna() | b.isna())
    if not bool(both.any()):
        return 0, 0, None
    d = (a.loc[both].astype(float) - b.loc[both].astype(float)).abs()
    return int(both.sum()), int((d > 1e-8).sum()), float(d.max())


def read_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing contract: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read contract {path}: {type(exc).__name__}: {exc}") from exc
    return obj


def validate_contract_object(obj: dict[str, Any], contract_path: Path) -> None:
    required = {
        "builder": BUILDER,
        "price_basis": PRICE_BASIS,
        "adjust_factor_mode": "raw_preclose",
        "feature_horizon": HORIZON,
        "model_data_key": "model_data",
    }
    for key, expected in required.items():
        got = obj.get(key)
        if got != expected:
            raise SystemExit(f"{contract_path}: bad {key}: got {got!r}, expected {expected!r}")

    outcomes = obj.get("outcomes")
    if outcomes != OUTCOMES:
        raise SystemExit(f"{contract_path}: bad outcomes: got {outcomes!r}, expected {OUTCOMES!r}")

    if int(obj.get("schema_columns", -1)) != 34:
        raise SystemExit(f"{contract_path}: bad schema_columns: {obj.get('schema_columns')!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate AS1455 adjusted model_data contract")
    ap.add_argument("--model-data", default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5")
    ap.add_argument("--require-contract", action="store_true")
    ap.add_argument("--write-contract", action="store_true")
    ap.add_argument("--require-adjusted-artifacts", action="store_true")
    ap.add_argument("--max-mainboard-abs-r01-fail", type=float, default=0.50,
                    help="Hard fail if any mainboard abs(r01) exceeds this; adjusted data should not show split-like jumps")
    ap.add_argument("--warn-mainboard-abs-r01", type=float, default=0.20,
                    help="Report count of mainboard abs(r01) above this warning threshold")
    args = ap.parse_args()

    model_path = Path(args.model_data)
    root = model_path.parent
    contract_path = root / CONTRACT_NAME
    audit_path = root / AUDIT_NAME
    adj_path = root / "as1455_ohlcv_adj.h5"

    if args.require_contract:
        validate_contract_object(read_contract(contract_path), contract_path)

    if args.require_adjusted_artifacts and (not adj_path.exists() or adj_path.stat().st_size == 0):
        raise SystemExit(f"missing adjusted artifact required by clean contract: {adj_path}")

    model = align_symbol_date_index(load_hdf(model_path, "model_data"), "model_data")

    problems: list[str] = []
    if list(model.columns) != EXPECTED_COLUMNS:
        problems.append("model_data columns do not exactly match expected 34-column schema")
    if model.shape[1] != 34:
        problems.append(f"model_data must have 34 columns, got {model.shape[1]}")
    outcomes = model.filter(like="fwd").columns.tolist()
    if outcomes != OUTCOMES:
        problems.append(f"bad outcomes: {outcomes}")

    dates = pd.to_datetime(model.index.get_level_values("date"))
    symbols = pd.Index(model.index.get_level_values("symbol"))
    sectors = pd.to_numeric(model["sector"], errors="coerce") if "sector" in model else pd.Series(dtype=float)
    sector_nunique = int(sectors.nunique(dropna=True)) if len(sectors) else 0
    if sector_nunique <= 1:
        problems.append(f"sector degenerated: sector_nunique={sector_nunique}")

    codes = pd.Series([code6(x) for x in symbols], index=model.index)
    main_mask = codes.map(is_mainboard_code).astype(bool)
    r01 = pd.to_numeric(model["r01"], errors="coerce")
    main_abs = r01.loc[main_mask].abs().dropna()
    warn_main_rows = int(main_abs.gt(args.warn_mainboard_abs_r01).sum())
    fail_main_rows = int(main_abs.gt(args.max_mainboard_abs_r01_fail).sum())
    if fail_main_rows:
        worst = model.loc[main_abs.nlargest(min(20, fail_main_rows)).index, ["r01"]].reset_index()
        problems.append(
            f"mainboard abs(r01)>{args.max_mainboard_abs_r01_fail:g} rows={fail_main_rows}; "
            f"worst={worst.to_dict(orient='records')}"
        )

    recalc_summary: dict[str, Any] = {}
    if adj_path.exists() and adj_path.stat().st_size > 0:
        adj = align_symbol_date_index(load_hdf(adj_path, "ohlcv"), "as1455_ohlcv_adj")
        if "adj_close_as1455" not in adj.columns:
            problems.append(f"{adj_path} missing adj_close_as1455")
        else:
            common = model.index.intersection(adj.index)
            if len(common) == 0:
                problems.append("model_data and as1455_ohlcv_adj.h5 have no common index")
            close = pd.to_numeric(adj.loc[common, "adj_close_as1455"], errors="coerce").sort_index()
            m_common = model.loc[common].sort_index()
            by_symbol_close = close.groupby(level="symbol")
            for t in RET_WINDOWS:
                col = f"r{t:02}"
                recalc = by_symbol_close.pct_change(t).reindex(m_common.index)
                compared, bad, maxdiff = max_abs_diff(pd.to_numeric(m_common[col], errors="coerce"), recalc)
                recalc_summary[col] = {"compared": compared, "diff_gt_1e-8": bad, "max_abs_diff": maxdiff}
                if bad:
                    problems.append(f"{col} differs from adj_close_as1455 recomputation: bad={bad}, maxdiff={maxdiff}")
            for t in FWD_WINDOWS:
                col = f"r{t:02}_fwd"
                base = pd.to_numeric(m_common[f"r{t:02}"], errors="coerce")
                recalc = base.groupby(level="symbol").shift(-t).reindex(m_common.index)
                compared, bad, maxdiff = max_abs_diff(pd.to_numeric(m_common[col], errors="coerce"), recalc)
                recalc_summary[col] = {"compared": compared, "diff_gt_1e-8": bad, "max_abs_diff": maxdiff}
                if bad:
                    problems.append(f"{col} differs from ML4T shift recomputation: bad={bad}, maxdiff={maxdiff}")
    elif args.require_adjusted_artifacts:
        problems.append(f"missing adjusted artifact: {adj_path}")

    contract = {
        "builder": BUILDER,
        "price_basis": PRICE_BASIS,
        "adjust_factor_mode": "raw_preclose",
        "feature_horizon": HORIZON,
        "model_data_key": "model_data",
        "schema_columns": 34,
        "outcomes": OUTCOMES,
        "model_data_path": str(model_path),
        "adjusted_ohlcv_path": str(adj_path),
        "rows": int(len(model)),
        "symbols": int(symbols.nunique()),
        "date_min": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
        "date_max": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
        "sector_nunique": sector_nunique,
    }

    audit = dict(contract)
    audit.update(
        {
            "contract_path": str(contract_path),
            "warn_mainboard_abs_r01_threshold": args.warn_mainboard_abs_r01,
            "warn_mainboard_abs_r01_rows": warn_main_rows,
            "fail_mainboard_abs_r01_threshold": args.max_mainboard_abs_r01_fail,
            "fail_mainboard_abs_r01_rows": fail_main_rows,
            "recalc_summary": recalc_summary,
            "passed": not problems,
            "problems": problems,
        }
    )

    if args.write_contract:
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if problems:
        raise SystemExit("AS1455 model_data contract validation failed; see " + str(audit_path))


if __name__ == "__main__":
    main()
