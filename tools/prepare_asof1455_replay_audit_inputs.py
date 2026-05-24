#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare snapshot-cache and temporary model metadata for replay reconstruction audits."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "saved_data" / "asof1455_audit_minidata"
OUT = ROOT / "saved_data" / "feature_reconstruction_audit" / "asof1455_replay_inputs"

SAMPLE_BY_STOCK = {
    "603308.SH": MINI / "603308_pipeline_out" / "04_external" / "aero_nuclear_equipment" / "training_samples_with_aero_nuclear_equipment_external.csv",
    "600487.SH": MINI / "600487_pipeline_out" / "04_external" / "optical_cable_grid" / "training_samples_with_optical_cable_grid_external.csv",
    "600312.SH": MINI / "600312_pipeline_out" / "03_sector" / "training_samples_with_sector.csv",
}
INTRADAY_BY_STOCK = {
    "603308.SH": MINI / "603308_pipeline_out" / "00_base" / "603308_5m.csv",
    "600487.SH": MINI / "600487_pipeline_out" / "00_base" / "600487_5m.csv",
    "600312.SH": MINI / "600312_pipeline_out" / "00_base" / "600312_5m.csv",
}


def normalize_symbol(value: str) -> str:
    s = str(value).strip().upper().replace("_", ".")
    if "." in s:
        a, b = s.split(".", 1)
        if a in {"SH", "SZ"}:
            return f"{b.zfill(6)}.{a}"
        return f"{a.zfill(6)}.{b}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def exchange(symbol: str) -> str:
    return normalize_symbol(symbol).split(".", 1)[1]


def write_snapshot_cache(dates: list[str], out_cache: Path) -> pd.DataFrame:
    rows = []
    for stock, intraday_path in INTRADAY_BY_STOCK.items():
        if not intraday_path.exists():
            rows.append({"stock_code": stock, "status": "missing_intraday", "path": str(intraday_path)})
            continue
        bars = pd.read_csv(intraday_path, parse_dates=["datetime"])
        bars["date"] = bars["datetime"].dt.strftime("%Y%m%d")
        for date in dates:
            day = bars[bars["date"] == date].sort_values("datetime").copy()
            if day.empty:
                rows.append({"stock_code": stock, "trade_date": date, "status": "missing_date"})
                continue
            day["cum_volume"] = pd.to_numeric(day["volume"], errors="coerce").fillna(0).cumsum()
            day["cum_amount"] = pd.to_numeric(day["amount"], errors="coerce").fillna(0).cumsum()
            prev = bars[bars["datetime"].dt.normalize() < day["datetime"].dt.normalize().iloc[0]]
            prev_close = float(prev.sort_values("datetime")["close"].iloc[-1]) if not prev.empty else np.nan
            out_rows = []
            for _, r in day.iterrows():
                dt = pd.to_datetime(r["datetime"])
                close = float(r["close"])
                pct = close / prev_close - 1.0 if np.isfinite(prev_close) and prev_close else np.nan
                out_rows.append(
                    {
                        "vendor": "synthetic_from_training_5m",
                        "source": "training_5m",
                        "quote_source": "training_5m",
                        "collected_at": dt.isoformat(),
                        "datetime": dt.isoformat(),
                        "symbol": stock,
                        "exchange": exchange(stock),
                        "trade_date": date,
                        "trade_time": dt.strftime("%H%M%S"),
                        "phase": "continuous",
                        "name": stock,
                        "last_price": close,
                        "open": float(r["open"]),
                        "high": float(day.loc[: r.name, "high"].max()),
                        "low": float(day.loc[: r.name, "low"].min()),
                        "prev_close": prev_close,
                        "volume": float(r["cum_volume"]),
                        "amount": float(r["cum_amount"]),
                        "turnover": np.nan,
                        "pct_chg": pct,
                        "pct_chg_raw": pct,
                        "pct_chg_norm": pct,
                        "pct_chg_source": "computed_from_prev_close",
                        "pct_chg_unit": "ratio",
                    }
                )
            sym_dir = out_cache / "pending" / date / stock
            sym_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(out_rows).to_csv(sym_dir / "snapshot_5level.csv", index=False, encoding="utf-8-sig")
            rows.append({"stock_code": stock, "trade_date": date, "status": "ok", "snapshot_rows": len(out_rows), "path": str(sym_dir / "snapshot_5level.csv")})
    return pd.DataFrame(rows)


def prepare_models(saved_models: Path, out_models: Path) -> pd.DataFrame:
    rows = []
    if out_models.exists():
        shutil.rmtree(out_models)
    for meta_path in sorted(saved_models.rglob("metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stock = normalize_symbol(str(meta.get("stock_code") or meta_path.parent.parent.name))
        if stock not in SAMPLE_BY_STOCK:
            continue
        sample = SAMPLE_BY_STOCK[stock]
        intraday = INTRADAY_BY_STOCK[stock]
        if not sample.exists() or not intraday.exists():
            rows.append({"stock_code": stock, "artifact_name": meta_path.parent.name, "status": "missing_minidata"})
            continue
        dst = out_models / stock / meta_path.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_path.parent / "feature_columns.txt", dst / "feature_columns.txt")
        meta["samples"] = str(sample.resolve())
        meta["intraday_bars"] = str(intraday.resolve())
        meta["artifact_name"] = str(meta.get("artifact_name") or meta_path.parent.name)
        meta["stock_code"] = stock
        (dst / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"stock_code": stock, "artifact_name": meta["artifact_name"], "status": "ok", "samples": str(sample), "intraday": str(intraday)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dates", default="20260519,20260520,20260521")
    p.add_argument("--out-root", default=str(OUT))
    p.add_argument("--saved-models", default=str(ROOT / "saved_models"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dates = [d.strip().replace("-", "") for d in args.dates.split(",") if d.strip()]
    cache = out_root / "synthetic_snapshot_cache"
    models = out_root / "temp_saved_models"
    snap_report = write_snapshot_cache(dates, cache)
    model_report = prepare_models(Path(args.saved_models), models)
    snap_report.to_csv(out_root / "snapshot_cache_report.csv", index=False, encoding="utf-8-sig")
    model_report.to_csv(out_root / "temp_model_report.csv", index=False, encoding="utf-8-sig")
    print(f"WROTE {out_root / 'snapshot_cache_report.csv'} rows={len(snap_report)}")
    print(f"WROTE {out_root / 'temp_model_report.csv'} rows={len(model_report)}")
    print(f"SNAPSHOT_CACHE={cache}")
    print(f"TEMP_SAVED_MODELS={models}")


if __name__ == "__main__":
    main()
