#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AS1455 = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_as1455"
DEFAULT_QFQ = PROJECT_DIR / "saved_data" / "ashare_ml4t" / "ch12_reproduce" / "baostock_qfq_daily_cache"


def normalize_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6)


def resolve_source_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def load_qfq(path: Path) -> pd.DataFrame:
    qfq = pd.read_csv(path, usecols=["date", "open", "close"])
    qfq["date"] = pd.to_datetime(qfq["date"], errors="coerce").dt.normalize()
    qfq["qfq_daily_open"] = pd.to_numeric(qfq.pop("open"), errors="coerce")
    qfq["qfq_daily_close"] = pd.to_numeric(qfq.pop("close"), errors="coerce")
    return qfq.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")


def inspect_5m(path: Path | None, dates: set[pd.Timestamp]) -> dict[pd.Timestamp, dict[str, object]]:
    if path is None or not path.exists() or not dates:
        return {}
    bars = pd.read_csv(path)
    if "datetime" not in bars:
        return {}
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    bars["date"] = bars["datetime"].dt.normalize()
    bars = bars[bars["date"].isin(dates)].copy()
    for col in ["open", "high", "low", "close"]:
        bars[col] = pd.to_numeric(bars.get(col), errors="coerce")
    result: dict[pd.Timestamp, dict[str, object]] = {}
    for date, group in bars.sort_values("datetime").groupby("date", sort=False):
        at_1455 = group[group["datetime"].dt.strftime("%H:%M").eq("14:55")]
        at_1500 = group[group["datetime"].dt.strftime("%H:%M").eq("15:00")]
        result[pd.Timestamp(date)] = {
            "source_5m_path": str(path),
            "source_5m_rows": int(len(group)),
            "source_first_time": group["datetime"].iloc[0].strftime("%H:%M"),
            "source_last_time": group["datetime"].iloc[-1].strftime("%H:%M"),
            "source_first_open": float(group["open"].iloc[0]) if pd.notna(group["open"].iloc[0]) else np.nan,
            "source_close_1455": float(at_1455["close"].iloc[-1]) if not at_1455.empty else np.nan,
            "source_close_1500": float(at_1500["close"].iloc[-1]) if not at_1500.empty else np.nan,
            "source_nonpositive_ohlc_rows": int((group[["open", "high", "low", "close"]].le(0).any(axis=1)).sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose as1455 adjustment-factor failures from existing local caches")
    parser.add_argument("--as1455-dir", default=str(DEFAULT_AS1455))
    parser.add_argument("--qfq-cache-dir", default=str(DEFAULT_QFQ))
    args = parser.parse_args()

    base = Path(args.as1455_dir)
    reports = base / "reports"
    daily_cache = base / "as1455_daily_cache"
    qfq_cache = Path(args.qfq_cache_dir)
    factor_report = pd.read_csv(reports / "as1455_adjust_factor_check.csv", dtype={"symbol": str})
    raw_obs = pd.to_numeric(factor_report.get("raw_obs"), errors="coerce")
    factor_obs = pd.to_numeric(factor_report.get("factor_obs"), errors="coerce")
    bad_symbols = factor_report.loc[raw_obs.gt(factor_obs), "symbol"].map(normalize_symbol).tolist()

    detail_rows: list[dict[str, object]] = []
    symbol_rows: list[dict[str, object]] = []
    for symbol in bad_symbols:
        daily_path = daily_cache / f"{symbol}_as1455_daily.csv"
        qfq_path = qfq_cache / f"{symbol}_qfq_daily.csv"
        if not daily_path.exists() or not qfq_path.exists():
            symbol_rows.append(
                {
                    "symbol": symbol,
                    "daily_cache_exists": daily_path.exists(),
                    "qfq_cache_exists": qfq_path.exists(),
                    "error": "required_file_missing",
                }
            )
            continue
        daily = pd.read_csv(daily_path, dtype={"symbol": str})
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        for col in ["raw_open_as1455", "raw_close_as1455", "raw_daily_close"]:
            if col in daily:
                daily[col] = pd.to_numeric(daily[col], errors="coerce")
        qfq = load_qfq(qfq_path)
        daily["qfq_daily_open"] = daily["date"].map(qfq["qfq_daily_open"])
        daily["qfq_daily_close"] = daily["date"].map(qfq["qfq_daily_close"])
        daily["open_factor"] = daily["qfq_daily_open"].div(daily["raw_open_as1455"])
        if "raw_daily_close" in daily:
            daily["old_close_factor"] = daily["qfq_daily_close"].div(daily["raw_daily_close"])
            bad_mask = ~np.isfinite(daily["old_close_factor"])
        else:
            daily["old_close_factor"] = np.nan
            bad_mask = ~np.isfinite(daily["open_factor"])
        bad = daily.loc[bad_mask].copy()
        source_path = resolve_source_path(daily["source_path"].dropna().iloc[0] if "source_path" in daily and daily["source_path"].notna().any() else "")
        source = inspect_5m(source_path, set(bad["date"].dropna()))
        for row in bad.itertuples(index=False):
            date = pd.Timestamp(row.date)
            item = {
                "symbol": symbol,
                "date": date.strftime("%Y-%m-%d"),
                "raw_open_as1455": getattr(row, "raw_open_as1455", np.nan),
                "raw_close_as1455": getattr(row, "raw_close_as1455", np.nan),
                "raw_daily_close": getattr(row, "raw_daily_close", np.nan),
                "qfq_daily_open": getattr(row, "qfq_daily_open", np.nan),
                "qfq_daily_close": getattr(row, "qfq_daily_close", np.nan),
                "open_factor": getattr(row, "open_factor", np.nan),
                "old_close_factor": getattr(row, "old_close_factor", np.nan),
            }
            item.update(source.get(date, {}))
            detail_rows.append(item)
        symbol_rows.append(
            {
                "symbol": symbol,
                "old_bad_rows": int(len(bad)),
                "new_open_factor_bad_rows": int((~np.isfinite(daily["open_factor"])).sum()),
                "source_5m_path": str(source_path or ""),
                "source_5m_exists": bool(source_path and source_path.exists()),
                "error": "",
            }
        )

    details = pd.DataFrame(detail_rows)
    symbols = pd.DataFrame(symbol_rows)
    details.to_csv(reports / "as1455_adjustment_failure_details.csv", index=False, encoding="utf-8-sig")
    symbols.to_csv(reports / "as1455_adjustment_failure_symbols.csv", index=False, encoding="utf-8-sig")
    new_bad = pd.to_numeric(symbols["new_open_factor_bad_rows"], errors="coerce").fillna(0) if "new_open_factor_bad_rows" in symbols else pd.Series(dtype=float)
    source_exists = symbols["source_5m_exists"].fillna(False).astype(bool) if "source_5m_exists" in symbols else pd.Series(dtype=bool)
    summary = {
        "bad_symbols_from_old_report": len(bad_symbols),
        "old_bad_rows_diagnosed": int(len(details)),
        "new_open_factor_bad_rows": int(new_bad.sum()),
        "source_5m_files_inspected": int(source_exists.sum()),
    }
    (reports / "as1455_adjustment_failure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
