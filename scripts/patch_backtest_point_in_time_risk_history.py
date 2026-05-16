#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path


TARGET = Path("portfolio_decision/backtest_historical_score_portfolio.py")


HELPER = r'''
def write_point_in_time_risk_history(
    history: pd.DataFrame,
    date: pd.Timestamp,
    out_dir: Path,
    include_current_day: bool = True,
    min_rows: int = 20,
) -> Optional[Path]:
    # Write the risk-history CSV visible to the optimizer on one backtest date.
    #
    # The optimizer is executed as a subprocess, so its covariance / correlation /
    # scenario-risk logic needs a CSV input. This function writes a point-in-time
    # wide close-price table clipped to the current decision date. It is an
    # intermediate risk input only, not a long-lived data source.
    if history is None or history.empty:
        return None

    cutoff = pd.Timestamp(date).normalize()
    hist = history.copy()
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    hist = hist[hist.index.notna()].sort_index()

    if include_current_day:
        hist = hist.loc[hist.index <= cutoff]
    else:
        hist = hist.loc[hist.index < cutoff]

    hist = hist.dropna(axis=1, how="all")
    if len(hist) < int(min_rows) or hist.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"risk_history_until_{cutoff.strftime('%Y%m%d')}.csv"

    export = hist.reset_index()
    first_col = export.columns[0]
    if first_col != "date":
        export = export.rename(columns={first_col: "date"})
    export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export.to_csv(path, index=False, encoding="utf-8-sig")
    return path

'''


def insert_helper(txt: str) -> str:
    if "def write_point_in_time_risk_history(" in txt:
        return txt

    marker = "def run_portfolio_adapter("
    if marker in txt:
        return txt.replace(marker, HELPER + "\n" + marker, 1)

    marker = "def simulate_portfolio("
    if marker in txt:
        return txt.replace(marker, HELPER + "\n" + marker, 1)

    raise SystemExit("[ERROR] cannot find run_portfolio_adapter or simulate_portfolio marker")


def patch_run_adapter_call(txt: str) -> str:
    if "risk_history_path = write_point_in_time_risk_history(history, date, day_out)" not in txt:
        anchor = (
            '        account_path = day_out / f"sim_account_{ymd}.json"\n'
            '        account_path.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")\n'
        )
        insert = anchor + "\n        risk_history_path = write_point_in_time_risk_history(history, date, day_out)\n"
        if anchor not in txt:
            raise SystemExit("[ERROR] cannot find account_path write block in simulate_portfolio")
        txt = txt.replace(anchor, insert, 1)

    pattern = re.compile(
        r"(orders_path\s*=\s*run_portfolio_adapter\(\s*.*?history_path\s*=\s*)history_path(,)",
        flags=re.DOTALL,
    )
    m = pattern.search(txt)
    if not m:
        if "history_path=risk_history_path" in txt or "history_path = risk_history_path" in txt:
            return txt
        raise SystemExit("[ERROR] cannot find run_portfolio_adapter history_path argument")
    txt = pattern.sub(r"\1risk_history_path\2", txt, count=1)
    return txt


def patch_summary(txt: str) -> str:
    if '"risk_history_mode": "point_in_time_daily_csv"' in txt:
        return txt

    old = (
        '        "history_source": history_source,\n'
        '        "signal_root": str(signal_root),\n'
    )
    new = (
        '        "history_source": history_source,\n'
        '        "risk_history_mode": "point_in_time_daily_csv",\n'
        '        "risk_history_include_current_day": True,\n'
        '        "signal_root": str(signal_root),\n'
    )
    if old in txt:
        return txt.replace(old, new, 1)

    old = (
        '        "source_mode": "historical_samples_generated_scores",\n'
        '        "model_policy": args.model_policy,\n'
    )
    new = (
        '        "source_mode": "historical_samples_generated_scores",\n'
        '        "risk_history_mode": "point_in_time_daily_csv",\n'
        '        "risk_history_include_current_day": True,\n'
        '        "model_policy": args.model_policy,\n'
    )
    if old in txt:
        return txt.replace(old, new, 1)

    print("[WARN] cannot add risk_history_mode to summary; continuing")
    return txt


def main() -> int:
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] missing {TARGET}")

    txt = TARGET.read_text(encoding="utf-8")
    txt = insert_helper(txt)
    txt = patch_run_adapter_call(txt)
    txt = patch_summary(txt)
    TARGET.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
