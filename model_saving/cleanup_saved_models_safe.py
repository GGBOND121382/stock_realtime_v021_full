#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, json, math, re, shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

def to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return float(x)
    except Exception:
        return default

def to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return int(float(x))
    except Exception:
        return default

@dataclass
class Row:
    stock_code: str
    artifact: str
    action: str
    reason: str
    artifact_dir: str
    moved_to: str = ""
    trades: int | str = ""
    win_rate: float | str = ""
    avg_return: float | str = ""
    median_return: float | str = ""
    max_drawdown: float | str = ""
    profit_factor: float | str = ""
    label_mode: str = ""

def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    for i in range(1, 10000):
        cand = Path(f"{dest}_{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(dest)

def evaluate(meta: dict, args):
    m = meta.get("validation_tail_trade_metrics", {}) or {}
    label = str(meta.get("label_mode") or "")
    trades = to_int(m.get("trades"), 0)
    wr = to_float(m.get("win_rate"))
    avg = to_float(m.get("avg_return"))
    med = to_float(m.get("median_return"))
    mdd = to_float(m.get("max_drawdown"))
    pf = to_float(m.get("profit_factor"))
    reasons = []
    if trades < args.min_trades_hard:
        reasons.append("too_few_trades")
    if label == "close_profit":
        if math.isfinite(avg) and math.isfinite(med) and avg < args.bad_avg_return and med < args.bad_median_return:
            reasons.append("weak_returns")
        if math.isfinite(pf) and pf < args.bad_profit_factor and math.isfinite(avg) and avg < args.min_avg_return_soft:
            reasons.append("weak_pf")
    if label == "hit":
        target = to_float(meta.get("target_hit_bps"), 50)
        min_wr = args.bad_hit_win_rate_80 if target >= 80 else args.bad_hit_win_rate_50
        if math.isfinite(wr) and wr < min_wr:
            reasons.append("weak_hit_rate")
    if math.isfinite(mdd) and mdd < args.bad_max_drawdown and ((not math.isfinite(pf)) or pf < args.drawdown_profit_factor_guard):
        reasons.append("drawdown_weak_pf")
    return ("move_candidate", ";".join(reasons)) if reasons else ("keep", "passes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--trash-root", default="cleanup_trash")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-trades-hard", type=int, default=30)
    ap.add_argument("--bad-avg-return", type=float, default=0.0)
    ap.add_argument("--bad-median-return", type=float, default=0.0)
    ap.add_argument("--bad-profit-factor", type=float, default=1.10)
    ap.add_argument("--min-avg-return-soft", type=float, default=0.001)
    ap.add_argument("--bad-hit-win-rate-50", type=float, default=0.55)
    ap.add_argument("--bad-hit-win-rate-80", type=float, default=0.62)
    ap.add_argument("--bad-max-drawdown", type=float, default=-0.45)
    ap.add_argument("--drawdown-profit-factor-guard", type=float, default=1.30)
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"saved_data/model_cleanup_logs/cleanup_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trash_base = Path(args.trash_root) / f"saved_models_cleanup_{ts}"
    only = {x.strip().upper() for x in args.only.replace(";", ",").split(",") if x.strip()}
    rows = []
    for artifact_dir in sorted(p for p in Path(args.models_dir).glob("*/*") if p.is_dir()):
        stock, artifact = artifact_dir.parent.name, artifact_dir.name
        if only and stock.upper() not in only and stock.split(".", 1)[0].upper() not in only:
            continue
        meta_path = artifact_dir / "metadata.json"
        if not meta_path.exists():
            row = Row(stock, artifact, "manual_review", "metadata_missing", str(artifact_dir))
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                action, reason = evaluate(meta, args)
                m = meta.get("validation_tail_trade_metrics", {}) or {}
                row = Row(stock, artifact, action, reason, str(artifact_dir),
                          trades=to_int(m.get("trades"), ""),
                          win_rate=to_float(m.get("win_rate"), ""),
                          avg_return=to_float(m.get("avg_return"), ""),
                          median_return=to_float(m.get("median_return"), ""),
                          max_drawdown=to_float(m.get("max_drawdown"), ""),
                          profit_factor=to_float(m.get("profit_factor"), ""),
                          label_mode=str(meta.get("label_mode") or ""))
            except Exception as exc:
                row = Row(stock, artifact, "manual_review", f"metadata_error:{type(exc).__name__}:{exc}", str(artifact_dir))
        if args.apply and row.action == "move_candidate":
            dest = unique_dest(trash_base / stock / artifact)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(row.artifact_dir, dest)
            row.action = "moved"
            row.moved_to = str(dest)
        rows.append(row)
    report = out_dir / "model_cleanup_report.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(Row("", "", "", "", "")).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    summary = {"apply": args.apply, "report_csv": str(report), "counts": {}}
    for r in rows:
        summary["counts"][r.action] = summary["counts"].get(r.action, 0) + 1
    (out_dir / "model_cleanup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
