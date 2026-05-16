#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def norm_symbol(s: str) -> str:
    s = str(s or "").strip().upper()
    if "." in s:
        code, mkt = s.split(".", 1)
        return f"{code.zfill(6)}.{mkt}"
    code = "".join(ch for ch in s if ch.isdigit()).zfill(6)
    if not code:
        return ""
    mkt = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{mkt}"


def safe_token(s: object) -> str:
    text = str(s or "").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def resolve_path(raw: object, stock_code: str = "") -> Optional[Path]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip().replace("\\", "/")
    if not text:
        return None
    p = Path(text)
    if p.exists():
        return p
    for marker in ["stock_realtime_v021_full/", "stock_realtime/"]:
        if marker in text:
            cand = ROOT / text.split(marker, 1)[1]
            if cand.exists():
                return cand
    if "saved_data/" in text:
        cand = ROOT / text[text.index("saved_data/") :]
        if cand.exists():
            return cand
    name = Path(text).name
    if name:
        code = norm_symbol(stock_code).split(".", 1)[0] if stock_code else ""
        roots = []
        if code:
            roots.extend(sorted((ROOT / "saved_data").glob(f"{code}_pipeline_out*")))
        roots.append(ROOT / "saved_data")
        for r in roots:
            if not r.exists():
                continue
            hits = list(r.rglob(name))
            if hits:
                return hits[0]
    return None


@dataclass
class Candidate:
    stock_code: str
    leaderboard: str
    label_mode: str
    entry_policy: str
    feature_group: str
    model_name: str
    target_hit_bps: float
    trades: int
    win_rate: float
    avg_return: float
    median_return: float
    max_drawdown: float
    profit_factor: float
    rank_score: float
    sample_file: str
    intraday_bars: str
    external: str
    artifact_name: str
    utility_score: float


def parse_float(x, default=float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def estimate_utility(row: pd.Series) -> float:
    avg = parse_float(row.get("avg_return"), 0.0)
    med = parse_float(row.get("median_return"), 0.0)
    wr = parse_float(row.get("win_rate"), 0.0)
    pf = parse_float(row.get("profit_factor"), 1.0)
    mdd = parse_float(row.get("max_drawdown"), -1.0)
    rank = parse_float(row.get("rank_score"), 0.0)
    trades = parse_int(row.get("trades"), 0)
    score = 5.0 * avg + 2.0 * med
    score += 0.04 * max(pf - 1.0, -1.0)
    score += 0.10 * max(wr - 0.50, -0.50)
    score += 0.80 * rank
    score += 0.02 * min(max(trades, 0), 250) / 250.0
    score += 0.08 * mdd
    return float(score)


def make_artifact(row: pd.Series, suffix: str) -> str:
    parts = [
        "nextday",
        safe_token(row.get("entry_policy")),
        safe_token(row.get("label_mode")),
        safe_token(row.get("model_name")),
        safe_token(row.get("feature_group")),
    ]
    ext = safe_token(row.get("external"))
    if ext not in {"", "na", "none", "nan"}:
        parts.append(ext)
    parts.append(safe_token(suffix))
    return "_".join(parts)


def load_candidates(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, rejected = [], []
    for lb in sorted(Path(args.saved_data_dir).glob("*_pipeline_out*/99_summary/final_leaderboard.csv")):
        try:
            df = pd.read_csv(lb)
        except Exception as exc:
            rejected.append({"leaderboard": str(lb), "reason": f"read_error:{type(exc).__name__}:{exc}"})
            continue
        if df.empty:
            rejected.append({"leaderboard": str(lb), "reason": "empty_leaderboard"})
            continue
        df["__leaderboard"] = str(lb)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if all_df.empty:
        return all_df, pd.DataFrame(rejected)

    only = set()
    if args.only:
        for x in str(args.only).replace(";", ",").split(","):
            y = norm_symbol(x)
            if y:
                only.add(y)
    all_df["stock_code"] = all_df["stock_code"].map(norm_symbol)
    if only:
        all_df = all_df[all_df["stock_code"].isin(only)].copy()

    candidates, rejs = [], []
    for _, r in all_df.iterrows():
        stock = norm_symbol(r.get("stock_code", ""))
        trades = parse_int(r.get("trades"), 0)
        avg = parse_float(r.get("avg_return"), float("nan"))
        mdd = parse_float(r.get("max_drawdown"), float("nan"))
        pf = parse_float(r.get("profit_factor"), float("nan"))
        rank = parse_float(r.get("rank_score"), float("nan"))
        reasons = []
        if not stock:
            reasons.append("missing_stock_code")
        if trades < args.min_trades:
            reasons.append(f"trades_lt_{args.min_trades}")
        if not math.isfinite(avg) or avg < args.min_avg_return:
            reasons.append(f"avg_return_lt_{args.min_avg_return}")
        if not math.isfinite(pf) or pf < args.min_profit_factor:
            reasons.append(f"profit_factor_lt_{args.min_profit_factor}")
        if math.isfinite(rank) and rank < args.min_rank_score:
            reasons.append(f"rank_score_lt_{args.min_rank_score}")
        if not math.isfinite(mdd) or mdd < args.max_drawdown_floor:
            reasons.append(f"max_drawdown_lt_{args.max_drawdown_floor}")
        sample = resolve_path(r.get("sample_file"), stock)
        intra = resolve_path(r.get("intraday_bars"), stock)
        if sample is None or not sample.exists():
            reasons.append("sample_file_missing")
        if intra is None or not intra.exists():
            reasons.append("intraday_bars_missing")
        if reasons:
            rr = r.to_dict()
            rr["reject_reason"] = ";".join(reasons)
            rejs.append(rr)
            continue
        candidates.append(Candidate(
            stock_code=stock,
            leaderboard=str(r.get("__leaderboard")),
            label_mode=str(r.get("label_mode")),
            entry_policy=str(r.get("entry_policy")),
            feature_group=str(r.get("feature_group")),
            model_name=str(r.get("model_name")),
            target_hit_bps=parse_float(r.get("target_hit_bps"), 50.0),
            trades=trades,
            win_rate=parse_float(r.get("win_rate"), float("nan")),
            avg_return=avg,
            median_return=parse_float(r.get("median_return"), float("nan")),
            max_drawdown=mdd,
            profit_factor=pf,
            rank_score=rank,
            sample_file=str(sample),
            intraday_bars=str(intra),
            external=str(r.get("external", "")),
            artifact_name=make_artifact(r, args.artifact_suffix),
            utility_score=estimate_utility(r),
        ))
    cand_df = pd.DataFrame([asdict(x) for x in candidates])
    rej_df = pd.DataFrame(rejs)
    if cand_df.empty:
        return cand_df, rej_df
    cand_df = cand_df.sort_values(["stock_code", "utility_score", "rank_score"], ascending=[True, False, False])
    selected = []
    for _, g in cand_df.groupby("stock_code", sort=True):
        selected.append(g.head(args.max_per_stock))
    selected_df = pd.concat(selected, ignore_index=True)
    selected_df = selected_df.sort_values(["utility_score", "rank_score"], ascending=[False, False]).reset_index(drop=True)
    return selected_df, rej_df


def save_candidate(row: pd.Series, args) -> dict:
    artifact_dir = Path(args.models_dir) / row["stock_code"] / row["artifact_name"]
    report = {
        "stock_code": row["stock_code"],
        "artifact_name": row["artifact_name"],
        "artifact_dir": str(artifact_dir),
        "sample_file": row["sample_file"],
        "intraday_bars": row["intraday_bars"],
        "status": "",
        "returncode": "",
    }
    if artifact_dir.exists() and not args.overwrite_existing:
        report["status"] = "skipped_existing_artifact"
        return report
    if artifact_dir.exists() and args.overwrite_existing:
        backup_dir = Path(args.out_dir) / "existing_artifact_backups" / row["stock_code"] / row["artifact_name"]
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir.rename(backup_dir)
        report["backup_dir"] = str(backup_dir)

    cmd = [
        sys.executable, "model_saving/save_nextday_model.py",
        "--stock-code", row["stock_code"],
        "--artifact-name", row["artifact_name"],
        "--samples", row["sample_file"],
        "--intraday-bars", row["intraday_bars"],
        "--out-dir", args.models_dir,
        "--feature-group", row["feature_group"],
        "--model-name", row["model_name"],
        "--label-mode", row["label_mode"],
        "--entry-policy", row["entry_policy"],
        "--target-hit-bps", str(row.get("target_hit_bps", 50.0)),
        "--entry-vwap-premium-bps", "50",
        "--round-trip-cost-bps", "1.7",
        "--valid-rows", "252",
        "--min-train-entries", "80",
        "--min-valid-trades", "8",
        "--quantiles", "0.5,0.6,0.7,0.8",
    ]
    report["cmd"] = " ".join(cmd)
    if args.dry_run:
        report["status"] = "dry_run"
        return report
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report["returncode"] = proc.returncode
    log_path = Path(args.out_dir) / f"save_{row['stock_code'].replace('.', '_')}_{row['artifact_name']}.log"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    report["log_path"] = str(log_path)
    report["status"] = "ok" if proc.returncode == 0 else "failed"
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-data-dir", default="saved_data")
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-suffix", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-per-stock", type=int, default=2)
    ap.add_argument("--min-rank-score", type=float, default=0.0)
    ap.add_argument("--min-trades", type=int, default=80)
    ap.add_argument("--min-avg-return", type=float, default=0.002)
    ap.add_argument("--min-profit-factor", type=float, default=1.35)
    ap.add_argument("--max-drawdown-floor", type=float, default=-0.35)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite-existing", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, rejected = load_candidates(args)
    selected_path = out_dir / "auto_model_selected.csv"
    rejected_path = out_dir / "auto_model_rejected.csv"
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    rejected.to_csv(rejected_path, index=False, encoding="utf-8-sig")
    reports = []
    if not selected.empty:
        for _, row in selected.iterrows():
            reports.append(save_candidate(row, args))
    report_df = pd.DataFrame(reports)
    report_path = out_dir / "auto_model_save_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    summary = {
        "selected_rows": int(len(selected)),
        "rejected_rows": int(len(rejected)),
        "save_rows": int(len(report_df)),
        "status_counts": report_df["status"].value_counts(dropna=False).to_dict() if not report_df.empty else {},
        "selected_csv": str(selected_path),
        "rejected_csv": str(rejected_path),
        "save_report_csv": str(report_path),
    }
    (out_dir / "auto_model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not (not report_df.empty and (report_df["status"] == "failed").any()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
