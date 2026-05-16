#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-select valuable next-day models from pipeline leaderboards and save artifacts.

This script is intentionally conservative.  It scans historical pipeline outputs,
normalizes validation metrics, rejects weak/high-risk rows, deduplicates by stock,
and invokes model_saving/save_nextday_model.py for selected rows.

It is designed to be called by scripts/update_ranked_models_latest.sh, but can be
run directly from the project root, e.g.

    python3 model_saving/auto_update_ranked_models.py \
      --saved-data-dir saved_data \
      --models-dir saved_models \
      --report-dir saved_data/model_search_queue_logs/manual_auto_select

Outputs:
  auto_model_candidates.csv   all parsed + scored leaderboard rows
  auto_model_selected.csv     rows selected for saving
  auto_model_save_report.csv  save command status and metadata metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@dataclass
class Thresholds:
    min_close_trades: int
    min_close_win_rate: float
    min_close_avg_bps: float
    min_close_median_bps: float
    min_close_profit_factor: float
    max_close_drawdown: float
    min_close_value_score: float
    min_hit_trades: int
    min_hit_win_rate: float
    min_hit_profit_factor: float
    max_hit_drawdown: float
    min_hit_value_score: float
    cost_bps: float


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def norm_stock_code(x: str) -> str:
    text = str(x or "").strip().upper().replace("_", ".")
    if not text:
        return text
    if "." in text:
        code, market = text.split(".", 1)
        return f"{code.zfill(6)}.{market}"
    code = "".join(ch for ch in text if ch.isdigit()).zfill(6)
    if code.startswith(("5", "6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def raw_code(stock_code: str) -> str:
    return norm_stock_code(stock_code).split(".", 1)[0]


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip().replace("%", "")
            if x in {"", "nan", "NaN", "None", "-", "--"}:
                return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    v = safe_float(x, float("nan"))
    return int(v) if math.isfinite(v) else default


def bps_from_decimal(x: Any) -> float:
    return safe_float(x) * 10000.0


def sanitize_token(value: str, max_len: int = 80) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_") or "na"


def model_short_name(model_name: str) -> str:
    name = str(model_name or "")
    replacements = {
        "xgb_d2_200_lr003_mcw5": "xgb_d2_200",
        "xgb_d3_400_lr003_mcw3": "xgb_d3_400",
        "xgb_d3_600_lr002_mcw3": "xgb_d3_600",
        "xgb_d4_500_lr002_mcw5": "xgb_d4_500",
        "lgbm_leaves7_400": "lgbm_l7_400",
        "lgbm_leaves15_700": "lgbm_l15_700",
        "extra_trees_600_d3": "extra_trees_d3",
        "random_forest_600_d4": "random_forest_d4",
    }
    return replacements.get(name, sanitize_token(name))


def artifact_name_for_row(row: pd.Series, version_suffix: str) -> str:
    entry_policy = sanitize_token(row.get("entry_policy", "all_days"))
    label_mode = sanitize_token(row.get("label_mode", "close_profit"))
    model = model_short_name(str(row.get("model_name", "model")))
    feature_group = sanitize_token(str(row.get("feature_group", "features")), max_len=70)
    external = sanitize_token(str(row.get("external", "") or ""), max_len=40)
    target_hit_bps = safe_int(row.get("target_hit_bps"), 50)

    label_part = label_mode
    if label_mode == "hit":
        label_part = f"hit{target_hit_bps}"

    if external and external not in {"nan", "none", "na"}:
        return f"nextday_{entry_policy}_{label_part}_{model}_{feature_group}_{external}_{version_suffix}"
    return f"nextday_{entry_policy}_{label_part}_{model}_{feature_group}_{version_suffix}"


def resolve_repo_path(raw: Any, project_dir: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path("")
    p = Path(text)
    if p.exists():
        return p
    if p.is_absolute() and p.exists():
        return p
    norm = text.replace("\\", "/")
    marker = "/saved_data/"
    if marker in norm:
        rel = "saved_data/" + norm.split(marker, 1)[1]
        cand = project_dir / rel
        if cand.exists():
            return cand
        return cand
    if norm.startswith("saved_data/") or norm.startswith("saved_models/"):
        return project_dir / norm
    return project_dir / norm


def find_leaderboards(saved_data_dir: Path) -> list[Path]:
    patterns = [
        "*_pipeline_out/99_summary/final_leaderboard.csv",
        "*_pipeline_out_*/99_summary/final_leaderboard.csv",
    ]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(saved_data_dir.glob(pat))
    return sorted(set(paths), key=lambda p: (p.parent.parent.name, p.stat().st_mtime if p.exists() else 0), reverse=True)


def load_all_leaderboards(saved_data_dir: Path, project_dir: Path) -> pd.DataFrame:
    frames = []
    for path in find_leaderboards(saved_data_dir):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[WARN] failed to read {path}: {type(exc).__name__}: {exc}")
            continue
        if df.empty:
            continue
        df = df.copy()
        df["leaderboard_path"] = str(path)
        df["leaderboard_mtime"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        df["pipeline_out_dir"] = str(path.parent.parent)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "stock_code" not in out.columns and "raw_code" in out.columns:
        out["stock_code"] = out["raw_code"].map(norm_stock_code)
    out["stock_code"] = out["stock_code"].map(norm_stock_code)
    out["sample_file_resolved"] = out.get("sample_file", "").map(lambda x: str(resolve_repo_path(x, project_dir)))
    out["intraday_bars_resolved"] = out.get("intraday_bars", "").map(lambda x: str(resolve_repo_path(x, project_dir)))
    return out


def load_saved_model_index(models_dir: Path) -> pd.DataFrame:
    rows = []
    for meta in models_dir.glob("*/*/metadata.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = data.get("validation_tail_trade_metrics") or {}
        rows.append({
            "stock_code": norm_stock_code(data.get("stock_code") or meta.parent.parent.name),
            "artifact_name": data.get("artifact_name") or meta.parent.name,
            "artifact_dir": str(meta.parent),
            "feature_group": data.get("feature_group", ""),
            "model_name": data.get("model_name", ""),
            "label_mode": data.get("label_mode", ""),
            "entry_policy": data.get("entry_policy", ""),
            "target_hit_bps": data.get("target_hit_bps", ""),
            "date_max": data.get("date_max", ""),
            "trades": metrics.get("trades", ""),
            "win_rate": metrics.get("win_rate", ""),
            "avg_return": metrics.get("avg_return", ""),
            "median_return": metrics.get("median_return", ""),
            "max_drawdown": metrics.get("max_drawdown", ""),
            "profit_factor": metrics.get("profit_factor", ""),
        })
    return pd.DataFrame(rows)


def compute_scores(df: pd.DataFrame, thresholds: Thresholds, include_high_drawdown: bool, allow_negative_rank_score: bool) -> pd.DataFrame:
    out = df.copy()
    for col in ["trades", "win_rate", "avg_return", "median_return", "max_drawdown", "profit_factor", "rank_score", "target_hit_bps"]:
        if col not in out.columns:
            out[col] = float("nan")
    out["trades_i"] = out["trades"].map(safe_int)
    out["win_rate_f"] = out["win_rate"].map(safe_float)
    out["avg_return_bps"] = out["avg_return"].map(bps_from_decimal)
    out["median_return_bps"] = out["median_return"].map(bps_from_decimal)
    out["max_drawdown_f"] = out["max_drawdown"].map(safe_float)
    out["profit_factor_f"] = out["profit_factor"].map(safe_float)
    out["rank_score_f"] = out["rank_score"].map(safe_float)
    out["target_hit_bps_f"] = out["target_hit_bps"].map(lambda x: safe_float(x, 50.0))
    out["label_mode"] = out["label_mode"].astype(str)
    out["feature_group"] = out.get("feature_group", "").astype(str)

    dd = out["max_drawdown_f"].fillna(-1.0)
    dd_penalty = (-dd.clip(upper=0.0) * 10000.0 * 0.18).clip(lower=0)
    median_bonus = out["median_return_bps"].fillna(0.0).clip(lower=-80, upper=120) * 0.18
    pf_bonus = (out["profit_factor_f"].fillna(1.0) - 1.0).clip(lower=-1, upper=3) * 18.0
    win_bonus = (out["win_rate_f"].fillna(0.5) - 0.5) * 80.0
    rank_bonus = out["rank_score_f"].fillna(0.0) * 180.0
    external_penalty = out["feature_group"].str.contains("external", na=False).astype(float) * 4.0

    out["close_value_score"] = (
        out["avg_return_bps"].fillna(-999.0)
        - thresholds.cost_bps
        - dd_penalty
        + median_bonus
        + pf_bonus
        + win_bonus
        + rank_bonus
        - external_penalty
    )
    # Hit model estimated value: target reward versus a conservative miss loss.
    miss_loss_bps = 60.0
    out["hit_value_score"] = (
        out["win_rate_f"].fillna(0.0) * out["target_hit_bps_f"].fillna(80.0)
        - (1.0 - out["win_rate_f"].fillna(0.0)) * miss_loss_bps
        - thresholds.cost_bps
        - dd_penalty
        + pf_bonus
        + rank_bonus
    )

    reasons = []
    eligible = []
    for _, r in out.iterrows():
        label = str(r.get("label_mode", ""))
        rs: list[str] = []
        ok = True
        if not Path(str(r.get("sample_file_resolved", ""))).exists():
            ok = False; rs.append("missing_sample_file")
        if not Path(str(r.get("intraday_bars_resolved", ""))).exists():
            ok = False; rs.append("missing_intraday_bars")
        if not allow_negative_rank_score and safe_float(r.get("rank_score_f"), 0.0) < 0:
            ok = False; rs.append("negative_rank_score")
        if label == "close_profit":
            if safe_int(r.get("trades_i")) < thresholds.min_close_trades:
                ok = False; rs.append("close_trades_low")
            if safe_float(r.get("win_rate_f")) < thresholds.min_close_win_rate:
                ok = False; rs.append("close_win_rate_low")
            if safe_float(r.get("avg_return_bps")) < thresholds.min_close_avg_bps:
                ok = False; rs.append("close_avg_bps_low")
            if safe_float(r.get("median_return_bps")) < thresholds.min_close_median_bps:
                ok = False; rs.append("close_median_bps_low")
            if safe_float(r.get("profit_factor_f")) < thresholds.min_close_profit_factor:
                ok = False; rs.append("close_pf_low")
            if (not include_high_drawdown) and safe_float(r.get("max_drawdown_f"), -1.0) < thresholds.max_close_drawdown:
                ok = False; rs.append("close_drawdown_too_large")
            if safe_float(r.get("close_value_score")) < thresholds.min_close_value_score:
                ok = False; rs.append("close_value_score_low")
        elif label == "hit":
            if safe_int(r.get("trades_i")) < thresholds.min_hit_trades:
                ok = False; rs.append("hit_trades_low")
            if safe_float(r.get("win_rate_f")) < thresholds.min_hit_win_rate:
                ok = False; rs.append("hit_win_rate_low")
            if safe_float(r.get("profit_factor_f")) < thresholds.min_hit_profit_factor:
                ok = False; rs.append("hit_pf_low")
            if (not include_high_drawdown) and safe_float(r.get("max_drawdown_f"), -1.0) < thresholds.max_hit_drawdown:
                ok = False; rs.append("hit_drawdown_too_large")
            if safe_float(r.get("hit_value_score")) < thresholds.min_hit_value_score:
                ok = False; rs.append("hit_value_score_low")
        else:
            ok = False; rs.append("unknown_label_mode")
        eligible.append(ok)
        reasons.append(";".join(rs))
    out["eligible"] = eligible
    out["reject_reasons"] = reasons
    out["value_score"] = out.apply(lambda r: r["close_value_score"] if str(r.get("label_mode")) == "close_profit" else r["hit_value_score"], axis=1)
    return out


def select_models(scored: pd.DataFrame, max_total: int, max_per_stock: int, include_hit_aux: bool) -> pd.DataFrame:
    if scored.empty:
        return scored
    cand = scored[scored["eligible"]].copy()
    if not include_hit_aux:
        cand = cand[cand["label_mode"].astype(str).eq("close_profit")].copy()
    if cand.empty:
        return cand

    # Avoid selecting duplicate rows from both canonical and old tagged pipeline dirs.
    dedupe_keys = [
        "stock_code", "label_mode", "entry_policy", "feature_group", "model_name",
        "target_hit_bps", "external",
    ]
    for k in dedupe_keys:
        if k not in cand.columns:
            cand[k] = ""
    cand = cand.sort_values(["value_score", "rank_score_f", "avg_return_bps"], ascending=False)
    cand = cand.drop_duplicates(dedupe_keys, keep="first")

    selected_rows = []
    per_stock: dict[str, int] = {}
    label_per_stock: set[tuple[str, str]] = set()
    for _, r in cand.iterrows():
        stock = str(r["stock_code"])
        label = str(r["label_mode"])
        if per_stock.get(stock, 0) >= max_per_stock:
            continue
        # Normally keep at most one close_profit and one hit per stock.
        if (stock, label) in label_per_stock:
            continue
        selected_rows.append(r)
        per_stock[stock] = per_stock.get(stock, 0) + 1
        label_per_stock.add((stock, label))
        if len(selected_rows) >= max_total:
            break
    return pd.DataFrame(selected_rows)


def build_save_command(row: pd.Series, args: argparse.Namespace, artifact_name: str) -> list[str]:
    return [
        args.python,
        "model_saving/save_nextday_model.py",
        "--stock-code", str(row["stock_code"]),
        "--artifact-name", artifact_name,
        "--samples", str(row["sample_file_resolved"]),
        "--intraday-bars", str(row["intraday_bars_resolved"]),
        "--out-dir", str(args.models_dir),
        "--feature-group", str(row["feature_group"]),
        "--model-name", str(row["model_name"]),
        "--label-mode", str(row["label_mode"]),
        "--entry-policy", str(row["entry_policy"]),
        "--target-hit-bps", str(safe_float(row.get("target_hit_bps"), 50.0)),
        "--entry-vwap-premium-bps", str(args.entry_vwap_premium_bps),
        "--round-trip-cost-bps", str(args.round_trip_cost_bps),
        "--valid-rows", str(args.valid_rows),
        "--min-train-entries", str(args.min_train_entries),
        "--min-valid-trades", str(args.min_valid_trades),
        "--quantiles", str(args.quantiles),
    ]


def read_saved_metadata(models_dir: Path, stock: str, artifact_name: str) -> dict[str, Any]:
    p = models_dir / stock / artifact_name / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_selected_models(selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        artifact_name = str(row["artifact_name_auto"])
        stock = str(row["stock_code"])
        artifact_dir = Path(args.models_dir) / stock / artifact_name
        cmd = build_save_command(row, args, artifact_name)
        started_at = now_ts()
        status = "dry_run" if args.dry_run or args.skip_save else "unknown"
        rc = 0
        error = ""
        if args.dry_run or args.skip_save:
            print("[DRY/SKIP SAVE] " + " ".join(cmd))
        else:
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            print("[SAVE] " + " ".join(cmd))
            proc = subprocess.run(cmd, cwd=str(PROJECT_DIR), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            rc = proc.returncode
            log_path = Path(args.report_dir) / f"save_{stock.replace('.', '_')}_{artifact_name}.log"
            log_path.write_text(proc.stdout or "", encoding="utf-8")
            if rc == 0:
                status = "ok"
            else:
                status = "failed"
                error = f"returncode={rc}; log={log_path}"
        finished_at = now_ts()
        meta = read_saved_metadata(Path(args.models_dir), stock, artifact_name) if status == "ok" else {}
        metrics = meta.get("validation_tail_trade_metrics") or {}
        rows.append({
            "stock_code": stock,
            "artifact_name": artifact_name,
            "status": status,
            "returncode": rc,
            "error": error,
            "started_at": started_at,
            "finished_at": finished_at,
            "source_leaderboard": row.get("leaderboard_path", ""),
            "samples": row.get("sample_file_resolved", ""),
            "intraday_bars": row.get("intraday_bars_resolved", ""),
            "feature_group": row.get("feature_group", ""),
            "model_name": row.get("model_name", ""),
            "label_mode": row.get("label_mode", ""),
            "entry_policy": row.get("entry_policy", ""),
            "selected_value_score": row.get("value_score", ""),
            "metadata_date_max": meta.get("date_max", ""),
            "metadata_trades": metrics.get("trades", ""),
            "metadata_win_rate": metrics.get("win_rate", ""),
            "metadata_avg_return": metrics.get("avg_return", ""),
            "metadata_median_return": metrics.get("median_return", ""),
            "metadata_max_drawdown": metrics.get("max_drawdown", ""),
            "metadata_profit_factor": metrics.get("profit_factor", ""),
            "artifact_dir": str(artifact_dir),
        })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] {path} rows={len(df)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto select and save valuable next-day model artifacts.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--saved-data-dir", default="saved_data")
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--only", default="", help="Comma-separated symbols to include, e.g. 603308.SH,600522.SH")
    ap.add_argument("--max-total", type=int, default=12)
    ap.add_argument("--max-per-stock", type=int, default=2)
    ap.add_argument("--version-suffix", default="auto_v1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-save", action="store_true")
    ap.add_argument("--include-hit-aux", action="store_true")
    ap.add_argument("--include-high-drawdown", action="store_true")
    ap.add_argument("--allow-negative-rank-score", action="store_true")

    ap.add_argument("--cost-bps", type=float, default=23.7)
    ap.add_argument("--min-close-trades", type=int, default=40)
    ap.add_argument("--min-close-win-rate", type=float, default=0.52)
    ap.add_argument("--min-close-avg-bps", type=float, default=30.0)
    ap.add_argument("--min-close-median-bps", type=float, default=10.0)
    ap.add_argument("--min-close-profit-factor", type=float, default=1.45)
    ap.add_argument("--max-close-drawdown", type=float, default=-0.20)
    ap.add_argument("--min-close-value-score", type=float, default=0.0)

    ap.add_argument("--min-hit-trades", type=int, default=40)
    ap.add_argument("--min-hit-win-rate", type=float, default=0.68)
    ap.add_argument("--min-hit-profit-factor", type=float, default=1.20)
    ap.add_argument("--max-hit-drawdown", type=float, default=-0.16)
    ap.add_argument("--min-hit-value-score", type=float, default=0.0)

    ap.add_argument("--entry-vwap-premium-bps", type=float, default=50.0)
    ap.add_argument("--round-trip-cost-bps", type=float, default=1.7)
    ap.add_argument("--valid-rows", type=int, default=252)
    ap.add_argument("--min-train-entries", type=int, default=80)
    ap.add_argument("--min-valid-trades", type=int, default=8)
    ap.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")

    args = ap.parse_args()
    saved_data_dir = Path(args.saved_data_dir)
    models_dir = Path(args.models_dir)
    if not saved_data_dir.is_absolute():
        saved_data_dir = PROJECT_DIR / saved_data_dir
    if not models_dir.is_absolute():
        models_dir = PROJECT_DIR / models_dir
    args.saved_data_dir = str(saved_data_dir)
    args.models_dir = str(models_dir)

    report_dir = Path(args.report_dir) if args.report_dir else saved_data_dir / "model_search_queue_logs" / f"auto_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not report_dir.is_absolute():
        report_dir = PROJECT_DIR / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir = str(report_dir)

    thresholds = Thresholds(
        min_close_trades=args.min_close_trades,
        min_close_win_rate=args.min_close_win_rate,
        min_close_avg_bps=args.min_close_avg_bps,
        min_close_median_bps=args.min_close_median_bps,
        min_close_profit_factor=args.min_close_profit_factor,
        max_close_drawdown=args.max_close_drawdown,
        min_close_value_score=args.min_close_value_score,
        min_hit_trades=args.min_hit_trades,
        min_hit_win_rate=args.min_hit_win_rate,
        min_hit_profit_factor=args.min_hit_profit_factor,
        max_hit_drawdown=args.max_hit_drawdown,
        min_hit_value_score=args.min_hit_value_score,
        cost_bps=args.cost_bps,
    )

    leaderboards = load_all_leaderboards(saved_data_dir, PROJECT_DIR)
    if leaderboards.empty:
        print(f"[ERROR] no final_leaderboard.csv found under {saved_data_dir}")
        return 2

    only_set = {norm_stock_code(x) for x in split_csv(args.only)}
    if only_set:
        leaderboards = leaderboards[leaderboards["stock_code"].isin(only_set)].copy()

    scored = compute_scores(
        leaderboards,
        thresholds,
        include_high_drawdown=args.include_high_drawdown,
        allow_negative_rank_score=args.allow_negative_rank_score,
    )

    # Make output stable and useful for inspection.
    sort_cols = ["eligible", "value_score", "rank_score_f", "avg_return_bps"]
    scored = scored.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)
    write_csv(scored, report_dir / "auto_model_candidates.csv")

    selected = select_models(scored, args.max_total, args.max_per_stock, args.include_hit_aux)
    if selected.empty:
        print("[WARN] no eligible models selected. See auto_model_candidates.csv reject_reasons.")
        write_csv(selected, report_dir / "auto_model_selected.csv")
        return 1

    selected = selected.copy().reset_index(drop=True)
    selected["artifact_name_auto"] = [artifact_name_for_row(r, args.version_suffix) for _, r in selected.iterrows()]
    selected["selected_at"] = now_ts()
    write_csv(selected, report_dir / "auto_model_selected.csv")

    saved_index = load_saved_model_index(models_dir)
    write_csv(saved_index, report_dir / "existing_saved_models_before.csv")

    save_report = save_selected_models(selected, args)
    write_csv(save_report, report_dir / "auto_model_save_report.csv")

    summary = {
        "created_at": now_ts(),
        "saved_data_dir": str(saved_data_dir),
        "models_dir": str(models_dir),
        "report_dir": str(report_dir),
        "leaderboard_rows": int(len(leaderboards)),
        "eligible_rows": int(scored["eligible"].sum()),
        "selected_rows": int(len(selected)),
        "save_status_counts": save_report["status"].value_counts(dropna=False).to_dict() if not save_report.empty else {},
        "thresholds": thresholds.__dict__,
        "include_hit_aux": bool(args.include_hit_aux),
        "include_high_drawdown": bool(args.include_high_drawdown),
        "allow_negative_rank_score": bool(args.allow_negative_rank_score),
        "dry_run": bool(args.dry_run),
        "skip_save": bool(args.skip_save),
    }
    (report_dir / "auto_model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failed = 0 if save_report.empty else int((save_report["status"] == "failed").sum())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
