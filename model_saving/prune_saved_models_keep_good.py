#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strictly prune saved_models to explicit keep families.

Dry-run by default. With --apply, non-kept artifact directories are moved to
cleanup_trash. Nothing is deleted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


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
class Family:
    family_id: str
    stock_code: str
    artifact_regex: str
    keep_count: int
    policy: str
    notes: str
    pattern: re.Pattern


@dataclass
class Artifact:
    stock_code: str
    artifact: str
    artifact_dir: str
    metadata_path: str
    family_id: str = ""
    keep_rank: int = 0
    action: str = ""
    reason: str = ""
    moved_to: str = ""
    created_at: str = ""
    mtime: float = 0.0
    score: float = float("-inf")
    trades: int | str = ""
    win_rate: float | str = ""
    avg_return: float | str = ""
    median_return: float | str = ""
    max_drawdown: float | str = ""
    profit_factor: float | str = ""
    label_mode: str = ""
    entry_policy: str = ""
    model_name: str = ""
    feature_group: str = ""


def read_families(path: Path) -> list[Family]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines, fieldnames=["family_id", "stock_code", "artifact_regex", "keep_count", "policy", "notes"])
    rows = []
    for r in reader:
        rows.append(Family(
            family_id=r["family_id"].strip(),
            stock_code=norm_symbol(r["stock_code"]),
            artifact_regex=r["artifact_regex"].strip(),
            keep_count=max(1, to_int(r.get("keep_count"), 1)),
            policy=(r.get("policy") or "best").strip().lower(),
            notes=(r.get("notes") or "").strip(),
            pattern=re.compile(r["artifact_regex"].strip()),
        ))
    return rows


def load_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def score_meta(meta: dict, fallback_mtime: float) -> tuple[float, dict]:
    m = meta.get("validation_tail_trade_metrics", {}) or {}
    avg = to_float(m.get("avg_return"), 0.0)
    med = to_float(m.get("median_return"), 0.0)
    pf = to_float(m.get("profit_factor"), 1.0)
    wr = to_float(m.get("win_rate"), 0.0)
    mdd = to_float(m.get("max_drawdown"), -1.0)
    trades = to_int(m.get("trades"), 0)
    score = 5.0 * avg + 2.0 * med + 0.05 * max(pf - 1.0, -1.0) + 0.10 * max(wr - 0.50, -0.5) + 0.02 * min(trades, 300) / 300.0 + 0.05 * mdd
    details = {
        "trades": trades,
        "win_rate": wr,
        "avg_return": avg,
        "median_return": med,
        "max_drawdown": mdd,
        "profit_factor": pf,
        "label_mode": str(meta.get("label_mode") or ""),
        "entry_policy": str(meta.get("entry_policy") or ""),
        "model_name": str(meta.get("model_name") or ""),
        "feature_group": str(meta.get("feature_group") or ""),
        "created_at": str(meta.get("artifact_created_at") or ""),
    }
    return score, details


def match_family(stock: str, artifact: str, families: list[Family]) -> Family | None:
    for fam in families:
        if fam.stock_code == stock and fam.pattern.search(artifact):
            return fam
    return None


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    for i in range(1, 10000):
        cand = Path(f"{dest}_{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"cannot create unique destination: {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--keep-config", default="configs/saved_models_keep_good_families.csv")
    ap.add_argument("--trash-root", default="cleanup_trash")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"saved_data/model_cleanup_logs/strict_keep_good_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trash_base = Path(args.trash_root) / f"strict_keep_good_{ts}"

    families = read_families(Path(args.keep_config))
    only = {norm_symbol(x) for x in args.only.replace(";", ",").split(",") if x.strip()}

    artifacts = []
    for artifact_dir in sorted(p for p in Path(args.models_dir).glob("*/*") if p.is_dir()):
        stock = norm_symbol(artifact_dir.parent.name)
        artifact = artifact_dir.name
        if only and stock not in only:
            continue
        meta_path = artifact_dir / "metadata.json"
        meta = load_metadata(meta_path)
        fam = match_family(stock, artifact, families)
        mtime = artifact_dir.stat().st_mtime
        score, details = score_meta(meta, mtime)
        artifacts.append(Artifact(
            stock_code=stock,
            artifact=artifact,
            artifact_dir=str(artifact_dir),
            metadata_path=str(meta_path),
            family_id=fam.family_id if fam else "",
            mtime=mtime,
            score=score,
            **details,
        ))

    fam_by_id = {f.family_id: f for f in families}
    by_family = {}
    unmatched = []
    for a in artifacts:
        if a.family_id:
            by_family.setdefault(a.family_id, []).append(a)
        else:
            unmatched.append(a)

    for fid, items in by_family.items():
        fam = fam_by_id[fid]
        if fam.policy == "latest":
            items_sorted = sorted(items, key=lambda x: (x.mtime, x.created_at), reverse=True)
        else:
            items_sorted = sorted(items, key=lambda x: (x.score, x.mtime), reverse=True)
        for idx, item in enumerate(items_sorted, start=1):
            item.keep_rank = idx
            if idx <= fam.keep_count:
                item.action = "keep"
                item.reason = f"keep_family:{fid}:rank_{idx}_of_{fam.keep_count}"
            else:
                item.action = "move_candidate"
                item.reason = f"duplicate_family:{fid}:rank_{idx}_gt_{fam.keep_count}"

    for item in unmatched:
        item.action = "move_candidate"
        item.reason = "not_in_keep_config"

    if args.apply:
        for r in artifacts:
            if r.action == "move_candidate":
                src = Path(r.artifact_dir)
                if not src.exists():
                    r.action = "missing_before_move"
                    continue
                dest = unique_dest(trash_base / r.stock_code / r.artifact)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                r.action = "moved"
                r.moved_to = str(dest)

    report_csv = out_dir / "strict_keep_good_report.csv"
    with report_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(Artifact("", "", "", "")).keys()))
        w.writeheader()
        for r in artifacts:
            w.writerow(asdict(r))

    summary = {
        "apply": bool(args.apply),
        "models_dir": args.models_dir,
        "keep_config": args.keep_config,
        "trash_base": str(trash_base),
        "report_csv": str(report_csv),
        "counts": {},
    }
    for r in artifacts:
        summary["counts"][r.action] = summary["counts"].get(r.action, 0) + 1
    (out_dir / "strict_keep_good_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
