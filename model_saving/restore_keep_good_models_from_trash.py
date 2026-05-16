#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restore keep-family artifacts from cleanup_trash if target is missing.

Dry-run by default. With --apply, matched artifacts are moved back to saved_models.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


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


@dataclass
class Rule:
    family_id: str
    stock_code: str
    pattern: re.Pattern


@dataclass
class Row:
    stock_code: str
    artifact: str
    source_dir: str
    target_dir: str
    action: str
    reason: str


def read_rules(path: Path) -> list[Rule]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    out = []
    for r in csv.DictReader(lines, fieldnames=["family_id", "stock_code", "artifact_regex", "keep_count", "policy", "notes"]):
        out.append(Rule(r["family_id"], norm_symbol(r["stock_code"]), re.compile(r["artifact_regex"])))
    return out


def matched(stock: str, artifact: str, rules: list[Rule]) -> bool:
    return any(r.stock_code == stock and r.pattern.search(artifact) for r in rules)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trash-root", default="cleanup_trash")
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--keep-config", default="configs/saved_models_keep_good_families.csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"saved_data/model_cleanup_logs/restore_keep_good_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = read_rules(Path(args.keep_config))

    rows = []
    for meta_path in sorted(Path(args.trash_root).glob("**/metadata.json")):
        artifact_dir = meta_path.parent
        artifact = artifact_dir.name
        stock = norm_symbol(artifact_dir.parent.name)
        if not stock:
            continue
        if not matched(stock, artifact, rules):
            continue
        target = Path(args.models_dir) / stock / artifact
        if target.exists():
            rows.append(Row(stock, artifact, str(artifact_dir), str(target), "skip_target_exists", "target_exists"))
        else:
            action = "restore_candidate"
            reason = "matched_keep_rule_target_missing"
            if args.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(artifact_dir), str(target))
                action = "restored"
            rows.append(Row(stock, artifact, str(artifact_dir), str(target), action, reason))

    report = out_dir / "restore_keep_good_report.csv"
    with report.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(Row("", "", "", "", "", "")).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    summary = {"apply": args.apply, "report_csv": str(report), "counts": {}}
    for r in rows:
        summary["counts"][r.action] = summary["counts"].get(r.action, 0) + 1
    (out_dir / "restore_keep_good_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
