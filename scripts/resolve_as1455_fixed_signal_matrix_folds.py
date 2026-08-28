#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve the largest valid one-fold-lag development set for each AS1455 target.

For each target, source folds 0..5 are mandatory: source_fold0 supplies the
strict-forward checkpoints and source_fold1..5 supply target_fold0..4.  Fold6
is optional.  When it contains at least ``top_n`` saved checkpoints, the
historical development set is extended to target_fold5; otherwise the target
uses target_fold0..4 and records the exact reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.as1455_ch17_common import default_fold_dir_template  # noqa: E402

TARGETS = ("r01_fwd", "r05_fwd", "r21_fwd")


def checkpoint_count(root: Path) -> tuple[int, str | None]:
    table_file = root / "search_best_checkpoints.csv"
    if not root.is_dir():
        return 0, f"missing directory: {root}"
    if not table_file.is_file():
        return 0, f"missing checkpoint table: {table_file}"
    try:
        table = pd.read_csv(table_file)
    except Exception as exc:  # pragma: no cover - server artifact dependent
        return 0, f"cannot read checkpoint table {table_file}: {exc}"
    if "checkpoint_saved" in table.columns:
        mask = table["checkpoint_saved"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes"]
        )
        table = table.loc[mask]
    return int(len(table)), None


def resolve_target(target_col: str, top_n: int) -> dict[str, Any]:
    template = default_fold_dir_template("rotation_addon_onehot", target_col)
    checks: list[dict[str, Any]] = []
    blocking: list[str] = []

    for source_fold in range(0, 7):
        root = Path(template.format(fold=source_fold)).expanduser().resolve()
        count, error = checkpoint_count(root)
        row = {
            "source_fold": source_fold,
            "root": str(root),
            "saved_checkpoint_count": count,
            "valid_for_top_n": error is None and count >= top_n,
            "error": error,
        }
        checks.append(row)
        if source_fold <= 5 and not row["valid_for_top_n"]:
            detail = error or f"need {top_n} saved checkpoints, got {count}"
            blocking.append(f"{target_col} source_fold{source_fold}: {detail}")

    if blocking:
        raise RuntimeError("\n".join(blocking))

    fold6 = checks[6]
    use_fold6 = bool(fold6["valid_for_top_n"])
    target_folds = list(range(0, 6 if use_fold6 else 5))
    source_folds = [fold + 1 for fold in target_folds]
    fold_label = f"fold0_{target_folds[-1]}"
    skip_reason = None
    if not use_fold6:
        skip_reason = fold6["error"] or (
            f"source_fold6 has {fold6['saved_checkpoint_count']} saved checkpoints; "
            f"need {top_n}"
        )

    return {
        "target_col": target_col,
        "top_n": top_n,
        "target_folds": target_folds,
        "target_folds_csv": ",".join(map(str, target_folds)),
        "source_folds": source_folds,
        "fold_label": fold_label,
        "source_fold6_used": use_fold6,
        "target_fold5_skipped": not use_fold6,
        "target_fold5_skipped_reason": skip_reason,
        "fold_dir_template": template,
        "checkpoint_checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve target_fold0..5 when source_fold6 is valid, else 0..4"
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--format",
        choices=["json", "shell"],
        default="json",
        help="shell emits fixed TARGET_FOLDS_Rxx/FOLD_LABEL_Rxx assignments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")

    try:
        plans = {target: resolve_target(target, args.top_n) for target in TARGETS}
    except RuntimeError as exc:
        print("[BLOCKED] required fold0..5 checkpoint coverage is incomplete:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"  - {line}", file=sys.stderr)
        raise SystemExit(3)

    payload = {
        "status": "ok",
        "feature_preset": "rotation_addon_onehot",
        "top_n": args.top_n,
        "targets": plans,
    }
    if args.output_json:
        path = Path(args.output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] fold availability plan={path}", file=sys.stderr)

    if args.format == "shell":
        for target in TARGETS:
            prefix = target.split("_", 1)[0].upper()
            plan = plans[target]
            print(f"TARGET_FOLDS_{prefix}={plan['target_folds_csv']}")
            print(f"FOLD_LABEL_{prefix}={plan['fold_label']}")
            print(f"USE_FOLD6_{prefix}={1 if plan['source_fold6_used'] else 0}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    for target in TARGETS:
        plan = plans[target]
        message = (
            f"[FOLDS] {target}: target={plan['target_folds_csv']} "
            f"source={','.join(map(str, plan['source_folds']))}"
        )
        if plan["target_fold5_skipped"]:
            message += f"; skip target_fold5: {plan['target_fold5_skipped_reason']}"
        print(message, file=sys.stderr)


if __name__ == "__main__":
    main()
