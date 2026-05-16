#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect saved_models feature dependencies."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Row:
    stock_code: str
    artifact: str
    feature_count: int
    board_cols: int
    stock_cols: int
    etf_cols: int
    future_cols: int
    metadata_samples: str
    feature_group: str
    model_name: str
    label_mode: str
    entry_policy: str
    examples_board: str


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="saved_models")
    ap.add_argument("--out-dir", default="saved_data/model_update_logs/model_feature_inspect")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    only = {x.strip().upper() for x in args.only.replace(";", ",").split(",") if x.strip()}
    rows = []
    for cols_path in sorted(Path(args.models_dir).glob("*/*/feature_columns.txt")):
        artifact_dir = cols_path.parent
        stock = artifact_dir.parent.name
        if only and stock.upper() not in only and stock.split(".", 1)[0].upper() not in only:
            continue
        cols = [x.strip() for x in cols_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        meta = {}
        meta_path = artifact_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        board = [c for c in cols if "_board_" in c or "board_basket" in c]
        stock_cols = [c for c in cols if "_stk_" in c or "stock_basket" in c]
        etf_cols = [c for c in cols if "_etf_" in c or "etf_basket" in c]
        fut_cols = [c for c in cols if "_fut_" in c or "future_basket" in c]
        rows.append(Row(
            stock_code=stock,
            artifact=artifact_dir.name,
            feature_count=len(cols),
            board_cols=len(board),
            stock_cols=len(stock_cols),
            etf_cols=len(etf_cols),
            future_cols=len(fut_cols),
            metadata_samples=str(meta.get("samples", "")),
            feature_group=str(meta.get("feature_group", "")),
            model_name=str(meta.get("model_name", "")),
            label_mode=str(meta.get("label_mode", "")),
            entry_policy=str(meta.get("entry_policy", "")),
            examples_board=" | ".join(board[:20]),
        ))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "saved_models_feature_dependencies.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(Row("", "", 0, 0, 0, 0, 0, "", "", "", "", "", "")).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(out)
    for r in rows:
        print(asdict(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
