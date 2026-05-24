#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit saved model feature dependencies for 14:55 as-of migration risk."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HIGH_RISK_PATTERNS = {
    "range_pct": re.compile(r"_range_pct(?:_|$)|^range_pct(?:_|$)"),
    "shock20": re.compile(r"_shock20$|shock20"),
    "z20": re.compile(r"_z20$|z20"),
    "z60": re.compile(r"_z60$|z60"),
    "ret3": re.compile(r"_ret3$|ret3"),
    "amount_z20": re.compile(r"_amount_z20$|amount_z20"),
    "volume_z20": re.compile(r"_volume_z20$|volume_z20"),
    "amount_shock20": re.compile(r"_amount_shock20$|amount_shock20"),
    "volume_shock20": re.compile(r"_volume_shock20$|volume_shock20"),
    "fut": re.compile(r"_fut_"),
    "future_basket": re.compile(r"_future_basket_"),
    "us": re.compile(r"_us_"),
    "us_basket": re.compile(r"_us_basket_"),
    "stock_vs_future_basket": re.compile(r"_stock_vs_future_basket_ret\d+$"),
    "stock_vs_us_basket": re.compile(r"_stock_vs_us_basket_ret\d+$"),
}

FIRST_ASOF_DROP_PATTERNS = {
    "range_pct_ret": re.compile(r"_range_pct_ret\d+$"),
    "range_pct_z20": re.compile(r"_range_pct_z20$"),
    "range_pct_z60": re.compile(r"_range_pct_z60$"),
    "range_pct_ma20_gap": re.compile(r"_range_pct_ma20_gap$"),
    "amount_shock20": re.compile(r"_amount_shock20$"),
    "amount_z20": re.compile(r"_amount_z20$"),
    "volume_shock20": re.compile(r"_volume_shock20$"),
    "volume_z20": re.compile(r"_volume_z20$"),
}


def normalize_symbol(symbol: object, fallback: str = "") -> str:
    text = str(symbol or fallback).strip().upper().replace("_", ".")
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ"}:
            return f"{right.zfill(6)}.{left}"
        return f"{left.zfill(6)}.{right}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    market = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits.zfill(6)}.{market}"


def resolve_repo_path(value: object) -> str:
    if not value:
        return ""
    text = str(value)
    path = Path(text)
    if path.exists():
        return str(path)
    marker = "stock_realtime_v021_full"
    if marker in path.parts:
        candidate = ROOT.joinpath(*path.parts[path.parts.index(marker) + 1 :])
        if candidate.exists():
            return str(candidate)
    candidate = ROOT / text
    return str(candidate) if candidate.exists() else text


def read_feature_columns(model_dir: Path) -> list[str]:
    path = model_dir / "feature_columns.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matching_patterns(feature: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(feature)]


def classify_feature(feature: str) -> str:
    if re.search(r"_fut_|_future_basket_|_us_|_us_basket_", feature) or re.search(
        r"_stock_vs_(future_basket|us_basket)_ret\d+$", feature
    ):
        return "lagged_daily_external"
    if feature.startswith(("sector_", "stock_vs_sector_", "board_")) or "_board_" in feature:
        return "sector_board_context"
    if re.match(r"^[A-Za-z0-9]+_(stock_basket|etf_basket|board_basket)_", feature):
        return "domestic_external_context"
    if re.match(r"^[A-Za-z0-9]+_", feature) and any(
        token in feature for token in ("_open", "_high", "_low", "_close", "_volume", "_amount")
    ):
        return "external_or_prefixed_raw"
    if matching_patterns(feature, FIRST_ASOF_DROP_PATTERNS):
        return "first_asof_drop_candidate"
    return "core_or_other"


def load_metadata(meta_path: Path) -> dict:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"metadata_read_error": f"{type(exc).__name__}: {exc}"}


def iter_model_rows(saved_models: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted(saved_models.rglob("metadata.json")):
        model_dir = meta_path.parent
        meta = load_metadata(meta_path)
        stock_code = normalize_symbol(meta.get("stock_code"), model_dir.parent.name)
        artifact_name = str(meta.get("artifact_name") or model_dir.name)
        features = read_feature_columns(model_dir)
        for idx, feature in enumerate(features, start=1):
            high_risk = matching_patterns(feature, HIGH_RISK_PATTERNS)
            first_drop = matching_patterns(feature, FIRST_ASOF_DROP_PATTERNS)
            rows.append(
                {
                    "stock_code": stock_code,
                    "artifact_name": artifact_name,
                    "model_dir": str(model_dir),
                    "feature_index": idx,
                    "feature": feature,
                    "feature_category": classify_feature(feature),
                    "is_high_risk_asof": bool(high_risk),
                    "high_risk_patterns": ",".join(high_risk),
                    "is_first_asof_drop_candidate": bool(first_drop),
                    "first_asof_drop_patterns": ",".join(first_drop),
                    "label_mode": meta.get("label_mode", ""),
                    "entry_policy": meta.get("entry_policy", ""),
                    "feature_group": meta.get("feature_group", ""),
                    "model_name": meta.get("model_name", ""),
                    "threshold": meta.get("threshold", ""),
                    "samples": resolve_repo_path(meta.get("samples", "")),
                    "metadata_path": str(meta_path),
                }
            )
        if not features:
            rows.append(
                {
                    "stock_code": stock_code,
                    "artifact_name": artifact_name,
                    "model_dir": str(model_dir),
                    "feature_index": "",
                    "feature": "",
                    "feature_category": "missing_feature_columns",
                    "is_high_risk_asof": False,
                    "high_risk_patterns": "",
                    "is_first_asof_drop_candidate": False,
                    "first_asof_drop_patterns": "",
                    "label_mode": meta.get("label_mode", ""),
                    "entry_policy": meta.get("entry_policy", ""),
                    "feature_group": meta.get("feature_group", ""),
                    "model_name": meta.get("model_name", ""),
                    "threshold": meta.get("threshold", ""),
                    "samples": resolve_repo_path(meta.get("samples", "")),
                    "metadata_path": str(meta_path),
                }
            )
    return rows


FIELDNAMES = [
    "stock_code",
    "artifact_name",
    "model_dir",
    "feature_index",
    "feature",
    "feature_category",
    "is_high_risk_asof",
    "high_risk_patterns",
    "is_first_asof_drop_candidate",
    "first_asof_drop_patterns",
    "label_mode",
    "entry_policy",
    "feature_group",
    "model_name",
    "threshold",
    "samples",
    "metadata_path",
]

SUMMARY_FIELDNAMES = [
    "stock_code",
    "artifact_name",
    "model_dir",
    "label_mode",
    "entry_policy",
    "feature_group",
    "model_name",
    "samples",
    "feature_count",
    "high_risk_feature_count",
    "high_risk_feature_ratio",
    "first_asof_drop_candidate_count",
    "first_asof_drop_candidate_ratio",
    "lagged_daily_external_count",
    "lagged_daily_external_ratio",
    "high_risk_features",
    "first_asof_drop_candidate_features",
    "lagged_daily_external_features",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_outputs(feature_rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    required_path = out_dir / "model_required_features.csv"
    risk_path = out_dir / "high_risk_feature_by_model.csv"
    write_csv(required_path, feature_rows, FIELDNAMES)

    if not feature_rows:
        write_csv(risk_path, [], SUMMARY_FIELDNAMES)
        return required_path, risk_path

    group_cols = [
        "stock_code",
        "artifact_name",
        "model_dir",
        "label_mode",
        "entry_policy",
        "feature_group",
        "model_name",
        "samples",
    ]
    grouped: dict[tuple, list[dict]] = {}
    for row in feature_rows:
        key = tuple(row.get(col, "") for col in group_cols)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for keys, part in grouped.items():
        data = dict(zip(group_cols, keys))
        features = [row for row in part if row.get("feature")]
        feature_count = len(features)
        high = [row for row in features if row.get("is_high_risk_asof")]
        first_drop = [row for row in features if row.get("is_first_asof_drop_candidate")]
        lagged = [row for row in features if row.get("feature_category") == "lagged_daily_external"]
        data.update(
            {
                "feature_count": feature_count,
                "high_risk_feature_count": len(high),
                "high_risk_feature_ratio": f"{(float(len(high)) / feature_count) if feature_count else 0.0:.6f}",
                "first_asof_drop_candidate_count": len(first_drop),
                "first_asof_drop_candidate_ratio": f"{(float(len(first_drop)) / feature_count) if feature_count else 0.0:.6f}",
                "lagged_daily_external_count": len(lagged),
                "lagged_daily_external_ratio": f"{(float(len(lagged)) / feature_count) if feature_count else 0.0:.6f}",
                "high_risk_features": ",".join(row["feature"] for row in high),
                "first_asof_drop_candidate_features": ",".join(row["feature"] for row in first_drop),
                "lagged_daily_external_features": ",".join(row["feature"] for row in lagged),
            }
        )
        summary_rows.append(data)

    summary_rows.sort(
        key=lambda row: (
            float(row["high_risk_feature_ratio"]),
            float(row["first_asof_drop_candidate_ratio"]),
            int(row["high_risk_feature_count"]),
        ),
        reverse=True,
    )
    write_csv(risk_path, summary_rows, SUMMARY_FIELDNAMES)
    return required_path, risk_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit saved model feature dependencies for as-of high-risk fields")
    p.add_argument("--saved-models", default=str(ROOT / "saved_models"))
    p.add_argument("--out-dir", default=str(ROOT / "reports" / "feature_audit"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = iter_model_rows(Path(args.saved_models))
    required_path, risk_path = write_outputs(rows, Path(args.out_dir))
    model_count = len({(row.get("stock_code", ""), row.get("artifact_name", "")) for row in rows})
    print(f"WROTE {required_path} rows={len(rows)} models={model_count}")
    print(f"WROTE {risk_path}")


if __name__ == "__main__":
    main()
