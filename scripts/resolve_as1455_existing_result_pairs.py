#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

TARGET_REBALANCE = {
    "r01_fwd": 1,
    "r05_fwd": 5,
    "r21_fwd": 21,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(root.resolve())
    except ValueError:
        return False


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return path


def normalize_signal_cols(value: Any) -> str:
    parts: list[str] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        number = float(token)
        if not number.is_integer() or number < 0:
            raise RuntimeError(f"invalid signal_cols={value!r}")
        parts.append(str(int(number)))
    if not parts:
        raise RuntimeError(f"invalid signal_cols={value!r}")
    return ",".join(parts)


def comparable_selection(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_name",
        "signal_name",
        "signal_mode",
        "historical_max_positions",
        "historical_sell_rank",
        "historical_rebalance_every",
        "historical_rebalance_offset",
    )
    result = {key: payload.get(key) for key in keys}
    for key in (
        "historical_max_positions",
        "historical_sell_rank",
        "historical_rebalance_every",
        "historical_rebalance_offset",
    ):
        if result[key] is not None:
            result[key] = int(result[key])
    result["signal_cols"] = normalize_signal_cols(payload.get("signal_cols"))
    return result


def resolve_recorded_file(
    *,
    root: Path,
    manifest: dict[str, Any],
    manifest_key: str,
    standard_names: tuple[str, ...],
    label: str,
) -> Path:
    candidates: list[Path] = []
    recorded = manifest.get(manifest_key)
    if recorded:
        path = Path(str(recorded)).expanduser()
        candidates.append(path)
        if not path.is_absolute():
            candidates.extend((root / path, root / "00_predictions" / path.name))
        else:
            candidates.append(root / "00_predictions" / path.name)
    candidates.extend(root / "00_predictions" / name for name in standard_names)

    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if is_within(resolved, root) and resolved.is_file() and resolved.stat().st_size > 0:
            return resolved

    h5_matches = sorted(
        path.resolve()
        for pattern in ("*.h5", "*.hdf", "*.hdf5")
        for path in (root / "00_predictions").glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    )
    if len(h5_matches) == 1:
        return h5_matches[0]
    raise RuntimeError(
        f"cannot resolve {label} under {root}; candidates={candidates} matches={h5_matches}"
    )


def load_config(run_dir: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = require_file(run_dir / "config.json", f"{label} config")
    return path, read_json(path)


def assert_config_matches_selection(
    config: dict[str, Any],
    selection: dict[str, Any],
    *,
    label: str,
) -> None:
    expected = comparable_selection(selection)
    checks = {
        "signal_name": config.get("signal_name"),
        "signal_mode": config.get("signal_mode"),
        "signal_cols": normalize_signal_cols(config.get("signal_cols")),
        "historical_max_positions": config.get("max_positions"),
        "historical_sell_rank": config.get("sell_rank"),
        "historical_rebalance_every": config.get("rebalance_every"),
        "historical_rebalance_offset": config.get("rebalance_offset"),
    }
    normalized = {
        key: int(value) if key.startswith("historical_") and value is not None else value
        for key, value in checks.items()
    }
    expected_config = {key: value for key, value in expected.items() if key != "run_name"}
    if normalized != expected_config:
        raise RuntimeError(
            f"{label} config does not match frozen selection:\n"
            f"expected={expected_config}\nactual={normalized}"
        )


def historical_nav(root: Path, run_name: str) -> Path:
    candidates = (
        root / "01_close_auction_grid" / "01_runs" / run_name / "close_auction_nav.csv",
        root / "01_close_auction_daily_grid" / "01_runs" / run_name / "close_auction_nav.csv",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    matches = sorted(root.glob(f"**/01_runs/{run_name}/close_auction_nav.csv"))
    if len(matches) == 1 and matches[0].stat().st_size > 0:
        return matches[0]
    raise RuntimeError(f"historical NAV not found for run={run_name} under {root}")


def validate_historical_root(
    root: Path,
    historical_base: Path,
    preset: str,
    target: str,
    rebalance_every: int,
    strict_selection: dict[str, Any],
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    expected_prefix = f"{preset}_{target}_reb{rebalance_every}_"
    if not is_within(root, historical_base):
        raise RuntimeError(f"historical root is outside base: {root}")
    if not root.name.startswith(expected_prefix):
        raise RuntimeError(
            f"historical root name mismatch: expected prefix={expected_prefix} actual={root.name}"
        )

    materialized_path = require_file(root / "materialized_best_run.json", "materialized manifest")
    materialized = read_json(materialized_path)
    selection = materialized.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError(f"materialized selection missing: {materialized_path}")
    if comparable_selection(selection) != comparable_selection(strict_selection):
        raise RuntimeError(
            "historical materialized selection does not match forward strict-OOS selection"
        )

    run_name = str(selection.get("run_name", "")).strip()
    if not run_name:
        raise RuntimeError(f"historical run_name missing: {materialized_path}")
    nav = historical_nav(root, run_name)
    run_dir = nav.parent
    config_path, config = load_config(run_dir, "historical")
    assert_config_matches_selection(config, selection, label="historical")

    mapping_path = require_file(
        root / "00_predictions" / "one_lag_prediction_manifest.json",
        "one-lag prediction manifest",
    )
    mapping_payload = read_json(mapping_path)
    prediction_file = resolve_recorded_file(
        root=root,
        manifest=mapping_payload,
        manifest_key="prediction_file",
        standard_names=("test_preds.h5", "one_lag_preds.h5"),
        label="historical prediction HDF",
    )

    fold_rows = mapping_payload.get("fold_mapping")
    if not isinstance(fold_rows, list):
        raise RuntimeError(f"fold_mapping missing: {mapping_path}")
    found_folds = {
        int(row["source_fold"])
        for row in fold_rows
        if isinstance(row, dict) and row.get("source_fold") is not None
    }
    expected_folds = set(range(1, 7)) if target in {"r01_fwd", "r05_fwd"} else set(range(1, 6))
    if found_folds != expected_folds:
        raise RuntimeError(
            f"historical fold mapping mismatch: expected={sorted(expected_folds)} "
            f"actual={sorted(found_folds)} path={mapping_path}"
        )

    normalized_rows: list[dict[str, Any]] = []
    for row in fold_rows:
        if not isinstance(row, dict):
            continue
        source_fold = int(row.get("source_fold", -1))
        if source_fold not in expected_folds:
            continue
        start = (
            row.get("target_fold_start")
            or row.get("target_validation_start")
            or row.get("target_test_start")
        )
        end = (
            row.get("target_fold_end")
            or row.get("target_validation_end")
            or row.get("target_test_end")
        )
        if not start or not end:
            raise RuntimeError(f"fold{source_fold} boundary missing: {mapping_path}")
        normalized_rows.append(
            {
                **row,
                "source_fold": source_fold,
                "start": str(start),
                "end": str(end),
            }
        )

    return {
        "root": str(root),
        "run_name": run_name,
        "nav_file": str(nav.resolve()),
        "config_file": str(config_path.resolve()),
        "config": config,
        "prediction_file": str(prediction_file),
        "materialized_manifest": str(materialized_path.resolve()),
        "prediction_manifest": str(mapping_path.resolve()),
        "fold_mapping": sorted(normalized_rows, key=lambda row: int(row["source_fold"])),
        "selection": comparable_selection(selection),
        "source_summary": materialized.get("source_summary"),
    }


def validate_forward_root(
    root: Path,
    forward_base: Path,
    historical_base: Path,
    preset: str,
    target: str,
    rebalance_every: int,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    expected_prefix = f"{preset}_{target}_reb{rebalance_every}_"
    if not is_within(root, forward_base):
        raise RuntimeError(f"forward root is outside base: {root}")
    if not root.name.startswith(expected_prefix):
        raise RuntimeError(
            f"forward root name mismatch: expected prefix={expected_prefix} actual={root.name}"
        )

    strict_path = require_file(
        root / "01_close_auction_grid" / "strict_oos_manifest.json",
        "strict-OOS manifest",
    )
    strict = read_json(strict_path)
    required_contract = {
        "evaluation_mode": "strict_oos",
        "historical_trading_parameters_reused": True,
        "historical_rebalance_phase_reused": True,
        "retained_config_count": 1,
    }
    for key, expected in required_contract.items():
        if strict.get(key) != expected:
            raise RuntimeError(
                f"strict-OOS contract mismatch: {key} expected={expected!r} "
                f"actual={strict.get(key)!r} path={strict_path}"
            )
    if int(strict.get("generated_config_count", -1)) != 1:
        raise RuntimeError(
            f"strict-OOS forward must contain exactly one generated config: {strict_path}"
        )

    strict_selection = strict.get("historical_selection")
    if not isinstance(strict_selection, dict):
        raise RuntimeError(f"historical_selection missing: {strict_path}")
    historical_root_text = str(strict_selection.get("backtest_root", "")).strip()
    if not historical_root_text:
        raise RuntimeError(f"historical backtest_root missing: {strict_path}")
    historical = validate_historical_root(
        Path(historical_root_text),
        historical_base,
        preset,
        target,
        rebalance_every,
        strict_selection,
    )

    retained_run = str(strict.get("retained_run_name", "")).strip()
    if not retained_run:
        raise RuntimeError(f"retained_run_name missing: {strict_path}")
    run_dir = root / "01_close_auction_grid" / "01_runs" / retained_run
    forward_nav = require_file(run_dir / "close_auction_nav.csv", "forward NAV")
    config_path, config = load_config(run_dir, "forward")

    retained_config = strict.get("retained_config")
    if not isinstance(retained_config, dict):
        raise RuntimeError(f"retained_config missing: {strict_path}")
    for key in ("max_positions", "sell_rank", "rebalance_every", "rebalance_offset"):
        if int(config.get(key, -1)) != int(retained_config.get(key, -2)):
            raise RuntimeError(
                f"forward config mismatch for {key}: config={config.get(key)} "
                f"strict={retained_config.get(key)}"
            )
    if str(config.get("signal_name")) != str(strict_selection.get("signal_name")):
        raise RuntimeError("forward signal_name does not match historical selection")
    if str(config.get("signal_mode")) != str(strict_selection.get("signal_mode")):
        raise RuntimeError("forward signal_mode does not match historical selection")
    if normalize_signal_cols(config.get("signal_cols")) != normalize_signal_cols(
        strict_selection.get("signal_cols")
    ):
        raise RuntimeError("forward signal_cols do not match historical selection")

    summary = require_file(
        root / "01_close_auction_grid" / "02_summary" / "grid_summary_compact.csv",
        "forward compact summary",
    )
    prediction_manifest_path = require_file(
        root / "00_predictions" / "fold0_forward_prediction_manifest.json",
        "forward prediction manifest",
    )
    prediction_manifest = read_json(prediction_manifest_path)
    prediction_file = resolve_recorded_file(
        root=root,
        manifest=prediction_manifest,
        manifest_key="prediction_file",
        standard_names=("fold0_forward_preds.h5",),
        label="forward prediction HDF",
    )

    return {
        "root": str(root),
        "run_name": retained_run,
        "nav_file": str(forward_nav.resolve()),
        "config_file": str(config_path.resolve()),
        "config": config,
        "prediction_file": str(prediction_file),
        "prediction_manifest": str(prediction_manifest_path.resolve()),
        "strict_oos_manifest": str(strict_path.resolve()),
        "summary_file": str(summary.resolve()),
        "historical": historical,
    }


def resolve_pair(
    forward_base: Path,
    historical_base: Path,
    preset: str,
    target: str,
) -> dict[str, Any]:
    rebalance_every = TARGET_REBALANCE[target]
    pattern = f"{preset}_{target}_reb{rebalance_every}_*"
    candidates = sorted(
        (path for path in forward_base.glob(pattern) if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    reasons: list[str] = []
    for candidate in candidates:
        try:
            forward = validate_forward_root(
                candidate,
                forward_base,
                historical_base,
                preset,
                target,
                rebalance_every,
            )
            return {
                "label": f"{target}_{preset}",
                "feature_preset": preset,
                "target_col": target,
                "rebalance_every": rebalance_every,
                "historical_root": forward["historical"]["root"],
                "forward_root": forward["root"],
                "historical": forward["historical"],
                "forward": {key: value for key, value in forward.items() if key != "historical"},
            }
        except Exception as exc:
            reasons.append(f"{candidate}: {exc}")
    detail = "\n".join(reasons[:10]) if reasons else "no candidate directories found"
    raise RuntimeError(
        f"no complete paired result for preset={preset} target={target} pattern={pattern}\n{detail}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve complete existing historical/strict-OOS AS1455 result pairs"
    )
    parser.add_argument("--historical-base", required=True)
    parser.add_argument("--forward-base", required=True)
    parser.add_argument(
        "--feature-presets",
        default="rotation_onehot rotation_addon_onehot",
    )
    parser.add_argument("--targets", default="r01_fwd r05_fwd r21_fwd")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--tsv-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_base = Path(args.historical_base).expanduser().resolve()
    forward_base = Path(args.forward_base).expanduser().resolve()
    if not historical_base.is_dir():
        raise FileNotFoundError(historical_base)
    if not forward_base.is_dir():
        raise FileNotFoundError(forward_base)

    presets = args.feature_presets.split()
    targets = args.targets.split()
    unsupported = sorted(set(targets) - set(TARGET_REBALANCE))
    if unsupported:
        raise SystemExit(f"unsupported targets: {unsupported}")

    pairs = [
        resolve_pair(forward_base, historical_base, preset, target)
        for target in targets
        for preset in presets
    ]
    expected = len(presets) * len(targets)
    if len(pairs) != expected:
        raise RuntimeError(f"pair count mismatch: expected={expected} actual={len(pairs)}")

    json_out = Path(args.json_out)
    tsv_out = Path(args.tsv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    tsv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "mode": "paired_frozen_results",
                "pair_count": len(pairs),
                "historical_base": str(historical_base),
                "forward_base": str(forward_base),
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with tsv_out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                f"{pair['label']}\t{pair['historical_root']}\t{pair['forward_root']}\n"
            )
    print(f"[OK] resolved complete result pairs: {len(pairs)}")
    print(f"[OK] json={json_out}")
    print(f"[OK] tsv={tsv_out}")


if __name__ == "__main__":
    main()
