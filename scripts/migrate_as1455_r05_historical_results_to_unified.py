#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Place validated legacy r05 historical grids under the unified matrix tree.

The default mode creates relative symbolic links.  This gives the unified refresh
workflow stable local paths without duplicating large historical grid artifacts or
moving the legacy results.  ``copy`` and ``hardlink`` modes are available when a
physical tree is required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.find_as1455_compatible_historical_result import (  # noqa: E402
    DEFAULT_SEARCH_ROOTS,
    SIGNALS,
    candidate_roots,
    validate_candidate,
)
from scripts.resolve_as1455_fixed_signal_matrix_folds import resolve_target  # noqa: E402

SIGNAL_KINDS = ("all5", "first3", "best")
DEFAULT_MATRIX_ROOT = Path(
    "saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1"
)


def parse_folds(text: str) -> list[int]:
    folds = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not folds or folds != list(range(folds[-1] + 1)):
        raise ValueError(
            "target folds must be a contiguous zero-based range such as 0,1,2,3,4,5"
        )
    return folds


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative, target_is_directory=True)


def materialize(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        relative_symlink(source, destination)
        return
    if mode == "copy":
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        return
    if mode == "hardlink":
        try:
            shutil.copytree(source, destination, copy_function=os.link)
        except Exception:
            if destination.exists():
                shutil.rmtree(destination)
            raise
        return
    raise ValueError(f"unsupported migration mode: {mode}")


def validated_local_result(
    destination: Path,
    *,
    signal_kind: str,
    target_folds: list[int],
) -> dict[str, Any] | None:
    if not destination.exists():
        return None
    try:
        return validate_candidate(
            destination,
            target_col="r05_fwd",
            signal_spec=SIGNALS[signal_kind],
            rebalance_every=5,
            expected_folds=target_folds,
        )
    except Exception:
        return None


def select_legacy_source(
    *,
    signal_kind: str,
    target_folds: list[int],
    search_roots: list[Path],
    destination: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    destination_resolved = destination.resolve() if destination.exists() else None
    for root in candidate_roots(search_roots, None):
        try:
            resolved = root.expanduser().resolve()
            if destination_resolved is not None and resolved == destination_resolved:
                continue
            item = validate_candidate(
                root,
                target_col="r05_fwd",
                signal_spec=SIGNALS[signal_kind],
                rebalance_every=5,
                expected_folds=target_folds,
            )
        except Exception as exc:
            rejected.append({"root": str(root), "reason": str(exc)})
            continue
        valid.append(item)
    valid.sort(key=lambda item: float(item.get("mtime", 0.0)), reverse=True)
    return (valid[0] if valid else None), rejected


def ensure_legacy_compatibility_link(experiment_root: Path, replace: bool) -> Path:
    canonical = experiment_root / "historical_fold_selection"
    legacy = experiment_root / "historical_fold0_to_fold5_selection"
    if legacy.is_symlink() and legacy.resolve() == canonical.resolve():
        return legacy
    if legacy.exists() or legacy.is_symlink():
        if not replace:
            raise RuntimeError(
                f"legacy compatibility path already exists and differs: {legacy}"
            )
        remove_path(legacy)
    relative_symlink(canonical, legacy)
    return legacy


def migrate_signal(
    *,
    matrix_root: Path,
    signal_kind: str,
    target_folds: list[int],
    search_roots: list[Path],
    mode: str,
    replace: bool,
) -> dict[str, Any]:
    fold_label = f"fold0_{target_folds[-1]}"
    experiment_name = f"r05_{signal_kind}_reb5_{fold_label}_forward"
    experiment_root = matrix_root / experiment_name
    destination = experiment_root / "historical_fold_selection"
    experiment_root.mkdir(parents=True, exist_ok=True)

    local = validated_local_result(
        destination,
        signal_kind=signal_kind,
        target_folds=target_folds,
    )
    if local is not None:
        legacy_link = ensure_legacy_compatibility_link(experiment_root, replace=replace)
        result = {
            "status": "already_present",
            "signal_kind": signal_kind,
            "signal_spec": SIGNALS[signal_kind],
            "target_folds": target_folds,
            "experiment_root": str(experiment_root.resolve()),
            "unified_historical_root": str(destination.resolve()),
            "legacy_compatibility_path": str(legacy_link),
            "migration_mode": "existing",
            "validated_result": local,
        }
    else:
        source, rejected = select_legacy_source(
            signal_kind=signal_kind,
            target_folds=target_folds,
            search_roots=search_roots,
            destination=destination,
        )
        if source is None:
            return {
                "status": "missing",
                "signal_kind": signal_kind,
                "signal_spec": SIGNALS[signal_kind],
                "target_folds": target_folds,
                "experiment_root": str(experiment_root.resolve()),
                "unified_historical_root": str(destination),
                "rejected_candidate_count": len(rejected),
                "rejected_candidates": rejected,
            }

        if destination.exists() or destination.is_symlink():
            if not replace:
                raise RuntimeError(
                    f"destination exists but is incompatible; use --replace: {destination}"
                )
            remove_path(destination)
        source_root = Path(source["historical_root"]).resolve()
        materialize(source_root, destination, mode)
        migrated = validate_candidate(
            destination,
            target_col="r05_fwd",
            signal_spec=SIGNALS[signal_kind],
            rebalance_every=5,
            expected_folds=target_folds,
        )
        legacy_link = ensure_legacy_compatibility_link(experiment_root, replace=replace)
        result = {
            "status": "migrated",
            "signal_kind": signal_kind,
            "signal_spec": SIGNALS[signal_kind],
            "target_folds": target_folds,
            "source_historical_root": str(source_root),
            "experiment_root": str(experiment_root.resolve()),
            "unified_historical_root": str(destination.resolve()),
            "unified_historical_path": str(destination),
            "legacy_compatibility_path": str(legacy_link),
            "migration_mode": mode,
            "validated_source": source,
            "validated_destination": migrated,
        }

    manifest_file = experiment_root / "historical_migration_manifest.json"
    manifest_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    result["manifest_file"] = str(manifest_file.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate validated legacy r05 historical grids into refresh_all_v1"
    )
    parser.add_argument("--matrix-root", default=str(DEFAULT_MATRIX_ROOT))
    parser.add_argument(
        "--target-folds",
        default=None,
        help="Default: auto use fold0..5 when r05 source_fold6 has five checkpoints, else fold0..4",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    matrix_root.mkdir(parents=True, exist_ok=True)
    if args.target_folds:
        target_folds = parse_folds(args.target_folds)
    else:
        plan = resolve_target("r05_fwd", args.top_n)
        target_folds = list(plan["target_folds"])
    search_roots = [Path(value) for value in args.search_root] or list(DEFAULT_SEARCH_ROOTS)

    results = [
        migrate_signal(
            matrix_root=matrix_root,
            signal_kind=signal_kind,
            target_folds=target_folds,
            search_roots=search_roots,
            mode=args.mode,
            replace=args.replace,
        )
        for signal_kind in SIGNAL_KINDS
    ]
    missing = [item for item in results if item["status"] == "missing"]
    payload = {
        "status": "partial" if missing else "ok",
        "matrix_root": str(matrix_root),
        "target_folds": target_folds,
        "mode": args.mode,
        "results": results,
        "missing_signal_kinds": [item["signal_kind"] for item in missing],
    }
    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else matrix_root / "r05_historical_migration_manifest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    for item in results:
        print(
            f"[R05 HISTORY] signal={item['signal_kind']} status={item['status']} "
            f"path={item['unified_historical_root']}"
        )
    print(f"[R05 HISTORY] manifest={output}")
    if missing and not args.allow_missing:
        raise SystemExit(
            "compatible legacy r05 historical grids were not found for: "
            + ",".join(item["signal_kind"] for item in missing)
        )


if __name__ == "__main__":
    main()
