#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def invalidate(out_root: Path) -> dict[str, Any]:
    out_root = Path(out_root).expanduser().resolve()
    strategies_root = out_root / "strategies"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_root = out_root / "_superseded_execution_batches" / stamp
    archived: list[str] = []

    if strategies_root.is_dir():
        for strategy_dir in sorted(p for p in strategies_root.iterdir() if p.is_dir()):
            batch = strategy_dir / "execution_batch.json"
            if batch.is_file():
                archive_root.mkdir(parents=True, exist_ok=True)
                target = archive_root / f"{strategy_dir.name}.json"
                shutil.move(str(batch), str(target))
                archived.append(str(target))
            manifest_file = strategy_dir / "strategy_manifest.json"
            if manifest_file.is_file():
                payload = json.loads(manifest_file.read_text(encoding="utf-8"))
                for key in (
                    "execution_batch_file",
                    "execution_batch_protocol",
                    "execution_price_source",
                ):
                    payload.pop(key, None)
                payload["execution_batch_state"] = "invalidated_for_recompute"
                atomic_json(manifest_file, payload)

    root_manifest_file = out_root / "live_nine_strategy_manifest.json"
    if root_manifest_file.is_file():
        payload = json.loads(root_manifest_file.read_text(encoding="utf-8"))
        payload.pop("execution_batch_protocol", None)
        payload["execution_batch_files"] = {}
        payload["execution_batch_state"] = "invalidated_for_recompute"
        atomic_json(root_manifest_file, payload)

    return {
        "status": "ok",
        "out_root": str(out_root),
        "archived_batch_count": len(archived),
        "archived_batches": archived,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    print(json.dumps(invalidate(Path(args.out_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
