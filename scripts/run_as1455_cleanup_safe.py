#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the AS1455 cleaner while preserving its external-process guard.

The shell maintenance wrapper necessarily contains ``as1455`` in its command
line.  The underlying cleaner therefore sees its own parent wrapper as an
active AS1455 process.  This launcher filters only the current process ancestry
from that check; unrelated AS1455 jobs remain blocking processes.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEANER_PATH = PROJECT_DIR / "scripts" / "cleanup_as1455_storage.py"


def parent_pid(pid: int) -> int | None:
    status = Path("/proc") / str(pid) / "status"
    try:
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PPid:"):
                value = int(line.split(":", 1)[1].strip())
                return value if value > 0 else None
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def ancestor_pids() -> set[int]:
    ancestors: set[int] = set()
    pid = os.getpid()
    while True:
        pid = parent_pid(pid) or 0
        if pid <= 0 or pid in ancestors:
            break
        ancestors.add(pid)
    return ancestors


def load_cleaner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("as1455_storage_cleaner", CLEANER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cleaner: {CLEANER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    cleaner = load_cleaner()
    original = cleaner.active_as1455_processes
    ignored = ancestor_pids()

    def external_active_as1455_processes() -> list[dict[str, Any]]:
        return [
            row
            for row in original()
            if int(row.get("pid", -1)) not in ignored
        ]

    cleaner.active_as1455_processes = external_active_as1455_processes
    cleaner.main()


if __name__ == "__main__":
    main()
