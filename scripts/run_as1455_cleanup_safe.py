#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the AS1455 cleaner while preserving its external-process guard.

The shell maintenance wrapper necessarily contains ``as1455`` in its command
line.  Bash process substitution may also create another process carrying the
same command line.  The underlying cleaner would therefore mistake its own
maintenance process family for an external AS1455 job.

This launcher excludes only the current maintenance wrapper and its descendants.
Unrelated training, backtest, data-refresh, live, or maintenance processes still
block ``--apply``.
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
MAINTENANCE_MARKER = "run_as1455_storage_maintenance.sh"


def process_snapshot() -> dict[int, tuple[int, str]]:
    rows: dict[int, tuple[int, str]] = {}
    proc = Path("/proc")
    if not proc.exists():
        return rows
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            ppid = 0
            for line in (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split(":", 1)[1].strip())
                    break
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        rows[pid] = (ppid, command)
    return rows


def current_ancestor_chain(snapshot: dict[int, tuple[int, str]]) -> set[int]:
    chain = {os.getpid()}
    pid = os.getpid()
    while pid in snapshot:
        parent = int(snapshot[pid][0])
        if parent <= 0 or parent in chain:
            break
        chain.add(parent)
        pid = parent
    return chain


def maintenance_family_pids(snapshot: dict[int, tuple[int, str]]) -> set[int]:
    chain = current_ancestor_chain(snapshot)
    roots = {
        pid
        for pid in chain
        if MAINTENANCE_MARKER in snapshot.get(pid, (0, ""))[1]
    }
    ignored = set(chain)
    if not roots:
        return ignored

    for candidate in snapshot:
        pid = candidate
        seen: set[int] = set()
        while pid > 0 and pid not in seen:
            if pid in roots:
                ignored.add(candidate)
                break
            seen.add(pid)
            pid = int(snapshot.get(pid, (0, ""))[0])
    return ignored


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
    snapshot = process_snapshot()
    ignored = maintenance_family_pids(snapshot)

    def external_active_as1455_processes() -> list[dict[str, Any]]:
        return [
            row
            for row in original()
            if int(row.get("pid", -1)) not in ignored
        ]

    ignored_as1455 = sorted(
        pid
        for pid in ignored
        if "as1455" in snapshot.get(pid, (0, ""))[1].lower()
    )
    print(
        "[PROCESS GUARD] ignored_current_maintenance_pids="
        f"{ignored_as1455}; unrelated AS1455 jobs still block apply"
    )
    cleaner.active_as1455_processes = external_active_as1455_processes
    cleaner.main()


if __name__ == "__main__":
    main()
