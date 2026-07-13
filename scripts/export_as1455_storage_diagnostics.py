#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export bounded AS1455 storage diagnostics to one text file.

The report is designed to be copied into a support conversation. It avoids
reading large model payloads into memory and limits directory/file listings so
that the output remains reviewable.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


GIB = 1024 ** 3
MIB = 1024 ** 2


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def run_command(command: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return int(completed.returncode), completed.stdout.rstrip()
    except FileNotFoundError as exc:
        return 127, f"command unavailable: {exc}"
    except Exception as exc:  # diagnostics must continue after one failed probe
        return 1, f"{type(exc).__name__}: {exc}"


def write_section(handle, title: str, body: str | Iterable[str]) -> None:
    handle.write(f"\n===== {title} =====\n")
    if isinstance(body, str):
        text = body
    else:
        text = "\n".join(str(line) for line in body)
    handle.write(text.rstrip() + "\n")


def active_as1455_processes() -> list[str]:
    rows: list[str] = []
    proc = Path("/proc")
    if not proc.exists():
        return rows
    current = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current:
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = command.lower()
        if "as1455" in lowered and "export_as1455_storage_diagnostics.py" not in lowered:
            rows.append(f"pid={entry.name} cmd={command}")
    return sorted(rows)


def scan_files(base: Path, top_n: int) -> dict[str, object]:
    top: list[tuple[int, str]] = []
    suffix_bytes: defaultdict[str, int] = defaultdict(int)
    suffix_count: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    permission_errors: list[str] = []

    for root, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [name for name in dirs if name != ".git"]
        for filename in files:
            path = Path(root) / filename
            try:
                stat = path.lstat()
            except (FileNotFoundError, PermissionError) as exc:
                if len(permission_errors) < 20:
                    permission_errors.append(f"{path}: {type(exc).__name__}")
                continue
            size = int(stat.st_size)
            total_files += 1
            total_bytes += size
            suffix = path.suffix.lower() or "<no_suffix>"
            suffix_bytes[suffix] += size
            suffix_count[suffix] += 1
            item = (size, str(path))
            if len(top) < top_n:
                heapq.heappush(top, item)
            elif item > top[0]:
                heapq.heapreplace(top, item)

    suffix_rows = sorted(
        (
            (size, suffix, int(suffix_count[suffix]))
            for suffix, size in suffix_bytes.items()
        ),
        reverse=True,
    )
    return {
        "top_files": sorted(top, reverse=True),
        "suffix_rows": suffix_rows,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "scan_errors": permission_errors,
    }


def important_paths(base: Path) -> list[str]:
    relative_paths = [
        "ch12_as1455/model_data_as1455.h5",
        "ch12_as1455/model_data_contract.json",
        "ch12_as1455/as1455_ohlcv_adj.h5",
        "ch12_as1455/baostock_5m_cache",
        "ch12_as1455/baostock_raw_daily_cache",
        "ch12_as1455/as1455_daily_cache",
        "ch12_as1455_forward_latest/model_data_as1455.h5",
        "ch12_as1455_forward_latest/as1455_ohlcv_raw.h5",
        "ch12_as1455_forward_latest/as1455_ohlcv_adj.h5",
        "ch12_as1455_forward_latest/as1455_execution_metadata.h5",
        "ch17_as1455_target_search",
        "ch17_as1455_target_backtest",
        "ch17_as1455_fold0_forward_backtest",
        "ch17_as1455_backtest_plots",
        "live_as1455",
    ]
    rows: list[str] = []
    for relative in relative_paths:
        path = base / relative
        if not path.exists():
            rows.append(f"MISSING\t{relative}")
            continue
        kind = "dir" if path.is_dir() else "file"
        size = path.stat().st_size if path.is_file() else 0
        rows.append(f"{kind}\t{human_bytes(size)}\t{relative}")
    return rows


def latest_json_summaries(base: Path, pattern: str, limit: int = 5) -> list[str]:
    rows: list[str] = []
    paths = sorted(base.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = {
                key: payload.get(key)
                for key in (
                    "created_at",
                    "mode",
                    "estimated_delete_gb",
                    "disk_before",
                    "disk_after",
                )
                if key in payload
            }
            rows.append(f"{path}: {json.dumps(summary, ensure_ascii=False)}")
        except Exception as exc:
            rows.append(f"{path}: unreadable {type(exc).__name__}: {exc}")
    return rows or ["none"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AS1455 storage diagnostics")
    parser.add_argument("--base", default="saved_data/ashare_ml4t")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-files", type=int, default=80)
    parser.add_argument("--du-depth", type=int, default=2)
    parser.add_argument("--du-lines", type=int, default=160)
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    usage_probe = base if base.exists() else base.parent
    usage = shutil.disk_usage(usage_probe)
    scan = scan_files(base, max(1, int(args.top_files))) if base.exists() else {
        "top_files": [],
        "suffix_rows": [],
        "total_files": 0,
        "total_bytes": 0,
        "scan_errors": [f"base does not exist: {base}"],
    }

    with out.open("w", encoding="utf-8") as handle:
        write_section(
            handle,
            "REPORT METADATA",
            [
                f"created_at={datetime.now().isoformat(timespec='seconds')}",
                f"hostname={socket.gethostname()}",
                f"platform={platform.platform()}",
                f"python={sys.version.replace(os.linesep, ' ')}",
                f"project_dir={PROJECT_DIR}",
                f"base={base}",
                f"report={out}",
            ],
        )

        for title, command in (
            ("GIT STATUS", ["git", "status", "--short", "--branch"]),
            ("GIT HEAD", ["git", "log", "-1", "--oneline", "--decorate"]),
            ("FILESYSTEM", ["df", "-h", str(usage_probe)]),
            ("INODES", ["df", "-i", str(usage_probe)]),
            ("MEMORY", ["free", "-h"]),
        ):
            code, output = run_command(command, cwd=PROJECT_DIR)
            write_section(handle, title, f"exit_code={code}\n{output}")

        write_section(
            handle,
            "DISK SNAPSHOT",
            [
                f"total={human_bytes(usage.total)}",
                f"used={human_bytes(usage.used)}",
                f"free={human_bytes(usage.free)}",
                f"scanned_files={scan['total_files']}",
                f"scanned_logical_bytes={human_bytes(int(scan['total_bytes']))}",
            ],
        )

        processes = active_as1455_processes()
        write_section(handle, "ACTIVE AS1455 PROCESSES", processes or ["none"])

        if base.exists():
            code, output = run_command(
                [
                    "du",
                    "-x",
                    "-B1",
                    f"--max-depth={max(0, int(args.du_depth))}",
                    str(base),
                ]
            )
            du_rows: list[tuple[int, str]] = []
            for line in output.splitlines():
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                try:
                    du_rows.append((int(parts[0]), parts[1]))
                except ValueError:
                    continue
            du_rows.sort(reverse=True)
            formatted = [
                f"{human_bytes(size):>12}\t{path}"
                for size, path in du_rows[: max(1, int(args.du_lines))]
            ]
            if code != 0:
                formatted.insert(0, f"du_exit_code={code}")
            write_section(handle, "LARGEST DIRECTORIES", formatted or [output or "none"])

        write_section(handle, "IMPORTANT PATHS", important_paths(base))
        write_section(
            handle,
            "LARGEST FILES",
            [
                f"{human_bytes(size):>12}\t{path}"
                for size, path in scan["top_files"]
            ] or ["none"],
        )
        write_section(
            handle,
            "FILE TYPES BY LOGICAL SIZE",
            [
                f"{human_bytes(size):>12}\tcount={count:>8}\t{suffix}"
                for size, suffix, count in scan["suffix_rows"]
            ] or ["none"],
        )
        write_section(handle, "SCAN ERRORS", scan["scan_errors"] or ["none"])
        write_section(
            handle,
            "RECENT CLEANUP AUDITS",
            latest_json_summaries(base, "cleanup_audit_*.json"),
        )

    print(f"[OK] AS1455 storage diagnostics: {out}")


if __name__ == "__main__":
    main()
