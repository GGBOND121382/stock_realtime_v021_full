#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail fast before an AS1455 job can fill the server filesystem."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check free disk space for AS1455 jobs")
    parser.add_argument("--path", default="saved_data/ashare_ml4t")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--label", default="as1455-job")
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    gib = 1024 ** 3
    payload = {
        "label": args.label,
        "path": str(path),
        "filesystem_probe": str(probe.resolve()),
        "total_gb": round(usage.total / gib, 3),
        "used_gb": round(usage.used / gib, 3),
        "free_gb": round(usage.free / gib, 3),
        "min_free_gb": float(args.min_free_gb),
        "passed": usage.free >= args.min_free_gb * gib,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit(
            f"[DISK BLOCK] {args.label}: free={payload['free_gb']} GiB "
            f"< required={args.min_free_gb} GiB; clean saved_data before running"
        )


if __name__ == "__main__":
    main()
