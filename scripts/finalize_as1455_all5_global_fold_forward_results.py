#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize the global-fold fixed top-five ensemble experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import finalize_as1455_global_fold_forward_results as base  # noqa: E402

FIXED_SIGNAL_SPEC = "ensemble_all5_mean:0,1,2,3,4:mean"


def argument_value(flag: str) -> str | None:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def main() -> None:
    out_root_value = argument_value("--out-root")
    base.FIXED_SIGNAL_SPEC = FIXED_SIGNAL_SPEC
    base.main()
    if not out_root_value:
        return
    manifest_file = (
        Path(out_root_value).expanduser().resolve()
        / "global_fold0_to_fold5_forward_manifest.json"
    )
    if not manifest_file.exists():
        return
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    payload["protocol"] = (
        "global_fold0_to_fold5_fixed_all5_ensemble_then_strict_forward"
    )
    payload["fixed_signal_spec"] = FIXED_SIGNAL_SPEC
    payload["fixed_signal_kind"] = "all5"
    manifest_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
