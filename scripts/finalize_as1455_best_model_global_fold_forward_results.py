#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize the global fold0..5 best-model experiment via the shared finalizer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import finalize_as1455_global_fold_forward_results as shared  # noqa: E402

FIXED_SIGNAL_SPEC = "model_0:0:single"
PROTOCOL = "global_fold0_to_fold5_fixed_best_model_then_strict_forward"


def argument_value(name: str) -> str | None:
    for index, token in enumerate(sys.argv[1:]):
        if token == name:
            position = index + 2
            if position >= len(sys.argv):
                raise SystemExit(f"{name} requires a value")
            return sys.argv[position]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def main() -> None:
    shared.FIXED_SIGNAL_SPEC = FIXED_SIGNAL_SPEC
    shared.main()

    out_root_value = argument_value("--out-root")
    if not out_root_value:
        raise SystemExit("--out-root is required")
    out_root = Path(out_root_value).expanduser().resolve()
    manifest_file = out_root / "global_fold0_to_fold5_forward_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["protocol"] = PROTOCOL
    manifest["fixed_signal_spec"] = FIXED_SIGNAL_SPEC
    manifest["fixed_model_semantics"] = (
        "prediction column 0 is each source fold's highest-ranked saved checkpoint"
    )
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
