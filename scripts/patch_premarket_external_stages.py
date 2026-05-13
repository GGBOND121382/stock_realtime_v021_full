#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch pipelines/run_premarket_history_update.py so external stages use
external_<profile> instead of generic external.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


NEW_FUNC_BODY = """def only_stages(self) -> list[str]:
        stages = ["update_data", "samples"]

        feature_pipeline = getattr(self, "feature_pipeline", None)
        if isinstance(feature_pipeline, str):
            feature_parts = {x.strip() for x in feature_pipeline.split(",") if x.strip()}
        elif feature_pipeline:
            feature_parts = {str(x).strip() for x in feature_pipeline if str(x).strip()}
        else:
            feature_parts = set()

        if (not feature_parts) or ("fundamental" in feature_parts):
            stages.append("fundamental")
        if (not feature_parts) or ("sector" in feature_parts):
            stages.append("sector")

        external = getattr(self, "external", None)
        if isinstance(external, str):
            external_items = [x.strip() for x in external.split(",") if x.strip()]
        elif external:
            external_items = [str(x).strip() for x in external if str(x).strip()]
        else:
            external_items = []

        for ext in external_items:
            stage_name = ext if ext.startswith("external_") else f"external_{ext}"
            if stage_name not in stages:
                stages.append(stage_name)

        return stages
"""


def patch_text(text: str) -> tuple[str, bool, str]:
    if 'stage_name = ext if ext.startswith("external_") else f"external_{ext}"' in text:
        return text, False, "already_patched"

    pattern = re.compile(
        r'(?ms)^    def only_stages\(self\)[^\n]*:\n'
        r'(?:^        .*\n|^\s*$)*?'
        r'(?=^    def |\nclass |\Z)'
    )

    match = pattern.search(text)
    if not match:
        return text, False, "only_stages_not_found"

    old_block = match.group(0)
    if "external" not in old_block or "stages" not in old_block:
        return text, False, "only_stages_unrecognized"

    new_block = "    " + NEW_FUNC_BODY
    patched = text[:match.start()] + new_block + text[match.end():]
    return patched, True, "patched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pipelines/run_premarket_history_update.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"[ERROR] target not found: {target}")

    text = target.read_text(encoding="utf-8")
    patched, changed, status = patch_text(text)

    print(f"[TARGET] {target}")
    print(f"[STATUS] {status}")

    if not changed:
        if status == "already_patched":
            print("[OK] No change needed.")
            return 0
        raise SystemExit(f"[ERROR] patch not applied: {status}")

    if args.dry_run:
        print("[DRY-RUN] Patch would be applied.")
        return 0

    backup = target.with_suffix(target.suffix + ".bak_premarket_external_stage")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")
    print(f"[BACKUP] {backup}")
    print("[OK] patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
