#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


BAD_NAMES = ["公用事业", "玻璃玻纤", "基础化工", "工程建设", "煤炭行业", "饮料乳品"]

# Exact THS board-name replacements based on the supplied valid ths_name list.
REPLACEMENTS = {
    # feature_building/build_stock_external_features.py
    'boards=("建筑材料", "玻璃玻纤", "风电设备", "电池")':
        'boards=("建筑材料", "非金属材料", "风电设备", "电池")',
    'boards=("电力", "公用事业")':
        'boards=("电力", "燃气")',
    'boards=("农化制品", "化学制品", "基础化工")':
        'boards=("农化制品", "化学制品", "化学原料")',

    # configs/realtime_context_sources.toml
    'symbols = ["农化制品", "化学制品", "基础化工"]':
        'symbols = ["农化制品", "化学制品", "化学原料"]',
    'symbols = ["电力", "公用事业"]':
        'symbols = ["电力", "燃气"]',

    # old wrapper/readme variants, if present
    "工程建设": "建筑装饰",
    "煤炭行业": "煤炭开采加工",
    "饮料乳品": "饮料制造",
}


def patch_file(path: Path) -> bool:
    if not path.exists():
        return False
    txt = path.read_text(encoding="utf-8")
    old = txt
    for a, b in REPLACEMENTS.items():
        txt = txt.replace(a, b)
    if txt != old:
        path.write_text(txt, encoding="utf-8")
        print(f"[PATCHED] {path}")
        return True
    print(f"[UNCHANGED] {path}")
    return False


def assert_no_bad_names(path: Path) -> None:
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8", errors="ignore")
    hits = [x for x in BAD_NAMES if x in txt]
    if hits:
        raise SystemExit(f"[ERROR] invalid THS board names still present in {path}: {hits}")


def main() -> int:
    targets = [
        Path("feature_building/build_stock_external_features.py"),
        Path("configs/realtime_context_sources.toml"),
        Path("scripts/run_new27_v2_full_pipelines.sh"),
        Path("README_RUN_NEW27_V2_FULL_PIPELINES.md"),
    ]

    any_changed = False
    for p in targets:
        any_changed = patch_file(p) or any_changed

    # Positive checks for the two active files that caused fetch failures.
    ext = Path("feature_building/build_stock_external_features.py")
    if ext.exists():
        txt = ext.read_text(encoding="utf-8")
        required = [
            'boards=("建筑材料", "非金属材料", "风电设备", "电池")',
            'boards=("电力", "燃气")',
            'boards=("农化制品", "化学制品", "化学原料")',
        ]
        for s in required:
            if s not in txt:
                raise SystemExit(f"[ERROR] missing expected board tuple in {ext}: {s}")

    cfg = Path("configs/realtime_context_sources.toml")
    if cfg.exists():
        txt = cfg.read_text(encoding="utf-8")
        required = [
            'symbols = ["农化制品", "化学制品", "化学原料"]',
            'symbols = ["电力", "燃气"]',
        ]
        for s in required:
            if s not in txt:
                raise SystemExit(f"[ERROR] missing expected context symbols in {cfg}: {s}")

    for p in targets:
        assert_no_bad_names(p)

    print("[OK] strict THS board-name patch applied" if any_changed else "[OK] already strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
