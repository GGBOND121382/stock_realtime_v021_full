#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path


def replace_default_max_positions(text: str) -> str:
    # Accept either old 3, prior patch 10, or already 7.
    if '"max_positions": 7,' in text:
        return text
    if '"max_positions": 10,' in text:
        return text.replace('"max_positions": 10,', '"max_positions": 7,', 1)
    if '"max_positions": 3,' in text:
        return text.replace('"max_positions": 3,', '"max_positions": 7,', 1)
    raise SystemExit('[ERROR] cannot find DEFAULT_CONFIG "max_positions" in optimizer')


# 1) Optimizer default
optimizer = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")
txt = optimizer.read_text(encoding="utf-8")
txt = replace_default_max_positions(txt)

# Keep account-level single-name cap.
if '"max_policy_weight": 0.15,' not in txt:
    txt = txt.replace(
        '"max_daily_buy_pct_of_cash": 0.70,\n',
        '"max_daily_buy_pct_of_cash": 0.70,\n    "max_policy_weight": 0.15,\n',
        1,
    )

# Ensure max_policy_weight is enforced inside get_tier_and_caps.
old = '''    if model_type == "hit":
        max_weight = min(max_weight, 0.10)
        max_add_weight = min(max_add_weight, 0.08)
    if model_type == "observation":
        max_weight = min(max_weight, 0.05)
        max_add_weight = min(max_add_weight, 0.05)
    return tier, max_weight, max_add_weight
'''
new = '''    max_policy_weight = cfg.get("max_policy_weight")
    if max_policy_weight is not None:
        max_weight = min(max_weight, as_float(max_policy_weight, max_weight))

    if model_type == "hit":
        max_weight = min(max_weight, 0.10)
        max_add_weight = min(max_add_weight, 0.08)
    if model_type == "observation":
        max_weight = min(max_weight, 0.05)
        max_add_weight = min(max_add_weight, 0.05)
    return tier, max_weight, max_add_weight
'''
if 'max_policy_weight = cfg.get("max_policy_weight")' not in txt:
    if old not in txt:
        raise SystemExit("[ERROR] cannot find get_tier_and_caps block to enforce max_policy_weight")
    txt = txt.replace(old, new, 1)

optimizer.write_text(txt, encoding="utf-8")
print("[PATCHED]", optimizer)

# 2) Runtime config
cfg_path = Path("configs/portfolio_confirm_config.json")
if not cfg_path.exists():
    raise SystemExit("[ERROR] missing configs/portfolio_confirm_config.json")

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["max_policy_weight"] = 0.15
cfg["max_positions"] = 7
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("[PATCHED]", cfg_path)
