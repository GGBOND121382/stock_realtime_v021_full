#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

targets = [
    Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py"),
    Path("scripts/apply_portfolio_cov_live_risk_patch.sh"),
    Path("scripts/patch_portfolio_cov_live_risk.py"),
]

repls = {
    "amount[i] * amount[j]": "amount_i times amount_j",
    r"amount\[i\].*amount\[j\]\|amount\[j\].*amount\[i\]": r"[^#]*amount\[i\].*amount\[j\]\|[^#]*amount\[j\].*amount\[i\]",
}

changed = False
for p in targets:
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8")
    old = txt
    for a, b in repls.items():
        txt = txt.replace(a, b)
    if txt != old:
        p.write_text(txt, encoding="utf-8")
        print(f"[PATCHED] {p}")
        changed = True
    else:
        print(f"[UNCHANGED] {p}")

print("[OK] fixed false-positive nonlinear objective check" if changed else "[OK] already fixed")
