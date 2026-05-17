#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def patch_adapter() -> None:
    path = Path("portfolio_decision/portfolio_confirm_from_buy_signals.py")
    if not path.exists():
        print(f"[SKIP] {path}")
        return
    txt = path.read_text(encoding="utf-8")
    old_txt = txt

    marker = '        sector = r.get("sector", r.get("industry", "UNKNOWN"))\n\n        sig_rows.append({\n'
    insert = '''        sector = r.get("sector", r.get("industry", "UNKNOWN"))

        entry_policy = str(r.get("entry_policy", meta.get("entry_policy", "")) or "").strip()
        entry_vwap_premium_bps = as_float(
            r.get("entry_vwap_premium_bps", meta.get("entry_vwap_premium_bps", 50.0)),
            50.0,
        )
        samples = str(
            r.get("samples", r.get("sample_file", meta.get("samples", meta.get("sample_file", "")))) or ""
        ).strip()
        expected_return_col = str(r.get("expected_return_col", "") or "").strip()
        if not expected_return_col:
            lm = str(label_mode).lower()
            expected_return_col = "trade_target_or_close_return" if lm.startswith("hit") or lm == "hit" else "trade_net_close_return"

        sig_rows.append({
'''
    if "expected_return_col = str(r.get(\"expected_return_col\"" not in txt and marker in txt:
        txt = txt.replace(marker, insert, 1)

    marker = '            "metadata_path": str(meta_path) if meta_path else "",\n        })\n'
    insert = '''            "entry_policy": entry_policy,
            "entry_vwap_premium_bps": entry_vwap_premium_bps,
            "samples": samples,
            "expected_return_col": expected_return_col,
            "metadata_path": str(meta_path) if meta_path else "",
        })
'''
    if '"expected_return_col": expected_return_col' not in txt and marker in txt:
        txt = txt.replace(marker, insert, 1)

    marker = '            "entry_policy": meta.get("entry_policy", ""),\n            "sector": sector,\n        })\n'
    insert = '''            "entry_policy": entry_policy,
            "entry_vwap_premium_bps": entry_vwap_premium_bps,
            "samples": samples,
            "expected_return_col": expected_return_col,
            "sector": sector,
        })
'''
    if '"samples": samples' not in txt and marker in txt:
        txt = txt.replace(marker, insert, 1)

    if txt != old_txt:
        path.write_text(txt, encoding="utf-8")
        print(f"[PATCHED] {path}")
    else:
        print(f"[UNCHANGED] {path}")


def patch_optimizer() -> None:
    path = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")
    if not path.exists():
        print(f"[SKIP] {path}")
        return
    txt = path.read_text(encoding="utf-8")
    old_txt = txt

    marker = '    fail_loss_bps: float\n    ev_bps: float\n'
    insert = '''    fail_loss_bps: float
    entry_policy: str
    entry_vwap_premium_bps: float
    samples: str
    expected_return_col: str
    metadata_path: str
    ev_bps: float
'''
    if "expected_return_col: str" not in txt and marker in txt:
        txt = txt.replace(marker, insert, 1)

    marker = '        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)\n        pred_prob = parse_rate_decimal(get_row_field(row, "pred_prob", get_row_field(row, "win_rate", cfg["default_hit_prob"])), cfg["default_hit_prob"])\n'
    insert = '''        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)
        entry_policy = str(get_row_field(row, "entry_policy", "") or "")
        entry_vwap_premium_bps = as_float(get_row_field(row, "entry_vwap_premium_bps", 50.0), 50.0)
        samples = str(get_row_field(row, "samples", "") or "")
        expected_return_col = str(get_row_field(row, "expected_return_col", "") or "")
        metadata_path = str(get_row_field(row, "metadata_path", "") or "")
        pred_prob = parse_rate_decimal(get_row_field(row, "pred_prob", get_row_field(row, "win_rate", cfg["default_hit_prob"])), cfg["default_hit_prob"])
'''
    if "entry_vwap_premium_bps = as_float(get_row_field(row" not in txt and marker in txt:
        txt = txt.replace(marker, insert, 1)

    marker = '            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps, ev_bps=ev_bps,\n            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,\n'
    insert = '''            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps,
            entry_policy=entry_policy, entry_vwap_premium_bps=entry_vwap_premium_bps,
            samples=samples, expected_return_col=expected_return_col, metadata_path=metadata_path,
            ev_bps=ev_bps,
            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,
'''
    if "entry_policy=entry_policy" not in txt and marker in txt:
        txt = txt.replace(marker, insert, 1)

    if txt != old_txt:
        path.write_text(txt, encoding="utf-8")
        print(f"[PATCHED] {path}")
    else:
        print(f"[UNCHANGED] {path}")


def main() -> int:
    patch_adapter()
    patch_optimizer()
    print("[OK] model usage consistency patch applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
